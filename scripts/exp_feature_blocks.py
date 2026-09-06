"""Two-stage ablation for several feature blocks at once.

Each block is tested INDEPENDENTLY against the current shipped configuration,
because a block that only helps when paired with another is a much weaker claim
than one that helps on its own, and testing them jointly hides which is which.

THE GATE
--------
Win on the validation season, then REPLICATE on a separate held-out season.
And per stratum, not pooled: pooled MAE is dominated by low-volume players where
any change looks harmless, so a block that improves the pooled number while
hurting starters must be caught. Strata are depth-rank-1 players (the ones a
board is read for) and cold-start players (fewer than three prior games, which is
what a mid-season promotion looks like).

Expect most blocks to fail. Already rejected on this project: recency weighting,
per-position models, extra capacity, longer rolling windows, and the red-zone
block. Shipped: depth rank, partially. That base rate is the point of the gate.

Run:
    python scripts/exp_feature_blocks.py                    # all blocks
    python scripts/exp_feature_blocks.py --blocks marcel,ngs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import features as F  # noqa: E402
from gridlib.cache import dc_path, read_parquet_or_none  # noqa: E402

# fold name -> (season scored, last training season). `holdout` is 2025 and has
# never been used to choose anything. Spend it only to confirm a decision that is
# already made, and only once, or it stops being a holdout.
FOLDS = {
    "validation": (2023, 2022),
    "test": (2024, 2023),
    "holdout": (2025, 2024),
}

# Each block is a flag on gridlib.features. The baseline turns them all OFF and
# each arm turns exactly one ON, so every comparison isolates one change.
BLOCKS = {
    "marcel": "USE_MARCEL",
    "ngs": "USE_NGS",
    "bio": "USE_BIO",
    "posdef": "USE_POSITIONAL_DEF",
    "venue": "USE_VENUE",
    "bioex": "USE_BIO_EXCLUDE",
    "posdefex": ("USE_POSITIONAL_DEF", "USE_POSDEF_EXCLUDE"),
    # The round-2 candidate as one arm: positional defence, kept away from the
    # two QB targets it regressed, plus the bio exclusion. Tested as a package
    # because that is what would actually ship, and because it lets the holdout
    # answer one question instead of three.
    "round2": ("USE_POSITIONAL_DEF", "USE_POSDEF_EXCLUDE", "USE_BIO_EXCLUDE"),
    # Round 3: the last three items of the programme.
    "scheme": "USE_SCHEME",
    "refcrew": "USE_REFEREE",
    "pfradv": "USE_PFR",
    # Round 4.
    "weather": "USE_WEATHER",
    "rolechg": "USE_ROLE_CHANGE",
}

# Blocks that have already passed the gate. `--base` puts them in BOTH the
# baseline and every arm, so a new block is measured against what is actually
# shipped rather than against a stripped-down model it will never run beside.
# Round 1 ran with an empty base; `bio` shipped, so round 2 uses `--base bio`.

# Blocks that REMOVE a feature for one target rather than adding any. Their
# feature count is unchanged at the default target, so the "block added nothing"
# plumbing warning does not apply to them.
EXCLUSION_BLOCKS = {"bioex"}

def _flags_of(block: str) -> tuple[str, ...]:
    """A block sets one flag or several. Normalise to a tuple."""
    v = BLOCKS[block]
    return (v,) if isinstance(v, str) else tuple(v)


PROBE = {
    "targets": ["WR", "TE", "RB"],
    "receptions": ["WR", "TE", "RB"],
    "receiving_yards": ["WR", "TE", "RB"],
    "carries": ["RB", "QB", "FB"],
    "rushing_yards": ["RB", "QB", "FB"],
    "attempts": ["QB"],
    "passing_yards": ["QB"],
}
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, random_state=0)


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def score(pw, lg, fold: int, train_max: int, flags: dict) -> dict:
    for attr, value in flags.items():
        setattr(F, attr, value)
    priors = F.league_priors(pw, fold)
    feat = F.build(pw[pw["season"] <= fold], lg, priors)

    out: dict[str, float] = {"_n_features": float(len(F.feature_columns(feat)))}
    for target, positions in PROBE.items():
        # Per target, because a block may be excluded for one target only.
        cols = F.feature_columns(feat, target)
        sub = feat[feat["position"].isin(positions) & feat[target].notna()]
        tr, te = sub[sub["season"] <= train_max], sub[sub["season"] == fold]
        if len(tr) < 500 or len(te) < 200:
            continue
        loss = "squared_error" if target.endswith("_yards") else "poisson"
        mdl = HistGradientBoostingRegressor(loss=loss, **PARAMS)
        mdl.fit(tr[cols], tr[target].astype(float))
        pred = np.clip(mdl.predict(te[cols]), 0, None)
        y = te[target].astype(float)
        out[target] = _mae(y, pred)

        rank = te.get("depth_rank_capped")
        if rank is not None and (rank == 1).sum() >= 50:
            m = (rank == 1).to_numpy()
            out[f"{target}::rank1"] = _mae(y[m], pred[m])
        prior = te.get("f_games_prior")
        if prior is not None and (prior.fillna(0) < 3).sum() >= 30:
            m = (prior.fillna(0) < 3).to_numpy()
            out[f"{target}::cold"] = _mae(y[m], pred[m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=",".join(BLOCKS))
    ap.add_argument("--base", default="",
                    help="already-shipped blocks, ON in the baseline and in "
                         "every arm (e.g. --base bio)")
    ap.add_argument("--folds", default="validation,test")
    ap.add_argument("--tag", default="",
                    help="suffix for the results CSV, so a later round cannot "
                         "silently overwrite an earlier one")
    args = ap.parse_args()
    blocks = [b.strip() for b in args.blocks.split(",") if b.strip() in BLOCKS]
    base_blocks = [b.strip() for b in args.base.split(",") if b.strip() in BLOCKS]
    blocks = [b for b in blocks if b not in base_blocks]
    fold_names = [f.strip() for f in args.folds.split(",") if f.strip() in FOLDS]
    if len(fold_names) != 2:
        sys.exit("--folds needs exactly two of: " + ", ".join(FOLDS))

    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    if pw is None or lg is None:
        print("Backfill missing.")
        sys.exit(2)

    off = {f: False for b in BLOCKS for f in _flags_of(b)}
    for b in base_blocks:
        for f in _flags_of(b):
            off[f] = True
    if base_blocks:
        print(f"baseline includes shipped blocks: {', '.join(base_blocks)}")
    t0 = time.time()
    rows = []
    for label in fold_names:
        fold, train_max = FOLDS[label]
        print(f"\n{'=' * 70}\n{label.upper()} season {fold} (train <= {train_max})\n{'=' * 70}")
        base = score(pw, lg, fold, train_max, dict(off))
        print(f"  baseline: {base['_n_features']:.0f} features")
        for b in blocks:
            flags = dict(off)
            for f in _flags_of(b):
                flags[f] = True
            arm = score(pw, lg, fold, train_max, flags)
            added = arm["_n_features"] - base["_n_features"]
            print(f"\n  --- {b}  (+{added:.0f} features) ---")
            if added == 0 and b not in EXCLUSION_BLOCKS:
                print("    WARNING: no features added. The block is not reaching "
                      "the model; suspect the plumbing, not the feature.")
            for k in sorted(k for k in arm if not k.startswith("_") and k in base):
                d = 100.0 * (1 - arm[k] / base[k]) if base[k] else np.nan
                rows.append({"fold": label, "block": b, "metric": k,
                             "base": base[k], "arm": arm[k], "delta_pct": d})
                if abs(d) >= 0.30:      # quiet the noise floor
                    tag = ("  <-- starters" if k.endswith("::rank1")
                           else "  <-- no history" if k.endswith("::cold") else "")
                    print(f"    {k:<28} {d:+6.2f}%{tag}")

    df = pd.DataFrame(rows)
    Path("docs").mkdir(exist_ok=True)
    # Tagged, because an untagged path let round 2 silently overwrite round 1's
    # results. The numbers survived only because they were in a receipt.
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = f"docs/exp_feature_blocks{suffix}.csv"
    df.to_csv(out_path, index=False)

    f1, f2 = fold_names
    print(f"\n{'=' * 70}\nTWO-STAGE VERDICT\n{'=' * 70}")
    for b in blocks:
        sub = df[df.block == b]
        piv = sub.pivot_table(index="metric", columns="fold", values="delta_pct")
        if piv.empty or f1 not in piv or f2 not in piv:
            continue

        # Score ONLY the metrics the block actually moved. An exclusion block
        # changes one target, so 18 of 21 metrics are identically zero by
        # construction, and averaging over all of them drags any real effect
        # toward zero. Round 2 called the bio exclusion a MIRAGE on a mean of
        # +0.07% that was 18/21 structural zeros.
        moved = piv[(piv[f1].abs() > 1e-9) | (piv[f2].abs() > 1e-9)]
        if moved.empty:
            print(f"\n  {b.upper()}: moved nothing. Suspect the plumbing.")
            continue

        ship = moved[(moved[f1] > 0) & (moved[f2] > 0)]
        # Replicated regression: negative on BOTH folds. This is the strongest
        # reject signal the gate produces, and a positive mean can hide it, so
        # it is reported separately rather than folded into the average.
        repeat_loss = moved[(moved[f1] < 0) & (moved[f2] < 0)]
        m1, m2 = moved[f1].mean(), moved[f2].mean()
        verdict = ("SHIP" if m1 > 0 and m2 > 0 and len(ship) >= 3
                   else f"MIRAGE (wins on {f1}, reverses on {f2})"
                   if m1 > 0 else "REJECT")
        print(f"\n  {b.upper()}: mean {m1:+.2f}% {f1}, {m2:+.2f}% {f2}"
              f"   -> {verdict}")
        print(f"    scored over {len(moved)} of {len(piv)} metrics "
              f"({len(piv) - len(moved)} unchanged by construction)")
        print(f"    improves on both folds: {len(ship)}")
        if len(ship):
            print("      " + ", ".join(sorted(ship.index)[:6]))
        if len(repeat_loss):
            print(f"    REPLICATED REGRESSION (negative on both folds): "
                  f"{', '.join(sorted(repeat_loss.index))}")

    print(f"\nwrote {out_path}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
