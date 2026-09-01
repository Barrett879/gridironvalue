"""Experiment: does prior red-zone opportunity improve touchdown projections?

MOTIVATION
----------
Receiving and rushing touchdowns were the only two targets to FAIL the ship gate
in the first validation run: both lost to a plain season-to-date per-game mean
(receiving TDs -2.8%, rushing TDs -3.4% against baseline 2).

That is the expected result for a model fed prior TD counts, because TD rate is
near-irreducibly noisy year to year (published stability 0.0155 for WR TD rate).
But the feature set was missing the signal that should actually carry TDs: WHERE
the prior opportunity happened. Six carries inside the five is a completely
different proposition from six carries at midfield, and the backfill already
stores carries and targets by yard-line band.

This ablates that block: `gridlib.features.REDZONE_STATS`, added as prior-game
history exactly like the volume block.

TWO-STAGE GATING
----------------
Winning on the validation season is not enough. The block must REPLICATE on a
separate held-out season before it ships. The MLB build shipped two false
confirmations from a harness whose confirmation run loaded thinner history than
the validator, so here BOTH stages call the identical build path with identical
history depth, and only the fold season differs.

If the confirmation shows EXACTLY zero change, suspect the plumbing rather than
the feature: that was the signature of the MLB build's grant-versus-veto column
bug, which produced byte-identical models and 0.00% deltas everywhere.

Run:
    python scripts/exp_redzone_block.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import features as F  # noqa: E402
from gridlib.cache import dc_path, read_parquet_or_none  # noqa: E402

VALID, TEST = 2023, 2024
TRAIN_END_VALID, TRAIN_END_TEST = 2022, 2023

# The TD targets that failed the gate, plus the volume targets they depend on,
# so a block that helps TDs but hurts volume is visible rather than hidden.
PROBE = {
    "receiving_tds": ["WR", "TE", "RB"],
    "rushing_tds": ["RB", "QB", "FB"],
    "passing_tds": ["QB"],
    "targets": ["WR", "TE", "RB"],
    "carries": ["RB", "QB", "FB"],
}

PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, random_state=0)


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def score(pw, lg, fold: int, train_max: int, use_block: bool) -> dict[str, float]:
    """MAE per probe target for one fold, with the block on or off.

    Both branches call the same build path with the same history depth; only the
    flag differs. That is the discipline that stops a false confirmation.
    """
    F.USE_REDZONE_BLOCK = use_block
    priors = F.league_priors(pw, fold)
    feat = F.build(pw[pw["season"] <= fold], lg, priors)
    cols = F.feature_columns(feat)

    out = {}
    for target, positions in PROBE.items():
        sub = feat[feat["position"].isin(positions) & feat[target].notna()]
        tr = sub[sub["season"] <= train_max]
        te = sub[sub["season"] == fold]
        if len(tr) < 500 or len(te) < 200:
            continue
        loss = "squared_error" if target.endswith("_yards") else "poisson"
        mdl = HistGradientBoostingRegressor(loss=loss, **PARAMS)
        mdl.fit(tr[cols], tr[target].astype(float))
        out[target] = _mae(te[target].astype(float),
                           np.clip(mdl.predict(te[cols]), 0, None))
    out["_n_features"] = float(len(cols))
    return out


def baseline2(pw, lg, fold: int, target: str, positions: list[str]) -> float:
    """The season-to-date mean the TD targets currently lose to."""
    F.USE_REDZONE_BLOCK = False
    priors = F.league_priors(pw, fold)
    feat = F.build(pw[pw["season"] <= fold], lg, priors)
    sub = feat[feat["position"].isin(positions) & feat[target].notna()]
    te = sub[sub["season"] == fold]
    tr = sub[sub["season"] < fold]
    b2 = te[f"f_std_{target}"].astype(float).to_numpy()
    b2 = np.where(np.isnan(b2), float(tr[target].mean()), b2)
    return _mae(te[target].astype(float), b2)


def main() -> None:
    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    if pw is None or lg is None:
        print("Backfill missing.")
        sys.exit(2)

    t0 = time.time()
    results = []
    for label, fold, train_max in (("validation", VALID, TRAIN_END_VALID),
                                   ("test", TEST, TRAIN_END_TEST)):
        print(f"\n{label.upper()} season {fold} (train <= {train_max})")
        off = score(pw, lg, fold, train_max, use_block=False)
        on = score(pw, lg, fold, train_max, use_block=True)
        print(f"  features: {off['_n_features']:.0f} without block, "
              f"{on['_n_features']:.0f} with it")
        if off["_n_features"] == on["_n_features"]:
            print("  WARNING: identical feature count. The block is not "
                  "reaching the model; suspect the plumbing, not the feature.")
        for target in PROBE:
            if target not in off or target not in on:
                continue
            delta = 100.0 * (1 - on[target] / off[target])
            results.append({"fold": label, "target": target,
                            "mae_off": off[target], "mae_on": on[target],
                            "delta_pct": delta})
            print(f"    {target:<20} off {off[target]:7.4f}  on {on[target]:7.4f}"
                  f"  {delta:+6.2f}%")

    df = pd.DataFrame(results)
    print("\n" + "=" * 66)
    print("TWO-STAGE VERDICT (a block must win on validation AND replicate)")
    print("=" * 66)
    for target in PROBE:
        v = df[(df.fold == "validation") & (df.target == target)]
        t = df[(df.fold == "test") & (df.target == target)]
        if v.empty or t.empty:
            continue
        vd, td = float(v.delta_pct.iloc[0]), float(t.delta_pct.iloc[0])
        if vd > 0 and td > 0:
            verdict = "SHIP: wins on both"
        elif vd > 0:
            verdict = "MIRAGE: wins on validation, reverses on test"
        elif td > 0:
            verdict = "no: loses on validation"
        else:
            verdict = "no: loses on both"
        print(f"  {target:<20} validation {vd:+6.2f}%  test {td:+6.2f}%   {verdict}")

    print("\nFor reference, the season-to-date baseline the TD targets must beat:")
    for target, positions in (("receiving_tds", ["WR", "TE", "RB"]),
                              ("rushing_tds", ["RB", "QB", "FB"])):
        b2 = baseline2(pw, lg, TEST, target, positions)
        on = df[(df.fold == "test") & (df.target == target)]
        if not on.empty:
            m = float(on.mae_on.iloc[0])
            print(f"  {target:<20} test: model-with-block {m:.4f} vs "
                  f"baseline2 {b2:.4f}  -> "
                  f"{'BEATS baseline' if m < b2 else 'still loses to baseline'}")

    Path("docs").mkdir(exist_ok=True)
    df.to_csv("docs/exp_redzone_block.csv", index=False)
    print(f"\nwrote docs/exp_redzone_block.csv   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
