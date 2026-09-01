"""Experiment: does depth-chart rank earn a place in the count models?

MOTIVATION
----------
Every feature the models use is derived from a player's OWN prior games, so a
player with no history has an essentially empty feature vector and collapses to
a base rate. That is exactly the case that matters most in practice, because it
is what a mid-season promotion looks like: the starter gets hurt, the chart
promotes the backup, and the backup is someone the model has never seen.

Measured on 2025, RB carries by depth rank and history depth:

    rank 1, 0-2 prior games   12.44 carries      rank 1, 3-16 games   13.76
    rank 2, 0-2 prior games    5.24              rank 2, 3-16 games    6.95
    rank 3, 0-2 prior games    2.61              rank 3, 3-16 games    2.40

Rank predicts volume almost independently of history. A rank-1 back the model
has never seen still gets 12.4 carries; the model currently projects Seattle's
rank-1 back at 3.08 because it can only see that he has no history.

WHY THIS IS THE INJURY FEATURE
-------------------------------
Teams demote injured players on the depth chart, and they do it BEFORE the
player appears on an injury report. Before the season starts there is no injury
report at all (the current season's file does not publish until week 1), so the
chart is the only availability signal in existence. Adding rank is therefore the
most direct way to make the projections respond to injuries.

THE GATE
--------
Two-stage: win on the validation season, then REPLICATE on a held-out season.
And per-stratum, not pooled: pooled MAE is dominated by the many low-volume
players where any change looks harmless, so the check that matters is whether
rank-1 players get better. A feature that improves the pooled number while
hurting starters is a regression dressed as a win.

Run:
    python scripts/exp_depth_rank.py
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

PROBE = {
    "carries": ["RB", "QB", "FB"],
    "rushing_yards": ["RB", "QB", "FB"],
    "targets": ["WR", "TE", "RB"],
    "receptions": ["WR", "TE", "RB"],
    "receiving_yards": ["WR", "TE", "RB"],
    "attempts": ["QB"],
    "passing_yards": ["QB"],
}

PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, random_state=0)


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def score(pw, lg, fold: int, train_max: int, use_rank: bool) -> dict:
    """MAE per target, plus the strata that actually matter."""
    F.USE_DEPTH_RANK = use_rank
    priors = F.league_priors(pw, fold)
    feat = F.build(pw[pw["season"] <= fold], lg, priors)
    cols = F.feature_columns(feat)

    out: dict[str, float] = {"_n_features": float(len(cols))}
    for target, positions in PROBE.items():
        sub = feat[feat["position"].isin(positions) & feat[target].notna()]
        tr = sub[sub["season"] <= train_max]
        te = sub[sub["season"] == fold]
        if len(tr) < 500 or len(te) < 200:
            continue
        loss = "squared_error" if target.endswith("_yards") else "poisson"
        mdl = HistGradientBoostingRegressor(loss=loss, **PARAMS)
        mdl.fit(tr[cols], tr[target].astype(float))
        pred = np.clip(mdl.predict(te[cols]), 0, None)
        y = te[target].astype(float)
        out[target] = _mae(y, pred)

        # The strata. Pooled MAE hides damage to the players who matter.
        rank = te.get("depth_rank_capped")
        if rank is not None:
            starters = (rank == 1).to_numpy()
            if starters.sum() >= 50:
                out[f"{target}::rank1"] = _mae(y[starters], pred[starters])
        prior = te.get("f_games_prior")
        if prior is not None:
            cold = (prior.fillna(0) < 3).to_numpy()
            if cold.sum() >= 30:
                out[f"{target}::coldstart"] = _mae(y[cold], pred[cold])
    return out


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
        off = score(pw, lg, fold, train_max, use_rank=False)
        on = score(pw, lg, fold, train_max, use_rank=True)
        print(f"  features: {off['_n_features']:.0f} without, "
              f"{on['_n_features']:.0f} with")
        if off["_n_features"] == on["_n_features"]:
            print("  WARNING: identical feature count. The block is not reaching "
                  "the model; suspect the plumbing, not the feature.")
        keys = [k for k in on if not k.startswith("_") and k in off]
        for k in sorted(keys):
            delta = 100.0 * (1 - on[k] / off[k]) if off[k] else np.nan
            results.append({"fold": label, "metric": k, "off": off[k],
                            "on": on[k], "delta_pct": delta})
            mark = "  <-- starters" if k.endswith("::rank1") else \
                   "  <-- no history" if k.endswith("::coldstart") else ""
            print(f"    {k:<32} off {off[k]:8.4f}  on {on[k]:8.4f}"
                  f"  {delta:+6.2f}%{mark}")

    df = pd.DataFrame(results)
    print("\n" + "=" * 72)
    print("TWO-STAGE VERDICT (must win on validation AND replicate on test)")
    print("=" * 72)
    ship, mirage = [], []
    for metric in sorted(df["metric"].unique()):
        v = df[(df.fold == "validation") & (df.metric == metric)]
        t = df[(df.fold == "test") & (df.metric == metric)]
        if v.empty or t.empty:
            continue
        vd, td = float(v.delta_pct.iloc[0]), float(t.delta_pct.iloc[0])
        if vd > 0 and td > 0:
            verdict, bucket = "SHIP", ship
        elif vd > 0:
            verdict, bucket = "MIRAGE (reverses on test)", mirage
        else:
            verdict, bucket = "no", mirage
        bucket.append(metric)
        print(f"  {metric:<32} val {vd:+6.2f}%  test {td:+6.2f}%   {verdict}")

    print(f"\n{len(ship)} metric(s) improve on both folds, {len(mirage)} do not.")
    starters = [m for m in ship if m.endswith("::rank1")]
    cold = [m for m in ship if m.endswith("::coldstart")]
    print(f"  of those, {len(starters)} are STARTER strata and {len(cold)} are "
          "no-history strata")
    print("\nThe no-history strata are the reason this feature exists: they are "
          "\\nthe promoted backups a mid-season injury creates.")

    Path("docs").mkdir(exist_ok=True)
    df.to_csv("docs/exp_depth_rank.csv", index=False)
    print(f"\nwrote docs/exp_depth_rank.csv   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
