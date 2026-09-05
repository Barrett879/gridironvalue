"""Empirical predictive distributions, so the board can say "61% Over".

WHY NOT A DISTRIBUTIONAL ASSUMPTION
-----------------------------------
The obvious move is to assume a family: Poisson for counts, Normal for yards.
Both are wrong here in ways that matter at the line.

  - Counts are OVER-dispersed. Targets and carries cluster with game script, so
    a Poisson tail is too thin and P(Over) on a high line comes out too low.
  - Yards are right-skewed and heteroscedastic. Yards are volume times
    efficiency, and both vary, so a symmetric interval around the mean puts
    mass below zero for small projections and understates the long right tail
    that a single 60-yard catch creates.

So nothing is assumed. This measures the out-of-sample distribution of
actual/projected and stores its quantiles, which captures the skew, the
over-dispersion and the discreteness of small counts for free.

THE RATIO IS BUCKETED BY PROJECTION LEVEL, WHICH IS THE WHOLE POINT
-------------------------------------------------------------------
`validate_uncertainty.py` already reports one standard deviation per target.
That number is real but it is POOLED, and pooling is exactly what makes it
useless at the line: a receiver projected for 2 targets and one projected for 10
do not share an error bar. Bucketing by projected value gives each its own.

FIT AND CHECK ARE DIFFERENT SEASONS
------------------------------------
Ratios are fitted on 2023 (train <= 2022) and the calibration is CHECKED on 2024
(train <= 2023) by `validate_calibration.py`. Fitting and checking on one season
would report the fit, not the calibration. 2025 stays untouched: it was spent
once on the feature programme and is not spent again here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import features as F  # noqa: E402
from gridlib.cache import atomic_to_parquet, dc_path, read_parquet_or_none  # noqa: E402

OUT = "prob_calibration_v1.parquet"
# FOUR fit seasons, each scored by a model that never saw it. One season with
# five bins left the tail quantiles resting on a few dozen rows, and the board
# was overconfident where it mattered most: a claimed 65% landed at 59%. On the
# 2023 development fold, one season -> four cut high-bin overconfidence from
# 5.94 points to 5.10, and five bins -> ten cut it again to 3.73.
FIT_SEASONS = (2020, 2021, 2022, 2023)

SPECS = {
    "attempts": (["QB"], "poisson"),
    "completions": (["QB"], "poisson"),
    "passing_yards": (["QB"], "squared_error"),
    "carries": (["RB", "QB", "FB"], "poisson"),
    "rushing_yards": (["RB", "QB", "FB"], "squared_error"),
    "targets": (["WR", "TE", "RB"], "poisson"),
    "receptions": (["WR", "TE", "RB"], "poisson"),
    "receiving_yards": (["WR", "TE", "RB"], "squared_error"),
}
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, random_state=0)

# Quantile grid of the ratio. Dense enough that interpolation between points is
# smooth, coarse enough that each bin's tail quantiles rest on real rows.
QUANTILES = np.round(np.arange(0.005, 1.0, 0.005), 3)
N_BINS = 10
# Below this the ratio explodes: a player projected for 0.2 targets who catches
# one has a ratio of 5, which says nothing about a player projected for 6.
MIN_PRED = 0.25


def _walk_forward(feats, target, positions, loss, season):
    """Score one season with a model trained strictly BEFORE it."""
    feat = feats[season]
    sub = feat[feat["position"].isin(positions) & feat[target].notna()]
    tr = sub[sub["season"] < season]
    te = sub[sub["season"] == season].copy()
    if len(tr) < 500 or len(te) < 200:
        return None
    cols = F.feature_columns(feat, target)
    mdl = HistGradientBoostingRegressor(loss=loss, **PARAMS)
    mdl.fit(tr[cols], tr[target].astype(float))
    te["pred"] = np.clip(mdl.predict(te[cols]), 0, None)
    # The ratios must describe the projection that is actually SERVED. The level
    # map is applied before serving, so fitting ratios on raw predictions would
    # make the probability layer correct a bias already removed, and P(Over)
    # would be wrong by exactly the size of that correction.
    from gridlib import uncertainty as U
    te["pred"] = U.apply_level(target, te["pred"].to_numpy())
    return te


def fit_target(feats, target: str, positions, loss) -> pd.DataFrame | None:
    frames = [f for f in (_walk_forward(feats, target, positions, loss, s)
                          for s in FIT_SEASONS) if f is not None]
    if not frames:
        return None
    te = pd.concat(frames, ignore_index=True)
    te = te[te["pred"] >= MIN_PRED]
    if len(te) < 400:
        return None

    te["ratio"] = te[target].astype(float) / te["pred"]
    # Quantile bins of the PROJECTION, so each bin holds players the model
    # thinks are comparable, and each gets its own error distribution.
    te["bin"], edges = pd.qcut(te["pred"], N_BINS, labels=False,
                               retbins=True, duplicates="drop")
    rows = []
    for b, g in te.groupby("bin"):
        if len(g) < 50:
            continue
        qs = np.quantile(g["ratio"].to_numpy(), QUANTILES)
        rows.append({
            "target": target, "bin": int(b),
            "lo": float(edges[int(b)]), "hi": float(edges[int(b) + 1]),
            "n": int(len(g)),
            "mean_pred": float(g["pred"].mean()),
            "mean_actual": float(g[target].mean()),
            "quantiles": qs.tolist(),
        })
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    if pw is None or lg is None:
        print("Backfill missing.")
        sys.exit(2)

    feats = {s: F.build(pw[pw["season"] <= s], lg, F.league_priors(pw, s))
             for s in FIT_SEASONS}

    frames = []
    print("Fitting ratio distributions on "
          f"{', '.join(str(x) for x in FIT_SEASONS)} (walk-forward each)\n")
    print(f"  {'target':<18}{'bins':>6}{'n':>8}   ratio spread per bin (p10 / p50 / p90)")
    for target, (positions, loss) in SPECS.items():
        df = fit_target(feats, target, positions, loss)
        if df is None:
            print(f"  {target:<18}  too few rows, skipped")
            continue
        frames.append(df)
        q = np.asarray(df.iloc[0]["quantiles"])
        i10, i50, i90 = (np.abs(QUANTILES - v).argmin() for v in (0.1, 0.5, 0.9))
        print(f"  {target:<18}{len(df):>6}{int(df['n'].sum()):>8}   "
              f"{q[i10]:.2f} / {q[i50]:.2f} / {q[i90]:.2f}  (lowest bin)")

    if not frames:
        print("Nothing fitted.")
        sys.exit(2)
    out = pd.concat(frames, ignore_index=True)
    out.attrs["quantiles"] = QUANTILES.tolist()
    # The grid is stored as a column so the artifact is self-describing: a
    # reader must never have to guess which quantiles the lists correspond to.
    out["q_grid"] = [QUANTILES.tolist()] * len(out)
    atomic_to_parquet(out, dc_path(OUT))
    print(f"\nwrote {OUT}  ({len(out)} target-bins)")


if __name__ == "__main__":
    main()
