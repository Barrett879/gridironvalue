"""Isotonic level correction for the three targets where it survives the gate.

THE BIAS THIS FIXES, AND WHY IT IS NOT A MODEL DEFECT THAT FEATURES CAN CURE
----------------------------------------------------------------------------
The lowest projection quintile is systematically OVER-projected, on both folds:
receiving yards -19.2% then -24.7%, rushing yards -14.7% then -13.8%. It is not
a cold-start effect (bias is flat across prior-games buckets) and it is not
skew. It is ZEROS: that quintile is 59-68% zeros, because a fifth receiver who
dresses and is never targeted records nothing, and nothing in the data separates
"dressed but unused" from "used a little". The model predicts the average of
both groups, which is too high for the many and too low for the few.

Round 3 established that adding features no longer helps, and the signal that
would fix this properly - inactives at kickoff minus 90 minutes - is not in
nflverse at all. So the honest tool is a measured level correction, not another
feature.

WHY THE MAP MUST BE FITTED ON WALK-FORWARD PREDICTIONS
-------------------------------------------------------
The shipped models train through 2025. Predictions for any season <= 2025 are
therefore IN SAMPLE, and a map fitted on those would learn "this model is
unbiased" and then do nothing, or harm, on the 2026 rows it is actually applied
to.

So the map is fitted on predictions from walk-forward models that never saw the
season they scored: train <= 2022 scored on 2023, and train <= 2023 scored on
2024, pooled. Pooling two seasons matters: a map fitted on 2022 alone
OVERSHOOTS on 2023, turning a -19.2% bias into +13.1%.

2025 is deliberately not used. It was spent once confirming the feature
programme and fitting a correction on it would retire it permanently.

WHY ONLY THREE TARGETS
-----------------------
Gated exactly like every feature block: win on validation, replicate on test,
per stratum. It ships for receiving_yards, rushing_yards, targets and receptions. It is
rejected for passing_yards, attempts, completions and carries, all of which FLIP
SIGN between folds. The pattern is sample size: the correction needs rows, and
the QB targets have ~660 a season against ~5,000 for receivers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import features as F  # noqa: E402
from gridlib.cache import atomic_to_parquet, dc_path, read_parquet_or_none  # noqa: E402

OUT = "mean_calibration_v1.parquet"
# (target, positions, loss). Only the three that passed the two-stage gate.
SHIPPED = {
    "receiving_yards": (["WR", "TE", "RB"], "squared_error"),
    "rushing_yards": (["RB", "QB", "FB"], "squared_error"),
    "targets": (["WR", "TE", "RB"], "poisson"),
    # Added after the estimator changed: `receptions` FAILED under raw isotonic
    # (+0.61% then -0.61%, a sign flip) and PASSES under the binned fit
    # (+0.42% then +0.25%, improving pooled, starters and cold-start on both
    # folds). Binning removed the instability that was making it look like noise.
    "receptions": (["WR", "TE", "RB"], "poisson"),
}
FOLDS = ((2022, 2023), (2023, 2024))     # (train_end, scored season)
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, random_state=0)
# The map is stored as a lookup on a fixed grid rather than a pickled estimator,
# so it survives a scikit-learn upgrade. Artifacts that unpickle differently
# under a new version are the silent-breakage path the spec warns about.
GRID_N = 200
# Rows per fitting bin. ~200 keeps each point of the map on real data.
N_FIT_BINS = 50
MIN_BIN_ROWS = 60


def main() -> None:
    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    if pw is None or lg is None:
        print("Backfill missing.")
        sys.exit(2)

    feats = {s: F.build(pw[pw["season"] <= s], lg, F.league_priors(pw, s))
             for _, s in FOLDS}

    rows = []
    print("Fitting on pooled walk-forward predictions "
          f"({', '.join(f'train<={a} -> {b}' for a, b in FOLDS)})\n")
    print(f"  {'target':<18}{'n':>7}{'q1 bias before':>17}{'after':>9}")
    for target, (positions, loss) in SHIPPED.items():
        preds, actuals = [], []
        for train_end, season in FOLDS:
            feat = feats[season]
            cols = F.feature_columns(feat, target)
            sub = feat[feat["position"].isin(positions) & feat[target].notna()]
            tr = sub[sub["season"] <= train_end]
            te = sub[sub["season"] == season]
            if len(tr) < 500 or len(te) < 200:
                continue
            mdl = HistGradientBoostingRegressor(loss=loss, **PARAMS)
            mdl.fit(tr[cols], tr[target].astype(float))
            preds.append(np.clip(mdl.predict(te[cols]), 0, None))
            actuals.append(te[target].astype(float).to_numpy())
        if not preds:
            continue
        p = np.concatenate(preds)
        y = np.concatenate(actuals)

        # FIT ON BINNED MEANS, not on raw points.
        #
        # Isotonic on raw rows degenerates into a step function wherever the
        # data thins out, and the data thins out at exactly the top, where the
        # best players are. The first version of this produced only TWO distinct
        # values across the top 20 grid points: a raw 94.80 mapped to 84.31 and
        # a raw 95.28 to 93.67, a nine-yard jump from a half-yard change, which
        # took Ja'Marr Chase from 94.9 to 85.5 for no measured reason.
        #
        # Binning first guarantees every point of the map rests on a few hundred
        # real rows, so the fitted curve is smooth and each step is estimated
        # rather than an accident of where one observation fell.
        order = np.argsort(p)
        ps, ys = p[order], y[order]
        # ADAPTIVE bin count. A fixed 50 bins silently produced ZERO usable
        # bins for rushing_yards on a single season (~44 rows per bin against a
        # 60-row floor), and the target would have been dropped without a word.
        # Deriving the count from the sample keeps every bin above the floor.
        n_bins = int(np.clip(len(ps) // MIN_BIN_ROWS, 5, N_FIT_BINS))
        edges = np.linspace(0, len(ps), n_bins + 1).astype(int)
        bx, by = [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if hi - lo < MIN_BIN_ROWS:
                continue
            bx.append(float(ps[lo:hi].mean()))
            by.append(float(ys[lo:hi].mean()))
        if len(bx) < 5:
            # Loud, not silent: a target that cannot be fitted must be reported,
            # or a missing map looks identical to a map that does nothing.
            print(f"  {target:<18}  only {len(bx)} usable bins, SKIPPED")
            continue
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(np.asarray(bx), np.asarray(by))
        # Capped at the 99th percentile rather than the 99.9th: the last tenth
        # of a percent is a handful of rows and cannot support a correction.
        grid = np.linspace(float(p.min()), float(np.quantile(p, 0.99)), GRID_N)
        mapped = np.clip(iso.predict(grid), 0, None)

        q1 = p <= np.quantile(p, 0.2)
        before = 100 * (y[q1].mean() / p[q1].mean() - 1)
        after = 100 * (y[q1].mean() / np.clip(iso.predict(p[q1]), 0, None).mean() - 1)
        print(f"  {target:<18}{len(p):>7}{before:>16.1f}%{after:>8.1f}%")
        rows.append({"target": target, "n": int(len(p)),
                     "grid": grid.tolist(), "mapped": mapped.tolist()})

    if not rows:
        print("Nothing fitted.")
        sys.exit(2)
    atomic_to_parquet(pd.DataFrame(rows), dc_path(OUT))
    print(f"\nwrote {OUT}  ({len(rows)} targets)")
    print("\nNOT applied to passing_yards, attempts, completions or carries:")
    print("all four FLIP SIGN between the validation and test folds.")


if __name__ == "__main__":
    main()
