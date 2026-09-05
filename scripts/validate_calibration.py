"""Does "61% Over" actually happen 61% of the time?

Side-picking accuracy says the probability ORDERS outcomes correctly. It says
nothing about whether the NUMBER is honest, and the number is what the board
shows. A model that says 61% and is right 52% of the time is well ordered and
badly calibrated, and showing that number would be a lie with a decimal point.

So this bins the predicted probabilities and compares each bin's claim to what
happened. Fitted on 2023, scored on 2024, which the calibration never saw.

Reported as expected calibration error: the average gap between claim and
reality, weighted by how many rows sit in each bin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import features as F, uncertainty as U  # noqa: E402
from gridlib.cache import dc_path, read_parquet_or_none  # noqa: E402

TRAIN_END, TEST = 2022, 2024
SPECS = [
    ("receptions", ["WR", "TE", "RB"], "poisson"),
    ("targets", ["WR", "TE", "RB"], "poisson"),
    ("carries", ["RB", "QB", "FB"], "poisson"),
    ("attempts", ["QB"], "poisson"),
    ("completions", ["QB"], "poisson"),
    ("receiving_yards", ["WR", "TE", "RB"], "squared_error"),
    ("rushing_yards", ["RB", "QB", "FB"], "squared_error"),
    ("passing_yards", ["QB"], "squared_error"),
]
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, random_state=0)
EDGES = np.array([0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0])


def main() -> None:
    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    if not U.available():
        print("Run scripts/build_prob_calibration.py first.")
        sys.exit(2)
    feat = F.build(pw[pw["season"] <= TEST], lg, F.league_priors(pw, TEST))

    all_p, all_y = [], []
    print(f"\nCalibration, fitted on 2023, scored on {TEST}\n")
    print(f"  {'target':<18}{'n':>7}{'ECE':>8}   worst bin")
    for target, positions, loss in SPECS:
        cols = F.feature_columns(feat, target)
        sub = feat[feat["position"].isin(positions) & feat[target].notna()]
        tr, te = sub[sub["season"] <= TRAIN_END], sub[sub["season"] == TEST].copy()
        if len(tr) < 500 or len(te) < 200:
            continue
        mdl = HistGradientBoostingRegressor(loss=loss, **PARAMS)
        mdl.fit(tr[cols], tr[target].astype(float))
        te["pred"] = np.clip(mdl.predict(te[cols]), 0, None)
        # Serving applies the level map, so measuring raw predictions here would
        # calibrate a pipeline that is not the one shipped.
        te["pred"] = U.apply_level(target, te["pred"].to_numpy())
        lc = f"f_std_{target}"
        if lc not in te.columns:
            continue
        te["line"] = np.floor(te[lc].astype(float)) + 0.5
        te = te[te["line"].notna() & (te["line"] > 0)]
        te["p"] = U.prob_series([target] * len(te), te["pred"], te["line"])
        te = te[te["p"].notna()]
        if len(te) < 200:
            continue
        y = (te[target].astype(float) > te["line"]).to_numpy().astype(float)
        p = te["p"].to_numpy()
        all_p.append(p)
        all_y.append(y)

        idx = np.digitize(p, EDGES) - 1
        ece, worst, worst_gap = 0.0, "", 0.0
        for b in range(len(EDGES) - 1):
            m = idx == b
            if m.sum() < 30:
                continue
            claim, real = p[m].mean(), y[m].mean()
            ece += (m.sum() / len(p)) * abs(claim - real)
            if abs(claim - real) > worst_gap:
                worst_gap = abs(claim - real)
                worst = (f"{100*EDGES[b]:.0f}-{100*EDGES[b+1]:.0f}%: "
                         f"claimed {100*claim:.0f}, actual {100*real:.0f} "
                         f"(n={int(m.sum())})")
        print(f"  {target:<18}{len(te):>7}{100*ece:>7.1f}%   {worst}")

    p = np.concatenate(all_p)
    y = np.concatenate(all_y)
    print(f"\n  Pooled reliability curve ({len(p)} rows)")
    print(f"  {'claimed':>12}{'actual':>10}{'n':>8}")
    rows = []
    idx = np.digitize(p, EDGES) - 1
    ece = 0.0
    for b in range(len(EDGES) - 1):
        m = idx == b
        if m.sum() < 30:
            continue
        claim, real = p[m].mean(), y[m].mean()
        ece += (m.sum() / len(p)) * abs(claim - real)
        rows.append({"bin_lo": EDGES[b], "bin_hi": EDGES[b + 1],
                     "claimed": round(100 * claim, 1),
                     "actual": round(100 * real, 1), "n": int(m.sum())})
        print(f"  {100*claim:>11.1f}%{100*real:>9.1f}%{int(m.sum()):>8}")
    print(f"\n  Expected calibration error: {100*ece:.1f} points")
    print("  Under 5 points is usable for a board; the claim and the outcome")
    print("  then agree to within a rounding of what a reader can act on.")
    Path("docs").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("docs/calibration.csv", index=False)
    print("\nwrote docs/calibration.csv")


if __name__ == "__main__":
    main()
