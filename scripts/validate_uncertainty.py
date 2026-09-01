"""How wrong is a projected stat line, in units a reader can feel?

Skill against a baseline says the model beats an average. It says nothing about
whether a single number is worth acting on. This measures the second thing:
the distribution of |actual - projected| out of sample, and how often the actual
lands within a band around the projection.

Walk-forward, train <= 2022, scored on 2024, which the model never saw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import features as F  # noqa: E402
from gridlib.cache import dc_path, read_parquet_or_none  # noqa: E402

TRAIN_END, TEST = 2022, 2024
SPECS = {
    "attempts": (["QB"], "poisson"),
    "passing_yards": (["QB"], "squared_error"),
    "completions": (["QB"], "poisson"),
    "carries": (["RB", "QB", "FB"], "poisson"),
    "rushing_yards": (["RB", "QB", "FB"], "squared_error"),
    "targets": (["WR", "TE", "RB"], "poisson"),
    "receptions": (["WR", "TE", "RB"], "poisson"),
    "receiving_yards": (["WR", "TE", "RB"], "squared_error"),
    "passing_tds": (["QB"], "poisson"),
    "receiving_tds": (["WR", "TE", "RB"], "poisson"),
}
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, random_state=0)


def main() -> None:
    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    priors = F.league_priors(pw, TEST)
    feat = F.build(pw[pw["season"] <= TEST], lg, priors)
    cols = F.feature_columns(feat)

    print(f"\nOut-of-sample error, train <= {TRAIN_END}, scored on {TEST}")
    print("Restricted to players with a real role (projection above the "
          "position's 60th percentile), because\nthe error on a 5th-string "
          "receiver projected at 0.3 targets is not what anyone is asking about.\n")
    print(f"  {'target':<18}{'n':>6}{'mean':>7}{'MAE':>8}{'MAE%':>7}"
          f"{'p50err':>8}{'p90err':>8}{'within20%':>11}{'within1sd':>10}")
    rows = []
    for target, (positions, loss) in SPECS.items():
        sub = feat[feat["position"].isin(positions) & feat[target].notna()]
        tr = sub[sub["season"] <= TRAIN_END]
        te = sub[sub["season"] == TEST]
        if len(tr) < 500 or len(te) < 200:
            continue
        mdl = HistGradientBoostingRegressor(loss=loss, **PARAMS)
        mdl.fit(tr[cols], tr[target].astype(float))
        pred = np.clip(mdl.predict(te[cols]), 0, None)
        y = te[target].astype(float).to_numpy()

        # Only players the model gives a real role. A board is read for these.
        cut = np.quantile(pred, 0.60)
        m = pred >= cut
        if m.sum() < 100:
            continue
        yy, pp = y[m], pred[m]
        err = np.abs(yy - pp)
        mean = yy.mean()
        within20 = float(np.mean(err <= 0.20 * np.maximum(pp, 1e-9)))
        sd = float(np.std(yy - pp))
        within1sd = float(np.mean(err <= sd))
        rows.append({"target": target, "n": int(m.sum()), "mean": mean,
                     "mae": err.mean(), "mae_pct": 100 * err.mean() / mean,
                     "p50": np.quantile(err, 0.5), "p90": np.quantile(err, 0.9),
                     "within20": 100 * within20, "within1sd": 100 * within1sd,
                     "sd": sd})
        print(f"  {target:<18}{int(m.sum()):>6}{mean:>7.1f}{err.mean():>8.2f}"
              f"{100*err.mean()/mean:>6.0f}%{np.quantile(err,0.5):>8.2f}"
              f"{np.quantile(err,0.9):>8.2f}{100*within20:>10.0f}%"
              f"{100*within1sd:>9.0f}%")

    df = pd.DataFrame(rows)
    Path("docs").mkdir(exist_ok=True)
    df.to_csv("docs/uncertainty.csv", index=False)
    print("\nMAE% is the mean absolute error as a share of the mean outcome.")
    print("within20% is how often the actual lands within a fifth of the "
          "projection.")
    print("\nwrote docs/uncertainty.csv")


if __name__ == "__main__":
    main()
