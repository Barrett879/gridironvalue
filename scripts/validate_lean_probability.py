"""Does picking sides by P(Over) beat picking by the gap?

THE HYPOTHESIS THIS TESTS
--------------------------
Yards props have a NEGATIVE measured side-picking edge: receiving -5.4 points,
rushing -7.7, both worse than always taking the more common side. The standing
explanation was the spec's "efficiency trap", which is true but vague.

A sharper mechanism: these stats are RIGHT-SKEWED, so the mean sits above the
median. A line set below the projected mean can still be above the median, and
the model then leans More on an outcome that is under a coin flip. On the real
2026 week 1 board, 84 of 415 lines (20%) have a gap and a probability that
disagree on the side, and 77 of those 84 are "gap says More, probability says
under 50%".

If that is the mechanism, picking by P(Over) should repair the yards props. If
the yards edge stays negative, the skew was not the problem and the honest
conclusion is that these props are unpredictable rather than mis-read.

Same protocol as validate_lean_accuracy.py: train <= 2022, scored on 2024, line
PROXIED by the player's season-to-date mean. The calibration artifact was fitted
on 2023, so it is out of sample for both the training data and the scored season.
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
    ("receptions", ["WR", "TE", "RB"], "poisson", "WR/TE receptions"),
    ("carries", ["RB", "QB", "FB"], "poisson", "RB rush attempts"),
    ("attempts", ["QB"], "poisson", "QB pass attempts"),
    ("completions", ["QB"], "poisson", "QB completions"),
    ("receiving_yards", ["WR", "TE", "RB"], "squared_error", "Receiving yards"),
    ("rushing_yards", ["RB", "QB", "FB"], "squared_error", "Rushing yards"),
    ("passing_yards", ["QB"], "squared_error", "Passing yards"),
    ("targets", ["WR", "TE", "RB"], "poisson", "Targets"),
]
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, random_state=0)
HALF = 0.5


def main() -> None:
    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    if not U.available():
        print("Run scripts/build_prob_calibration.py first.")
        sys.exit(2)
    feat = F.build(pw[pw["season"] <= TEST], lg, F.league_priors(pw, TEST))

    print(f"\nSide-picking: GAP versus P(Over), train <= {TRAIN_END}, scored on {TEST}")
    print("Both columns pick on the SAME rows, so the difference is the rule,")
    print("not the sample. 'coin' is always taking the more common side.\n")
    print(f"  {'prop':<22}{'n':>6}{'coin':>7}{'gap':>8}{'prob':>8}"
          f"{'gapEdge':>9}{'probEdge':>10}{'change':>8}")
    rows = []
    for target, positions, loss, label in SPECS:
        cols = F.feature_columns(feat, target)
        sub = feat[feat["position"].isin(positions) & feat[target].notna()]
        tr, te = sub[sub["season"] <= TRAIN_END], sub[sub["season"] == TEST].copy()
        if len(tr) < 500 or len(te) < 200:
            continue
        mdl = HistGradientBoostingRegressor(loss=loss, **PARAMS)
        mdl.fit(tr[cols], tr[target].astype(float))
        te["pred"] = np.clip(mdl.predict(te[cols]), 0, None)
        # Apply the level map, because serving does. Measuring raw predictions
        # here would score a pipeline that is not the one shipped.
        te["pred"] = U.apply_level(target, te["pred"].to_numpy())

        line_col = f"f_std_{target}"
        if line_col not in te.columns:
            continue
        te["line"] = np.floor(te[line_col].astype(float)) + HALF
        te = te[te["line"].notna() & (te["line"] > 0)]
        te = te[(te["pred"] - te["line"]).abs() >= 0.05 * te["line"]]
        te["p_over"] = U.prob_series([target] * len(te), te["pred"], te["line"])
        te = te[te["p_over"].notna()]
        if len(te) < 150:
            continue

        actual_over = te[target].astype(float).to_numpy() > te["line"].to_numpy()
        gap_over = (te["pred"] > te["line"]).to_numpy()
        prob_over = (te["p_over"] > 0.5).to_numpy()
        base = max(actual_over.mean(), 1 - actual_over.mean())
        gap_acc = float((gap_over == actual_over).mean())
        prob_acc = float((prob_over == actual_over).mean())
        rows.append({"prop": label, "n": len(te),
                     "coin_pct": 100 * base,
                     "gap_pct": 100 * gap_acc, "prob_pct": 100 * prob_acc,
                     "gap_edge": 100 * (gap_acc - base),
                     "prob_edge": 100 * (prob_acc - base),
                     "change": 100 * (prob_acc - gap_acc),
                     "disagreed": int((gap_over != prob_over).sum())})
        print(f"  {label:<22}{len(te):>6}{100*base:>6.1f}%{100*gap_acc:>7.1f}%"
              f"{100*prob_acc:>7.1f}%{100*(gap_acc-base):>+9.1f}"
              f"{100*(prob_acc-base):>+10.1f}{100*(prob_acc-gap_acc):>+8.1f}")

    df = pd.DataFrame(rows)
    Path("docs").mkdir(exist_ok=True)
    df.to_csv("docs/lean_probability.csv", index=False)
    if not df.empty:
        print(f"\n  {'MEAN':<22}{'':>6}{'':>7}{'':>7}{'':>7}"
              f"{df.gap_edge.mean():>+9.1f}{df.prob_edge.mean():>+10.1f}"
              f"{df.change.mean():>+8.1f}")
        yards = df[df.prop.str.contains("yards", case=False)]
        print(f"  {'  yards props only':<22}{'':>6}{'':>7}{'':>7}{'':>7}"
              f"{yards.gap_edge.mean():>+9.1f}{yards.prob_edge.mean():>+10.1f}"
              f"{yards.change.mean():>+8.1f}")
    print("\nwrote docs/lean_probability.csv")


if __name__ == "__main__":
    main()
