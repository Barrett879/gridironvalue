"""When the model disagrees with a line, how often is it right?

MAE answers "how close is the number". A prop is a different question: given a
posted line, does the model pick the correct side? A model can have mediocre MAE
and still pick well, or excellent MAE and pick badly, because what matters is
which side of ONE number the outcome lands on.

No historical PrizePicks lines exist (their API blocks server-side access and
their terms prohibit automated collection), so the line is PROXIED by the
player's season-to-date per-game mean, which is roughly what a naive book would
post and is exactly mandatory baseline 2. Reading: "when the model disagrees
with a naive line, how often does it win?" A real sharp line is harder to beat
than this proxy, so treat these as an OPTIMISTIC ceiling, not an expectation.

Walk-forward: train <= 2022, scored on 2024.
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
# Ordered as the strategy document ranks them, so the two can be compared.
SPECS = [
    ("receptions", ["WR", "TE", "RB"], "poisson", "WR/TE receptions", "*****"),
    ("carries", ["RB", "QB", "FB"], "poisson", "RB rush attempts", "*****"),
    ("attempts", ["QB"], "poisson", "QB pass attempts", "****+"),
    ("completions", ["QB"], "poisson", "QB completions", "****+"),
    ("receiving_yards", ["WR", "TE", "RB"], "squared_error", "Receiving yards", "****"),
    ("rushing_yards", ["RB", "QB", "FB"], "squared_error", "Rushing yards", "****"),
    ("passing_yards", ["QB"], "squared_error", "Passing yards", "-"),
    ("targets", ["WR", "TE", "RB"], "poisson", "Targets", "-"),
    ("receiving_tds", ["WR", "TE", "RB"], "poisson", "Receiving TD", "**"),
]
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, random_state=0)
# A line is posted at a half-point, so ties cannot happen against a real book.
HALF = 0.5


def main() -> None:
    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    priors = F.league_priors(pw, TEST)
    feat = F.build(pw[pw["season"] <= TEST], lg, priors)

    print(f"\nSide-picking accuracy, train <= {TRAIN_END}, scored on {TEST}")
    print("Line PROXIED by the player's season-to-date mean, set to the nearest")
    print("half point. A real book's line is sharper, so read these as a ceiling.\n")
    print(f"  {'prop':<22}{'doc':>7}{'n':>7}{'model':>8}{'coin':>7}"
          f"{'edge':>7}   verdict")
    rows = []
    for target, positions, loss, label, stars in SPECS:
        sub = feat[feat["position"].isin(positions) & feat[target].notna()]
        tr = sub[sub["season"] <= TRAIN_END]
        te = sub[sub["season"] == TEST].copy()
        if len(tr) < 500 or len(te) < 200:
            continue
        # Per target, matching what ships: the QB models are trained without
        # the positional-defence block. Selecting once for every target would
        # measure a configuration that is not the one being served.
        cols = F.feature_columns(feat, target)
        mdl = HistGradientBoostingRegressor(loss=loss, **PARAMS)
        mdl.fit(tr[cols], tr[target].astype(float))
        te["pred"] = np.clip(mdl.predict(te[cols]), 0, None)
        line_col = f"f_std_{target}"
        if line_col not in te.columns:
            continue
        te["line"] = np.floor(te[line_col].astype(float)) + HALF
        te = te[te["line"].notna() & (te["line"] > 0)]
        # Only rows where the model actually disagrees enough to have a view.
        te = te[(te["pred"] - te["line"]).abs() >= 0.05 * te["line"]]
        if len(te) < 150:
            continue
        actual = te[target].astype(float).to_numpy()
        line = te["line"].to_numpy()
        pred = te["pred"].to_numpy()
        model_over = pred > line
        actual_over = actual > line
        correct = float(np.mean(model_over == actual_over))
        # What a coin flip gets: always pick the more common side.
        base = max(float(np.mean(actual_over)), 1 - float(np.mean(actual_over)))
        rows.append({"prop": label, "doc_stars": stars, "n": len(te),
                     "model_pct": 100 * correct, "base_pct": 100 * base,
                     "edge_pts": 100 * (correct - base)})
        verdict = ("beats the naive line" if correct - base > 0.02
                   else "no better than the line" if correct - base > -0.02
                   else "WORSE than the line")
        print(f"  {label:<22}{stars:>7}{len(te):>7}{100*correct:>7.1f}%"
              f"{100*base:>6.1f}%{100*(correct-base):>+6.1f}   {verdict}")

    Path("docs").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("docs/lean_accuracy.csv", index=False)
    print("\n'coin' is always picking the more common side, which is the bar a")
    print("side-picker must clear to be worth anything at all.")
    print("\nwrote docs/lean_accuracy.csv")


if __name__ == "__main__":
    main()
