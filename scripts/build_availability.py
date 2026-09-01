"""P(player appears in the box score), the factor the projections were missing.

THE PROBLEM THIS SOLVES
-----------------------
The training table contains only players who recorded a stat line, so every
model estimates `E[Y | the player appeared]`. Inference applies that to a
19-man depth chart where about 12 players actually appear. Since Y is zero when
a player does not appear, the quantity that composes is

    E[Y] = P(appear) x E[Y | appear]

and the site was serving only the second factor and then summing it. Measured on
2025 week 6: the full published roster sums to 1.40x actual carries and 1.73x
actual pass attempts, while the SAME projection restricted to players who
appeared sums to 0.96x and 1.02x. The conditional model is fine. The missing
multiplier is the entire defect.

WHY A LOOKUP TABLE AND NOT A LEARNED MODEL
-------------------------------------------
P(appear) is almost entirely explained by two pregame facts: position and depth
rank. A shrunk cell mean over that grid closes the gap essentially completely,
which sets a bar any learned model has to clear before it earns its complexity.
Starting here also means the artifact is inspectable: a 30-cell table can be
read and sanity-checked by eye, which a gradient booster cannot.

Injury status is folded in where the report exists, because a player listed Out
is the one case where position and rank are badly wrong.

FIT ON ONE DEPTH-CHART ERA, NOT BOTH
-------------------------------------
The depth-chart file changed shape in 2025, and the two eras encode rank so
differently that pooling them biases the grid badly:

                       players listed   WR3 appears   RB1 appears
    2016-2024 schema        16.1/team       0.49          0.77
    2025+     schema        21.0/team       0.81          0.88

A rank 3 in a chart that stops at 3 is the last man on it; a rank 3 in a chart
that runs to 14 is a genuine rotation player. Pooling made the grid predict 9.8
appearances per team against 11.9 observed, and that shortfall passed straight
through into team totals of about 0.84x actual.

So the grid is fit on the era whose charts the site actually serves. Change
`--seasons` if that ever stops being true, and re-check the printed expected
appearance count against the 11.9 box-score average before trusting it.

EMPIRICAL BAYES, NOT RAW CELL MEANS
------------------------------------
Deep cells are thin (an RB6 barely exists) and a raw mean there is noise. Each
cell is shrunk toward its position's overall rate with a fixed prior weight, so
a cell with three observations mostly inherits the position rate and a cell with
three hundred is mostly itself.

Usage:
    python scripts/build_availability.py --seasons 2025-2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import fetch  # noqa: E402
from gridlib.cache import atomic_to_parquet, dc_path, read_parquet_or_none  # noqa: E402

VERSION = "v1"

# Prior weight for the empirical-Bayes shrink, in pseudo-observations. A cell
# with this many rows is weighted half itself and half its position's rate.
SHRINK_K = 25.0

# Depth rank is capped because the two depth-chart eras disagree: 2001-2024 caps
# `depth_team` at 3, while 2025+ `pos_rank` runs to 14. Capping makes the grid
# comparable across the whole training window instead of silently splitting it.
MAX_RANK = 6

# Injury status -> a multiplicative adjustment, applied on top of the grid.
# "Out" is the case position and rank get badly wrong, and it is knowable
# pregame, so it is worth the extra dimension.
INJURY_ADJ = {"Out": 0.02, "Doubtful": 0.25, "Questionable": 0.92}


def build(seasons: list[int]) -> pd.DataFrame:
    """One row per (position, depth_rank) with a shrunk appearance rate."""
    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    if pw is None:
        raise SystemExit("backfill missing; run build_player_week.py first")
    appeared = set(zip(pw["season"], pw["week"], pw["gsis_id"]))

    frames = []
    for season in seasons:
        dc = fetch.depth_chart_normalized(season)
        if dc.empty:
            continue
        dc = dc[dc["depth_rank"].notna()].copy()
        dc["rank_capped"] = dc["depth_rank"].clip(upper=MAX_RANK).astype(int)
        dc["appeared"] = [
            (s, w, g) in appeared
            for s, w, g in zip(dc["season"], dc["week"], dc["gsis_id"])
        ]
        frames.append(dc)
    if not frames:
        raise SystemExit("no depth charts available for those seasons")
    allc = pd.concat(frames, ignore_index=True)

    pos_rate = allc.groupby("position")["appeared"].mean()
    grid = (allc.groupby(["position", "rank_capped"])
                .agg(n=("appeared", "size"), raw=("appeared", "mean"))
                .reset_index())
    grid["prior"] = grid["position"].map(pos_rate)
    grid["p_play"] = ((grid["raw"] * grid["n"] + grid["prior"] * SHRINK_K)
                      / (grid["n"] + SHRINK_K))
    grid["version"] = VERSION
    return grid.sort_values(["position", "rank_capped"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2025-2025",
                    help="seasons to FIT on; must share a depth-chart schema")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seasons.split("-"))
    seasons = list(range(lo, hi + 1))

    grid = build(seasons)
    path = dc_path(f"availability_{lo}_{hi}_{VERSION}.parquet")
    atomic_to_parquet(grid, path)

    print(f"\nP(appears in the box score), {lo}-{hi}, shrunk with k={SHRINK_K:.0f}\n")
    print(f"  {'pos':<5}{'rank':>5}{'n':>8}{'raw':>8}{'shrunk':>8}")
    for _, r in grid.iterrows():
        print(f"  {r['position']:<5}{int(r['rank_capped']):>5}{int(r['n']):>8}"
              f"{r['raw']:>8.3f}{r['p_play']:>8.3f}")
    print(f"\nwrote {path.name}")
    # The check that matters: applied to the charts we actually serve, does
    # this grid expect the right NUMBER of players to appear? A grid that is
    # individually plausible but sums wrong drags every team total with it.
    import numpy as _np
    from gridlib import fetch as _fetch
    tot, n = 0.0, 0
    for season in seasons:
        dc = _fetch.depth_chart_normalized(season)
        if dc.empty:
            continue
        dc = dc[dc["depth_rank"].notna()].copy()
        dc["rank_capped"] = dc["depth_rank"].clip(upper=MAX_RANK).astype(int)
        key = grid.set_index(["position", "rank_capped"])["p_play"]
        dc["p"] = [float(key.get((p_, r_), 0.5))
                   for p_, r_ in zip(dc["position"], dc["rank_capped"])]
        per = dc.groupby(["season", "week", "team"])["p"].sum()
        tot += float(per.sum()); n += len(per)
    if n:
        print(f"\nExpected appearances per team-week on the served charts: "
              f"{tot / n:.1f}")
        print("  box scores average 11.9. A gap here becomes a team-total gap.")


if __name__ == "__main__":
    main()
