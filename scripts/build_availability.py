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

TWO POPULATIONS THAT MUST MATCH, AND DID NOT
---------------------------------------------
A rate is only meaningful against the population it will be applied to, and v1
was fit on a population inference never sees. Two contaminations, both one-way:

  BYE WEEKS. A team's depth chart is published every week including its bye,
  and on a bye nobody can appear. 645 of 12099 fitted rows (5.3%) were bye rows
  with an appearance rate of 0.0016, which pulled every cell down by a factor
  of 0.947. Inference only ever scores teams that are playing, so that
  deflation was pure bias.

  INJURY, COUNTED TWICE. The cell rate was fit over all rows, injured ones
  included, so it was already the UNCONDITIONAL P(appear). Serving then
  multiplied it by INJURY_ADJ a second time. Healthy players inherited a rate
  dragged down by weeks they were hurt, and designated players were penalised
  twice.

Together those compounded to 0.947 x 0.958 = 0.907, and the served board came
in at 0.904 of actual appearances. So the grid is now fit on rows whose team
was PLAYING and who carried NO injury designation, which is exactly the row the
multiplier is applied to.

WHEN THERE IS NO INJURY REPORT
-------------------------------
A healthy-baseline grid over-counts if the report has not been filed yet, since
some of those players will be ruled out on Friday. So a second column,
`p_play_any`, is fit over playing rows REGARDLESS of status: the marginal rate,
correct to serve when there is nothing to condition on. Serving picks per team.

THE MULTIPLIERS ARE MEASURED, NOT ASSUMED
------------------------------------------
v1 hardcoded Questionable at 0.92. Measured against players of the same
position and rank it is 0.78 (95% CI 0.71 to 0.85), so 0.92 was outside the
interval. The naive split is worse still and says 1.01, because Questionable
players skew toward starters and rank has to be controlled for. They are now
estimated from the data and written beside the grid.

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

# v2: fit on playing, non-designated rows. v1 was fit on a population that
# included bye weeks and injured players, and serving then applied the injury
# multiplier on top of a rate that already contained it. The version bump is
# load-bearing: v2 code reading a v1 file would silently serve the old bias.
VERSION = "v2"

# Prior weight for the empirical-Bayes shrink, in pseudo-observations. A cell
# with this many rows is weighted half itself and half its position's rate.
SHRINK_K = 25.0

# Depth rank is capped because the two depth-chart eras disagree: 2001-2024 caps
# `depth_team` at 3, while 2025+ `pos_rank` runs to 14. Capping makes the grid
# comparable across the whole training window instead of silently splitting it.
MAX_RANK = 6

# Floors for the measured multipliers. Out and Doubtful are observed at exactly
# 0 appearances (0/459 and 0/27), and a hard zero asserts more than 459 rows can
# support: by the rule of three the upper 95% bound is 0.007 and 0.111. These
# floors keep the estimate honest without pretending to certainty.
ADJ_FLOOR = {"Out": 0.02, "Doubtful": 0.10, "Questionable": 0.0}
STATUSES = ("Out", "Doubtful", "Questionable")


def _playing_pairs(seasons: list[int]) -> set[tuple[int, int, str]]:
    """(season, week, team) for every team that actually had a game.

    A depth chart is published on a bye week too, and on a bye nobody appears.
    Fitting over those rows is what pulled the whole grid down by 5.3%.
    """
    sch = fetch.load_schedules()
    sch = sch[sch["season"].isin(seasons)]
    pairs: set[tuple[int, int, str]] = set()
    for col in ("home_team", "away_team"):
        pairs |= set(zip(sch["season"].astype(int), sch["week"].astype(int),
                         sch[col]))
    return pairs


def _observed(seasons: list[int]) -> pd.DataFrame:
    """Depth-chart rows for the fit seasons, with `appeared` and injury status.

    One row per (season, week, player, position) as published, restricted to
    teams that were playing that week.
    """
    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    if pw is None:
        raise SystemExit("backfill missing; run build_player_week.py first")
    appeared = set(zip(pw["season"], pw["week"], pw["gsis_id"]))
    playing = _playing_pairs(seasons)

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
        dc["playing"] = [
            (int(s), int(w), t) in playing
            for s, w, t in zip(dc["season"], dc["week"], dc["team"])
        ]
        inj = fetch.load_injuries(season)
        if inj is not None and not inj.empty:
            rep = (inj[["season", "week", "gsis_id", "report_status"]]
                   .dropna(subset=["gsis_id"])
                   .drop_duplicates(["season", "week", "gsis_id"]))
            dc = dc.merge(rep, on=["season", "week", "gsis_id"], how="left")
        else:
            dc["report_status"] = pd.NA
        frames.append(dc)
    if not frames:
        raise SystemExit("no depth charts available for those seasons")

    allc = pd.concat(frames, ignore_index=True)
    n_bye = int((~allc["playing"]).sum())
    if n_bye:
        bye_rate = float(allc.loc[~allc["playing"], "appeared"].mean())
        print(f"  dropping {n_bye} bye-week rows ({100 * n_bye / len(allc):.1f}%"
              f" of the chart), appearance rate {bye_rate:.4f}")
    return allc[allc["playing"]].reset_index(drop=True)


def _shrunk(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Empirical-Bayes cell rates over (position, rank), shrunk to the position."""
    pos_rate = d.groupby("position")["appeared"].mean()
    g = (d.groupby(["position", "rank_capped"])
          .agg(n=("appeared", "size"), raw=("appeared", "mean"))
          .reset_index())
    g["prior"] = g["position"].map(pos_rate)
    g[col] = ((g["raw"] * g["n"] + g["prior"] * SHRINK_K)
              / (g["n"] + SHRINK_K))
    return g


def measure_adjustments(obs: pd.DataFrame) -> pd.DataFrame:
    """The injury multipliers, estimated against same-rank healthy players.

    Rank has to be controlled for. Pooled, Questionable players appear at 0.575
    against a healthy 0.567, which reads as no effect at all; but they skew
    toward starters, and against players of the SAME position and rank the true
    figure is 0.78. The pooled comparison is not a weaker version of this one,
    it points the wrong way.
    """
    healthy = obs[obs["report_status"].isna()]
    base = (healthy.groupby(["position", "rank_capped"])["appeared"]
            .mean().rename("base"))
    rows = []
    for status in STATUSES:
        d = (obs[obs["report_status"] == status]
             .join(base, on=["position", "rank_capped"]).dropna(subset=["base"]))
        floor = ADJ_FLOOR.get(status, 0.0)
        if len(d) < 20 or not d["base"].mean():
            rows.append({"report_status": status, "n": len(d),
                         "measured": np.nan, "multiplier": floor,
                         "source": "floor (too few observations)"})
            continue
        obs_rate = float(d["appeared"].mean())
        measured = obs_rate / float(d["base"].mean())
        rows.append({"report_status": status, "n": len(d),
                     "measured": round(measured, 4),
                     "multiplier": round(max(measured, floor), 4),
                     "source": "measured" if measured > floor else "floor"})
    out = pd.DataFrame(rows)
    out["version"] = VERSION
    return out


def build(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The grid and the injury multipliers.

    Two rate columns, because serving faces two different situations:
      p_play      fit on players with NO injury designation. The baseline the
                  multiplier is applied to, and the only one that composes
                  with it without counting injury twice.
      p_play_any  fit on every playing row regardless of status: the marginal
                  rate. What to serve for a team that has not filed a report
                  yet, where there is nothing to condition on.
    """
    obs = _observed(seasons)
    healthy = obs[obs["report_status"].isna()]

    grid = _shrunk(healthy, "p_play")[
        ["position", "rank_capped", "n", "raw", "prior", "p_play"]]
    marg = _shrunk(obs, "p_play_any")[["position", "rank_capped", "p_play_any"]]
    grid = grid.merge(marg, on=["position", "rank_capped"], how="left")
    # A cell can exist in one population and not the other (every player at
    # that rank happened to be designated). Falling back to the healthy rate is
    # the conservative direction: it never invents availability.
    grid["p_play_any"] = grid["p_play_any"].fillna(grid["p_play"])
    grid["version"] = VERSION
    grid = grid.sort_values(["position", "rank_capped"]).reset_index(drop=True)
    return grid, measure_adjustments(obs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2025-2025",
                    help="seasons to FIT on; must share a depth-chart schema")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seasons.split("-"))
    seasons = list(range(lo, hi + 1))

    print(f"\nfitting on {lo}-{hi}")
    grid, adj = build(seasons)

    print(f"\nP(appears | team playing), {lo}-{hi}, shrunk with k={SHRINK_K:.0f}\n")
    print(f"  {'pos':<5}{'rank':>5}{'n':>8}{'raw':>8}{'healthy':>9}{'any':>8}")
    for _, r in grid.iterrows():
        print(f"  {r['position']:<5}{int(r['rank_capped']):>5}{int(r['n']):>8}"
              f"{r['raw']:>8.3f}{r['p_play']:>9.3f}{r['p_play_any']:>8.3f}")

    print("\nInjury multipliers, measured against same-rank healthy players:\n")
    print(f"  {'status':<14}{'n':>6}{'measured':>10}{'used':>8}  source")
    for _, r in adj.iterrows():
        m = "n/a" if pd.isna(r["measured"]) else f"{r['measured']:.3f}"
        print(f"  {r['report_status']:<14}{int(r['n']):>6}{m:>10}"
              f"{r['multiplier']:>8.3f}  {r['source']}")

    # ── The check that matters ──
    # Applied to the charts we actually serve, does this grid expect the right
    # NUMBER of players to appear? A grid that is individually plausible but
    # sums wrong drags every team total with it.
    #
    # v1's version of this check ran over ALL chart rows, byes included, and so
    # divided by team-weeks that could not produce an appearance. It compared
    # the result against 11.9 and passed, while the served board was at 0.90 of
    # actual. The check has to score the population inference scores: teams
    # that are playing, with the multipliers applied.
    obs = _observed(seasons)
    mult = dict(zip(adj["report_status"], adj["multiplier"]))
    key_h = grid.set_index(["position", "rank_capped"])["p_play"]
    p = np.array([float(key_h.get((po, rk), 0.5))
                  for po, rk in zip(obs["position"], obs["rank_capped"])])
    p = p * obs["report_status"].map(mult).fillna(1.0).to_numpy()
    obs = obs.assign(p=np.clip(p, 0.0, 1.0))
    per = obs.groupby(["season", "week", "team"]).agg(
        p=("p", "sum"), actual=("appeared", "sum"))
    exp, act = float(per["p"].mean()), float(per["actual"].mean())
    print(f"\nExpected appearances per team-week on the served charts: {exp:.2f}")
    print(f"  box scores on the same rows average {act:.2f}"
          f"   ratio {exp / act:.3f}")
    if not 0.95 <= exp / act <= 1.05:
        raise SystemExit(
            f"grid sums to {exp / act:.3f} of actual appearances. That gap "
            "becomes a team-total gap on every projection; do not ship it. "
            "Nothing was written.")
    print("  within 5%, so team totals will not be dragged by availability.")

    # WRITTEN LAST, and that ordering is the point. Both artifacts used to be
    # written before this gate ran, and atomic_to_parquet is a tmp-write plus
    # os.replace, so a grid the gate then refused with "do not ship it" had
    # already clobbered the good one on disk. The next run would read the bad
    # file, and the refusal message would be false.
    atomic_to_parquet(grid, dc_path(f"availability_{lo}_{hi}_{VERSION}.parquet"))
    atomic_to_parquet(adj, dc_path(f"availability_adj_{lo}_{hi}_{VERSION}.parquet"))
    print(f"\nwrote availability_{lo}_{hi}_{VERSION}.parquet "
          f"and availability_adj_{lo}_{hi}_{VERSION}.parquet")


if __name__ == "__main__":
    main()
