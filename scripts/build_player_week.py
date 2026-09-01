"""Build the player-week training table: one row per (game, player).

This is the spine of the whole model. Levels 3 and 4 of the decomposition:

    team plays -> pass/run split -> PLAYER SHARE -> PLAYER RATE

TARGETS COME FROM THE BOX SCORE, NOT FROM PLAY-BY-PLAY
------------------------------------------------------
Every quantity the model predicts is taken from `stats_player_week`, the
official weekly box score. Play-by-play is used only for EXPOSURE and CONTEXT.
This is deliberate: pbp and the box score disagree in small, systematic ways
(two-point conversions, laterals, nullified plays), and a target that disagrees
with the box score is a target that will never match what a user sees.

THREE EXPOSURE MEASURES, KEPT SEPARATE ON PURPOSE
-------------------------------------------------
  - `off_snaps` / `off_pct` from PFR snap counts. Updated several times daily in
    season, so this is the LIVE exposure proxy and the only one available at
    inference time.
  - `routes_run_proxy`: plays the player was on the field for a team dropback,
    from participation data. This is the correct WR/TE exposure (targets are
    then routes x target-per-route-run), but participation publishes only AFTER
    the postseason and never updates in season, so it is TRAINING-ONLY.
  - Carries and targets themselves, from the box score.

**Rush attempts and routes are never merged into "touches."** They respond to
game script with opposite signs: a leading team runs more and throws less.
Merging them cancels the script signal and produces a model that looks fine on
aggregate MAE while being useless in exactly the blowout and comeback games
where a projection matters.

THE SNAP-COUNT JOIN NEEDS A CROSSWALK
--------------------------------------
`snap_counts` carries `pfr_player_id` and a display name, but NO gsis_id. It is
joined through `players.parquet`'s `pfr_id` (2186 of 2192 ids matched for 2024).
Never join on date plus player name: that silently drops every player whose name
carries a suffix or a diacritic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import fetch  # noqa: E402
from gridlib.cache import atomic_to_parquet, dc_path, logger, read_parquet_or_none  # noqa: E402

VERSION = "v1"
TEAM_WEEK_VERSION = "v1"

# The five positions in scope, plus FB (the feed groups fullbacks with backs).
SKILL = ["QB", "RB", "FB", "WR", "TE", "K"]

NEUTRAL_WP_LO, NEUTRAL_WP_HI = 0.05, 0.95

# The stat table from the spec: what we predict, taken from the box score.
TARGET_COLS = [
    # QB
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "passing_air_yards",
    # rushing (every position)
    "carries", "rushing_yards", "rushing_tds",
    # receiving
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_yards_after_catch",
    # kicking
    "fg_att", "fg_made", "pat_att", "pat_made",
    # turnovers, needed for the fantasy composite
    "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost",
    # shares the feed already computes, useful as validation of our own
    "target_share", "air_yards_share",
]


def _pfr_crosswalk() -> pd.Series:
    """pfr_id -> gsis_id, from the player master table."""
    players = fetch.load_players()
    if players is None or "pfr_id" not in players.columns:
        logger.warning("players.parquet unavailable; snap counts will not join")
        return pd.Series(dtype=str)
    x = players[["pfr_id", "gsis_id"]].dropna()
    return x.drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]


def _snaps(season: int) -> pd.DataFrame:
    """Per-player offensive snaps, keyed to gsis_id through the crosswalk."""
    sn = fetch.load_snap_counts(season)
    if sn is None or sn.empty:
        return pd.DataFrame(columns=["game_id", "gsis_id", "off_snaps", "off_pct"])
    sn = sn[sn["game_type"] == "REG"].copy()
    sn["gsis_id"] = sn["pfr_player_id"].map(_pfr_crosswalk())
    sn = sn[sn["gsis_id"].notna()]
    return (
        sn.groupby(["game_id", "gsis_id"], as_index=False)
        .agg(off_snaps=("offense_snaps", "max"), off_pct=("offense_pct", "max"))
    )


def _routes(season: int) -> pd.DataFrame:
    """Plays each player was on the field for a team DROPBACK.

    For a WR or TE this is the routes-run proxy, the true plate-appearance
    analog: targets = routes x target-per-route-run. It overcounts slightly for
    players who stay in to block, which is why it is named a proxy and why RB
    and TE values deserve more suspicion than WR ones.

    TRAINING ONLY. Participation publishes after the postseason and never
    updates in season, so this column is unavailable for any live projection.
    Snap counts are the in-season stand-in.
    """
    part = fetch.load_participation(season)
    pbp = fetch.load_pbp(season)
    if part is None or part.empty or pbp is None or pbp.empty:
        logger.warning("participation or pbp missing for %d; no routes", season)
        return pd.DataFrame(columns=["game_id", "gsis_id", "routes_run_proxy",
                                     "pass_snaps", "off_snaps_pbp"])

    keep = pbp[pbp["season_type"] == "REG"][
        ["game_id", "play_id", "qb_dropback", "play_type", "posteam"]
    ]
    j = keep.merge(
        part[["nflverse_game_id", "play_id", "offense_players"]],
        left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "play_id"],
        how="inner",
    )
    j = j[j["offense_players"].notna()]
    j = j[j["play_type"].isin(["pass", "run"])]
    if j.empty:
        return pd.DataFrame(columns=["game_id", "gsis_id", "routes_run_proxy",
                                     "pass_snaps", "off_snaps_pbp"])

    j["gsis_id"] = j["offense_players"].str.split(";")
    ex = j.explode("gsis_id")
    ex = ex[ex["gsis_id"].astype(str).str.len() > 0]
    ex["is_db"] = ex["qb_dropback"].fillna(0)

    out = ex.groupby(["game_id", "gsis_id"], as_index=False).agg(
        off_snaps_pbp=("play_id", "count"),
        routes_run_proxy=("is_db", "sum"),
    )
    out["pass_snaps"] = out["routes_run_proxy"]
    return out


def _player_pbp_opportunity(season: int) -> pd.DataFrame:
    """Per-player opportunity detail that the box score does not carry.

    Chiefly WHERE the opportunity happened. Touchdowns are near-irreducibly
    noisy, so the model must regress them toward opportunity-implied rates, and
    that needs red-zone and goal-line counts per player, not just season TDs.
    """
    pbp = fetch.load_pbp(season)
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=["game_id", "gsis_id"])
    p = pbp[(pbp["season_type"] == "REG") & pbp["posteam"].notna()].copy()
    two_pt = p["two_point_attempt"].fillna(0) == 0
    p["is_pass_att"] = (
        p["complete_pass"].fillna(0) + p["incomplete_pass"].fillna(0)
        + p["interception"].fillna(0)
    ) * two_pt
    p["is_rush_att"] = p["rush_attempt"].fillna(0) * two_pt
    p["neutral"] = p["wp"].between(NEUTRAL_WP_LO, NEUTRAL_WP_HI, inclusive="both")

    frames = []

    rush = p[(p["is_rush_att"] > 0) & p["rusher_player_id"].notna()].copy()
    if not rush.empty:
        rush["rz"] = (rush["yardline_100"] <= 20).astype(float)
        rush["i10"] = (rush["yardline_100"] <= 10).astype(float)
        rush["i5"] = (rush["yardline_100"] <= 5).astype(float)
        rush["neu"] = rush["neutral"].astype(float)
        frames.append(
            rush.groupby(["game_id", "rusher_player_id"], as_index=False).agg(
                pbp_carries=("is_rush_att", "sum"),
                carries_rz=("rz", "sum"),
                carries_i10=("i10", "sum"),
                carries_i5=("i5", "sum"),
                carries_neutral=("neu", "sum"),
                rush_epa=("epa", "mean"),
            ).rename(columns={"rusher_player_id": "gsis_id"})
        )

    rec = p[(p["is_pass_att"] > 0) & p["receiver_player_id"].notna()].copy()
    if not rec.empty:
        rec["rz"] = (rec["yardline_100"] <= 20).astype(float)
        rec["i10"] = (rec["yardline_100"] <= 10).astype(float)
        rec["i5"] = (rec["yardline_100"] <= 5).astype(float)
        rec["neu"] = rec["neutral"].astype(float)
        frames.append(
            rec.groupby(["game_id", "receiver_player_id"], as_index=False).agg(
                pbp_targets=("is_pass_att", "sum"),
                targets_rz=("rz", "sum"),
                targets_i10=("i10", "sum"),
                targets_i5=("i5", "sum"),
                targets_neutral=("neu", "sum"),
                adot=("air_yards", "mean"),
                target_epa=("epa", "mean"),
            ).rename(columns={"receiver_player_id": "gsis_id"})
        )

    if not frames:
        return pd.DataFrame(columns=["game_id", "gsis_id"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["game_id", "gsis_id"], how="outer")
    return out


def build_season(season: int) -> pd.DataFrame | None:
    stats = fetch.load_player_week(season)
    if stats is None or stats.empty:
        logger.warning("no weekly stats for %d", season)
        return None

    base = stats[(stats["season_type"] == "REG") & stats["position"].isin(SKILL)].copy()
    if base.empty:
        return None

    keep = ["player_id", "player_display_name", "position", "position_group",
            "season", "week", "game_id", "team", "opponent_team"]
    have_targets = [c for c in TARGET_COLS if c in base.columns]
    missing = [c for c in TARGET_COLS if c not in base.columns]
    if missing:
        logger.warning("%d: weekly stats missing %s", season, missing)
    df = base[keep + have_targets].rename(columns={"player_id": "gsis_id"})

    # ── Exposure ──
    df = df.merge(_snaps(season), on=["game_id", "gsis_id"], how="left")
    df = df.merge(_routes(season), on=["game_id", "gsis_id"], how="left")
    df = df.merge(_player_pbp_opportunity(season), on=["game_id", "gsis_id"], how="left")
    # A player-game with no carries genuinely had ZERO red-zone carries, so the
    # count columns are filled with 0 rather than left NaN. This is not cosmetic:
    # left as NaN, the missingness is a perfect indicator of "this player had no
    # opportunity in the game being predicted", and any prior-history feature
    # derived from the column inherits that and leaks the answer. Verified:
    # P(targets_rz is NaN | this game's targets == 0) was exactly 1.0.
    # Rate columns (adot, the EPAs) stay NaN, because "no opportunity" genuinely
    # means the rate is undefined rather than zero.
    _OPP_COUNTS = ["pbp_carries", "carries_rz", "carries_i10", "carries_i5",
                   "carries_neutral", "pbp_targets", "targets_rz", "targets_i10",
                   "targets_i5", "targets_neutral"]
    for c in _OPP_COUNTS:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # ── Depth-chart rank: where the injury information actually lives ──
    # The chart is re-published continuously and demotes injured players, so it
    # carries availability news EARLIER than the injury report and, before the
    # season starts, is the ONLY such signal (the current season's injuries file
    # does not exist until week 1).
    #
    # Capped at 3 deliberately. The two depth-chart eras are not comparable
    # beyond that: 2001-2024 files stop at rank 3, so their "3" pools everyone
    # third-or-deeper, while 2025+ files run to 14. Capping makes 1/2/3 mean the
    # same thing in both eras (starter / backup / third or deeper) instead of
    # handing the model a feature whose meaning changes mid-history.
    ranks = fetch.depth_chart_normalized(season)
    if ranks is not None and not ranks.empty:
        r = ranks[["season", "week", "team", "gsis_id", "depth_rank"]].copy()
        df = df.merge(r, on=["season", "week", "team", "gsis_id"], how="left")
    else:
        df["depth_rank"] = pd.NA
        logger.warning("no depth chart for %d; rank features unavailable", season)
    df["depth_rank_capped"] = (pd.to_numeric(df["depth_rank"], errors="coerce")
                               .clip(upper=3))
    df["is_starter"] = (df["depth_rank_capped"] == 1).astype(float)

    # ── Availability: the injury report, which IS pregame ──
    # Published Wednesday to Friday for a Sunday game (and shifted for other
    # slates), so unlike almost everything else in this table it is genuinely
    # knowable before kickoff and is safe to use as a feature. Absent for
    # seasons before 2009 and for the current season until it starts.
    inj = fetch.load_injuries(season)
    if inj is not None and not inj.empty:
        rep = (inj[["season", "week", "gsis_id", "report_status",
                    "practice_status"]]
               .dropna(subset=["gsis_id"])
               .drop_duplicates(["season", "week", "gsis_id"]))
        df = df.merge(rep, on=["season", "week", "gsis_id"], how="left")
    else:
        df["report_status"] = pd.NA
        df["practice_status"] = pd.NA
        logger.warning("no injury report for %d; availability features unavailable",
                       season)

    # Ordinal severity, so a model sees a scale rather than three unrelated
    # categories. Not on the report at all is the healthy baseline, 0.
    _SEVERITY = {"Questionable": 1.0, "Doubtful": 2.0, "Out": 3.0}
    df["injury_severity"] = df["report_status"].map(_SEVERITY).fillna(0.0)
    _PRACTICE = {"Full Participation in Practice": 0.0,
                 "Limited Participation in Practice": 1.0,
                 "Did Not Participate In Practice": 2.0}
    df["practice_limitation"] = df["practice_status"].map(_PRACTICE).fillna(0.0)
    df["on_injury_report"] = df["report_status"].notna().astype(float)

    # ── Team context (own offense, and the defense faced) ──
    tw = read_parquet_or_none(dc_path(f"team_week_{season}_{TEAM_WEEK_VERSION}.parquet"))
    if tw is None:
        logger.warning("team_week for %d missing; run build_team_week.py first", season)
        return None
    tcols = [c for c in tw.columns if c.startswith(("off_", "def_allowed_", "realized_"))]
    df = df.merge(
        tw[["game_id", "team", "is_home"] + tcols],
        on=["game_id", "team"], how="left", suffixes=("", "_tw"),
    )

    # ── Game context: market, venue, rest. The market's implied team total is
    #    the EXOGENOUS instrument for game script; realized win probability
    #    would leak the outcome we are trying to project. ──
    sched = fetch.load_schedules()
    g = sched[sched["season"] == season].copy()
    g = fetch.implied_team_totals(g)
    g = fetch.apply_stadium_corrections(g)
    gm = g[["game_id", "gameday", "weekday", "spread_line", "total_line",
            "home_implied_total", "away_implied_total", "home_team", "away_team",
            "roof_type", "surface", "div_game", "home_rest", "away_rest",
            "temp", "wind"]].copy()
    df = df.merge(gm, on="game_id", how="left")

    is_home = df["team"] == df["home_team"]
    df["team_implied_total"] = np.where(is_home, df["home_implied_total"],
                                        df["away_implied_total"])
    df["opp_implied_total"] = np.where(is_home, df["away_implied_total"],
                                       df["home_implied_total"])
    # Spread from THIS team's perspective: negative means this team is favored.
    df["team_spread"] = np.where(is_home, -df["spread_line"], df["spread_line"])
    df["rest_days"] = np.where(is_home, df["home_rest"], df["away_rest"])
    df = df.drop(columns=["home_implied_total", "away_implied_total",
                          "home_rest", "away_rest", "home_team", "away_team"])

    df["fumbles_lost"] = (
        df.get("rushing_fumbles_lost", 0).fillna(0)
        + df.get("receiving_fumbles_lost", 0).fillna(0)
        + df.get("sack_fumbles_lost", 0).fillna(0)
    )
    return df.sort_values(["week", "team", "position", "gsis_id"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2016-2025")
    args = ap.parse_args()
    if "-" in args.seasons:
        lo, hi = (int(x) for x in args.seasons.split("-"))
        seasons = list(range(lo, hi + 1))
    else:
        seasons = [int(args.seasons)]

    frames = []
    for season in seasons:
        df = build_season(season)
        if df is None:
            print(f"  {season}: SKIPPED")
            continue
        path = dc_path(f"player_week_{season}_{VERSION}.parquet")
        atomic_to_parquet(df, path)
        routes = int(df["routes_run_proxy"].notna().sum())
        snaps = int(df["off_snaps"].notna().sum())
        print(f"  {season}: {len(df):5d} player-games  "
              f"snaps {100*snaps/len(df):5.1f}%  routes {100*routes/len(df):5.1f}%"
              f"  -> {path.name}")
        frames.append(df)

    if frames:
        allf = pd.concat(frames, ignore_index=True)
        path = dc_path(f"player_week_{seasons[0]}_{seasons[-1]}_{VERSION}.parquet")
        atomic_to_parquet(allf, path)
        print(f"\ncombined: {len(allf)} player-games, {len(allf.columns)} cols "
              f"-> {path.name}")
        print("\nby position:")
        print(allf["position"].value_counts().to_string())


if __name__ == "__main__":
    main()
