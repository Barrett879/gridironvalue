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

    # ── Bio: age and draft pedigree ──
    # Age curves are steeper in football than in baseball. Running backs decline
    # sharply in their late twenties and receivers commonly break out in year
    # three, so a 24-year-old and a 31-year-old with identical prior games are
    # not the same projection. `players.parquet` has carried birth_date the
    # whole time and nothing used it.
    bio = fetch.player_bio()
    if not bio.empty:
        df = df.merge(bio, on="gsis_id", how="left")
        # `age` needs `gameday`, which arrives with the schedule merge further
        # down, so it is derived there. Everything date-free is derived here.
        df["experience"] = (df["season"]
                            - pd.to_numeric(df.get("rookie_season"),
                                            errors="coerce")).clip(lower=0)
        df["draft_round"] = pd.to_numeric(df.get("draft_round"), errors="coerce")
        # Undrafted is a real signal, not missing data, so it gets its own value
        # rather than a NaN a tree will route arbitrarily.
        df["draft_round"] = df["draft_round"].fillna(8.0)
    else:
        for c in ("experience", "draft_round", "height", "weight",
                  "birth_date", "rookie_season"):
            df[c] = pd.NA

    # ── NextGen Stats: the tracking layer ──
    # Separation, cushion, air-yards share and YAC-over-expected are the closest
    # football analog to Statcast quality-of-contact. Coverage is QUALIFIED
    # players only (about 73 receivers a week), so most of a depth chart is
    # missing here by design; the block is judged on the rows it covers.
    _NGS = {
        "receiving": ["avg_cushion", "avg_separation", "avg_intended_air_yards",
                      "percent_share_of_intended_air_yards",
                      "avg_yac_above_expectation"],
        "rushing": ["efficiency", "percent_attempts_gte_eight_defenders",
                    "avg_time_to_los", "rush_yards_over_expected_per_att"],
        "passing": ["avg_time_to_throw", "avg_air_yards_differential",
                    "aggressiveness", "avg_air_yards_to_sticks",
                    "completion_percentage_above_expectation"],
    }
    for kind, keep in _NGS.items():
        ngs = fetch.load_nextgen(kind)
        if ngs is None or ngs.empty:
            for c in keep:
                df[f"ngs_{c}"] = pd.NA
            continue
        sub = ngs[ngs["season"] == season]
        cols = [c for c in keep if c in sub.columns]
        if sub.empty or not cols:
            for c in keep:
                df[f"ngs_{c}"] = pd.NA
            continue
        sub = (sub[["season", "week", "player_gsis_id"] + cols]
               .rename(columns={"player_gsis_id": "gsis_id"})
               .drop_duplicates(["season", "week", "gsis_id"]))
        sub = sub.rename(columns={c: f"ngs_{c}" for c in cols})
        df = df.merge(sub, on=["season", "week", "gsis_id"], how="left")

    # ── PFR advanced: broken tackles and drops ──
    # The two things the box score cannot see. Nothing here scrapes PFR; these
    # are nflverse's MIRRORED release parquets, which is why they are usable at
    # all, since PFR itself sits behind bot verification this project does not
    # work around. One file per season and coverage starts in 2018, so 2016-17
    # rows carry NaN by construction rather than by failure.
    #
    # SAME-GAME measurements, exactly like NextGen: a receiver's drops in week 6
    # are counted during week 6. They enter only as lagged history.
    _PFR = {
        "rec": ["receiving_broken_tackles", "receiving_drop",
                "receiving_drop_pct", "receiving_rat"],
        "rush": ["rushing_broken_tackles"],
        "pass": ["passing_drops", "passing_drop_pct"],
    }
    _xw = _pfr_crosswalk()
    for kind, keep in _PFR.items():
        adv = fetch.load_pfr_advstats(kind, season)
        if adv is None or adv.empty or _xw.empty:
            for c in keep:
                df[f"pfr_{c}"] = pd.NA
            continue
        cols = [c for c in keep if c in adv.columns]
        if not cols or "pfr_player_id" not in adv.columns:
            for c in keep:
                df[f"pfr_{c}"] = pd.NA
            continue
        sub = adv[["week", "pfr_player_id"] + cols].copy()
        sub["gsis_id"] = sub["pfr_player_id"].map(_xw)
        sub = (sub.dropna(subset=["gsis_id"])
                  .drop(columns=["pfr_player_id"])
                  .drop_duplicates(["week", "gsis_id"])
                  .rename(columns={c: f"pfr_{c}" for c in cols}))
        df = df.merge(sub, on=["week", "gsis_id"], how="left")
        for c in keep:
            if f"pfr_{c}" not in df.columns:
                df[f"pfr_{c}"] = pd.NA

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
            "stadium_id", "old_game_id", "temp", "wind"]].copy()
    df = df.merge(gm, on="game_id", how="left")

    # Canonical on BOTH sides. games.parquet spells the Raiders OAK (2016-2019)
    # and the Chargers SD (2016) while stats_player_week spells them LV and LAC,
    # so a raw comparison was False for every Raiders and Chargers row including
    # their HOME games. 477 player-games that are home games had
    # team_implied_total and opp_implied_total swapped, rest_days and
    # opp_rest_days swapped, and team_spread carrying the opposite sign, with a
    # signed error from -28 to +28. The same rows carry is_home = 1 from the
    # team_week merge, which is derived from play-by-play and is correct, so the
    # row contradicted itself: a home game whose spread said the team was a
    # 3-point underdog when the market had it a 3-point favourite.
    from gridlib.teams import canonical
    is_home = df["team"].map(canonical) == df["home_team"].map(canonical)
    df["team_implied_total"] = np.where(is_home, df["home_implied_total"],
                                        df["away_implied_total"])
    df["opp_implied_total"] = np.where(is_home, df["away_implied_total"],
                                       df["home_implied_total"])
    # Spread from THIS team's perspective: negative means this team is favored.
    df["team_spread"] = np.where(is_home, -df["spread_line"], df["spread_line"])
    df["rest_days"] = np.where(is_home, df["home_rest"], df["away_rest"])
    df["opp_rest_days"] = np.where(is_home, df["away_rest"], df["home_rest"])
    df = df.drop(columns=["home_implied_total", "away_implied_total",
                          "home_rest", "away_rest", "home_team", "away_team"])

    # ── Referee crew ──
    # From the OFFICIALS feed, never `schedules.referee`: that column is 100%
    # populated for completed seasons and 0% for the upcoming one, making it a
    # postgame field like temp and wind. The officials file publishes crews a
    # few days before kickoff, so it is the only source that exists at inference
    # as well as in training. Joined on `old_game_id`, the NFL numeric id;
    # joining on `game_id` matches nothing and looks like an unpublished file.
    _ref = fetch.referee_by_game()
    if not _ref.empty and "old_game_id" in df.columns:
        df["old_game_id"] = df["old_game_id"].astype(str)
        df = df.merge(_ref, on="old_game_id", how="left")
    else:
        df["referee"] = pd.NA

    # ── Weather at kickoff ──
    # `schedules.parquet` carries temp and wind and BOTH ARE POSTGAME: 66.7%
    # populated for 2025's completed games, 0% for 2026's unplayed ones, which
    # is why they sit in columns.POSTGAME_TRAP. These come from Open-Meteo
    # instead (scripts/build_weather.py), which covers the whole backfill and
    # can be forecast for an upcoming game.
    #
    # Raw wind moves scoring hard: combined points fall from 46.4 under 6mph to
    # 41.4 at 12-18mph across 2016-2025. Whether it moves a PROJECTION is a
    # different question, because the model already sees the Vegas total and
    # Vegas prices weather. That is what the ablation is for.
    _wx = read_parquet_or_none(dc_path("weather_2016_2026_v1.parquet"))
    if _wx is not None and not _wx.empty:
        _wx = _wx.drop(columns=[c for c in ("stadium_id",) if c in _wx.columns])
        df = df.merge(_wx, on="game_id", how="left")
    else:
        for c in ("wx_temp", "wx_wind", "wx_gust", "wx_wind_max",
                  "wx_gust_max", "wx_precip"):
            df[c] = np.nan

    # ── Venue and schedule context ──
    # The football analog of park factors and day-versus-night. All of it is
    # knowable weeks ahead, and none of it was encoded: roof_type and surface
    # were carried as strings and dropped by the feature builder for being
    # non-numeric, so the model has never known whether a game is indoors.
    _ROOF = {"outdoors": 0.0, "retractable": 1.0, "dome": 2.0}
    _unmapped = set(df["roof_type"].dropna().unique()) - set(_ROOF)
    if _unmapped:
        raise ValueError(
            f"Unmapped roof_type values {_unmapped} in season {season}. They "
            "would silently encode as outdoors. Extend _ROOF."
        )
    df["roof_indoor"] = df["roof_type"].map(_ROOF).fillna(0.0)

    # `.strip()` is load-bearing: the feed carries both 'grass' and 'grass ',
    # and the trailing-space variant (93 games) was being classified as TURF,
    # because lowercasing does not remove whitespace. An unknown surface (44
    # games, empty string) stays 0, which is the majority class.
    df["is_turf"] = (~df["surface"].astype(str).str.lower().str.strip()
                     .isin(["grass", "nan", "none", ""])).astype(float)

    # Altitude, which is a real and very local effect: only two venues in the
    # league sit high enough to matter, and both are extreme rather than
    # marginal. A general elevation column would be almost all zeros, so this is
    # explicit about the two that count.
    #
    # This column was ALL ZEROS on first build: `stadium_id` was not in the
    # schedule merge above, and the `if "stadium_id" in df.columns` guard that
    # used to sit here turned the missing column into a plausible constant
    # instead of an error. No silent fallback now. Denver hosts games in every
    # season, so a season with no high-altitude rows means the join broke.
    _ALTITUDE_M = {"DEN00": 1610.0, "MEX00": 2240.0}
    df["altitude_m"] = df["stadium_id"].map(_ALTITUDE_M).fillna(0.0)
    # Check the MAP, not the calendar.
    #
    # This used to raise when a season had no high-altitude rows, on the
    # reasoning that Denver hosts games every year. True for a full season and
    # false for a PARTIAL one: Denver's first 2026 home game is week 2, so the
    # weekly refresh job's very first run, a week-1-only rebuild, would have
    # raised and killed the automation before it ever committed anything.
    #
    # Comparing rows that SHOULD have altitude against rows that DO catches a
    # broken join on any slate, including one with no high-altitude game at all.
    # Two checks, because the equality alone is not enough: if EVERY stadium_id
    # were wrong, expected and got would both be zero and it would pass. The
    # original failure was the merge not happening at all, which shows up as a
    # column of nulls, so assert the column is populated first.
    _null = float(df["stadium_id"].isna().mean())
    if _null > 0.05:
        raise ValueError(
            f"stadium_id is {_null:.0%} null in season {season}; the schedule "
            "merge did not bring it through."
        )
    _expected = int(df["stadium_id"].isin(_ALTITUDE_M).sum())
    _got = int((df["altitude_m"] > 0).sum())
    if _expected != _got:
        raise ValueError(
            f"Altitude join broken in season {season}: {_expected} rows are at "
            f"a high-altitude venue but {_got} carry an altitude."
        )

    # Rest DIFFERENTIAL, not just own rest. A team on six days against one on
    # thirteen is a different game from both teams on seven, and only the
    # difference carries that.
    df["rest_diff"] = (pd.to_numeric(df.get("rest_days"), errors="coerce")
                       - pd.to_numeric(df.get("opp_rest_days"), errors="coerce")
                       ).fillna(0.0)

    # Primetime. A standalone national window is a different game script from a
    # one-of-eight Sunday afternoon kickoff: the baseball analog is day versus
    # night, and it is the one schedule fact a viewer would name first.
    _wd = df.get("weekday").astype(str) if "weekday" in df.columns else None
    df["is_primetime"] = (_wd.isin(["Thursday", "Monday"]).astype(float)
                          if _wd is not None else 0.0)
    df["week_of_season"] = pd.to_numeric(df["week"], errors="coerce")

    # Age, now that the schedule has supplied a kickoff date. Computed against
    # the actual game date rather than a season midpoint, because a rookie born
    # in September and one born in March are nearly a year apart on the curve.
    _born = pd.to_datetime(df.get("birth_date"), errors="coerce")
    _gameday = pd.to_datetime(df.get("gameday"), errors="coerce")
    df["age"] = ((_gameday - _born).dt.days / 365.25
                 if _born.notna().any() else pd.NA)

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
