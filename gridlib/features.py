"""Point-in-time features, shared by training and inference.

ONE code path builds features for a training row and for a live projection. That
is the only way to guarantee they agree; two implementations drift, and the drift
shows up as a model that validates well and projects badly.

THE RULE EVERY FUNCTION HERE OBEYS
-----------------------------------
A feature for the game in row i may use data from games STRICTLY BEFORE row i.
Not "up to and including". The MLB build shipped that comparison as `<=` twice,
and both times a row counted its own game as a prior appearance, which inflates
validation scores and cannot be reproduced live.

Implemented as `cumsum() - current` rather than a shifted rolling window, because
the subtraction is exact and self-documenting: the current game is removed by
construction, not by an off-by-one that a reader has to verify.

WINDOWS COUNT GAMES, NOT WEEKS
-------------------------------
Every team has a bye, so a player has 17 games across 18 weeks. A window defined
in weeks silently becomes a shorter window around a bye. Every window here is
over ROWS of a player's game log, and each row is one game played, so this is
correct by construction.

EFFICIENCY IS SHRUNK HARD, ON PURPOSE
--------------------------------------
Published year-over-year stability (Sharp Football): WR targets per game 0.585
against yards per target 0.0395 and TD rate 0.0155; RB touches per game 0.567
against rush TD per game 0.229. Yards per carry is the worst offender: it needs
roughly 1,978 same-team carries to stabilise, which no career supplies, and an
8-game 5.00 YPC regresses to about 4.37.

A gradient booster handed raw yards-per-carry will select it and will not
generalise. So every efficiency feature here is an empirical-Bayes posterior
shrunk toward a league prior, with the prior weight set from the published
stabilisation points rather than tuned. Volume features are left alone; volume
is where the signal is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Shrinkage constants ──────────────────────────────────────────────────────
# k is the number of prior observations at which the estimate is weighted half
# player and half league prior: posterior = (n*obs + k*prior) / (n + k).
# These come from published stabilisation points, not from tuning, so they
# cannot overfit the validation season.
SHRINK_K = {
    "yards_per_target": 60.0,     # targets; y/t stability is ~0.04 year to year
    "catch_rate": 50.0,           # targets
    "yards_per_carry": 200.0,     # carries; the single worst trap in football
    "yards_per_attempt": 200.0,   # pass attempts
    "completion_rate": 100.0,     # pass attempts
    "target_share": 25.0,         # team pass attempts faced
    "rush_share": 20.0,           # team rush attempts
    "td_rate": 150.0,             # opportunities; TDs are near-irreducible noise
}

# Volume stats we build prior-history features for.
VOLUME_STATS = [
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "carries", "rushing_yards", "rushing_tds",
    "attempts", "completions", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered",
    "fg_att", "fg_made", "pat_att",
    "off_snaps", "routes_run_proxy",
]

# Prior RED-ZONE opportunity, kept as a separately switchable block so its value
# can be ablated rather than assumed. Touchdowns are the targets that failed the
# ship gate against a plain season-to-date average, and where the opportunity
# happened is the obvious missing signal: a back with six carries inside the
# five is a different proposition from a back with six carries at midfield.
# Modelling TDs from raw prior TDs is exactly what the spec warns against.
REDZONE_STATS = [
    "carries_rz", "carries_i10", "carries_i5",
    "targets_rz", "targets_i10", "targets_i5",
]

ROLL_WINDOWS = (3, 8)

# Toggled by the ablation harness (scripts/exp_redzone_block.py). Default ON
# only after the block has replicated on a held-out season; see docs/.
USE_REDZONE_BLOCK = False

# Depth-chart rank as a model feature. Toggled by the ablation harness
# (scripts/exp_depth_rank.py) so the block can be gated rather than assumed.
USE_DEPTH_RANK = True


def _prior_mean(df: pd.DataFrame, key: str, col: str) -> pd.Series:
    """Expanding mean over all STRICTLY prior rows for each key.

    cumsum minus the current value, divided by the count of prior rows. A player
    with no prior games gets NaN, never 0, so "no history" stays distinguishable
    from "history of zero".

    THE CURRENT ROW IS FILLED BEFORE SUBTRACTION, AND THAT MATTERS.
    `cumsum() - current` propagates a NaN in the CURRENT row straight into the
    feature, so the feature becomes NaN exactly when the current game has no
    value. That is a leak: the model reads missingness as the answer. It was
    caught on the red-zone block, where
    P(f_career_targets_rz is NaN | this game's targets == 0) was exactly 1.0
    and the "improvement" was 13% on targets. Filling with 0 before the
    subtraction makes the feature depend only on prior rows.
    """
    v = df[col].astype(float).fillna(0.0)
    prior_sum = v.groupby(df[key], sort=False).cumsum() - v
    prior_n = df.groupby(key, sort=False).cumcount()
    return prior_sum / prior_n.replace(0, np.nan)


def _prior_sum_and_n(df: pd.DataFrame, key: str, col: str):
    """Prior total and prior count. Same NaN discipline as `_prior_mean`."""
    v = df[col].astype(float).fillna(0.0)
    prior_sum = v.groupby(df[key], sort=False).cumsum() - v
    return prior_sum, df.groupby(key, sort=False).cumcount().astype(float)


def _prior_roll_mean(df: pd.DataFrame, key: str, col: str, window: int) -> pd.Series:
    """Mean over the last `window` STRICTLY prior GAMES (rows), per key.

    shift(1) already excludes the current row, so this never had the leak that
    `_prior_mean` did, but the fill keeps the two definitions consistent.
    """
    filled = df[col].astype(float).fillna(0.0)
    return (
        filled.groupby(df[key], sort=False)
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )


def _shrunk(numer: pd.Series, denom: pd.Series, prior: float, k: float) -> pd.Series:
    """Empirical-Bayes posterior: (numer + k*prior) / (denom + k).

    With no observations this returns the league prior exactly, which is the
    correct answer for a debut rather than a NaN the model has to impute.
    """
    return (numer.fillna(0) + k * prior) / (denom.fillna(0) + k)


def add_player_history(df: pd.DataFrame) -> pd.DataFrame:
    """Prior-game volume and shrunk-efficiency features, per player.

    Requires the frame sorted by (season, week). Adds only columns prefixed
    `f_`, so the feature set is separable from the backfill by prefix.
    """
    out = df.sort_values(["gsis_id", "season", "week"]).copy()

    grp = out.groupby("gsis_id", sort=False)
    out["f_games_prior"] = grp.cumcount().astype(float)
    # Games earlier in THIS season, which is what baseline 2 uses.
    out["f_games_prior_season"] = (
        out.groupby(["gsis_id", "season"], sort=False).cumcount().astype(float)
    )

    stats = VOLUME_STATS + (REDZONE_STATS if USE_REDZONE_BLOCK else [])
    for col in stats:
        if col not in out.columns:
            continue
        out[f"f_career_{col}"] = _prior_mean(out, "gsis_id", col)
        for w in ROLL_WINDOWS:
            out[f"f_r{w}_{col}"] = _prior_roll_mean(out, "gsis_id", col, w)
        # Season-to-date, excluding the current game. This is mandatory
        # baseline 2 and also a feature in its own right.
        s_sum, s_n = _prior_sum_and_n(
            out.assign(_k=out["gsis_id"].astype(str) + "_" + out["season"].astype(str)),
            "_k", col)
        out[f"f_std_{col}"] = s_sum / s_n.replace(0, np.nan)

    return out


def add_shrunk_efficiency(df: pd.DataFrame, league: dict[str, float]) -> pd.DataFrame:
    """Efficiency rates as empirical-Bayes posteriors over prior games only.

    `league` supplies the priors and MUST be computed from seasons strictly
    before the row's season. Passing a league table that includes the current
    season leaks, subtly and invisibly.
    """
    out = df.copy()
    key = "gsis_id"

    def prior_totals(col: str) -> pd.Series:
        s, _ = _prior_sum_and_n(out, key, col)
        return s

    tgt = prior_totals("targets")
    rec = prior_totals("receptions")
    rec_yds = prior_totals("receiving_yards")
    car = prior_totals("carries")
    rush_yds = prior_totals("rushing_yards")
    att = prior_totals("attempts")
    cmp_ = prior_totals("completions")
    pass_yds = prior_totals("passing_yards")

    out["f_shrunk_catch_rate"] = _shrunk(
        rec, tgt, league["catch_rate"], SHRINK_K["catch_rate"])
    out["f_shrunk_ypt"] = _shrunk(
        rec_yds, tgt, league["yards_per_target"], SHRINK_K["yards_per_target"])
    out["f_shrunk_ypc"] = _shrunk(
        rush_yds, car, league["yards_per_carry"], SHRINK_K["yards_per_carry"])
    out["f_shrunk_ypa"] = _shrunk(
        pass_yds, att, league["yards_per_attempt"], SHRINK_K["yards_per_attempt"])
    out["f_shrunk_comp_rate"] = _shrunk(
        cmp_, att, league["completion_rate"], SHRINK_K["completion_rate"])

    # Scoring rates, shrunk hardest of all. Expected TDs are stickier than raw
    # TDs (r^2 0.382 vs 0.276) but barely better at PREDICTING next-season TDs
    # (0.283 vs 0.275), so the honest move is to regress toward opportunity and
    # let the uncertainty band carry the rest.
    rec_td = prior_totals("receiving_tds")
    rush_td = prior_totals("rushing_tds")
    out["f_shrunk_rec_td_rate"] = _shrunk(
        rec_td, tgt, league["rec_td_per_target"], SHRINK_K["td_rate"])
    out["f_shrunk_rush_td_rate"] = _shrunk(
        rush_td, car, league["rush_td_per_carry"], SHRINK_K["td_rate"])
    return out


def add_share_history(df: pd.DataFrame) -> pd.DataFrame:
    """Prior target share and rush share, shrunk.

    Shares are the right currency for the player level: a player's targets are
    his share of a team quantity, and the share is far more stable than the raw
    count when the team's own volume moves.
    """
    out = df.copy()
    tgt_sum, _ = _prior_sum_and_n(out, "gsis_id", "targets")
    car_sum, _ = _prior_sum_and_n(out, "gsis_id", "carries")
    team_pass_sum, _ = _prior_sum_and_n(out, "gsis_id", "off_pass_att")
    team_rush_sum, _ = _prior_sum_and_n(out, "gsis_id", "off_designed_rush")

    out["f_shrunk_target_share"] = _shrunk(
        tgt_sum, team_pass_sum, 0.12, SHRINK_K["target_share"])
    out["f_shrunk_rush_share"] = _shrunk(
        car_sum, team_rush_sum, 0.25, SHRINK_K["rush_share"])
    return out


def add_team_history(df: pd.DataFrame, league_by_season: pd.DataFrame) -> pd.DataFrame:
    """Team tendency and opponent defence, from strictly prior team-games.

    PROE is centred on the PRIOR season's league mean, never the current one.
    nflfastR's xpass model has drifted (a flat ~0.627 prediction against a league
    pass rate that fell from 0.613 to 0.594), so raw PROE carries a season-level
    offset; centring on the current season would leak unplayed games.
    """
    out = df.sort_values(["team", "season", "week"]).copy()

    # One row per team-game, so team history is not weighted by roster size.
    tg = (out[["team", "season", "week", "game_id", "off_plays", "off_pass_rate",
               "off_proe", "off_dropbacks", "off_designed_rush", "off_epa_play",
               "off_pass_att"]]
          .drop_duplicates(["game_id", "team"])
          .sort_values(["team", "season", "week"])
          .reset_index(drop=True))

    for col in ["off_plays", "off_pass_rate", "off_proe", "off_dropbacks",
                "off_designed_rush", "off_epa_play", "off_pass_att"]:
        tg[f"f_team_prior_{col}"] = _prior_mean(tg, "team", col)
        tg[f"f_team_r4_{col}"] = _prior_roll_mean(tg, "team", col, 4)

    # Centre PROE on the prior season's league mean.
    lg = league_by_season.set_index("season")["league_off_proe"]
    prior_lg = lg.shift(1)
    tg["f_team_proe_centered"] = (
        tg["f_team_prior_off_proe"] - tg["season"].map(prior_lg)
    )

    keep = ["game_id", "team"] + [c for c in tg.columns if c.startswith("f_team_")]
    out = out.merge(tg[keep], on=["game_id", "team"], how="left")

    # Opponent defence, from the same table read from the other side. Built as
    # a standalone frame keyed on (game_id, defending team) so the merge cannot
    # collide with the `team` column already on `out`.
    dg = (out[["opponent_team", "season", "week", "game_id",
               "def_allowed_epa_play", "def_allowed_pass_att",
               "def_allowed_designed_rush", "def_allowed_plays"]]
          .drop_duplicates(["game_id", "opponent_team"])
          .rename(columns={"opponent_team": "_def_team"})
          .sort_values(["_def_team", "season", "week"])
          .reset_index(drop=True))
    for col in ["def_allowed_epa_play", "def_allowed_pass_att",
                "def_allowed_designed_rush", "def_allowed_plays"]:
        dg[f"f_opp_prior_{col}"] = _prior_mean(dg, "_def_team", col)
    dkeep = ["game_id", "_def_team"] + [c for c in dg.columns if c.startswith("f_opp_")]
    out = out.merge(dg[dkeep], left_on=["game_id", "opponent_team"],
                    right_on=["game_id", "_def_team"], how="left")
    return out.drop(columns=["_def_team"])


def league_priors(df: pd.DataFrame, before_season: int) -> dict[str, float]:
    """League efficiency priors from seasons STRICTLY BEFORE `before_season`.

    Recomputed per validation fold rather than once over everything, because a
    prior that saw the test season is a leak even though it is only a mean.
    """
    h = df[df["season"] < before_season]
    if h.empty:
        h = df  # first fold has no history; documented, not silent
    def rate(num: str, den: str, fallback: float) -> float:
        d = h[den].sum()
        return float(h[num].sum() / d) if d else fallback
    return {
        "catch_rate": rate("receptions", "targets", 0.64),
        "yards_per_target": rate("receiving_yards", "targets", 7.9),
        "yards_per_carry": rate("rushing_yards", "carries", 4.3),
        "yards_per_attempt": rate("passing_yards", "attempts", 7.1),
        "completion_rate": rate("completions", "attempts", 0.64),
        "rec_td_per_target": rate("receiving_tds", "targets", 0.055),
        "rush_td_per_carry": rate("rushing_tds", "carries", 0.023),
    }


def build(df: pd.DataFrame, league_by_season: pd.DataFrame,
          priors: dict[str, float]) -> pd.DataFrame:
    """The one entry point. Training and inference both call this."""
    out = add_player_history(df)
    out = add_shrunk_efficiency(out, priors)
    out = add_share_history(out)
    out = add_team_history(out, league_by_season)
    return out.sort_values(["season", "week", "team", "gsis_id"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every model-ready feature: the `f_` history block plus the pregame block.

    Nothing from the backfill's same-game columns can appear here, which is what
    makes the prefix convention worth having.
    """
    from .columns import PREGAME
    # String-valued pregame columns are excluded: HistGradientBoosting needs
    # numeric input, and each of these has a numeric encoding alongside it
    # (roof_type/surface are not encoded yet; report_status and
    # practice_status become injury_severity and practice_limitation).
    NON_NUMERIC = {"gameday", "weekday", "roof_type", "surface",
                   "report_status", "practice_status",
                   # The raw rank is era-dependent (old charts stop at 3, new
                   # ones run to 14). Only the capped version and the starter
                   # flag are comparable across the training window.
                   "depth_rank"}
    if not USE_DEPTH_RANK:
        NON_NUMERIC |= {"depth_rank_capped", "is_starter"}
    hist = [c for c in df.columns if c.startswith("f_")]
    pre = [c for c in df.columns if c in PREGAME and c not in NON_NUMERIC]
    return hist + pre
