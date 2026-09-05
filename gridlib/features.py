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

# NextGen Stats: the tracking layer, and the closest football analog to
# Statcast quality-of-contact. These are SAME-GAME measurements (a receiver's
# separation in week 6 is measured during week 6), so they enter only as lagged
# history, exactly like the volume block.
#
# Coverage is qualified players only: 36% of WR/TE rows, 36% of RB, and 84% of
# QB. That is not a defect. The covered rows are the players who carry prop
# lines, and it shows: a WR with NGS data averages 7.87 targets against 1.80 for
# one without. The block must be judged on the rows it covers.
NGS_STATS = [
    # receiving: how open he gets, how far downfield, what he does after
    "ngs_avg_cushion", "ngs_avg_separation", "ngs_avg_intended_air_yards",
    "ngs_percent_share_of_intended_air_yards", "ngs_avg_yac_above_expectation",
    # rushing: box counts faced, time to the line, yards over expected
    "ngs_efficiency", "ngs_percent_attempts_gte_eight_defenders",
    "ngs_avg_time_to_los", "ngs_rush_yards_over_expected_per_att",
    # passing: how long he holds it, how aggressive, completion over expected
    "ngs_avg_time_to_throw", "ngs_avg_air_yards_differential",
    "ngs_aggressiveness", "ngs_avg_air_yards_to_sticks",
    "ngs_completion_percentage_above_expectation",
]

# Bio. Static or knowable years ahead, so these are used DIRECTLY rather than
# lagged. Football age curves are steeper than baseball's, and draft round is
# the only pedigree signal that exists for a player with no NFL history at all,
# which is precisely the case every history-based feature fails on.
BIO_FEATURES = ["age", "experience", "draft_round", "height", "weight"]

# Venue and schedule context: park factors and day-versus-night, in football
# terms. All knowable weeks ahead, so used directly rather than lagged.
VENUE_FEATURES = ["roof_indoor", "is_turf", "altitude_m", "rest_diff",
                  "opp_rest_days", "is_primetime", "week_of_season"]

ROLL_WINDOWS = (3, 8)

# ── Ablation flags ───────────────────────────────────────────────────────────
# Every block is switchable so it can be gated rather than assumed. A block
# ships only after it wins on the validation season AND replicates on a separate
# held-out season, checked per stratum rather than pooled. Most do not ship:
# recency weighting, per-position models, extra capacity, longer windows and the
# red-zone block were all tested and rejected.

# Rejected: no effect on validation, reversed on test. Kept for the record.
USE_REDZONE_BLOCK = False

# Shipped: wins for QB volume (+3.1% on starters) and all four cold-start
# strata. See docs/exp_depth_rank.csv.
USE_DEPTH_RANK = True

# Rejected: NextGen Stats. Lost on validation (-0.22% mean) so the gate stopped
# there. Worth recording WHY, because the raw numbers look tempting: `attempts`
# moved -0.79% on validation and +1.67% on test. A block whose headline metric
# swings 2.5 points between two adjacent seasons is measuring the fold, not the
# feature. Only 3 of 21 metrics improved on both folds, none by more than 0.4%.
# The likely cause is coverage: NGS carries qualified players only, so for the
# 64% of receivers without it every column is null, and the model already knows
# those players are low-volume from their own history.
USE_NGS = False

# Rejected: Marcel volume regression. Mean -0.09% validation, -0.11% test, and
# it REGRESSED on both folds at exactly the strata it was built to fix
# (carries::cold -1.40%/-1.24%, attempts::rank1 -1.77%/-0.44%). A replicated
# regression is the strongest reject signal the gate produces. The diagnosis is
# that the shrinkage was redundant and lossy: `f_games_prior` is already a
# feature, so the model could learn "few prior games, distrust this mean" by
# itself and with a split it chose, whereas pre-averaging destroyed the raw
# counts it was learning from. Shrinkage helps when the model cannot see the
# sample size. This one could.
USE_MARCEL = False

# Shipped: bio (age, experience, draft round, height, weight). Mean +0.64%
# validation, +0.32% test, improving on both folds for 8 of 21 metrics, and the
# wins land where the mechanism predicts: a player with fewer than three games
# has no history to average, so pedigree and age are the only signal there is.
# targets::cold +2.74%/+5.87%, receptions::cold +3.18%/+1.89%,
# receiving_yards::cold +2.10%/+1.10%. See BIO_EXCLUDE_TARGETS for the one place
# it does not hold.
USE_BIO = True

# Bio's one replicated regression: passing_yards for depth-rank-1 quarterbacks,
# -0.60% on validation and -0.31% on test. Pooled `passing_yards` and `attempts`
# both FLIP sign between folds, so those are noise and are not excluded; this one
# metric is not. A plausible mechanism is that a franchise quarterback's age and
# draft round correlate with his team's era and roster quality rather than with
# next Sunday, and starters are exactly the cohort with enough history that
# pedigree adds nothing. Under test, not assumed: the exclusion is a block in its
# own right, measured against the shipped configuration.
#
# REFUTED ON THE 2025 HOLDOUT, and it is the cautionary result of the programme.
# On validation+test it looked right: passing_yards::rank1 +0.60%/+0.31%,
# improving on both folds. But those are the SAME two folds whose round-1
# numbers generated the hypothesis, and +0.60/+0.31 is near-exactly the mirror
# of round 1's -0.60/-0.31, as it must be, since removing bio returns roughly to
# the no-bio model. That was not a replication, it was an algebraic restatement
# of the thing being tested. On the untouched 2025 season the exclusion LOSES:
# passing_yards -0.94%, passing_yards::cold -3.10%, and passing_yards::rank1
# -0.47%, the very metric it was built to fix. Two folds do not protect against
# a hypothesis that was read off those two folds. Left wired and OFF.
USE_BIO_EXCLUDE = False
BIO_EXCLUDE_TARGETS = frozenset({"passing_yards"})

MARCEL_K = 6.0          # games at which a player is half himself, half cohort
MARCEL_STATS = ["targets", "carries", "attempts", "receptions", "off_snaps"]

# Defence-versus-position, shrunk. Deliberately heavy: the spec's warning is
# that unshrunk DVP is mostly schedule noise and actively degrades projections,
# so a defence needs a lot of prior evidence before it moves off the league rate.
# SHIPPED, with the QB exclusion below. Confirmed on the untouched 2025
# holdout: +0.39% test, +0.67% HOLDOUT across non-QB metrics, and the holdout
# gains are LARGER than the test ones. Five cold-start strata improve on both
# folds (rushing_yards::cold +3.53%, receiving_yards::cold +2.48% on holdout),
# which is the mid-season-promotion case: a player with no history of his own
# still has an opponent, and that matchup is the only signal available.
USE_POSITIONAL_DEF = True
DEF_SHRINK_K = 12.0     # games against this position before half-weight

# Defence-versus-position is computed for targets, receiving_yards, carries and
# rushing_yards. None of those is QB passing volume, so for an attempts or
# passing_yards model the four columns are matchup noise about OTHER positions.
# Measured, and it shows: posdef regressed on both folds for `attempts::rank1`
# (-1.06%/-0.62%) and `passing_yards::rank1` (-0.15%/-0.67%), while winning on
# both folds across four cold-start skill-position strata.
DVP_PREFIX = "f_dvp_"
USE_POSDEF_EXCLUDE = True
# EVERY QB passing target, not just the two the ablation probed. The probe set
# covers 7 targets; `completions`, `passing_tds`, `passing_interceptions` and
# `sacks_suffered` are not among them, so the first exclusion list applied the
# rule only where the ablation happened to look. That inconsistency cost 2.7
# points of side-picking edge on QB completions (9.4 -> 6.7) before it was
# caught by the lean-accuracy A/B.
#
# The rule is mechanical, not fitted: DVP is computed for targets,
# receiving_yards, carries and rushing_yards. For a QB row those columns
# describe what a defence allows quarterbacks on the GROUND. None of them
# describes passing, so for any passing target they are noise.
POSDEF_EXCLUDE_TARGETS = frozenset({
    "attempts", "completions", "passing_yards",
    "passing_tds", "passing_interceptions", "sacks_suffered",
})

# Defensive scheme from participation: man/zone, pressure, blitz, box count.
# The strategy document's section 3 and the last rich unused source. Built as
# STRICTLY PRIOR-SEASON, league-relative rates; see add_def_scheme for why
# both of those are load-bearing rather than stylistic.
USE_SCHEME = False

# Referee crew tendencies. Availability is the constraint: see
# add_referee_tendency. Prior for this helping an individual player line is
# weak, which is exactly why it gets measured rather than argued about.
USE_REFEREE = False

# PFR advanced: broken tackles and drops, lagged like every other same-game
# measurement. Same family as NextGen, which was rejected, so the prior is low.
USE_PFR = False
PFR_STATS = ["pfr_receiving_broken_tackles", "pfr_receiving_drop",
             "pfr_receiving_drop_pct", "pfr_receiving_rat",
             "pfr_rushing_broken_tackles", "pfr_passing_drops",
             "pfr_passing_drop_pct"]

# Venue and schedule block.
# Rejected: venue and schedule. Mean -0.06% validation, so the gate stopped at
# stage one. It is genuinely split rather than merely flat, which is the
# interesting part: passing helped on both folds (passing_yards +0.65%/+0.81%)
# while rushing was hurt (carries::cold -3.60% validation). A block that helps
# one phase and hurts another nets to nothing, and splitting it per target
# would be the same selection-on-the-fold mistake the bio exclusion just made.
# The venue COLUMNS stay in the backfill: they cost nothing there and three
# real data bugs were found by encoding them.
USE_VENUE = False


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


def _prior_mean_skipna(df: pd.DataFrame, key: str, col: str) -> pd.Series:
    """Expanding mean over prior rows where the value EXISTS.

    The zero-filling in `_prior_mean` is right for counts, where a missing value
    means the player did not do the thing. It is wrong for a measured RATE: a
    missing NextGen separation means the player was not tracked that week, not
    that he got zero separation, and averaging in a zero would corrupt it.

    The current row is still excluded by construction, both its value and its
    presence, so this keeps the property that made `_prior_mean` safe: nothing
    about row i can reach feature i.
    """
    v = df[col].astype(float)
    present = v.notna().astype(float)
    filled = v.fillna(0.0)
    grp_v = filled.groupby(df[key], sort=False).cumsum() - filled
    grp_n = present.groupby(df[key], sort=False).cumsum() - present
    return grp_v / grp_n.replace(0, np.nan)


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

    stats = (VOLUME_STATS
             + (REDZONE_STATS if USE_REDZONE_BLOCK else [])
             + (NGS_STATS if USE_NGS else [])
             + (PFR_STATS if USE_PFR else []))
    for col in stats:
        if col not in out.columns:
            continue
        # NGS values are measured RATES that are simply absent for unqualified
        # players, so their history skips missing weeks instead of scoring them
        # as zero. Counts keep the zero-filling, where absent genuinely means 0.
        # PFR columns behave the same way: a player with no PFR row for a week
        # has an ABSENT drop rate, not a zero one, and zero-filling would tell
        # the model that every uncovered week was a flawless one.
        is_rate = col.startswith("ngs_") or col.startswith("pfr_")
        if is_rate:
            out[f"f_career_{col}"] = _prior_mean_skipna(out, "gsis_id", col)
            # A rolling window over a sparse column is mostly empty, and the
            # career mean already carries the signal, so rates get one feature
            # each rather than four. Fewer, better-populated columns.
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


def add_marcel_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Regress a player's prior volume toward his positional cohort.

    The efficiency features have always been shrunk; the volume features were
    raw. So a player with two prior games carried a "career average" built from
    two games, and the model had no way to know it was thin. Every projection
    for a newly promoted starter went through that hole.

    The cohort is (position, capped depth rank), which is the right reference
    class: what a rank-1 running back does is a far better prior for a rank-1
    running back with two games than the league mean over all backs. The cohort
    mean is computed from STRICTLY PRIOR SEASONS so it cannot see the row.

    Weight is games played, not a tuned parameter: at MARCEL_K games a player is
    half himself and half his cohort.
    """
    out = df.copy()
    if "depth_rank_capped" in out.columns:
        cohort_key = (out["position"].astype(str) + "|"
                      + out["depth_rank_capped"].fillna(9).astype(int).astype(str))
    else:
        cohort_key = out["position"].astype(str)
    out["_cohort"] = cohort_key

    n = out["f_games_prior"].fillna(0).astype(float)
    for col in MARCEL_STATS:
        career = f"f_career_{col}"
        if career not in out.columns:
            continue
        # Cohort mean from PRIOR seasons only. Using the current season would
        # let a player's own later games inform his earlier ones through the
        # cohort, which is the subtlest form of the leak this project keeps
        # finding.
        prior_seasons = out.groupby(["_cohort", "season"])[col].mean().reset_index()
        prior_seasons = prior_seasons.sort_values("season")
        prior_seasons["cohort_prior"] = (
            prior_seasons.groupby("_cohort")[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        )
        out = out.merge(prior_seasons[["_cohort", "season", "cohort_prior"]],
                        on=["_cohort", "season"], how="left")
        prior = out["cohort_prior"]
        obs = out[career]
        # Where there is no cohort history at all, fall back to the player's own
        # mean rather than inventing a number.
        prior = prior.fillna(obs)
        out[f"f_marcel_{col}"] = ((obs.fillna(0) * n + prior.fillna(0) * MARCEL_K)
                                  / (n + MARCEL_K))
        out = out.drop(columns=["cohort_prior"])
    return out.drop(columns=["_cohort"])


# ── Defensive scheme, from participation ─────────────────────────────────────
SCHEME_ARTIFACT = "def_scheme_2016_2025_v1.parquet"
SCHEME_RATES = ["man_rate", "pressure_rate", "blitz_rate", "box8_rate", "box_avg"]
_SCHEME_CACHE: dict[str, pd.DataFrame] = {}


def _scheme_table() -> pd.DataFrame:
    """The per-(season, defence) scheme artifact, cached.

    RAISES when the flag is on and the file is missing, rather than returning
    empty and letting the block quietly no-op. A block that silently does
    nothing is indistinguishable from a block that does not help, and this
    project has already rejected one feature for the wrong reason that way.
    """
    if "df" not in _SCHEME_CACHE:
        from .cache import dc_path, read_parquet_or_none
        t = read_parquet_or_none(dc_path(SCHEME_ARTIFACT))
        if t is None or t.empty:
            raise FileNotFoundError(
                f"USE_SCHEME is on but {SCHEME_ARTIFACT} is missing. Run "
                "scripts/build_def_scheme.py, or set USE_SCHEME = False."
            )
        _SCHEME_CACHE["df"] = t
    return _SCHEME_CACHE["df"]


def add_def_scheme(df: pd.DataFrame) -> pd.DataFrame:
    """What this opponent's defence DOES, from strictly prior seasons.

    Man versus zone, pressure, blitz and box count: the strategy document's
    section 3, and the last unused rich source in the pipeline. A receiver who
    beats man coverage and a defence that plays 63% man (Denver 2024) against
    one that plays 33% is a real matchup difference that a
    defence-versus-position multiplier averages away.

    TWO DESIGN CHOICES THAT MATTER MORE THAN THE FEATURE ITSELF
    -----------------------------------------------------------
    1. STRICTLY PRIOR SEASONS. Participation publishes only after a season's
       postseason, so "this defence so far this year" cannot exist at inference.
       Using the previous season keeps training and serving identical.

    2. RELATIVE TO THE LEAGUE THAT SEASON, never the raw rate. The charting
       shifts underneath these columns: league man rate reads 0.286 in 2022,
       0.492 in 2024 and 0.318 in 2025, and the 8-in-the-box rate HALVES between
       2023 (0.208) and 2024 (0.101). Defences did not change that fast; the
       charting did. A raw rate hands the model an era trend and calls it
       scheme. A ratio to the same season's league mean keeps the only thing
       that is actually comparable, which is where a defence sits relative to
       its peers. This is the same correction the PROE features already use.
    """
    out = df.copy()
    if "opponent_team" not in out.columns or "season" not in out.columns:
        return out
    tab = _scheme_table().copy()

    # Centre each rate on its own season, then lag by one season.
    for c in SCHEME_RATES:
        if c not in tab.columns:
            continue
        lg = tab.groupby("season")[c].transform("mean")
        tab[f"rel_{c}"] = tab[c] / lg.replace(0, np.nan)
    rel_cols = [f"rel_{c}" for c in SCHEME_RATES if f"rel_{c}" in tab.columns]
    lagged = tab[["season", "def_team"] + rel_cols].copy()
    lagged["season"] = lagged["season"] + 1      # available to the NEXT season
    lagged = lagged.rename(columns={c: f"f_sch_{c[4:]}" for c in rel_cols})

    return out.merge(lagged, how="left",
                     left_on=["season", "opponent_team"],
                     right_on=["season", "def_team"]) \
              .drop(columns=["def_team"])


REFEREE_ARTIFACT = "referee_2016_2025_v1.parquet"
REFEREE_RATES = ["ref_pen_per_game", "ref_pen_yards_per_game", "ref_dpi_per_game"]
_REF_CACHE: dict[str, pd.DataFrame] = {}


def _referee_table() -> pd.DataFrame:
    if "df" not in _REF_CACHE:
        from .cache import dc_path, read_parquet_or_none
        t = read_parquet_or_none(dc_path(REFEREE_ARTIFACT))
        if t is None or t.empty:
            raise FileNotFoundError(
                f"USE_REFEREE is on but {REFEREE_ARTIFACT} is missing. Run "
                "scripts/build_referee.py, or set USE_REFEREE = False."
            )
        _REF_CACHE["df"] = t
    return _REF_CACHE["df"]


def add_referee_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """This crew's prior-season flag rate, relative to the league that season.

    The theory is that a flag-heavy crew extends drives and adds plays, which
    would lift passing volume. The spread is real at the game level: in 2025 the
    lightest crew called 10.13 penalties a game and the heaviest 16.12.

    Centred per season for the same reason as everything else here. The league
    rate moved from 13.46 penalties a game in 2019 to 11.18 in 2022 and back to
    12.86 in 2024, which is points of emphasis changing, not crews.

    AVAILABILITY IS THIS BLOCK'S WEAK POINT and it is not hidden: crews are
    announced only a few days out, so the feature exists for the imminent week
    and is null beyond it. If it ever ships, the null-crew case needs a stated
    decision rather than a tree's learned default branch.
    """
    out = df.copy()
    if "referee" not in out.columns:
        return out
    tab = _referee_table().copy()
    for c in REFEREE_RATES:
        if c not in tab.columns:
            continue
        lg = tab.groupby("season")[c].transform("mean")
        tab[f"rel_{c}"] = tab[c] / lg.replace(0, np.nan)
    rel = [f"rel_{c}" for c in REFEREE_RATES if f"rel_{c}" in tab.columns]
    lagged = tab[["season", "referee"] + rel].copy()
    lagged["season"] = lagged["season"] + 1
    lagged = lagged.rename(columns={c: f"f_{c[4:]}" for c in rel})
    return out.merge(lagged, on=["season", "referee"], how="left")


def add_positional_defense(df: pd.DataFrame) -> pd.DataFrame:
    """What this opponent's defence has allowed TO THIS POSITION, shrunk.

    The team-level defensive features already here (`f_opp_prior_*`) treat a
    defence as one number. Football does not work that way: a team can be stout
    against the run and porous to tight ends, and a projection for a tight end
    should know which.

    SHRUNK HARD, AND THAT IS THE WHOLE POINT. Raw defence-versus-position
    multipliers are mostly opponent-SCHEDULE artifacts over a 17-game season,
    they are worst exactly where they are most used (receivers), and unshrunk
    they actively degrade projections. So each defence's allowed rate is an
    empirical-Bayes posterior against the league rate for that position, with a
    prior weight in games rather than a tuned constant.

    Everything is computed from STRICTLY PRIOR games of the defending team.
    """
    out = df.copy()
    stats = ["targets", "receiving_yards", "carries", "rushing_yards"]
    have = [s for s in stats if s in out.columns]
    if not have or "opponent_team" not in out.columns:
        return out

    # One row per (game, defence, position): what that defence allowed.
    allowed = (out.groupby(["season", "week", "game_id", "opponent_team",
                            "position"], as_index=False)[have].sum()
                  .rename(columns={"opponent_team": "def_team"})
                  .sort_values(["def_team", "position", "season", "week"])
                  .reset_index(drop=True))
    allowed["_key"] = allowed["def_team"].astype(str) + "|" + allowed["position"].astype(str)

    for s in have:
        prior_sum, prior_n = _prior_sum_and_n(allowed, "_key", s)
        # The league rate for this position, from prior seasons only.
        lg = (allowed.groupby(["position", "season"])[s].mean().reset_index()
                     .sort_values("season"))
        lg["lg_prior"] = (lg.groupby("position")[s]
                            .apply(lambda x: x.shift(1).expanding().mean())
                            .reset_index(level=0, drop=True))
        allowed = allowed.merge(lg[["position", "season", "lg_prior"]],
                                on=["position", "season"], how="left")
        prior_rate = allowed["lg_prior"].fillna(
            allowed.groupby("position")[s].transform("mean"))
        # k in GAMES: a defence with DEF_SHRINK_K prior games against this
        # position is weighted half itself, half the league.
        allowed[f"f_dvp_{s}"] = ((prior_sum + prior_rate * DEF_SHRINK_K)
                                 / (prior_n + DEF_SHRINK_K))
        allowed = allowed.drop(columns=["lg_prior"])

    keep = ["season", "week", "game_id", "def_team", "position"] + \
           [f"f_dvp_{s}" for s in have]
    return out.merge(allowed[keep], how="left",
                     left_on=["season", "week", "game_id", "opponent_team", "position"],
                     right_on=["season", "week", "game_id", "def_team", "position"]) \
              .drop(columns=["def_team"])


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
    if USE_MARCEL:
        out = add_marcel_volume(out)
    out = add_shrunk_efficiency(out, priors)
    out = add_share_history(out)
    if USE_POSITIONAL_DEF:
        out = add_positional_defense(out)
    if USE_SCHEME:
        out = add_def_scheme(out)
    if USE_REFEREE:
        out = add_referee_tendency(out)
    out = add_team_history(out, league_by_season)
    return out.sort_values(["season", "week", "team", "gsis_id"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Every model-ready feature: the `f_` history block plus the pregame block.

    Nothing from the backfill's same-game columns can appear here, which is what
    makes the prefix convention worth having.

    `target` lets one target drop a block that helps everywhere else. Passing it
    is optional so existing callers keep working, but a caller that trains a
    model per target should pass it, or a per-target exclusion silently does
    nothing.
    """
    from .columns import PREGAME
    # String-valued pregame columns are excluded: HistGradientBoosting needs
    # numeric input, and each of these has a numeric encoding alongside it
    # (roof_type/surface are not encoded yet; report_status and
    # practice_status become injury_severity and practice_limitation).
    NON_NUMERIC = {"gameday", "weekday", "roof_type", "surface",
                   "referee",
                   "report_status", "practice_status",
                   # The raw rank is era-dependent (old charts stop at 3, new
                   # ones run to 14). Only the capped version and the starter
                   # flag are comparable across the training window.
                   "depth_rank"}
    if not USE_DEPTH_RANK:
        NON_NUMERIC |= {"depth_rank_capped", "is_starter"}
    if not USE_BIO or (USE_BIO_EXCLUDE and target in BIO_EXCLUDE_TARGETS):
        NON_NUMERIC |= set(BIO_FEATURES)
    if not USE_VENUE:
        NON_NUMERIC |= set(VENUE_FEATURES)
    hist = [c for c in df.columns if c.startswith("f_")]
    # The DVP block is `f_`-prefixed history, so it is excluded here rather than
    # through NON_NUMERIC, which only covers the pregame block.
    if (USE_POSITIONAL_DEF and USE_POSDEF_EXCLUDE
            and target in POSDEF_EXCLUDE_TARGETS):
        hist = [c for c in hist if not c.startswith(DVP_PREFIX)]
    pre = [c for c in df.columns if c in PREGAME and c not in NON_NUMERIC]
    return hist + pre
