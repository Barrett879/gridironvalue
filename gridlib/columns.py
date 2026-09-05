"""Column contract for the training table: what each column IS, and whether it
may ever be used as a model feature.

This module exists because the single most dangerous thing about the backfill is
that most of it is NOT safe to feed a model. The team-week block describes the
SAME GAME being predicted: `off_dropbacks` is how many dropbacks the team
actually had in the game whose stat line we are projecting. It is indispensable
for building the model (it is the denominator of a realized share, and it is the
raw material for lagged features) and it is catastrophic as an inference
feature, because at prediction time it does not exist.

Nothing about a column's NAME makes that obvious, and a gradient booster handed
`off_dropbacks` will happily achieve a beautiful validation score and then be
useless in production. So the classification is explicit, exhaustive and tested:
`tests/test_leakage.py` asserts that every column in the built table appears in
exactly one bucket, and that no bucket has drifted.

THE BUCKETS
-----------
KEYS
    Identifiers and the join spine. Not features.

TARGETS
    What the model predicts, taken from the official box score.

SAME_GAME_OUTCOME
    Known only after the game. Legitimate for deriving targets and for building
    LAGGED features (last week's value, a rolling mean over prior games), but
    never usable at its own row's index. This is the big, dangerous bucket.

PREGAME
    Genuinely knowable before kickoff: the market, the venue, rest, the schedule.
    These are the only columns that may be used directly as features.

POSTGAME_TRAP
    Columns that LOOK pregame and are not. `temp` and `wind` are the important
    ones: games.parquet records weather only after the game is played (0 of 272
    rows populated for the 2026 season, 190 of 285 for 2025). A wind feature
    would train beautifully and be null for every live projection.
"""
from __future__ import annotations

KEYS = frozenset({
    "gsis_id", "player_display_name", "position", "position_group",
    "season", "week", "game_id", "team", "opponent_team",
    # Bio source fields. `age` and `experience` are DERIVED from these and are
    # the usable form; the raw fields are a date string and a year and are kept
    # only so the derivation is reproducible from the table.
    "birth_date", "rookie_season",
    # Venue identity. A KEY, never a feature: fed to a model directly it would
    # memorise venues, and `altitude_m` is the derived form that generalises.
    # Kept so that derivation is reproducible from the table, same as birth_date.
    "stadium_id",
    # The NFL numeric game id. A KEY: it is only the join used to attach the
    # officiating crew, which the officials feed publishes against this id
    # rather than the nflverse one.
    "old_game_id",
})

TARGETS = frozenset({
    # QB
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "passing_air_yards",
    # rushing
    "carries", "rushing_yards", "rushing_tds",
    # receiving
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_yards_after_catch",
    # kicking
    "fg_att", "fg_made", "pat_att", "pat_made",
    # turnovers, for the fantasy composite
    "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost",
    "fumbles_lost",
})

# Same-game exposure. These ARE the opportunity we want to model, so they are
# targets in their own right for the exposure stage AND leak for the rate stage.
SAME_GAME_EXPOSURE = frozenset({
    "off_snaps", "off_pct", "off_snaps_pbp", "routes_run_proxy", "pass_snaps",
    "pbp_carries", "pbp_targets",
    "carries_rz", "carries_i10", "carries_i5", "carries_neutral",
    "targets_rz", "targets_i10", "targets_i5", "targets_neutral",
    "adot", "rush_epa", "target_epa",
    "target_share", "air_yards_share",
})

# Everything the team did in THIS game, plus what the opponent's defense allowed
# in THIS game. Prefix-based so a new aggregate cannot silently escape.
SAME_GAME_TEAM_PREFIXES = ("off_", "def_allowed_", "realized_")

PREGAME = frozenset({
    "gameday", "weekday", "is_home", "spread_line", "total_line",
    "team_implied_total", "opp_implied_total", "team_spread",
    "roof_type", "surface", "div_game", "rest_days",
    # The injury report is published Wednesday to Friday for a Sunday game, so
    # it is genuinely knowable before kickoff. It is the ONLY availability
    # signal here; inactives (kickoff minus 90 minutes) are not in nflverse at
    # all and would have to be scraped.
    "report_status", "practice_status", "injury_severity",
    "practice_limitation", "on_injury_report",
    # Depth-chart rank. Genuinely pregame: the chart is published before the
    # game and is re-published continuously. It is also the EARLIEST
    # availability signal there is, because a team demotes an injured player on
    # the chart before he appears on an injury report, and before the season
    # starts it is the only such signal at all.
    "depth_rank", "depth_rank_capped", "is_starter",
    # Bio. Static or knowable years in advance, so unambiguously pregame. Age
    # curves are steep in football (backs decline in their late twenties,
    # receivers often break out in year three), and draft round is the one
    # pedigree signal that exists for a player with no NFL history at all.
    "age", "experience", "draft_round", "height", "weight",
    # Venue and schedule: the football analog of park factors and day-versus-
    # night. Known weeks ahead. `roof_type` and `surface` were carried as
    # strings and silently dropped for being non-numeric, so until now the model
    # did not know whether a game was indoors.
    "roof_indoor", "is_turf", "altitude_m", "opp_rest_days", "rest_diff",
    "is_primetime", "week_of_season",
    # The officiating crew. Genuinely pregame, but ONLY from the officials
    # feed: `schedules.referee` is 100% populated for completed seasons and
    # 0% for the upcoming one, making it a postgame field like temp and wind.
    # Carried as a NAME and never used as a feature directly; the feature
    # layer turns it into the crew's prior-season tendencies.
    "referee",
})

# NextGen Stats are SAME-GAME measurements. A receiver's average separation in
# week 6 is measured during week 6's game, so these are outcomes exactly like
# targets are, and they are usable only as lagged history. Filed here so the
# feature builder must lag them and `safe_feature_columns` can never return one.
SAME_GAME_NGS_PREFIX = "ngs_"

# PFR advanced stats are SAME-GAME too: a receiver's drops in week 6 are
# counted during week 6. Same treatment as NextGen, lagged or not used.
SAME_GAME_PFR_PREFIX = "pfr_"

# Looks pregame, is not. See the module docstring.
POSTGAME_TRAP = frozenset({"temp", "wind"})


def _is_same_game_team(col: str) -> bool:
    return col.startswith(SAME_GAME_TEAM_PREFIXES) and col not in SAME_GAME_EXPOSURE


def _is_same_game_ngs(col: str) -> bool:
    return (col.startswith(SAME_GAME_NGS_PREFIX)
            or col.startswith(SAME_GAME_PFR_PREFIX))


def classify(col: str) -> str:
    """Which bucket a column belongs to. Raises on an unknown column.

    Raising is deliberate: a new column added to the backfill must be classified
    before anything can train on the table. Silently defaulting an unclassified
    column to "safe" is how leaks ship.
    """
    if col in KEYS:
        return "key"
    if col in TARGETS:
        return "target"
    if col in POSTGAME_TRAP:
        return "postgame_trap"
    if col in SAME_GAME_EXPOSURE:
        return "same_game_exposure"
    if _is_same_game_ngs(col):
        return "same_game_exposure"
    if _is_same_game_team(col):
        return "same_game_team"
    if col in PREGAME:
        return "pregame"
    raise KeyError(
        f"Column {col!r} is not classified in gridlib/columns.py. Add it to a "
        "bucket before training on this table."
    )


def safe_feature_columns(columns) -> list[str]:
    """The only columns that may be fed to a model AS THEY STAND.

    Everything else needs a lag or a rolling window applied first, which is the
    feature layer's job, not the model's.
    """
    return [c for c in columns if classify(c) == "pregame"]


def leaky_columns(columns) -> list[str]:
    """Columns that must never reach a model at their own row's index."""
    return [c for c in columns
            if classify(c) in ("same_game_team", "same_game_exposure",
                               "postgame_trap", "target")]


def audit(columns) -> dict[str, list[str]]:
    """Group every column by bucket. Used by the leakage tests and the report."""
    out: dict[str, list[str]] = {}
    for c in columns:
        out.setdefault(classify(c), []).append(c)
    return out
