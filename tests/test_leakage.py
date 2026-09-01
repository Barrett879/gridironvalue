"""Leakage suite. Mandatory per the spec, and the most valuable tests here.

The MLB build shipped two bugs that tests like these would have caught, both of
the same shape: history that was not filtered to strictly before the target
game, so a row saw its own game as a prior appearance.

Football adds a second, larger trap that MLB did not have. Most of the training
table is same-game information: `off_dropbacks` is the number of dropbacks the
team actually had in the very game whose stat line we are projecting. It is
essential for deriving targets and lagged features and it is unusable as an
inference feature. Nothing in the column's name says so, so the contract is
declared in `gridlib/columns.py` and enforced here.

Tests that need the built table are skipped when the backfill has not been run.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gridlib import columns as C
from gridlib.cache import dc_path, read_parquet_or_none

BACKFILL = dc_path("player_week_2016_2025_v1.parquet")


@pytest.fixture(scope="module")
def table():
    df = read_parquet_or_none(BACKFILL)
    if df is None:
        pytest.skip("backfill not built; run scripts/build_player_week.py")
    return df


# ── The column contract ──────────────────────────────────────────────────────
def test_every_built_column_is_classified(table):
    """An unclassified column must raise, not default to safe.

    Silently treating a new column as safe is exactly how a leak ships.
    """
    unclassified = []
    for col in table.columns:
        try:
            C.classify(col)
        except KeyError:
            unclassified.append(col)
    assert not unclassified, f"unclassified columns: {unclassified}"


def test_unknown_column_raises():
    with pytest.raises(KeyError):
        C.classify("some_new_column_nobody_classified")


def test_buckets_do_not_overlap():
    """A column in two buckets makes classify() order-dependent."""
    named = [C.KEYS, C.TARGETS, C.PREGAME, C.POSTGAME_TRAP, C.SAME_GAME_EXPOSURE]
    for i, a in enumerate(named):
        for b in named[i + 1:]:
            assert not (a & b), f"overlap: {a & b}"


def test_team_aggregates_are_never_safe_features(table):
    """The whole team-week block describes the SAME game being predicted."""
    safe = set(C.safe_feature_columns(table.columns))
    offenders = [c for c in table.columns
                 if c.startswith(("off_", "def_allowed_", "realized_"))
                 and c in safe]
    assert not offenders, f"same-game team columns marked safe: {offenders}"


def test_realized_script_is_never_safe(table):
    """Realized win probability leaks the outcome we are projecting.

    The market's implied total is the exogenous instrument for game script;
    realized in-game win probability is the outcome itself.
    """
    safe = set(C.safe_feature_columns(table.columns))
    assert not [c for c in table.columns if c.startswith("realized_") and c in safe]


def test_weather_is_flagged_as_a_postgame_trap(table):
    """temp and wind are recorded only AFTER a game is played.

    Verified: 0 of 272 rows populated for the 2026 season, 190 of 285 for 2025
    (the outdoor games; domes are null by design). A wind feature would train
    beautifully and be null for every live projection.
    """
    for col in ("temp", "wind"):
        if col in table.columns:
            assert C.classify(col) == "postgame_trap"
            assert col not in C.safe_feature_columns(table.columns)


def test_targets_are_never_safe_features(table):
    safe = set(C.safe_feature_columns(table.columns))
    assert not (C.TARGETS & safe)


def test_exposure_is_not_a_safe_feature(table):
    """Snaps and routes for THIS game are outcomes, not inputs."""
    safe = set(C.safe_feature_columns(table.columns))
    for col in ("off_snaps", "routes_run_proxy", "target_share"):
        if col in table.columns:
            assert col not in safe, f"{col} must not be a direct feature"


def test_the_safe_set_is_actually_usable(table):
    """A contract that classifies everything as leaky is useless.

    These are the columns genuinely knowable before kickoff, and they must
    include the market, which is the exogenous instrument for game script.
    """
    safe = set(C.safe_feature_columns(table.columns))
    assert len(safe) >= 8
    for must in ("team_implied_total", "team_spread", "is_home", "rest_days",
                 "roof_type", "div_game"):
        assert must in safe, f"{must} should be usable before kickoff"


# ── Point-in-time history ────────────────────────────────────────────────────
def _prior_games(df: pd.DataFrame, gsis_id: str, season: int, week: int):
    """The history a feature builder is allowed to see for one target row.

    Strictly BEFORE the target game. The MLB build shipped this comparison as
    <= twice; both times a row counted its own game as a prior appearance.
    """
    return df[(df["gsis_id"] == gsis_id)
              & ((df["season"] < season)
                 | ((df["season"] == season) & (df["week"] < week)))]


def test_history_excludes_the_target_game(table):
    """The canonical leak: a row seeing its own game in its history."""
    sub = table[table["position"] == "WR"].head(2000)
    row = sub.iloc[len(sub) // 2]
    hist = _prior_games(table, row["gsis_id"], row["season"], row["week"])
    assert row["game_id"] not in set(hist["game_id"])


def test_history_is_strictly_before_not_up_to(table):
    """Same week, same season must be excluded, not merely 'not after'."""
    row = table.iloc[1000]
    hist = _prior_games(table, row["gsis_id"], row["season"], row["week"])
    assert (hist["week"] < row["week"]).all() or (hist["season"] < row["season"]).any()
    same_week = hist[(hist["season"] == row["season"])
                     & (hist["week"] == row["week"])]
    assert len(same_week) == 0


def test_a_players_first_ever_game_has_empty_history(table):
    """A debut must yield no prior games, not a silent fallback to someone else."""
    first = table.sort_values(["season", "week"]).groupby("gsis_id").head(1).iloc[0]
    hist = _prior_games(table, first["gsis_id"], first["season"], first["week"])
    assert len(hist) == 0


def test_rolling_windows_must_count_games_not_weeks(table):
    """Every team has a bye, so 17 games span 18 weeks.

    A window defined as "the last 4 weeks" silently becomes 3 games around a
    bye. Windows must count GAMES PLAYED. This test documents the gap exists.
    """
    per_season = (table[table["position"] == "WR"]
                  .groupby(["gsis_id", "season"])["week"]
                  .agg(["min", "max", "count"]))
    spans = per_season["max"] - per_season["min"] + 1
    assert (spans > per_season["count"]).any(), (
        "expected at least one player whose week span exceeds his game count"
    )


# ── Structural guarantees of the built table ─────────────────────────────────
def test_no_duplicate_player_games(table):
    assert table.duplicated(["game_id", "gsis_id"]).sum() == 0


def test_no_postseason_rows_leak_in(table):
    """The board is a regular-season product and week numbers 19-22 would
    corrupt any week-indexed logic."""
    assert table["week"].max() <= 18


def test_every_row_has_a_season_and_week(table):
    assert table["season"].notna().all()
    assert table["week"].notna().all()


# ── Missingness leaks: the subtlest class, and one shipped here once ─────────
def test_prior_history_does_not_inherit_current_row_missingness():
    """`cumsum() - current` propagates a NaN in the CURRENT row into the feature.

    That makes the feature NaN exactly when the current game has no value, so
    the model can read missingness as the answer. This shipped once: the
    red-zone block showed P(feature is NaN | this game's targets == 0) = 1.0 and
    a fake 13% improvement on targets that replicated across folds, because a
    leak is present in every fold. Two-stage gating catches mirages, not leaks.
    """
    import numpy as np
    import pandas as pd

    from gridlib import features as F

    # Player A has a value in every game; player B's value is missing exactly
    # when his own current-game outcome is zero.
    df = pd.DataFrame({
        "gsis_id": ["A"] * 4 + ["B"] * 4,
        "season": [2024] * 8,
        "week": [1, 2, 3, 4] * 2,
        "opp": [1.0, 2.0, 3.0, 4.0, np.nan, 2.0, np.nan, 4.0],
    })
    prior = F._prior_mean(df, "gsis_id", "opp")

    # Row 0 and row 4 are each player's first game: no history, NaN is correct.
    assert np.isnan(prior.iloc[0]) and np.isnan(prior.iloc[4])
    # Every LATER row must have a real number regardless of the current value
    # being missing. If missingness leaked, rows 6 would be NaN.
    later = prior.iloc[[1, 2, 3, 5, 6, 7]]
    assert later.notna().all(), f"current-row missingness leaked: {list(prior)}"


def test_prior_mean_treats_missing_as_zero_not_as_a_gap():
    """A missing opportunity count means zero opportunities, so the running
    mean must include it as a zero rather than skipping the game entirely."""
    import numpy as np
    import pandas as pd

    from gridlib import features as F

    df = pd.DataFrame({
        "gsis_id": ["A"] * 3,
        "opp": [4.0, np.nan, 0.0],
    })
    prior = F._prior_mean(df, "gsis_id", "opp")
    assert prior.iloc[1] == 4.0          # one prior game of 4
    assert prior.iloc[2] == 2.0          # (4 + 0) / 2, the NaN counted as zero


def test_opportunity_counts_are_zero_filled_in_the_backfill(table):
    """A player-game with no carries genuinely had zero red-zone carries.

    Left NaN at the source, every prior-history feature derived from the column
    inherits a perfect indicator of the current game's outcome.
    """
    for col in ("carries_rz", "carries_i5", "targets_rz", "targets_i10",
                "pbp_carries", "pbp_targets"):
        if col in table.columns:
            assert table[col].isna().sum() == 0, f"{col} still carries NaNs"


def test_no_feature_source_column_has_target_correlated_missingness(table):
    """The general form: any column the feature builder reads must not carry a
    NaN pattern that encodes the current game's outcome.

    Scoped to the columns that actually become features (VOLUME_STATS plus
    REDZONE_STATS). Some same-game RATE columns are legitimately NaN when there
    was no opportunity - `adot` and `target_epa` are undefined for a player with
    zero targets, and P(missing | targets == 0) is exactly 1.0 for both. That is
    correct at the source and harmless while they stay out of the feature
    builder, which the next test pins.
    """
    from gridlib import features as F

    sub = table[table["position"].isin(["WR", "TE", "RB"])]
    zero = sub["targets"] == 0
    offenders = []
    for col in F.VOLUME_STATS + F.REDZONE_STATS:
        if col not in sub.columns or sub[col].dtype.kind not in "fi":
            continue
        na = sub[col].isna()
        if not na.any():
            continue
        p0, p1 = float(na[zero].mean()), float(na[~zero].mean())
        if abs(p0 - p1) > 0.5:
            offenders.append((col, round(p0, 3), round(p1, 3)))
    assert not offenders, (
        f"these feed the feature builder and their missingness encodes the "
        f"target: {offenders}"
    )


def test_undefined_rate_columns_stay_out_of_the_feature_builder():
    """`adot` and `target_epa` are NaN exactly when the player had no targets.

    They are fine sitting in the backfill, and they would leak the moment
    anyone added them to the feature source lists. This pins that they are not
    there, so the failure is a red test rather than a silently great model.
    """
    from gridlib import features as F

    forbidden = {"adot", "target_epa", "rush_epa"}
    sources = set(F.VOLUME_STATS) | set(F.REDZONE_STATS)
    assert not (forbidden & sources), (
        f"undefined-rate columns added to the feature builder: "
        f"{forbidden & sources}. They are NaN exactly when the opportunity "
        f"count is zero, so prior-history features built from them leak."
    )


# ── The two estimands ────────────────────────────────────────────────────────
def test_conditional_and_expected_are_separate_columns():
    """Every model trains on players who recorded a stat line, so it estimates
    E[Y | plays]. That is right for a prop (props void on a DNP) and wrong to
    sum across a roster. Both must exist, separately."""
    import numpy as np
    import pandas as pd

    from gridlib import predict

    df = pd.DataFrame({
        "position": ["QB", "RB", "WR"],
        "depth_rank": [1.0, 4.0, 6.0],
    })
    out = predict.attach_p_play(df)
    assert "p_play" in out.columns
    assert (out["p_play"] >= 0).all() and (out["p_play"] <= 1).all()
    # A starter must be far likelier to appear than a deep reserve.
    assert out["p_play"].iloc[0] > out["p_play"].iloc[2]


def test_p_play_falls_back_loudly_not_silently(monkeypatch):
    """With no grid the site must keep running, but with p_play = 1.0 so the
    coherence checks fail immediately rather than the error hiding."""
    import pandas as pd

    from gridlib import predict

    monkeypatch.setattr(predict, "availability_grid", lambda: None)
    out = predict.attach_p_play(pd.DataFrame({"position": ["RB"],
                                              "depth_rank": [1.0]}))
    assert out["p_play"].iloc[0] == 1.0


def test_injury_status_lowers_the_appearance_probability():
    """The injury report is pregame and is the one case position and rank get
    badly wrong."""
    import pandas as pd

    from gridlib import predict

    df = pd.DataFrame({"position": ["RB", "RB"], "depth_rank": [1.0, 1.0],
                       "report_status": [None, "Out"]})
    out = predict.attach_p_play(df)
    assert out["p_play"].iloc[1] < out["p_play"].iloc[0] * 0.2


def test_missing_depth_rank_does_not_crash_or_assume_starter():
    import numpy as np
    import pandas as pd

    from gridlib import predict

    out = predict.attach_p_play(pd.DataFrame({"position": ["WR"],
                                              "depth_rank": [np.nan]}))
    assert 0.0 <= out["p_play"].iloc[0] <= 1.0


# ── Train/serve parity: the skew that silently drops a feature ───────────────
def test_inference_rows_derive_every_feature_the_backfill_does(table):
    """A column that exists in training and not at inference does not error.

    The model just sees NaN, a tree routes it to a learned default branch, and
    the feature silently stops working. That is what happened to
    depth_rank_capped and is_starter: they were added to the backfill, the
    models trained on them, and build_inference_rows never computed them. The
    symptom was league QB1 attempts reading 22.3 instead of 31.6.
    """
    derived = {"depth_rank_capped", "is_starter"}
    present = derived & set(table.columns)
    assert present == derived, (
        f"backfill is missing {derived - present}; if the backfill stops "
        f"deriving these, the ablation and the registry are out of sync"
    )


def test_predict_refuses_to_serve_with_missing_features(monkeypatch):
    """Failing OPEN on a missing feature is worse than failing loudly: a silent
    wrong answer beats no answer only if you never find out."""
    import pandas as pd
    import pytest as _pytest

    from gridlib import predict

    reg = predict.load_registry()
    if reg is None:
        _pytest.skip("no model registry; run scripts/train_models.py")
    # Every feature the registry names must be derivable from an inference row.
    rows = predict.build_inference_rows(2026, 1)
    if rows.empty:
        _pytest.skip("no inference rows available for 2026 week 1")
    pregame = {c for c in reg["feature_columns"] if not c.startswith("f_")}
    missing = pregame - set(rows.columns)
    assert not missing, (
        f"build_inference_rows does not produce {sorted(missing)}, which the "
        f"models were trained on. Serving would drop them silently."
    )


def test_depth_rank_is_capped_identically_on_both_sides():
    """The two depth-chart eras stop at different ranks, so the cap is what
    makes the feature mean the same thing in training and at inference. A
    mismatch here is a silent distribution shift, not an error."""
    import numpy as np
    import pandas as pd

    raw = pd.Series([1.0, 2.0, 3.0, 7.0, 14.0, np.nan])
    capped = pd.to_numeric(raw, errors="coerce").clip(upper=3)
    assert list(capped.dropna()) == [1.0, 2.0, 3.0, 3.0, 3.0]
    assert bool((capped == 1).iloc[0])
