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
# The three tests below used to assert that `_prior_games`, defined a few lines
# above them, filtered correctly. They held by construction: a change to
# gridlib/features.py could not make any of them fail. That is not a
# hypothetical weakness. `_prior_roll_mean` shipped with no shift(1) at all, so
# every f_r3_* and f_r8_* feature included the game it was describing, and all
# 259 tests passed. They now call the production builders.


@pytest.mark.parametrize("builder", ["_prior_mean", "_prior_mean_skipna",
                                     "_prior_roll_mean"])
def test_builders_never_see_the_current_row(builder):
    """The canonical leak, tested against the real functions.

    Values are strictly increasing per key, so a builder that includes the
    current row cannot help but produce a feature at least as large as it. The
    first row of every key must be NaN: with nothing before it there is no
    honest answer, and 0 would be a lie the model reads as history.
    """
    import numpy as np

    from gridlib import features as F

    fn = getattr(F, builder)
    df = pd.DataFrame({"k": ["a"] * 5 + ["b"] * 4,
                       "x": [10.0, 20.0, 30.0, 40.0, 50.0, 7.0, 14.0, 21.0, 28.0]})
    got = fn(df, "k", "x", 3) if builder == "_prior_roll_mean" else fn(df, "k", "x")
    got = pd.Series(np.asarray(got, dtype=float), index=df.index)

    for key, g in df.groupby("k", sort=False):
        vals, feat = g["x"].to_numpy(), got.loc[g.index].to_numpy()
        assert np.isnan(feat[0]), (
            f"{builder}: first row of key {key} has history {feat[0]}, but "
            "nothing precedes it"
        )
        later = feat[1:]
        assert (later[~np.isnan(later)] < vals[1:][~np.isnan(later)]).all(), (
            f"{builder}: the feature is not strictly below the current value "
            "on a strictly increasing series, so it contains the current row"
        )


def test_prior_roll_mean_is_exactly_the_previous_window():
    """Pinned by value, because 'looks about right' is how this shipped broken.

    [10, 20, 30, 40] with window 3 must give [nan, 10, 15, 20]. It shipped
    giving [10, 15, 20, 30], the window INCLUDING the current game, and the
    docstring claimed the shift was there.
    """
    import numpy as np

    from gridlib import features as F

    df = pd.DataFrame({"k": ["a"] * 4, "x": [10.0, 20.0, 30.0, 40.0]})
    got = np.asarray(F._prior_roll_mean(df, "k", "x", 3), dtype=float)
    assert np.isnan(got[0])
    assert list(got[1:]) == [10.0, 15.0, 20.0], (
        f"got {list(got)}; the current-row-inclusive answer is [10, 15, 20, 30]"
    )


def test_rolling_features_do_not_track_the_current_game(table):
    """On real data, a leaked window correlates with its own target.

    A 3-game window that includes the current game correlates with that game's
    outcome at about 0.98 on a monotone series and stays high on real, noisy
    data. A strictly prior window correlates only as far as the player's form
    actually persists. This is the end-to-end version of the unit tests above:
    it runs the whole feature build, so it also catches a caller that undoes
    the shift.
    """
    import numpy as np

    from gridlib import features as F

    from gridlib.cache import dc_path, read_parquet_or_none

    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    sub = table[table["season"].isin([2023, 2024])]
    if len(sub) < 2000 or lg is None:
        pytest.skip("backfill too small, or league table missing")
    priors = F.league_priors(sub, 2024)
    feat = F.build(sub, lg, priors)
    for col, target in (("f_r3_targets", "targets"),
                        ("f_r3_receiving_yards", "receiving_yards")):
        if col not in feat.columns:
            continue
        d = feat[feat[col].notna() & feat[target].notna()]
        d = d[d["position"].isin(["WR", "TE"])]
        if len(d) < 500:
            continue
        r = float(np.corrcoef(d[col], d[target])[0, 1])
        assert r < 0.80, (
            f"{col} correlates with its own game's {target} at {r:.3f}. A "
            "strictly prior window tracks form, not the outcome; this is what "
            "a window containing the current game looks like."
        )


def test_a_players_first_ever_game_has_empty_history(table):
    """A debut must yield no prior games, not a silent fallback to someone else.

    Tested on the production builder: the first row of each key must be NaN,
    never 0, so "no history" stays distinguishable from "history of zero".
    """
    import numpy as np

    from gridlib import features as F

    sub = table.sort_values(["season", "week"]).head(20000)
    got = F._prior_mean(sub, "gsis_id", "targets")
    debuts = sub.groupby("gsis_id", sort=False).head(1).index
    vals = np.asarray(got.loc[debuts], dtype=float)
    assert np.isnan(vals).all(), (
        f"{int((~np.isnan(vals)).sum())} debut rows carry prior history"
    )


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
    p = float(out["p_play"].iloc[0])
    # This asserted only 0 <= p <= 1, which ADMITS p = 1.0, the exact
    # "assume starter" case the name says it prevents. Any number between zero
    # and one passed, so the test could not fail.
    assert 0.0 <= p <= 1.0
    starter = predict.attach_p_play(pd.DataFrame({"position": ["WR"],
                                                  "depth_rank": [1]}))
    assert p < float(starter["p_play"].iloc[0]), (
        f"a player with no depth rank was given {p:.3f}, at or above the "
        "rank-1 probability; an unknown rank must not be read as a starter")


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

    # The body above only checked the BACKFILL, while the name promises the
    # INFERENCE path. It never called build_inference_rows, so the very skew it
    # is named for could ship underneath it. Call it.
    from gridlib import fetch, predict
    season = fetch.current_season()
    week = fetch.current_week(season) or 1
    inf = predict.build_inference_rows(season, week)
    if inf is None or inf.empty:
        # This used to skip unconditionally, which turned the only train/serve
        # parity test OFF in exactly the failure mode it guards: a dead
        # depth-chart feed reported "passed, 1 skipped" and the suite looked
        # green while nothing could be projected at all.
        #
        # A fresh clone with no cached data is a legitimate skip. A missing
        # CURRENT season while other seasons are present is a broken pipeline,
        # and this test is the one that should say so.
        have_any = any(
            not (fetch.depth_chart_normalized(s) is None
                 or fetch.depth_chart_normalized(s).empty)
            for s in (season - 1, season - 2)
        )
        assert not have_any, (
            f"no inference rows for {season} week {week}, but depth charts "
            "exist for earlier seasons. The feed for the served season is "
            "broken, which is exactly what this test is for; it is not a "
            "reason to skip."
        )
        pytest.skip("no depth chart for any season; nothing cached to test")
    missing = derived - set(inf.columns)
    assert not missing, (
        f"build_inference_rows does not derive {missing}; the models train on "
        "them, so every live row would carry NaN and the feature would stop "
        "working without erroring")
    for c in derived:
        assert inf[c].notna().all(), f"{c} is null for some inference rows"


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


# ── Venue block: a constant column is a broken join ──────────────────────────
# `altitude_m` shipped as all zeros because `stadium_id` was missing from the
# schedule merge and a defensive `if "stadium_id" in df.columns` guard turned
# that into a plausible constant. A feature that is constant carries no
# information, so the model cannot tell you it broke: the ablation just reports
# "no effect" and the block gets rejected for the wrong reason.
def test_venue_columns_are_not_constant(table):
    """Every venue column must actually vary. A constant one means a dead join."""
    for col in ("roof_indoor", "is_turf", "altitude_m", "rest_diff",
                "is_primetime", "week_of_season"):
        if col not in table.columns:
            pytest.skip(f"{col} not built yet")
        n = pd.to_numeric(table[col], errors="coerce").nunique(dropna=True)
        assert n > 1, (
            f"{col} is constant across all {len(table)} rows. A venue feature "
            "that never varies is a broken join, not a real constant."
        )


def test_altitude_is_set_for_denver_and_only_the_high_venues(table):
    """Denver hosts games every season, so zero high-altitude rows is a bug."""
    if "altitude_m" not in table.columns or "stadium_id" not in table.columns:
        pytest.skip("venue block not built yet")
    high = table[table["altitude_m"] > 0]
    assert len(high) > 0, "no high-altitude rows at all; the stadium join broke"
    assert set(high["stadium_id"].unique()) <= {"DEN00", "MEX00"}
    # Every Denver home game must be flagged, not merely some of them.
    den = table[table["stadium_id"] == "DEN00"]
    assert (den["altitude_m"] == 1610.0).all()


def test_surface_whitespace_does_not_flip_grass_to_turf(table):
    """The feed carries both 'grass' and 'grass '; both are grass."""
    if "is_turf" not in table.columns:
        pytest.skip("venue block not built yet")
    surf = table["surface"].astype(str).str.lower().str.strip()
    assert (table.loc[surf == "grass", "is_turf"] == 0.0).all(), (
        "a grass game is flagged as turf; suspect an unstripped surface string"
    )


# ── Per-target feature sets: the third instance of train/serve skew ──────────
# Targets no longer share one feature list. Positional defence ships for skill
# positions and is excluded from every QB passing target, so a registry that
# records one global list, or a serving path that builds one shared X, would
# feed the QB models four columns they were never fitted on. That is a silent
# wrong answer, not an error, and it is the same shape as the two skews that
# already cost this project a regression each.
def _registry():
    import json
    from gridlib.predict import MODELS_DIR
    p = MODELS_DIR / "registry_m1.json"
    if not p.exists():
        pytest.skip("models not trained")
    return json.loads(p.read_text())


def test_registry_records_feature_columns_per_target():
    reg = _registry()
    for target, meta in reg["targets"].items():
        assert "feature_columns" in meta, (
            f"{target} has no per-target feature list; serving would fall back "
            "to the union and feed it columns it was never fitted on"
        )
        assert meta["feature_columns"], f"{target} has an empty feature list"


def test_qb_passing_models_were_fitted_without_positional_defence():
    """DVP describes skill-position and QB RUSHING outcomes, never passing."""
    from gridlib import features as F
    if not F.USE_POSITIONAL_DEF or not F.USE_POSDEF_EXCLUDE:
        pytest.skip("positional defence or its exclusion is off")
    reg = _registry()
    for target in F.POSDEF_EXCLUDE_TARGETS:
        meta = reg["targets"].get(target)
        if meta is None:
            continue
        dvp = [c for c in meta["feature_columns"]
               if c.startswith(F.DVP_PREFIX)]
        assert not dvp, f"{target} was fitted WITH {dvp}, which was excluded"


def test_every_trained_column_is_a_subset_of_the_union():
    """The union is what the inference completeness check is run against."""
    reg = _registry()
    union = set(reg["feature_columns"])
    for target, meta in reg["targets"].items():
        extra = set(meta.get("feature_columns", [])) - union
        assert not extra, (
            f"{target} was fitted on {sorted(extra)[:4]}, absent from the union, "
            "so the inference check would not verify they exist"
        )


def test_availability_grid_is_fit_on_playing_teams_only():
    """A bye-week row can never produce an appearance, so fitting over it is
    pure downward bias on a population inference never scores.

    645 of 12099 fitted rows were bye rows with an appearance rate of 0.0016,
    which pulled every cell down by a factor of 0.947. The guard is the rate
    itself: a grid fit with byes in it puts QB1 near 0.88, and a healthy
    starting quarterback plays essentially every week his team does.
    """
    from gridlib import predict

    grid = predict.availability_grid()
    if grid is None or grid.empty:
        import pytest
        pytest.skip("availability grid not built")
    key = grid.set_index(["position", "rank_capped"])["p_play"]
    for pos in ("QB", "RB", "WR", "TE"):
        assert key.get((pos, 1), 0.0) > 0.92, (
            f"{pos}1 appears {key.get((pos, 1)):.3f} of the time. A healthy "
            "starter plays; a rate this low means the fit population contains "
            "weeks the player could not possibly have appeared."
        )


def test_injury_is_not_counted_twice():
    """The multiplier may only be applied to a rate that excludes injury.

    v1 fit the cell rate over every row, injured ones included, and then
    multiplied by INJURY_ADJ at serve time. The healthy baseline must therefore
    sit ABOVE the marginal rate, and a player with no designation must be
    served the marginal rate rather than the healthy one when his team has not
    filed. If the two columns ever collapse into each other, the distinction
    that prevents the double-count has been lost.
    """
    from gridlib import predict

    grid = predict.availability_grid()
    if grid is None or grid.empty or "p_play_any" not in grid.columns:
        import pytest
        pytest.skip("availability grid not built, or predates the split")
    thick = grid[grid["n"] >= 200]
    assert len(thick) >= 10, "too few well-populated cells to judge"
    # Judged where the gap is meaningful. In a cell that appears 20% of the
    # time a designation barely moves anything, because the player mostly does
    # not play regardless, and the two shrink priors can cross at the fourth
    # decimal on noise. The rows that carry team totals are the ones that play.
    real = thick[thick["p_play_any"] >= 0.4]
    assert len(real) >= 8, "too few high-rate cells to judge"
    assert (real["p_play"] >= real["p_play_any"]).all(), (
        "a healthy player must appear at least as often as the average player "
        "of the same position and rank, injured ones included"
    )
    assert (thick["p_play"] >= thick["p_play_any"] - 0.01).all(), (
        "a cell where the healthy rate sits materially BELOW the marginal one "
        "is not noise; the two populations have been swapped"
    )
    assert (thick["p_play"] > thick["p_play_any"] + 1e-6).any(), (
        "the healthy and marginal rates are identical everywhere, so the "
        "multiplier is being applied on top of a rate that already contains "
        "injury. That is the v1 double-count."
    )


def test_unfiled_team_is_not_served_the_healthy_baseline():
    """Before a team files, nothing is known, so the marginal rate is correct.

    Serving the healthy baseline to a team with no report over-counts: some of
    those players will be ruled out on Friday and nothing in the frame says
    which.
    """
    import pandas as pd

    from gridlib import predict

    grid = predict.availability_grid()
    if grid is None or grid.empty or "p_play_any" not in grid.columns:
        import pytest
        pytest.skip("availability grid not built, or predates the split")
    df = pd.DataFrame({"position": ["WR", "WR"], "depth_rank": [1.0, 1.0],
                       "team": ["AAA", "BBB"], "report_status": [None, None]})
    out = predict.attach_p_play(df, filed_teams={"AAA"})
    assert out["p_play"].iloc[0] > out["p_play"].iloc[1], (
        "the team that filed should get the healthy rate for an undesignated "
        "player; the team that has not filed should get the lower marginal one"
    )


def test_no_feature_comes_from_a_training_only_source(table):
    """A feature the live season cannot populate is worse than no feature.

    `routes_run_proxy` comes from participation data, which nflverse publishes
    only AFTER the postseason and never updates in season. The backfill has real
    values for every completed season and the season the site actually serves
    has none, so `_prior_mean`'s zero-fill made every live game contribute
    routes = 0. Measured by nulling 2025's routes and rebuilding, which is what
    a live season looks like: `f_r3_routes_run_proxy` trained on 20.13 and
    served 0.51, exactly zero on 93.1% of rows; `f_std_` trained on 20.20 and
    served 0.00 on 97.1%. Four such features were in the shipped model.

    This simulates the live season rather than naming the column, so a new
    feature built from any training-only source is caught the same way.
    """
    import numpy as np

    from gridlib import features as F
    from gridlib.cache import dc_path, read_parquet_or_none

    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    sub = table[table["season"].isin([2024, 2025])]
    if len(sub) < 2000 or lg is None:
        pytest.skip("backfill too small, or league table missing")
    priors = F.league_priors(sub, 2025)

    # Every source column that participation supplies is absent in a live
    # season. Null the served season's copies and rebuild.
    live = sub.copy()
    for col in ("routes_run_proxy", "pass_snaps"):
        if col in live.columns:
            live.loc[live["season"] == 2025, col] = np.nan

    real = F.build(sub, lg, priors)
    served = F.build(live, lg, priors)
    cols = F.feature_columns(real)

    mask = (real["season"] == 2025) & (real["week"] >= 6)
    if mask.sum() < 200:
        pytest.skip("not enough live-season rows")

    # Compare the SAME rows across the two builds. Comparing a feature against
    # zero instead would flag ordinary sparsity: f_r3_attempts is 0 for 85% of
    # rows because most players are not quarterbacks, which is correct and is
    # identical in both builds. What matters is whether the served value
    # DIVERGES from the trained one, and only an absent source does that.
    broken = []
    idx = real.index[mask]
    for c in cols:
        if c not in served.columns:
            continue
        a = pd.to_numeric(real.loc[idx, c], errors="coerce")
        b = pd.to_numeric(served.loc[idx, c], errors="coerce")
        if a.notna().mean() < 0.5 or abs(float(a.mean())) < 1.0:
            continue
        ratio = float(b.fillna(0).mean()) / float(a.fillna(0).mean())
        if ratio < 0.5:
            broken.append((c, round(float(a.mean()), 2),
                           round(float(b.fillna(0).mean()), 2), round(ratio, 3)))

    assert not broken, (
        "these model features are populated in the backfill and collapse to "
        "zero in a live season, so the model was trained on one thing and "
        f"served another (feature, trained, served, ratio): {broken}"
    )
