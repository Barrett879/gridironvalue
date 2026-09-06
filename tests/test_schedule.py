"""Schedule, market and timing logic.

These use synthetic frames rather than the network so they run offline and
deterministically. The one live-data test is marked `network` and can be
deselected with `-m "not network"`.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from gridlib import fetch
from gridlib.fetch import EASTERN


def _games(rows):
    """Build a minimal games frame with the columns the code actually reads."""
    cols = [
        "game_id", "season", "game_type", "week", "gameday", "weekday",
        "gametime", "away_team", "home_team", "away_score", "home_score",
        "spread_line", "total_line", "stadium_id", "stadium", "roof", "location",
    ]
    return pd.DataFrame(rows, columns=cols)


ROW = dict(
    game_id="2026_01_NE_SEA", season=2026, game_type="REG", week=1,
    gameday="2026-09-13", weekday="Sunday", gametime="13:00",
    away_team="NE", home_team="SEA", away_score=None, home_score=None,
    spread_line=3.5, total_line=44.5, stadium_id="SEA00", stadium="Lumen Field",
    roof="outdoors", location="Home",
)


# ── Implied totals: the sign convention is the whole test ────────────────────
def test_spread_line_is_from_the_home_perspective():
    """Positive spread_line means the HOME team is favored.

    total/2 + spread/2 = home, total/2 - spread/2 = away. Getting this
    backwards inverts every game script in the model, so it is pinned here.
    """
    g = _games([{**ROW, "spread_line": 7.0, "total_line": 49.5}])
    out = fetch.implied_team_totals(g)
    assert out["home_implied_total"].iloc[0] == pytest.approx(28.25)
    assert out["away_implied_total"].iloc[0] == pytest.approx(21.25)
    # The favorite gets the bigger number.
    assert out["home_implied_total"].iloc[0] > out["away_implied_total"].iloc[0]


def test_negative_spread_favors_the_away_team():
    g = _games([{**ROW, "spread_line": -2.5, "total_line": 47.5}])
    out = fetch.implied_team_totals(g)
    assert out["away_implied_total"].iloc[0] == pytest.approx(25.0)
    assert out["home_implied_total"].iloc[0] == pytest.approx(22.5)


def test_implied_totals_sum_to_the_game_total():
    g = _games([{**ROW, "spread_line": -6.5, "total_line": 41.0}])
    out = fetch.implied_team_totals(g)
    total = out["home_implied_total"].iloc[0] + out["away_implied_total"].iloc[0]
    assert total == pytest.approx(41.0)


def test_missing_line_yields_missing_totals_not_zero():
    """A game with no posted line must not silently imply a 0-0 script."""
    g = _games([{**ROW, "spread_line": None, "total_line": None}])
    out = fetch.implied_team_totals(g)
    assert pd.isna(out["home_implied_total"].iloc[0])
    assert pd.isna(out["away_implied_total"].iloc[0])


# ── Kickoff time ─────────────────────────────────────────────────────────────
def test_kickoff_is_eastern_regardless_of_venue():
    """gametime is US Eastern for every game, including the ones abroad."""
    g = _games([{**ROW, "gameday": "2026-10-05", "gametime": "09:30",
                 "stadium": "Tottenham Hotspur Stadium", "location": "Neutral"}])
    k = fetch.kickoff_series(g).iloc[0]
    assert k.hour == 9 and k.minute == 30
    assert k.tzinfo is not None
    assert k.tz_convert(EASTERN).hour == 9


def test_missing_gametime_falls_back_rather_than_vanishing():
    """A NaT kickoff would drop the game out of a sorted slate entirely."""
    g = _games([{**ROW, "gametime": None}])
    k = fetch.kickoff_series(g).iloc[0]
    assert not pd.isna(k)
    assert k.hour == 13


def test_lock_is_ninety_minutes_before_each_kickoff():
    """Inactives publish per game, not on a wall-clock Sunday cutoff."""
    g = _games([
        {**ROW, "gametime": "13:00"},
        {**ROW, "game_id": "x", "gametime": "20:20", "away_team": "DAL",
         "home_team": "NYG"},
    ])
    k = fetch.kickoff_series(g)
    locks = k - pd.Timedelta(minutes=90)
    assert locks.iloc[0].hour == 11 and locks.iloc[0].minute == 30
    assert locks.iloc[1].hour == 18 and locks.iloc[1].minute == 50


# ── Broadcast windows ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "gameday,gametime,expected",
    [
        ("2026-09-10", "20:15", "Thursday night"),
        ("2026-09-12", "16:30", "Saturday"),
        ("2026-09-13", "09:30", "Sunday morning"),   # the London slot
        ("2026-09-13", "13:00", "Sunday early"),
        ("2026-09-13", "16:25", "Sunday late"),
        ("2026-09-13", "20:20", "Sunday night"),
        ("2026-09-14", "20:15", "Monday night"),
    ],
)
def test_window_labels(gameday, gametime, expected):
    g = _games([{**ROW, "gameday": gameday, "gametime": gametime}])
    k = fetch.kickoff_series(g).iloc[0]
    assert fetch._window_label(k) == expected


# ── Season and week resolution ───────────────────────────────────────────────
def _two_season_frame():
    rows = []
    for wk in range(1, 4):
        rows.append({**ROW, "season": 2025, "week": wk,
                     "gameday": f"2025-09-{7 + wk * 7:02d}",
                     "game_id": f"2025_0{wk}", "home_score": 20, "away_score": 17})
    for wk in range(1, 4):
        rows.append({**ROW, "season": 2026, "week": wk,
                     "gameday": f"2026-09-{6 + wk * 7:02d}",
                     "game_id": f"2026_0{wk}"})
    return _games(rows)


def test_current_season_rolls_over_before_kickoff():
    """nflreadpy.get_current_season() returned 2025 on 2026-08-31, ten days
    before the 2026 opener, which is why we resolve this from the schedule."""
    g = _two_season_frame()
    now = dt.datetime(2026, 8, 31, 12, 0, tzinfo=EASTERN)
    assert fetch.current_season(g, now) == 2026


def test_current_season_stays_put_deep_in_the_offseason():
    g = _two_season_frame()
    now = dt.datetime(2026, 3, 1, 12, 0, tzinfo=EASTERN)
    assert fetch.current_season(g, now) == 2025


def test_current_week_is_the_first_week_still_to_kick_off():
    g = _two_season_frame()
    # After week 1 (Sep 13) but before week 2 (Sep 20).
    now = dt.datetime(2026, 9, 15, 12, 0, tzinfo=EASTERN)
    assert fetch.current_week(2026, g, now) == 2


def test_current_week_pins_to_the_last_week_once_the_season_ends():
    g = _two_season_frame()
    now = dt.datetime(2026, 12, 31, 12, 0, tzinfo=EASTERN)
    assert fetch.current_week(2026, g, now) == 3


# ── Corrections layer ────────────────────────────────────────────────────────
def test_retractable_roof_is_structure_not_game_state():
    """The feed leaves `roof` empty for a retractable until after the game, then
    writes closed/open. `roof_type` must carry the structural fact instead."""
    g = _games([{**ROW, "stadium_id": "HOU00", "stadium": "Reliant Stadium",
                 "roof": ""}])
    out = fetch.apply_stadium_corrections(g)
    assert out["roof_type"].iloc[0] == "retractable"


def test_stale_stadium_name_is_corrected():
    """The 2026 file regressed Houston to a name retired in 2014."""
    g = _games([{**ROW, "stadium_id": "HOU00", "stadium": "Reliant Stadium"}])
    out = fetch.apply_stadium_corrections(g)
    assert out["stadium"].iloc[0] == "NRG Stadium"


def test_wrong_roof_on_an_international_venue_is_corrected():
    """The feed calls the open-air MCG a dome."""
    g = _games([{**ROW, "stadium_id": "MEL00",
                 "stadium": "Melbourne Cricket Ground", "roof": "dome"}])
    out = fetch.apply_stadium_corrections(g)
    assert out["roof_type"].iloc[0] == "outdoors"


def test_uncorrected_stadium_falls_back_to_the_feed():
    g = _games([{**ROW, "stadium_id": "SEA00", "roof": "outdoors"}])
    out = fetch.apply_stadium_corrections(g)
    assert out["roof_type"].iloc[0] == "outdoors"


def test_post_game_roof_state_maps_to_structure():
    """A completed retractable game says closed/open; both mean retractable."""
    for state in ("closed", "open"):
        g = _games([{**ROW, "stadium_id": "UNKNOWN00", "roof": state}])
        out = fetch.apply_stadium_corrections(g)
        assert out["roof_type"].iloc[0] == "retractable"


# ── One live check, opt-out with -m "not network" ────────────────────────────
@pytest.mark.network
def test_live_schedule_has_the_shape_we_depend_on():
    g = fetch.load_schedules()
    assert {"spread_line", "total_line", "stadium_id", "espn"} <= set(g.columns)
    # 272 regular-season games a year since the 17-game schedule arrived.
    n2025 = len(g[(g["season"] == 2025) & (g["game_type"] == "REG")])
    assert n2025 == 272


# ── Deep-link resolution ─────────────────────────────────────────────────────
def _week1_frame():
    return _games([
        {**ROW, "game_id": "2026_01_NE_SEA", "away_team": "NE", "home_team": "SEA"},
        {**ROW, "game_id": "2026_01_CLE_JAX", "away_team": "CLE", "home_team": "JAX"},
    ])


def test_link_resolves_to_the_right_game():
    g = _week1_frame()
    hit = fetch.find_game(2026, 1, "NE", "SEA", g)
    assert hit is not None and hit["game_id"] == "2026_01_NE_SEA"


def test_link_resolution_folds_aliases():
    """A link written with an old abbreviation still opens the game."""
    g = _games([{**ROW, "game_id": "2026_01_LAC_LA", "away_team": "LAC",
                 "home_team": "LA"}])
    assert fetch.find_game(2026, 1, "SD", "LAR", g) is not None


def test_link_resolution_tolerates_reversed_teams():
    """The matchup is unambiguous either way; dead-ending would be unhelpful."""
    g = _week1_frame()
    hit = fetch.find_game(2026, 1, "SEA", "NE", g)
    assert hit is not None and hit["game_id"] == "2026_01_NE_SEA"


def test_unknown_matchup_returns_none_so_the_page_can_explain():
    g = _week1_frame()
    assert fetch.find_game(2026, 1, "NE", "JAX", g) is None


def test_missing_team_param_returns_none():
    g = _week1_frame()
    assert fetch.find_game(2026, 1, "", "SEA", g) is None
    assert fetch.find_game(2026, 1, "NE", None, g) is None


def test_week_with_no_games_returns_none():
    g = _week1_frame()
    assert fetch.find_game(2026, 14, "NE", "SEA", g) is None


def test_depth_chart_memo_returns_the_same_data_and_stays_bounded():
    """The memo exists for memory, so it must not become a memory leak, and it
    must not hand back a frame someone else mutated.

    `build_inference_rows` asks for the depth chart 32 times a week, once per
    team per game, each time to keep about 21 rows out of 554,215. Measured
    before the memo: 0.45 s and ~100 MB of RSS per call that never came back,
    17.3 s and a 476 MB peak for one week's board, on a container with about
    1 GB. The risk a memo introduces is a stale or shared-and-mutated frame, so
    that is what this pins.
    """
    from gridlib import fetch

    fetch.clear_depth_chart_memo()
    first = fetch.load_depth_charts(2025)
    if first is None or first.empty:
        pytest.skip("2025 depth chart not cached")
    baseline = first.copy()

    assert fetch.load_depth_charts(2025).equals(baseline), "memo hit differs"
    fetch.clear_depth_chart_memo()
    assert fetch.load_depth_charts(2025).equals(baseline), "re-derived differs"

    # The one consumer that overwrites a column must not reach the shared frame.
    fetch.depth_chart_normalized(2025)
    assert fetch.load_depth_charts(2025).equals(baseline), (
        "depth_chart_normalized mutated the shared memoized chart"
    )

    fetch.clear_depth_chart_memo()
    for season in (2023, 2024, 2025):
        fetch.load_depth_charts(season)
    assert len(fetch._DEPTH_MEMO) <= fetch._DEPTH_MEMO_MAX, (
        "the memo is unbounded, which is a slower version of the leak it "
        "replaces"
    )


def test_latest_depth_chart_is_unchanged_by_the_memo():
    """Slices must be identical whether or not a previous call warmed the memo."""
    from gridlib import fetch

    fresh, memoed = {}, {}
    for week in (1, 6, 12):
        for team in ("KC", "SF", "PHI", "LV", "LAC"):
            fetch.clear_depth_chart_memo()
            fresh[(week, team)] = fetch.latest_depth_chart(2025, week, team)
    if all(v.empty for v in fresh.values()):
        pytest.skip("2025 depth chart not cached")

    fetch.clear_depth_chart_memo()
    for week in (1, 6, 12):
        for team in ("KC", "SF", "PHI", "LV", "LAC"):
            memoed[(week, team)] = fetch.latest_depth_chart(2025, week, team)

    bad = [k for k in fresh if not fresh[k].equals(memoed[k])]
    assert not bad, f"memoized slices differ for {bad}"
