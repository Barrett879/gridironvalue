"""Cache freshness around kickoff, and the rendering that a user actually reads."""
from __future__ import annotations

import datetime as dt
import os
import time

import pandas as pd
import pytest

from gridlib import store
from gridlib.cache import atomic_to_parquet, dc_fresh, lock_time, read_parquet_or_none

UTC = dt.timezone.utc


@pytest.fixture()
def aged_file(tmp_path):
    """A file whose mtime we can set precisely, in seconds of age."""
    def _make(age_seconds: float):
        p = tmp_path / "x.parquet"
        p.write_bytes(b"x")
        t = time.time() - age_seconds
        os.utime(p, (t, t))
        return p
    return _make


# ── Freshness ramps as kickoff approaches ────────────────────────────────────
def test_missing_file_is_never_fresh(tmp_path):
    assert dc_fresh(tmp_path / "nope.parquet") is False


def test_explicit_ttl_wins_over_kickoff(aged_file):
    p = aged_file(60)
    kick = dt.datetime.now(UTC) + dt.timedelta(days=5)
    assert dc_fresh(p, kickoff=kick, ttl=30) is False
    assert dc_fresh(p, kickoff=kick, ttl=120) is True


def test_far_from_kickoff_tolerates_a_stale_hour(aged_file):
    """Days out, news moves slowly; a one-hour-old file is fine."""
    kick = dt.datetime.now(UTC) + dt.timedelta(days=3)
    assert dc_fresh(aged_file(3600), kickoff=kick) is True
    assert dc_fresh(aged_file(7 * 3600), kickoff=kick) is False


def test_inside_a_day_the_window_tightens(aged_file):
    """Practice reports and downgrades land; an hour old is now too old."""
    kick = dt.datetime.now(UTC) + dt.timedelta(hours=6)
    assert dc_fresh(aged_file(600), kickoff=kick) is True
    assert dc_fresh(aged_file(3600), kickoff=kick) is False


def test_inside_the_inactives_window_it_tightens_again(aged_file):
    """Kickoff minus 90 minutes is when inactives publish."""
    kick = dt.datetime.now(UTC) + dt.timedelta(minutes=45)
    assert dc_fresh(aged_file(120), kickoff=kick) is True
    assert dc_fresh(aged_file(900), kickoff=kick) is False


def test_in_progress_refreshes_fastest(aged_file):
    kick = dt.datetime.now(UTC) - dt.timedelta(hours=1)
    assert dc_fresh(aged_file(60), kickoff=kick) is True
    assert dc_fresh(aged_file(600), kickoff=kick) is False


def test_a_finished_game_becomes_immutable(aged_file):
    """Well after the final whistle, an ancient file is still correct, so the
    only way to invalidate it is a filename version bump."""
    kick = dt.datetime.now(UTC) - dt.timedelta(days=4)
    assert dc_fresh(aged_file(90 * 24 * 3600), kickoff=kick) is True


def test_post_game_grace_still_refreshes(aged_file):
    """Stat corrections land for a while after the whistle."""
    kick = dt.datetime.now(UTC) - dt.timedelta(hours=6)
    assert dc_fresh(aged_file(600), kickoff=kick) is True
    assert dc_fresh(aged_file(4 * 3600), kickoff=kick) is False


def test_naive_kickoff_does_not_raise(aged_file):
    """A naive datetime must be coerced, not blow up mid-render."""
    kick = dt.datetime.now() + dt.timedelta(days=2)
    assert dc_fresh(aged_file(60), kickoff=kick) in (True, False)


def test_lock_time_is_ninety_minutes():
    kick = dt.datetime(2026, 9, 13, 13, 0, tzinfo=UTC)
    assert lock_time(kick) == dt.datetime(2026, 9, 13, 11, 30, tzinfo=UTC)


# ── Atomic writes ────────────────────────────────────────────────────────────
def test_atomic_write_round_trips(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = tmp_path / "t.parquet"
    atomic_to_parquet(df, p)
    assert read_parquet_or_none(p).equals(df)
    # No temp siblings left behind.
    assert [f.name for f in tmp_path.iterdir()] == ["t.parquet"]


def test_corrupt_parquet_returns_none_rather_than_raising(tmp_path):
    p = tmp_path / "bad.parquet"
    p.write_bytes(b"not a parquet")
    assert read_parquet_or_none(p) is None


# ── Rendering ────────────────────────────────────────────────────────────────
def _row(**kw):
    base = dict(
        game_id="2026_01_CLE_JAX", away_team="CLE", home_team="JAX",
        spread_line=7.5, total_line=40.5, away_score=None, home_score=None,
        kickoff=pd.Timestamp("2026-09-13 13:00", tz="America/New_York"),
    )
    base.update(kw)
    return pd.Series(base)


def test_spread_is_quoted_on_the_home_team_when_home_is_favored():
    """spread_line +7.5 means the HOME team is favored by 7.5."""
    html = store._market_html(_row(spread_line=7.5))
    assert "JAX -7.5" in html
    assert "CLE" not in html


def test_spread_is_quoted_on_the_away_team_when_away_is_favored():
    html = store._market_html(_row(spread_line=-2.5))
    assert "CLE -2.5" in html
    assert "JAX" not in html


def test_a_pick_em_is_labelled_not_shown_as_minus_zero():
    html = store._market_html(_row(spread_line=0.0))
    assert "PK" in html
    assert "-0.0" not in html


def test_no_line_says_so_rather_than_rendering_blank():
    """Never let an empty value render as silence."""
    html = store._market_html(_row(spread_line=None, total_line=None))
    assert "Line not posted" in html


def test_total_shows_even_when_the_spread_is_missing():
    html = store._market_html(_row(spread_line=None, total_line=44.5))
    assert "O/U 44.5" in html


def test_status_final_beats_a_missing_kickoff():
    assert store.game_status(_row(away_score=17, home_score=34, kickoff=pd.NaT)) == "final"


def test_status_live_after_kickoff_without_a_score():
    kick = pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(hours=1)
    assert store.game_status(_row(kickoff=kick)) == "live"


def test_status_upcoming_before_kickoff():
    kick = pd.Timestamp.now(tz="America/New_York") + pd.Timedelta(hours=1)
    assert store.game_status(_row(kickoff=kick)) == "upcoming"


def test_final_score_emphasises_the_winner():
    html = store._final_html(_row(away_score=17, home_score=34))
    assert '<span class="w">34</span>' in html
    assert "<span>17</span>" in html


def test_card_is_a_plain_anchor_with_a_readable_url():
    """Deep links must be minted explicitly and be human readable, because the
    address bar never follows in-app navigation inside Streamlit's iframe."""
    html = store.render_slate_card(_row(), season=2026, week=1)
    assert html.startswith("<a class=\"gv-slate-card\"")
    assert 'target="_self"' in html
    assert "season=2026" in html and "week=1" in html
    assert "away=CLE" in html and "home=JAX" in html


def test_card_carries_both_team_colors():
    html = store.render_slate_card(_row(), season=2026, week=1)
    assert "--away:#006778" not in html          # JAX is the HOME team here
    assert "--home:#006778" in html              # so its teal is the home var
    assert "--away:#FF3C00" in html              # Cleveland orange, away


def test_window_ordering_follows_the_week_not_the_alphabet():
    order = [store._window_rank(w) for w in
             ["Thursday night", "Sunday early", "Sunday night", "Monday night"]]
    assert order == sorted(order)


def test_unknown_window_sorts_last_rather_than_being_dropped():
    assert store._window_rank("Some new slot") > store._window_rank("Monday night")


def test_render_slate_groups_by_window():
    games = pd.DataFrame([
        _row(window="Thursday night", lock_at=pd.Timestamp("2026-09-10 18:45",
             tz="America/New_York")),
        _row(window="Sunday early", lock_at=pd.Timestamp("2026-09-13 11:30",
             tz="America/New_York")),
    ])
    html = store.render_slate(games, 2026, 1)
    assert html.index("Thursday night") < html.index("Sunday early")
    assert html.count("gv-slate-grid") == 2


def test_empty_slate_renders_empty_string_so_the_page_can_explain():
    assert store.render_slate(pd.DataFrame(), 2026, 1) == ""


# ── Depth chart rendering ────────────────────────────────────────────────────
def _depth_frame(rows):
    return pd.DataFrame(rows, columns=["pos_abb", "pos_rank", "player_name",
                                       "gsis_id", "status"])


_ORDER = [("QB", "Quarterback"), ("RB", "Running back"), ("PK", "Kicker")]


def test_depth_chart_groups_and_orders_by_position():
    """Positions render in football order, not alphabetically."""
    d = _depth_frame([
        ("PK", 1, "Jason Myers", "00-1", ""),
        ("RB", 1, "Zach Charbonnet", "00-2", ""),
        ("QB", 1, "Sam Darnold", "00-3", ""),
    ])
    html = store.render_depth_chart(d, "SEA", _ORDER)
    assert html.index("Quarterback") < html.index("Running back") < html.index("Kicker")


def test_depth_chart_marks_the_starter():
    d = _depth_frame([("QB", 1, "Sam Darnold", "00-3", ""),
                      ("QB", 2, "Drew Lock", "00-4", "")])
    html = store.render_depth_chart(d, "SEA", _ORDER)
    assert "Sam Darnold" in html and "Drew Lock" in html


def test_injury_status_renders_as_a_badge():
    d = _depth_frame([("QB", 1, "Sam Darnold", "00-3", "Out")])
    html = store.render_depth_chart(d, "SEA", _ORDER)
    assert 'class="gv-inj gv-inj-out"' in html
    assert ">Out<" in html


def test_no_injury_status_renders_no_badge():
    d = _depth_frame([("QB", 1, "Sam Darnold", "00-3", "")])
    assert "gv-inj" not in store.render_depth_chart(d, "SEA", _ORDER)


def test_empty_depth_chart_explains_itself():
    """Silence would leave a user unable to tell a bye from a broken feed."""
    html = store.render_depth_chart(pd.DataFrame(), "SEA", _ORDER)
    assert "No depth chart published" in html


def test_depth_panel_accepts_a_disambiguated_color():
    """NE at SEA share #002244, so the page passes distinguish()'s colors in."""
    d = _depth_frame([("QB", 1, "Drake Maye", "00-9", "")])
    html = store.render_depth_chart(d, "NE", _ORDER, color="#C60C30")
    assert "--c:#C60C30" in html
    assert "--c:#002244" not in html


def test_player_names_are_escaped():
    """Feed strings are third-party input even though they are not user input."""
    d = _depth_frame([("QB", 1, "<script>x</script>", "00-9", "")])
    html = store.render_depth_chart(d, "SEA", _ORDER)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ── Matchup header ───────────────────────────────────────────────────────────
def test_matchup_header_pulls_colliding_teams_apart():
    row = _row(away_team="NE", home_team="SEA")
    html = store.render_matchup_header(row)
    assert html.count("--c:#002244") == 1      # only the home team keeps navy


def test_matchup_header_shows_a_final_score_with_the_winner_marked():
    row = _row(away_team="PHI", home_team="NYG", away_score=17, home_score=34)
    html = store.render_matchup_header(row)
    assert "Final" in html
    assert '<span class="w">34</span>' in html


def test_context_panel_says_not_posted_rather_than_showing_nothing():
    html = store.render_context_panel(_row(spread_line=None, total_line=None))
    assert "not posted" in html
    assert "needs a posted line" in html


def test_context_panel_flags_a_retractable_roof_as_unknowable():
    row = _row(roof_type="retractable")
    html = store.render_context_panel(row)
    assert "decided on gameday" in html


def test_project_week_cached_builds_once_under_concurrency(monkeypatch):
    """A cold build peaks around 1.3 GB and the container has about 1 GB.

    Streamlit gives every visitor their own script thread, so a cold cache plus
    two arrivals in the same second meant two simultaneous builds, which is an
    OOM that presents as a random restart. It also repeated 25 seconds of work
    whose answer the first caller was about to write to disk.

    The build itself is stubbed here: the property under test is how many times
    it is entered, not what it returns.
    """
    import threading
    import time

    import pandas as pd

    from gridlib import predict

    calls = []

    def slow_build(season, week, games=None):
        calls.append((season, week))
        time.sleep(0.4)          # long enough for the others to pile up
        return pd.DataFrame({"gsis_id": ["x"], "season": [season],
                             "week": [week]})

    monkeypatch.setattr(predict, "project_week", slow_build)
    monkeypatch.setattr(predict, "atomic_to_parquet",
                        lambda df, path: None)          # no disk write
    monkeypatch.setattr(predict, "read_parquet_or_none", lambda path: None)

    results = {}

    def call(i):
        results[i] = predict.project_week_cached(2099, 1)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1, (
        f"{len(calls)} concurrent builds ran; each peaks around 1.3 GB on a "
        "1 GB container"
    )
    assert len(results) == 5
    assert all(len(r) == 1 for r in results.values()), (
        "a caller that waited on the lock got nothing back"
    )
