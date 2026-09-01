"""GridironValue home / slate page.

Pick a week, see every game grouped by broadcast window, click one to open its
detail page. This page is the week index; per-game projections live on the Game
page at their own URL.

Milestone 1 of the build order: data layer plus slate viewer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import props_ui  # noqa: E402
from gridlib import fetch, predict, store  # noqa: E402
from gridlib.theme import (  # noqa: E402
    SITE_NAME,
    render_footer,
    render_nav,
    render_page_chrome,
)

st.set_page_config(page_title=SITE_NAME, page_icon="static/favicon.svg", layout="wide")
render_page_chrome()

# ── Load the schedule once per session-ish (the loader has its own disk TTL) ──
@st.cache_data(ttl=fetch.SCHEDULE_TTL, show_spinner=False)
def _schedule():
    return fetch.load_schedules()


try:
    SCHED = _schedule()
except RuntimeError as e:
    render_nav("Home")
    st.markdown(
        store.render_notice(
            "<b>The schedule feed is unavailable and has never been cached.</b> "
            f"{SITE_NAME} cannot draw a week board without it. This is a data "
            "source outage, not an empty week. Details: " + str(e)
        ),
        unsafe_allow_html=True,
    )
    render_footer()
    st.stop()

SEASONS = sorted(SCHED.loc[SCHED["game_type"] == "REG", "season"].unique().tolist())
LIVE_SEASON = fetch.current_season(SCHED)
LIVE_WEEK = fetch.current_week(LIVE_SEASON, SCHED)


def _weeks_for(season: int) -> list[int]:
    """Regular-season week numbers for a season, ascending.

    Read from the schedule rather than assumed, because the count is not a
    constant: seasons through 2020 had 17 weeks and 2021+ have 18.
    """
    w = SCHED.loc[
        (SCHED["season"] == season) & (SCHED["game_type"] == "REG"), "week"
    ]
    return sorted(int(x) for x in w.unique())


# ── View state: seed once from the URL, mirror every change back to it ───────
def _seed_int(param: str, default: int, allowed: list[int]) -> int:
    raw = st.query_params.get(param)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default
    return val if val in allowed else default


if "season" not in st.session_state:
    st.session_state["season"] = _seed_int("season", LIVE_SEASON, SEASONS)
if "week" not in st.session_state:
    st.session_state["week"] = _seed_int(
        "week", LIVE_WEEK if st.session_state["season"] == LIVE_SEASON else 1,
        _weeks_for(st.session_state["season"]),
    )

season = int(st.session_state["season"])
weeks = _weeks_for(season)
# A season switch can strand a week number that season never had.
if st.session_state["week"] not in weeks:
    st.session_state["week"] = weeks[0]
week = int(st.session_state["week"])

# URL hygiene: this URL owns {season, week, theme}, so opening or sharing it
# reproduces exactly this view.
if st.query_params.get("season") != str(season):
    st.query_params["season"] = str(season)
if st.query_params.get("week") != str(week):
    st.query_params["week"] = str(week)

# Persist any freshly-pasted lines BEFORE the header controls render, so the
# popover's counts and every board below reflect this run's paste.
props_ui.resolve_and_persist(season, week, SCHED)

render_nav("Home", keep={"season": str(season), "week": str(week)})
dark = bool(st.session_state.get("theme_dark", False))

# Nav-bar action: the line input lives in the header, not on a page of its own,
# so the feature is wherever the user already is.
with st.container(key="gv_nav_actions"):
    with st.popover("Update lines", help="Add or update PrizePicks lines"):
        props_ui.render_input(season, week)


# ── Week bar ─────────────────────────────────────────────────────────────────
def _step_week(delta: int):
    ws = _weeks_for(int(st.session_state["season"]))
    cur = int(st.session_state["week"])
    idx = ws.index(cur) if cur in ws else 0
    st.session_state["week"] = ws[max(0, min(len(ws) - 1, idx + delta))]


def _jump_now():
    st.session_state["season"] = LIVE_SEASON
    st.session_state["week"] = LIVE_WEEK


def _on_season_change():
    ws = _weeks_for(int(st.session_state["season"]))
    if int(st.session_state.get("week", 0)) not in ws:
        st.session_state["week"] = ws[0]


with st.container(key="gv_weekbar"):
    c_prev, c_next, c_week, c_season, c_now, _ = st.columns(
        [0.05, 0.05, 0.16, 0.18, 0.14, 0.42]
    )
    with c_prev:
        st.button("‹", key="gv_week_prev", on_click=_step_week, args=(-1,),
                  disabled=week == weeks[0], help="Previous week")
    with c_next:
        st.button("›", key="gv_week_next", on_click=_step_week, args=(1,),
                  disabled=week == weeks[-1], help="Next week")
    with c_week:
        # NOTE: the options list is built ascending and never reordered between
        # reruns. Streamlit hashes str(options) into the element id, so a
        # reorder orphans the selection and the user sees "my pick does not
        # register." Keep this stable.
        st.selectbox("Week", weeks, key="week",
                     format_func=lambda w: f"Week {w}")
    with c_season:
        st.selectbox("Season", SEASONS[::-1], key="season",
                     on_change=_on_season_change)
    with c_now:
        st.button("This week", key="gv_week_now", on_click=_jump_now,
                  disabled=(season == LIVE_SEASON and week == LIVE_WEEK),
                  help=f"Jump to week {LIVE_WEEK} of {LIVE_SEASON}")

st.markdown('<div class="gv-bar-rule"></div>', unsafe_allow_html=True)

# ── Board ────────────────────────────────────────────────────────────────────
games = fetch.week_games(season, week, SCHED)
st.markdown(store.render_masthead(season, week, games), unsafe_allow_html=True)

if games.empty:
    st.markdown(
        store.render_notice(
            f"<b>No games found for week {week} of {season}.</b> The schedule "
            "feed has no rows for this week. If the season has not been "
            "published yet, that is expected."
        ),
        unsafe_allow_html=True,
    )
else:
    on_bye = 32 - len(games) * 2
    if on_bye > 0:
        st.markdown(
            store.render_notice(
                f"<b>{on_bye} teams are on bye this week.</b> Every team has one "
                "bye, so a player plays 17 games across 18 weeks."
            ),
            unsafe_allow_html=True,
        )
    # Projections drive both the per-card line counts and the board below.
    # Cached to disk, so this is a fast read after the first build of a week.
    _proj = predict.project_week_cached(season, week, games=SCHED)
    _counts = props_ui.line_counts_by_game(_proj, season, week)
    st.markdown(
        store.render_slate(games, season, week, dark, _counts),
        unsafe_allow_html=True,
    )

    no_line = int(games["total_line"].isna().sum())
    if no_line:
        st.markdown(
            store.render_notice(
                f"<b>{no_line} of {len(games)} games have no market line yet.</b> "
                "Lines populate closer to kickoff. Projections for those games "
                "fall back to a neutral game script instead of a market-implied "
                "one, which widens their uncertainty."
            ),
            unsafe_allow_html=True,
        )

# ── Model vs market, for the whole week, in place ────────────────────────────
if not games.empty:
    _matched = props_ui.render_board(_proj, season, week,
                                     scope_label=f"week {week}",
                                     warn_on_empty=True)
    if _matched == 0 and props_ui.saved_count(season, week) == 0:
        st.markdown(
            store.render_notice(
                "<b>No PrizePicks lines loaded for this week.</b> Use "
                "<b>Update lines</b> in the top bar to paste a board. Lines "
                "then show here, as a count on each game card, and under each "
                "player on the game pages."),
            unsafe_allow_html=True,
        )
    props_ui.render_week_record(season, weeks)

render_footer()
