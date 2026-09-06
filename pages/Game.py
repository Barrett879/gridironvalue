"""GridironValue game page.

Resolves a shareable ?season=&week=&away=&home= link back to one game and shows
what is known about it: the matchup, the market's view of the game script, the
venue, and both teams' skill-position depth charts.

Per-player projections are NOT here yet. They arrive at spec step 4, after the
models clear the validation gate and Barrett makes the per-target column
decision. Until then this page shows the opportunity picture, which is already
the useful half of a football projection, and says plainly what is missing
rather than rendering a placeholder number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import props_ui  # noqa: E402
from gridlib import fetch, predict, props, store  # noqa: E402
from gridlib.teams import canonical, distinguish  # noqa: E402
from gridlib.util import esc  # noqa: E402
from gridlib.theme import (  # noqa: E402
    SITE_NAME,
    render_footer,
    render_nav,
    render_page_chrome,
)

st.set_page_config(page_title=f"Game · {SITE_NAME}", page_icon="static/favicon.svg",
                   layout="wide")
render_page_chrome()


@st.cache_data(ttl=fetch.SCHEDULE_TTL, show_spinner=False)
def _schedule():
    return fetch.load_schedules()


@st.cache_data(ttl=1800, show_spinner=False)
def _depth(season: int, week: int, team: str):
    return fetch.latest_depth_chart(season, week, team)


@st.cache_data(ttl=1800, show_spinner=False)
def _injuries(season: int, week: int, teams: tuple[str, ...]):
    return fetch.injury_report(season, week, list(teams))


def _bail(message_html: str, keep: dict | None = None) -> None:
    """Render a self-explaining dead end and stop.

    A game link that cannot be resolved must say why. Silence here is the worst
    outcome: the user cannot tell a bad link from a broken feed.
    """
    render_nav("Home", keep=keep or {})
    st.markdown(store.render_notice(message_html), unsafe_allow_html=True)
    st.markdown(
        '<a class="gv-back" href="/" target="_self">Back to the week board</a>',
        unsafe_allow_html=True,
    )
    render_footer()
    st.stop()


# ── Resolve the link ─────────────────────────────────────────────────────────
try:
    SCHED = _schedule()
except RuntimeError as e:
    _bail(
        "<b>The schedule feed is unavailable and has never been cached.</b> "
        f"No game can be resolved without it. Details: {e}"
    )


def _int_param(name: str):
    try:
        return int(st.query_params.get(name))
    except (TypeError, ValueError):
        return None


season = _int_param("season")
week = _int_param("week")
away = canonical(st.query_params.get("away"))
home = canonical(st.query_params.get("home"))
keep = {k: str(v) for k, v in (("season", season), ("week", week)) if v is not None}

if season is None or week is None or not away or not home:
    _bail(
        "<b>This game link is incomplete.</b> A game URL needs a season, a week "
        "and both teams, like "
        "<code>/Game?season=2026&amp;week=1&amp;away=NE&amp;home=SEA</code>. "
        "Pick a game from the week board instead."
    )

GAME = fetch.find_game(season, week, away, home, SCHED)
if GAME is None:
    _bail(
        f"<b>No game found for {esc(away)} at {esc(home)} in week "
        f"{week} of {season}.</b> "
        "Either the teams did not play each other that week, or one of them was "
        "on bye. Every team has one bye, so a matchup that exists in one week "
        "will not exist in another.",
        keep,
    )

# Persist any freshly-pasted lines BEFORE the header renders, so the popover's
# counts and the boards below reflect this run's paste.
props_ui.resolve_and_persist(season, week, SCHED)

render_nav("Home", keep=keep)

# Nav-bar action: the same paste flow the week board has, so lines can be added
# from wherever the user happens to be.
with st.container(key="gv_nav_actions"):
    with st.popover("Update lines", help="Add or update PrizePicks lines"):
        props_ui.render_input(season, week)

# ── Header, context ──────────────────────────────────────────────────────────
st.markdown(store.render_matchup_header(GAME), unsafe_allow_html=True)
st.markdown(store.render_context_panel(GAME), unsafe_allow_html=True)

# ── Depth charts, with injury status folded in where it exists ───────────────
teams = (GAME["away_team"], GAME["home_team"])
inj = _injuries(season, week, teams)

st.markdown(
    '<div class="gv-window"><span class="lab">Projected to play</span>'
    '<span class="meta">skill positions</span></div>',
    unsafe_allow_html=True,
)

# Projections for this game only, which scope both the per-player lines under
# each roster row and the ledger below.
_all_proj = predict.project_week_cached(season, week, games=SCHED)
GAME_PROJ = (_all_proj[_all_proj["game_id"] == GAME["game_id"]]
             if not _all_proj.empty else _all_proj)
_pbn_raw = props_ui.props_by_name(GAME_PROJ, season, week)
PBN = {name: props_ui.render_player_lines(rows)
       for name, rows in _pbn_raw.items()}

cols = st.columns(2)
# Same disambiguated pair the header uses, so the panels match it.
panel_colors = distinguish(*teams)
for col, team, color in zip(cols, teams, panel_colors):
    depth = _depth(season, week, team)
    if not depth.empty and not inj.empty:
        # Join on gsis_id, the canonical player id, never on name.
        status = (
            inj[inj["team"].map(canonical) == canonical(team)]
            .set_index("gsis_id")["report_status"]
        )
        depth = depth.copy()
        depth["status"] = depth["gsis_id"].map(status).fillna("")
    with col:
        st.markdown(
            store.render_depth_chart(depth, team, fetch.POSITION_ORDER, color,
                                     props_by_name=PBN),
            unsafe_allow_html=True,
        )

# ── Model vs market for THIS game, in place ──────────────────────────────────
_matched = props_ui.render_board(GAME_PROJ, season, week,
                                 scope_label="this game")
if _matched == 0 and props_ui.saved_count(season, week) == 0:
    st.markdown(
        store.render_notice(
            "<b>No PrizePicks lines loaded for this week.</b> Use "
            "<b>Update lines</b> in the top bar to paste a board. Matching "
            "lines then appear under each player above and in a ledger here."),
        unsafe_allow_html=True,
    )

# ── Say what is missing, and why ─────────────────────────────────────────────
notes = []
if inj.empty:
    notes.append(
        f"<b>No injury report for week {week} of {season} yet.</b> nflverse "
        "publishes injuries from 2009 on, but the current season's file does "
        "not appear until the season starts. The report cadence is also "
        "slate-specific: Wednesday to Friday for Sunday games, but Monday to "
        "Wednesday for a Thursday game and Thursday to Saturday for a Monday "
        "one. Inactives are final 90 minutes before kickoff, at "
        f"{GAME['lock_at'].strftime('%-I:%M %p')} ET for this game."
    )

# This used to say "per-player projections are not built yet", unconditionally,
# and it kept saying it for months after they shipped: every visitor read it
# directly BENEATH the projections it denied the existence of.
notes.append(
    "<b>Model values are medians, not forecasts.</b> A projection of 68 "
    "receiving yards means half of this player's outcomes in this spot land "
    "above it and half below, which is the number a half-point line actually "
    "asks about. Two stats are served from a season-to-date average instead of "
    "the model, because the model loses to that average out of sample; they are "
    "marked where they appear."
)

if str(GAME.get("roof_type")) == "retractable":
    notes.append(
        "<b>This stadium has a retractable roof.</b> The feed records whether "
        "it was open or closed only after the game is played, so the state is "
        "not knowable in advance. Recent seasons are lopsided (no open roofs "
        "recorded in 2024 or 2025), so the model will assume closed and say so."
    )

for n in notes:
    st.markdown(store.render_notice(n), unsafe_allow_html=True)

# ── Share: the address bar never follows in-app navigation ───────────────────
st.markdown(
    '<div class="gv-window"><span class="lab">Share this game</span></div>',
    unsafe_allow_html=True,
)
st.code(
    f"?season={season}&week={week}&away={GAME['away_team']}&home={GAME['home_team']}",
    language=None,
)
st.markdown(
    '<div class="gv-note">Streamlit serves the app inside an iframe, so the '
    "browser address bar does not follow in-app navigation. Append the text "
    "above to the site URL to link straight to this game.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f'<a class="gv-back" href="/?season={season}&week={week}" target="_self">'
    "Back to the week board</a>",
    unsafe_allow_html=True,
)
render_footer()
