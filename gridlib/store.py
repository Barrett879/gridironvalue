"""HTML rendering for GridironValue: the slate grid, window headers, notices.

Everything here returns a string and touches no Streamlit state, so the pieces
are unit-testable without a running app. Rendered fragments go out through
st.markdown(..., unsafe_allow_html=True), which is why every interpolated value
passes through util.esc first.

These are plain <a> anchors, NOT components.html iframes. On a cold start an
iframe request can be answered by the app homepage itself, which is a genuinely
confusing failure mode, so critical navigation never goes through one.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .teams import distinguish, on_color, team_color, team_name
from .theme import SENTINEL
from .util import esc, fmt_kick, game_url

# Windows in the order a week actually unfolds. Anything unrecognized sorts
# last rather than being dropped.
_WINDOW_ORDER = [
    "Wednesday",
    "Thursday night",
    "Friday",
    "Saturday",
    "Sunday morning",
    "Sunday early",
    "Sunday late",
    "Sunday night",
    "Monday night",
]


def _window_rank(label: str) -> int:
    try:
        return _WINDOW_ORDER.index(label)
    except ValueError:
        return len(_WINDOW_ORDER)


def game_status(row, now: dt.datetime | None = None) -> str:
    """'final', 'live', or 'upcoming' for one game row.

    A game is final when a score is recorded. It is live when kickoff has passed
    but no score has landed yet. The score check comes first because a very old
    game with a missing kickoff time should still read as final.
    """
    if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
        return "final"
    kick = row.get("kickoff")
    if kick is None or pd.isna(kick):
        return "upcoming"
    _now = now or dt.datetime.now(kick.tzinfo)
    return "live" if kick <= _now else "upcoming"


def _market_html(row) -> str:
    """The market readout: the spread quoted on the favorite, then the total.

    `spread_line` is from the HOME team's perspective and positive means home is
    favored, so a positive line is quoted as HOME -x and a negative one as
    AWAY -x. Quoting a spread against the wrong team is the classic way to make
    a football page look amateur, and the sign convention here is the reason.
    """
    spread = row.get("spread_line")
    total = row.get("total_line")
    if pd.isna(spread) and pd.isna(total):
        return (
            '<div class="gv-slate-mkt"><span>Line not posted</span></div>'
        )
    bits = []
    if pd.notna(spread):
        if float(spread) == 0.0:
            bits.append("<b>PK</b>")
        else:
            fav = row["home_team"] if float(spread) > 0 else row["away_team"]
            bits.append(f"<b>{esc(fav)} {-abs(float(spread)):.1f}</b>")
    if pd.notna(total):
        if bits:
            bits.append('<span class="sep">/</span>')
        bits.append(f"<span>O/U {float(total):.1f}</span>")
    return f'<div class="gv-slate-mkt">{"".join(bits)}</div>'


def _final_html(row) -> str:
    """Final score, with the winner's side emphasized."""
    a, h = row.get("away_score"), row.get("home_score")
    if pd.isna(a) or pd.isna(h):
        return ""
    a, h = int(a), int(h)
    away_cls = ' class="w"' if a > h else ""
    home_cls = ' class="w"' if h > a else ""
    return (
        f'<div class="gv-slate-final">'
        f"<span{away_cls}>{a}</span> - <span{home_cls}>{h}</span>"
        f"</div>"
    )


def render_slate_card(row, season: int, week: int, theme_dark: bool = False,
                      n_lines: int = 0) -> str:
    """One clickable game card: team colors, kickoff, and market context.

    `n_lines` is the count of posted PrizePicks lines that map to a stat we
    project, across BOTH teams. Rendered as a quiet top-left micro-pill and only
    when it is above zero, so a card with no lines shows no badge.
    """
    away, home = row["away_team"], row["home_team"]
    # Not team_color twice: four clubs share the exact primary #002244, so a
    # NE at SEA card would draw a single-color stripe. distinguish() falls back
    # to a secondary when the two collide.
    away_c, home_c = distinguish(away, home)
    status = game_status(row)

    if status == "final":
        tag = '<span class="gv-slate-tag final">Final</span>'
        bottom = _final_html(row)
    elif status == "live":
        tag = '<span class="gv-slate-tag live">Live</span>'
        bottom = _market_html(row)
    else:
        tag = ""
        bottom = _market_html(row)

    lines_html = ""
    if n_lines > 0:
        unit = "line" if n_lines == 1 else "lines"
        lines_html = (f'<span class="gv-slate-lines"><b>{n_lines}</b>'
                      f'<span class="u">{unit}</span></span>')

    href = game_url(season, week, away, home, theme_dark)
    title = f"{team_name(away)} at {team_name(home)}"
    return (
        f'<a class="gv-slate-card" href="{href}" target="_self" '
        f'title="{esc(title)}" style="--away:{away_c};--home:{home_c}">'
        f"{tag}{lines_html}"
        f'<div class="gv-slate-teams">'
        f'<span class="gv-slate-away">{esc(away)}</span>'
        f'<span class="gv-slate-at">at</span>'
        f'<span class="gv-slate-home">{esc(home)}</span>'
        f"</div>"
        f'<div class="gv-slate-time">{esc(fmt_kick(row.get("kickoff")))}</div>'
        f"{bottom}"
        f"</a>"
    )


def render_window_header(label: str, n_games: int, lock_at) -> str:
    """A group header for one broadcast window.

    The lock time is shown because it is the single most useful timestamp on the
    page: inactives publish 90 minutes before kickoff, and that is when a
    projection stops being a guess about availability. It is computed per game,
    never as a wall-clock Sunday-morning cutoff.
    """
    games_word = "game" if n_games == 1 else "games"
    lock = ""
    if lock_at is not None and not pd.isna(lock_at):
        hour = lock_at.strftime("%I").lstrip("0") or "12"
        lock = (
            f'<span class="lock">Inactives {hour}:{lock_at.strftime("%M %p")} ET</span>'
        )
    return (
        f'<div class="gv-window">'
        f'<span class="lab">{esc(label)}</span>'
        f'<span class="meta">{n_games} {games_word}</span>'
        f"{lock}"
        f"</div>"
    )


def render_slate(games: pd.DataFrame, season: int, week: int,
                 theme_dark: bool = False,
                 line_counts: dict | None = None) -> str:
    """The full week board, grouped into broadcast windows.

    Grouping is the point. The NFL board is one week with staggered locks, so a
    flat 16-card grid hides the thing that actually matters about it: which
    games are already locked and which are days away.
    """
    if games.empty:
        return ""
    parts = []
    order = sorted(games["window"].unique(), key=_window_rank)
    for label in order:
        block = games[games["window"] == label]
        parts.append(render_window_header(label, len(block), block["lock_at"].min()))
        counts = line_counts or {}
        cards = "".join(
            render_slate_card(r, season, week, theme_dark,
                              int(counts.get(str(r["game_id"]), 0)))
            for _, r in block.iterrows()
        )
        parts.append(f'<div class="gv-slate-grid">{cards}</div>')
    return "".join(parts)


def render_masthead(season: int, week: int, games: pd.DataFrame) -> str:
    """Editorial masthead: the week, and a few honest counts about it.

    'Lines posted' is a real and useful count during the preseason window,
    when only part of the season carries a market. Showing it beats implying
    every game has one.
    """
    n = len(games)
    with_lines = int(games["total_line"].notna().sum()) if n else 0
    finals = int(games["home_score"].notna().sum()) if n else 0

    if n:
        d0 = pd.to_datetime(games["gameday"]).min()
        d1 = pd.to_datetime(games["gameday"]).max()
        span = (
            d0.strftime("%b %-d")
            if d0.date() == d1.date()
            else f"{d0.strftime('%b %-d')} to {d1.strftime('%b %-d')}"
        )
    else:
        span = SENTINEL

    summary = (
        f'<div class="gv-sum"><div class="gv-sum-num">{n}</div>'
        f'<div class="gv-sum-lab">Games</div></div>'
        f'<div class="gv-sum"><div class="gv-sum-num">{with_lines}</div>'
        f'<div class="gv-sum-lab">Lines posted</div></div>'
        f'<div class="gv-sum"><div class="gv-sum-num">{finals}</div>'
        f'<div class="gv-sum-lab">Played</div></div>'
    )
    return (
        f'<div class="gv-masthead">'
        f'<div class="gv-kicker"><span class="gv-pip"></span>'
        f"Season {season}</div>"
        f'<div class="gv-mast-rule"></div>'
        f'<div class="gv-mast-row">'
        f'<div><div class="gv-brand">Week <span class="accent">{week}</span></div>'
        f'<div class="gv-sum-lab" style="margin-top:.35rem">{esc(span)}</div></div>'
        f'<div class="gv-mast-summary">{summary}</div>'
        f"</div></div>"
    )


def render_notice(body_html: str) -> str:
    """A quiet panel for an empty or degraded state.

    Never let an empty result render as silence: say why it is empty. A user
    who sees a blank week cannot tell a bye-heavy slate from a broken fetch.
    """
    return f'<div class="gv-notice">{body_html}</div>'


# ── Game page ────────────────────────────────────────────────────────────────
def render_matchup_header(row) -> str:
    """The game's identity: both teams in their colors, kickoff, window."""
    away, home = row["away_team"], row["home_team"]
    away_c, home_c = distinguish(away, home)
    away_on, home_on = team_color(away)[1], team_color(home)[1]
    status = game_status(row)

    if status == "final":
        a, h = int(row["away_score"]), int(row["home_score"])
        score = (f'<div class="gv-mh-score">'
                 f'<span class="{"w" if a > h else ""}">{a}</span>'
                 f'<span class="dash">-</span>'
                 f'<span class="{"w" if h > a else ""}">{h}</span></div>')
        tag = "Final"
    else:
        score = f'<div class="gv-mh-at">at</div>'
        tag = "Live" if status == "live" else esc(row.get("window", ""))

    return (
        f'<div class="gv-mh">'
        f'<div class="gv-mh-tag">{tag}</div>'
        f'<div class="gv-mh-row">'
        f'<div class="gv-mh-team" style="--c:{away_c};--on:{away_on}">'
        f'<div class="gv-mh-abbr">{esc(away)}</div>'
        f'<div class="gv-mh-name">{esc(team_name(away))}</div></div>'
        f"{score}"
        f'<div class="gv-mh-team" style="--c:{home_c};--on:{home_on}">'
        f'<div class="gv-mh-abbr">{esc(home)}</div>'
        f'<div class="gv-mh-name">{esc(team_name(home))}</div></div>'
        f"</div>"
        f'<div class="gv-mh-when">{esc(fmt_kick(row.get("kickoff")))}'
        f'<span class="sep">/</span>{esc(row.get("stadium", ""))}</div>'
        f"</div>"
    )


def _fact(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="gv-fact-hint">{esc(hint)}</div>' if hint else ""
    return (
        f'<div class="gv-fact"><div class="gv-fact-lab">{esc(label)}</div>'
        f'<div class="gv-fact-val">{value}</div>{hint_html}</div>'
    )


def render_context_panel(row) -> str:
    """Market, venue and rest: the exogenous inputs to a game script.

    The implied team totals are the important pair. They are the market's view
    of the game script, and script is what drives opportunity in football. They
    are shown as context for a projection, never as a wager prompt.
    """
    spread, total = row.get("spread_line"), row.get("total_line")
    facts = []

    if pd.notna(spread) and float(spread) != 0:
        fav = row["home_team"] if float(spread) > 0 else row["away_team"]
        facts.append(_fact("Spread", f"{esc(fav)} {-abs(float(spread)):.1f}"))
    elif pd.notna(spread):
        facts.append(_fact("Spread", "Pick em"))
    else:
        facts.append(_fact("Spread", SENTINEL, "not posted"))

    facts.append(
        _fact("Total", f"{float(total):.1f}" if pd.notna(total) else SENTINEL,
              "" if pd.notna(total) else "not posted")
    )

    ai, hi = row.get("away_implied_total"), row.get("home_implied_total")
    if pd.notna(ai) and pd.notna(hi):
        facts.append(_fact(
            "Implied totals",
            f'{esc(row["away_team"])} {float(ai):.1f}'
            f'<span class="sep"> / </span>'
            f'{esc(row["home_team"])} {float(hi):.1f}',
            "the market's view of the game script",
        ))
    else:
        facts.append(_fact("Implied totals", SENTINEL, "needs a posted line"))

    roof = str(row.get("roof_type") or "")
    roof_hint = ("open or closed is decided on gameday and is not in the feed"
                 if roof == "retractable" else "")
    facts.append(_fact("Venue", esc(roof.title() or SENTINEL), roof_hint))

    ar, hr = row.get("away_rest"), row.get("home_rest")
    if pd.notna(ar) and pd.notna(hr):
        facts.append(_fact(
            "Rest",
            f'{esc(row["away_team"])} {int(ar)}d'
            f'<span class="sep"> / </span>'
            f'{esc(row["home_team"])} {int(hr)}d',
        ))

    if pd.notna(row.get("div_game")) and int(row.get("div_game") or 0) == 1:
        facts.append(_fact("Matchup", "Division game"))

    return f'<div class="gv-facts">{"".join(facts)}</div>'


def render_depth_chart(depth: pd.DataFrame, team: str, position_order,
                       color: str | None = None,
                       props_by_name: dict | None = None) -> str:
    """One team's skill-position depth chart, grouped by position.

    This is the OPPORTUNITY picture, which is most of what a football
    projection is: who is in line for the touches. It is shown before any
    projection exists because it is already the useful half of the answer.

    `color` overrides the team primary. The caller passes the disambiguated
    color from distinguish() so that a NE at SEA page does not draw two
    identical navy headers.

    `props_by_name` maps a player's display name to a pre-rendered block of his
    posted PrizePicks lines, which is inserted directly beneath his row. Players
    who have lines get a small teal marker so a reader can see which rows carry
    something without scanning every one.
    """
    primary = color or team_color(team)[0]
    on = on_color(primary)
    if depth.empty:
        body = (
            '<div class="gv-depth-empty">No depth chart published for this '
            "team and week yet.</div>"
        )
    else:
        blocks = []
        for abbr, label in position_order:
            grp = depth[depth["pos_abb"] == abbr]
            if grp.empty:
                continue
            rows = []
            for _, p in grp.iterrows():
                rank = p.get("pos_rank")
                rank_txt = str(int(rank)) if pd.notna(rank) else SENTINEL
                status = str(p.get("status") or "")
                badge = (f'<span class="gv-inj gv-inj-{status.lower()}">{esc(status)}</span>'
                         if status else "")
                name = str(p["player_name"])
                # A player's own PrizePicks lines open under his own row, so
                # the comparison lives where the reader already is rather than
                # in a separate table they have to cross-reference.
                lines_html = (props_by_name or {}).get(name) or ""
                marker = " has-lines" if lines_html else ""
                rows.append(
                    f'<div class="gv-depth-row{marker}">'
                    f'<span class="rk">{rank_txt}</span>'
                    f'<span class="nm">{esc(name)}{badge}</span>'
                    f"</div>{lines_html}"
                )
            blocks.append(
                f'<div class="gv-depth-grp">'
                f'<div class="gv-depth-pos">{esc(label)}</div>'
                f'{"".join(rows)}</div>'
            )
        body = "".join(blocks)

    return (
        f'<div class="gv-depth" style="--c:{primary};--on:{on}">'
        f'<div class="gv-depth-head">{esc(team_name(team))}</div>'
        f"{body}</div>"
    )
