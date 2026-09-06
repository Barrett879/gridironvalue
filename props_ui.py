"""Shared PrizePicks UI: line input, per-game boards, expandable player rows.

Deliberately NOT a page. The whole feature lives wherever the user already is:
the week board and the game detail page. Line input persists per WEEK, so a
paste on any page flows to every game in that week. This mirrors DiamondValue,
whose props started as a standalone tab and were consolidated for exactly this
reason.

An informational model-versus-market view, never a wager recommendation. The
gap is reported as a difference and never as dollar value: the Arena format
computes payouts from the entry mix rather than a fixed multiplier table, so any
EV figure would be fiction.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from gridlib import predict, props, store
from gridlib.theme import SENTINEL
from gridlib.util import esc

# The user opens this themselves. Nothing here fetches it: PrizePicks blocks
# server-side requests AND their terms prohibit automated access.
FEED_URL = "https://api.prizepicks.com/projections?league_id=9&per_page=1000"


def _read_only() -> bool:
    from gridlib.cache import READ_ONLY
    return READ_ONLY


def resolve_and_persist(season: int, week: int, games: pd.DataFrame | None = None):
    """Merge any freshly-pasted text into the saved set and persist it.

    Safe to call at the top of a page before the input widgets render: it reads
    their committed session_state values. Lines are saved under the week of the
    GAME they belong to, not the week on screen, because a board pasted midweek
    carries more than one slate.

    Returns the frame saved for THIS week (or None).
    """
    note = ""
    txt = (st.session_state.get("pp_paste") or "").strip()
    if txt:
        got = props.parse_any(txt)
        if got is not None and got.empty:
            skipped = got.attrs.get("skipped_noname", 0)
            note = (
                f"That JSON has {skipped} props but no player names (its "
                "included player list is missing), so nothing could be saved. "
                "Re-open the feed and copy the WHOLE page, or paste the board "
                "text instead."
            ) if skipped else (
                "Nothing readable in that paste. Paste the feed JSON or the "
                "board text."
            )
        elif got is not None and not got.empty:
            got = props.collapse_alt_lines(got)
            routed = props.route_to_weeks(got, games, season) if games is not None else {}
            if not routed:
                routed = {week: got}
            for wk, batch in routed.items():
                dest = week if wk == "_unrouted" else int(wk)
                existing = props.load_lines(season, dest)
                merged = (pd.concat([existing, batch], ignore_index=True)
                          if existing is not None else batch)
                props.save_lines(season, dest, props.collapse_alt_lines(merged))
                # Freeze what the model says RIGHT NOW for these lines. Doing
                # it at grading time instead would score a three-week-old board
                # against a model retrained since, so the record would improve
                # every time the models are refit. First seen wins, so this
                # cannot overwrite an earlier paste's projection.
                _freeze_week(season, dest)
            elsewhere = {int(w): len(b) for w, b in routed.items()
                         if w != "_unrouted" and int(w) != week}
            st.session_state["pp_routed_elsewhere"] = elsewhere
    st.session_state["pp_parse_note"] = note
    return props.load_lines(season, week)


def _freeze_week(season: int, week: int) -> int:
    """Snapshot the model's number for every not-yet-frozen line in a week.

    Best effort by design: a board pasted for a week with no depth chart yet
    has nothing to project against, which is normal rather than an error, and
    the lines freeze on the next paste once the chart publishes. Never let this
    break the paste itself, because losing the lines is worse than losing the
    snapshot.
    """
    try:
        lines = props.load_lines(season, week)
        if lines is None or lines.empty:
            return 0
        proj = predict.project_week_cached(season, week)
        if proj is None or proj.empty:
            return 0
        reg = predict.load_registry()
        table, _meta = props.compare(lines, proj, reg)
        if table.empty:
            return 0
        stamp = (reg or {}).get("trained_at") or (reg or {}).get("version")
        return props.freeze_projections(season, week, table, stamp)
    except Exception as e:  # noqa: BLE001
        props.logger.warning("could not freeze week %s w%s: %s", season, week, e)
        return 0


def saved_count(season: int, week: int) -> int:
    saved = props.load_lines(season, week)
    return 0 if saved is None or saved.empty else len(saved)


def line_counts_by_game(scope_proj: pd.DataFrame, season: int, week: int) -> dict:
    """{game_id: posted lines that map to a projected stat, both teams}.

    Feeds the micro-pill on each slate card. A "line" is one posted
    (player, stat) prop that resolves to something we actually project, so the
    count matches exactly what the game page can show.
    """
    lines = props.load_lines(season, week)
    if (lines is None or lines.empty or scope_proj is None or scope_proj.empty
            or "game_id" not in scope_proj.columns):
        return {}
    reg = predict.load_registry()
    out: dict[str, int] = {}
    for gid, gp in scope_proj.groupby("game_id"):
        table, _meta = props.compare(lines, gp, reg)
        # Count only what the game page will actually SHOW. Counting matches
        # instead would promise a card of lines and then deliver fewer, which is
        # the kind of small inconsistency that makes a page feel broken.
        pickable, _hidden = props.filter_pickable(table)
        if len(pickable):
            out[str(gid)] = int(len(pickable))
    return out


def props_by_name(scope_proj: pd.DataFrame, season: int, week: int) -> dict:
    """{player_display_name: [ {stat, model, line, diff, lean, source}, ... ]}.

    Feeds the expandable roster rows on the game page, so a player's lines open
    underneath his own row instead of living in a separate table.
    """
    lines = props.load_lines(season, week)
    if lines is None or lines.empty or scope_proj is None or scope_proj.empty:
        return {}
    table, _ = props.compare(lines, scope_proj, predict.load_registry())
    # Same rule as the ledger: a lean the board does not offer is not a pick, so
    # it does not belong under a player's row either. Filtering in one place and
    # not the other would be worse than not filtering at all.
    table, _hidden = props.filter_pickable(table)
    if table.empty:
        return {}
    out: dict[str, list] = {}
    for _, r in table.iterrows():
        out.setdefault(str(r["player"]), []).append({
            "stat": r["stat"], "model": r["model"], "line": r["line"],
            "diff": r["diff"], "lean": r["lean"], "source": r["source"],
            "odds_type": r.get("odds_type", "standard"),
            "sides": r.get("sides", "both"),
            "tier": r.get("tier"), "tier_edge": r.get("tier_edge"),
            # Without this the per-player rows fall back to mean formatting and
            # a 58% touchdown chance renders as "0.6", which is the exact
            # confusion the probability treatment exists to prevent.
            "kind": r.get("kind", "mean"),
            "p_over": r.get("p_over"),
        })
    return out


_TIER_CLASS = {"strong": "t-strong", "moderate": "t-mod", "weak": "t-weak",
               "none": "t-none", "negative": "t-neg", "unmeasured": "t-unk"}


def _tier_badge(tier, edge) -> str:
    """How much this stat's lean is actually worth, measured.

    Not a decoration. A receiving-yards gap and a pass-attempts gap of the same
    size are opposite in sign as evidence (-5.7 against +10.1 points of
    side-picking edge), and an unlabelled board presents them identically.
    """
    t = str(tier or "unmeasured").lower()
    cls = _TIER_CLASS.get(t, "t-unk")
    label = props.TIER_LABEL.get(t, "unmeasured")
    tip = (f"Measured side-picking edge {edge:+.1f} points against always "
           "taking the more common side, out of sample."
           if edge is not None and edge == edge
           else "Side-picking accuracy not measured for this stat.")
    return f'<span class="gv-tier {cls}" title="{esc(tip)}">{esc(label)}</span>'


def _odds_badge(odds_type) -> str:
    """The Demon / Goblin marker.

    These are NOT alternative prices on the same number. They are deliberately
    SHIFTED lines: a Goblin sits easier than the standard line and pays less, a
    Demon sits harder and pays more. So a large model-versus-line gap on one of
    them is manufactured by the line itself rather than a market error, and
    showing it unmarked next to standard lines would read as an edge that is not
    there. That is the whole reason this badge exists.

    They survive `collapse_alt_lines` only when no standard line exists for that
    player and stat, so they are uncommon. That makes marking them cheap and
    leaving them unmarked all the more misleading.
    """
    o = str(odds_type or "standard").lower()
    if o == "demon":
        return ('<span class="gv-odds demon" title="Demon: set harder than the '
                'standard line, pays more">Demon</span>')
    if o == "goblin":
        return ('<span class="gv-odds goblin" title="Goblin: set easier than '
                'the standard line, pays less">Goblin</span>')
    return ""


def _odds_footnote(table) -> str:
    """The footnote behind the marker. Shown only when one is on screen."""
    if table is None or "odds_type" not in getattr(table, "columns", []):
        return ""
    kinds = {str(o).lower() for o in table["odds_type"]} & {"demon", "goblin"}
    if not kinds:
        return ""
    names = " and ".join(sorted(k.capitalize() + "s" for k in kinds))
    return (
        f'<div class="gv-note"><b>{names} on this board.</b> Those lines are '
        "deliberately shifted off the standard number: a Goblin is set easier "
        "and pays less, a Demon is set harder and pays more. A large gap "
        "against one of them is manufactured by the line, not evidence that the "
        "market is wrong, so read those rows differently from the rest. "
        "They were More-only until 21 August 2026, when PrizePicks began "
        "allowing Less on them, but only on selected sports and stat types "
        "with NFL listed as coming soon. So a football Demon or Goblin may "
        "still offer More and not Less. Which sides a line actually sells is "
        "read per line from the board rather than assumed, and any row whose "
        "lean is not on offer is left out.</div>"
    )


def render_board(scope_proj: pd.DataFrame, season: int, week: int,
                 scope_label: str = "this game",
                 warn_on_empty: bool = False) -> int:
    """The model-versus-market ledger for a scope, biggest gaps first.

    Renders nothing and returns 0 when no lines are stored. With
    `warn_on_empty` (the week-wide call), saved-but-matchless lines get an
    EXPLANATION rather than silence: a board that says nothing after "600 lines
    saved" reads as broken.
    """
    lines = props.load_lines(season, week)
    if lines is None or lines.empty:
        return 0
    reg = predict.load_registry()
    table, meta = props.compare(lines, scope_proj, reg)

    if table.empty:
        if warn_on_empty:
            bits = []
            if meta["unmatched_players"]:
                bits.append(f"{meta['unmatched_players']} lines name a player "
                            "who is not on a projected depth chart this week")
            if meta["unmapped_stats"]:
                bits.append("unmapped stats: " + ", ".join(
                    f"{esc(k)} ({v})"
                    for k, v in list(meta["unmapped_stats"].items())[:5]))
            if meta["refused"]:
                bits.append("refused by design: " + ", ".join(
                    f"{esc(k)} ({v})" for k, v in meta["refused"].items()))
            st.markdown(
                store.render_notice(
                    f"<b>{len(lines)} lines are saved for week {week}, but none "
                    f"matched {esc(scope_label)}.</b> "
                    + ("; ".join(bits) or "No line resolved to a projected "
                                          "player and stat.") + "."),
                unsafe_allow_html=True)
        return 0

    st.markdown(
        f'<div class="gv-window"><span class="lab">Model vs market</span>'
        f'<span class="meta">{len(table)} lines on {esc(scope_label)}</span>'
        f'<span class="lock">biggest gaps first</span></div>',
        unsafe_allow_html=True)

    if meta["refused"]:
        st.markdown(
            store.render_notice(
                "<b>Not priced: </b>"
                + ", ".join(f"{esc(k)} ({v})" for k, v in meta["refused"].items())
                + ". Longest-anything props are maxima over plays, and a "
                "mean-projection model cannot price a maximum."),
            unsafe_allow_html=True)

    # Drop rows whose lean is a side the board does not offer. Showing a "Less"
    # on a prop that only sells "More" invites someone to act on a pick that
    # does not exist. The hidden rows are counted, never silently swallowed.
    table, hidden = props.filter_pickable(table)
    # Then drop rows the model has no business ranking. An edge computed from an
    # empty feature vector is the model's error, not the market's.
    table, thin = props.filter_informed(table)
    if table.empty and not thin.empty:
        st.markdown(
            store.render_notice(
                f"<b>All {len(thin)} matching lines are on players with almost "
                "no prior games.</b> The projection for a player the model has "
                "not seen is close to a league base rate, so the gap against "
                "the line measures the model's ignorance rather than the "
                "market's. Nothing worth showing here."),
            unsafe_allow_html=True)
        return 0
    if table.empty:
        st.markdown(
            store.render_notice(
                f"<b>All {len(hidden)} matching lines were one-sided against "
                "the model.</b> On every one, the side the projection leans is "
                "not offered, so there is no pick to show."),
            unsafe_allow_html=True)
        return 0

    st.markdown(_ledger_html(table), unsafe_allow_html=True)

    # Name the stats where a lean is not evidence. Sorting them to the bottom
    # is not enough on its own: a reader who scrolls still meets a confident
    # looking row and has no way to know it is not worth acting on.
    #
    # The numbers come from data/stat_reliability.csv rather than being written
    # here. This text hardcoded them twice and went stale twice, most recently
    # when moving the lean from the gap to P(Over) turned rushing yards from
    # -7.7 into +2.3 and left the page warning against its own fix.
    rel = props.reliability_table()
    weak_tiers = {"negative", "none"}
    low = (table[table["tier"].isin(weak_tiers)] if "tier" in table.columns
           else table.iloc[0:0])
    if not low.empty:
        by_stat = []
        for stat in sorted(set(low["stat"])):
            sub = low[low["stat"] == stat]
            edge = sub["tier_edge"].dropna()
            e = f" ({float(edge.iloc[0]):+.1f} pts)" if len(edge) else ""
            by_stat.append(f"{esc(str(stat))}{e}")
        st.markdown(
            store.render_notice(
                f"<b>{len(low)} of these are stats where a lean is not "
                "evidence:</b> " + ", ".join(by_stat) + ". The edge shown is "
                "measured out of sample against simply always taking the more "
                "common side, so a number at or below zero means the lean adds "
                "nothing. Touchdown lines are the clearest case: they hit about "
                "86% of the time because they mostly resolve Under, which is a "
                "base rate rather than skill. They sort to the bottom and are "
                "shown for completeness, not as signal."),
            unsafe_allow_html=True)

    if not thin.empty:
        n = len(thin)
        names = ", ".join(sorted(set(thin["player"]))[:4])
        st.markdown(
            f'<div class="gv-note"><b>{n} line{"" if n == 1 else "s"} withheld '
            f"for thin history</b> ({esc(names)}"
            f"{', and others' if len(set(thin['player'])) > 4 else ''}). "
            "The model has fewer than three prior games for those players, so "
            "its projection is close to a positional base rate. Ranking a board "
            "by the size of the gap would put exactly those rows on top, "
            "because that is where the model is least informed.</div>",
            unsafe_allow_html=True)
    if not hidden.empty:
        n = len(hidden)
        st.markdown(
            f'<div class="gv-note"><b>{n} line{"" if n == 1 else "s"} hidden.'
            "</b> On those the projection leans a side the board does not "
            "offer, so the pick cannot be made. They are left out rather than "
            "shown as an opportunity that does not exist.</div>",
            unsafe_allow_html=True)
    foot = _odds_footnote(table)
    if foot:
        st.markdown(foot, unsafe_allow_html=True)
    st.markdown(
        '<div class="gv-note">Model values are distribution means, not '
        "forecasts. The gap is a difference, never a dollar value. Rows marked "
        "<b>baseline</b> come from a season average because the model for that "
        "stat lost to one out of sample.</div>",
        unsafe_allow_html=True)
    return len(table)


def _ledger_html(table: pd.DataFrame) -> str:
    """The ledger as a plain HTML table, styled by the theme's tokens."""
    rows = []
    for _, r in table.iterrows():
        # From the LEAN, never from the gap. The lean now comes from P(Over),
        # so colouring by `diff` painted a green "Less" next to a red 52% on
        # exactly the rows where the two disagree, which is the whole point of
        # the change.
        cls = _lean_cls(r.get("lean"))
        badge = ""
        if r["source"] == "baseline":
            badge = '<span class="gv-src base">baseline</span>'
        elif r["source"] == "model_low_confidence":
            badge = '<span class="gv-src low">low conf</span>'
        elif r["source"] == "composite":
            badge = '<span class="gv-src comp">composed</span>'
        rows.append(
            f'<div class="gv-ledger-row">'
            f'<span class="nm">{esc(r["player"])}'
            f'<i>{esc(r["team"])} vs {esc(r.get("opponent") or "")}</i></span>'
            f'<span class="st">{esc(r["stat"])}'
            f'{_tier_badge(r.get("tier"), r.get("tier_edge"))}'
            f'{_odds_badge(r.get("odds_type"))}{badge}</span>'
            f'{_fmt_model(r)}'
            f'<span class="num line">{r["line"]:.1f}</span>'
            f'{_fmt_gap(r)}'
            f'<span class="lean {cls}">{esc(r["lean"])}</span>'
            f"</div>")
    head = ('<div class="gv-ledger-head">'
            '<span class="nm">Player</span><span class="st">Stat</span>'
            '<span class="num">Model</span><span class="num">Line</span>'
            '<span class="num">Gap</span><span class="lean">Lean</span></div>')
    return f'<div class="gv-ledger">{head}{"".join(rows)}</div>'


def _lean_cls(lean) -> str:
    """Colour class for a lean. One source of truth, used by every cell in the
    row, so a row can never contradict itself."""
    return "over" if lean == "More" else "under" if lean == "Less" else ""


def _fmt_model(r) -> str:
    """A touchdown prop's model value is a PROBABILITY, so it must not render as
    a stat line. "0.6" next to a 0.5 line reads as expected touchdowns, which is
    a different and materially larger number: 0.69 expected touchdowns is only a
    50% chance of scoring one. Percent makes the units unmistakable."""
    if r.get("kind") == "probability":
        return f'<span class="num">{100 * float(r["model"]):.0f}%</span>'
    return f'<span class="num">{r["model"]:.1f}</span>'


def _fmt_gap(r) -> str:
    """How far the projection sits from the line, in the stat's own units.

    This was a percentage, and the percentage was the problem: 29% of the board
    lands between 50 and 55, where "52%" dresses a coin flip in a precise
    looking number. The gap says the same thing in units a reader already
    understands ("0.2 receptions under the line") without implying precision
    that is not there.

    THE GAP IS ONLY SAFE TO SHOW BECAUSE MODEL IS NOW THE MEDIAN. As the mean it
    was the biased quantity that made receiving yards -5.4 and rushing -7.7,
    and it contradicted the lean on 84 rows. Median minus line agrees with the
    lean by construction, so the sign and the verdict can never disagree.

    The probability still decides the lean and still ranks the board; it is just
    no longer printed.
    """
    cls = _lean_cls(r.get("lean"))
    if r.get("kind") == "probability":
        # A touchdown row's model value IS a probability, so its distance from
        # the line is measured in percentage points away from a coin flip.
        return (f'<span class="num diff {cls}">'
                f'{100 * float(r["model"]) - 50:+.0f}pt</span>')
    d = r.get("diff")
    if d is None or pd.isna(d):
        return f'<span class="num diff">{SENTINEL}</span>'
    # Yards move in whole numbers; receptions and attempts do not.
    dec = 0 if abs(float(d)) >= 10 else 1
    v = round(float(d), dec)
    if v == 0:
        # Avoid "-0.0", which reads as a typo. A gap that rounds away is zero.
        return f'<span class="num diff">0.0</span>'
    return f'<span class="num diff {cls}">{v:+.{dec}f}</span>'


def render_player_lines(lines_for_player: list) -> str:
    """The lines for ONE player, rendered inside his expandable roster row."""
    if not lines_for_player:
        return ""
    rows = []
    for p in lines_for_player:
        cls = _lean_cls(p.get("lean"))
        badge = (' <span class="gv-src base">baseline</span>'
                 if p["source"] == "baseline" else "")
        rows.append(
            f'<div class="gv-pl-row">'
            f'<span class="st">{esc(p["stat"])}'
            f'{_tier_badge(p.get("tier"), p.get("tier_edge"))}'
            f'{_odds_badge(p.get("odds_type"))}{badge}</span>'
            f'{_fmt_model(p)}'
            f'<span class="num line">{p["line"]:.1f}</span>'
            f'{_fmt_gap(p)}'
            f'<span class="lean {cls}">{esc(p["lean"])}</span>'
            f"</div>")
    head = ('<div class="gv-pl-head"><span class="st">Stat</span>'
            '<span class="num">Model</span><span class="num">Line</span>'
            '<span class="num">Gap</span><span class="lean">Lean</span></div>')
    return f'<div class="gv-pl">{head}{"".join(rows)}</div>'


def _clear_lines(season: int, week: int) -> None:
    for w in props.saved_weeks(season):
        props.lines_path(season, w).unlink(missing_ok=True)


def render_input(season: int, week: int) -> None:
    """The paste controls. Drawn INSIDE the header's popover.

    A bordered container, not an expander: Streamlit forbids nesting expanders
    inside popovers, and this renders inside one.
    """
    if _read_only():
        st.info(
            "Paste a board here and it will show for a while, then clear when "
            "the site restarts. That is fine for looking at a slate. The "
            "accuracy record is separate: it only changes when the site is "
            "updated, so it stays the same for everyone and cannot be moved by "
            "anything pasted here."
        )

    with st.container(border=True):
        st.markdown("**Copy the PrizePicks board and paste it here**")
        st.markdown(
            f'1. Open the <a href="{FEED_URL}" target="_blank" rel="noopener">'
            "<b>PrizePicks NFL feed &#8599;</b></a> in a new tab, or just the "
            "board page. The feed looks like a wall of code; that IS the whole "
            "board.<br>"
            "2. Select all and copy.<br>"
            "3. Paste below and click <b>Add these lines</b>. Every line lands "
            "on the week of its own game, so a Wednesday copy of Sunday's board "
            "files itself correctly.",
            unsafe_allow_html=True)
    st.caption(
        "Nothing here fetches from PrizePicks. Their API blocks server-side "
        "requests, and separately their terms prohibit automated access, so "
        "pasting is the only path that respects both.")
    st.text_area(
        "Paste the feed JSON or the board text", height=150, key="pp_paste",
        placeholder=("Paste the copied feed page here, or the board text:\n"
                     "Patrick Mahomes\nKC - QB\n245.5\nPass Yards"))

    n_saved = saved_count(season, week)
    c_add, c_clear = st.columns([2.4, 1], gap="small")
    with c_add:
        added = st.button("Add these lines", type="primary", key="pp_add",
                          width="stretch")
    with c_clear:
        st.button("Clear all", key="pp_clear", on_click=_clear_lines,
                  args=(season, week), width="stretch",
                  disabled=not props.saved_weeks(season),
                  help="Remove every saved line, this week and every other")
    st.caption(
        "A paste ADDS to what is saved, so stat tabs can come in one at a "
        "time. Alt-line ladders collapse to one line per player and stat. "
        "Longest-anything props are skipped: they are maxima over plays and a "
        "mean model cannot price them.")

    note = st.session_state.get("pp_parse_note")
    if note:
        st.warning(note)
    elsewhere = st.session_state.get("pp_routed_elsewhere") or {}
    for w, n in sorted(elsewhere.items())[:3]:
        st.info(f"{n} line(s) belong to games in **week {w}** and were saved "
                "there. Step to that week to see their board.")
    if added:
        st.toast(f"{n_saved} PrizePicks line(s) on file for week {week}."
                 if n_saved else "Couldn't read any lines from that paste.")
    if n_saved:
        st.success(f"{n_saved} line(s) loaded for week {week}. They show on the "
                   "week board and on every game page.")


def render_week_record(season: int, weeks: list[int]) -> None:
    """The running week-by-week record, in place under the board.

    Only weeks with saved lines appear. Grading fills in once games are played.
    """
    if not weeks:
        return
    reg = predict.load_registry()
    rows = []
    oos: list = []
    for w in weeks:
        ln = props.load_lines(season, w)
        if ln is None or ln.empty:
            continue
        # Prefer the FROZEN comparison: what the model said when the line was
        # first seen. Recomputing scores an old board against a model retrained
        # since, which makes the record improve every time the models are refit
        # and is therefore not a record of anything. Weeks with no snapshot are
        # still shown, labelled, rather than hidden.
        frozen = props.load_frozen(season, w)
        if frozen is not None and not frozen.empty:
            table, basis = frozen, "frozen"
        else:
            proj = predict.project_week_cached(season, w)
            table, _ = props.compare(ln, proj, reg)
            basis = "recomputed"

        # Count what the board actually SHOWS, so this table reconciles with the
        # ledger above it and with the per-game card pills. Counting matches
        # instead would report a number the reader cannot find anywhere.
        table, hidden_rows = props.filter_pickable(table)
        act = _actuals_for(season, w)
        graded = (props.grade(table, act)
                  if act is not None and not act.empty else pd.DataFrame())
        # Only OUT-OF-SAMPLE weeks count toward the season record. Grading a
        # week inside the training window is the model marking its own
        # homework, and folding it into a headline rate would inflate it.
        if basis == "frozen" and not props.is_in_training_window(season, reg):
            oos.append(graded)
        card = props.week_scorecard(graded) if not graded.empty else {"n": 0}
        rows.append({
            "Week": w,
            "Lines": len(ln),
            "Shown": len(table),
            "Hidden": len(hidden_rows),
            "Graded": card.get("decided", 0) or 0,
            "Model hit %": card.get("hit_rate") if card.get("decided") else SENTINEL,
            "Model MAE": card.get("mae") if card.get("decided") else SENTINEL,
            "Line MAE": card.get("line_mae") if card.get("decided") else SENTINEL,
            "In sample": "yes" if props.is_in_training_window(season, reg) else "no",
            "Basis": basis,
        })
    if not rows:
        return
    rec = props.season_record(oos)
    if rec.get("decided"):
        verdict = ("an edge, on this evidence" if rec["beats_coin"]
                   else "not yet distinguishable from a coin")
        st.markdown(
            '<div class="gv-window"><span class="lab">Season record, '
            'out of sample</span></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="gv-note"><b>{rec["hits"]} of {rec["decided"]} '
            f'({rec["hit_rate"]}%)</b>, 95% confidence interval '
            f'{rec["ci_low"]}% to {rec["ci_high"]}%. Always taking the more '
            f'common side would have won {rec.get("baseline", 50)}% for free, '
            f'so this is <b>{verdict}</b>. Graded against the projection frozen '
            f'when each line was first seen, never a recomputed one.'
            + (f' {rec["ties"]} line(s) landed exactly on the number and are '
               'counted as neither.' if rec.get("ties") else "")
            + '</div>', unsafe_allow_html=True)
    elif oos:
        st.markdown(
            '<div class="gv-note">Boards are on file but no out-of-sample week '
            'has been played and graded yet, so there is no record to show.'
            '</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="gv-window"><span class="lab">Week by week</span>'
        '<span class="meta">fills in as boards are pasted and games are played'
        "</span></div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown(
        '<div class="gv-note">A hit rate above 50% over a handful of weeks is '
        "not evidence of an edge. The NFL plays 272 regular-season games a year "
        "against baseball's 2,430, so the confidence intervals here are wide "
        "and any small edge claimed on half a season is noise. Weeks marked "
        "<b>in sample</b> are inside the model's training window, which makes "
        "them a plumbing check rather than a test.</div>",
        unsafe_allow_html=True)


def _actuals_for(season: int, week: int):
    from gridlib import fetch
    df = fetch.load_player_week(season)
    if df is None or df.empty:
        return None
    sub = df[(df["week"] == week) & (df["season_type"] == "REG")]
    return sub.rename(columns={"player_id": "gsis_id"}) if not sub.empty else None
