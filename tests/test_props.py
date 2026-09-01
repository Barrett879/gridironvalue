"""PrizePicks ingestion, matching, comparison and grading.

Every test here pins a lesson that cost real debugging time in the MLB build or
is called out explicitly in the spec. None of them touch the network: there is
no fetcher to test, on purpose.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from gridlib import props


# ── Name normalisation ───────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("Ja'Marr Chase", "ja marr chase"),
    ("Amon-Ra St. Brown", "amon ra st brown"),
    ("Marvin Harrison Jr.", "marvin harrison"),
    ("Michael Pittman Jr", "michael pittman"),
    ("Kenneth Walker III", "kenneth walker"),
    ("  Travis   Kelce  ", "travis kelce"),
])
def test_normalize_handles_suffixes_and_punctuation(raw, expected):
    """Without this, every Jr., III and apostrophe silently fails to match."""
    assert props.normalize_name(raw) == expected


def test_normalize_strips_diacritics():
    assert props.normalize_name("Jose Ramirez") == props.normalize_name("José Ramírez")


def test_normalize_is_safe_on_none():
    assert props.normalize_name(None) == ""


# ── Rejecting non-person names ───────────────────────────────────────────────
@pytest.mark.parametrize("name", ["KC", "SF", "NYG", "TB", "", None, "QB"])
def test_team_codes_are_not_people(name):
    """A feed pasted without its included[] player list puts the TEAM code in
    `description`. Thousands of lines keyed to team codes match nothing."""
    assert props._person_like(name) is False


@pytest.mark.parametrize("name", ["Patrick Mahomes", "Ja'Marr Chase",
                                  "Amon-Ra St. Brown"])
def test_real_names_are_people(name):
    assert props._person_like(name) is True


# ── Direction: the Demons and Goblins trap ───────────────────────────────────
def test_direction_is_never_inferred_from_odds_type():
    """PrizePicks now allows More OR Less on Demons and Goblins. Most guides
    still say More-only. Inferring from odds_type hides real lines."""
    assert props.offered_sides(None, "demon") == "both"
    assert props.offered_sides(None, "goblin") == "both"
    assert props.offered_sides("", "demon") == "both"


def test_explicit_direction_is_respected():
    """Only the board's own Less/More buttons count as a direction."""
    assert props.offered_sides("more", "demon") == "more"
    assert props.offered_sides("less", "standard") == "less"


def test_unknown_direction_defaults_to_both():
    """Showing a side the market may not offer is a much smaller failure than
    silently hiding a real line."""
    assert props.offered_sides("garbage", "standard") == "both"


# ── Feed parsing ─────────────────────────────────────────────────────────────
def _feed(with_players: bool = True):
    payload = {
        "data": [{
            "type": "projection", "id": "1",
            "attributes": {"stat_type": "Pass Yards", "line_score": 245.5,
                           "odds_type": "standard", "description": "KC",
                           "start_time": "2026-09-13T17:00:00Z"},
            "relationships": {"new_player": {"data": {"id": "9"}}},
        }],
        "included": ([{"type": "new_player", "id": "9",
                       "attributes": {"name": "Patrick Mahomes", "team": "KC",
                                      "position": "QB"}}]
                     if with_players else []),
    }
    return json.dumps(payload)


def test_feed_parses_with_the_included_player_list():
    df = props.parse_prizepicks_json(_feed(True))
    assert len(df) == 1
    assert df.iloc[0]["name"] == "Patrick Mahomes"
    assert df.iloc[0]["line"] == 245.5


def test_feed_without_players_skips_team_codes_and_counts_them():
    """The description falls back to the TEAM code, which is not a person.

    The count must survive so the UI can say WHY a paste produced nothing.
    """
    df = props.parse_prizepicks_json(_feed(False))
    assert df.empty
    assert df.attrs["skipped_noname"] == 1


def test_feed_never_sets_a_one_sided_direction():
    df = props.parse_prizepicks_json(_feed(True))
    assert df.iloc[0]["direction"] == "both"


# ── Board text parsing ───────────────────────────────────────────────────────
BOARD = """Patrick Mahomes
KC - QB
245.5
Pass Yards
Rashee Rice
KC - WR
64.5
Receiving YardsDemon
KC
KC - TM
44.5
Team Total
"""


def test_board_text_parses_players_and_lines():
    df = props.parse_prizepicks_board(BOARD)
    assert set(df["name"]) == {"Patrick Mahomes", "Rashee Rice"}
    assert df.iloc[0]["line"] == 245.5


def test_board_text_drops_the_team_code_row():
    df = props.parse_prizepicks_board(BOARD)
    assert "KC" not in set(df["name"])


def test_board_text_strips_the_demon_suffix_from_the_stat():
    df = props.parse_prizepicks_board(BOARD)
    rice = df[df["name"] == "Rashee Rice"].iloc[0]
    assert rice["stat_type"] == "Receiving Yards"
    assert rice["odds_type"] == "demon"


def test_parse_any_routes_json_and_text():
    assert len(props.parse_any(_feed(True))) == 1
    assert len(props.parse_any(BOARD)) == 2
    assert props.parse_any("").empty


# ── The alt-line ladder ──────────────────────────────────────────────────────
def test_alt_line_ladder_collapses_to_one_line_per_player_stat():
    """Left alone the ladder triple-counts every player and makes any accuracy
    number meaningless."""
    df = pd.DataFrame([
        {"name": "Patrick Mahomes", "stat_type": "Pass Yards", "line": 245.5,
         "odds_type": "standard"},
        {"name": "Patrick Mahomes", "stat_type": "Pass Yards", "line": 265.5,
         "odds_type": "demon"},
        {"name": "Patrick Mahomes", "stat_type": "Pass Yards", "line": 225.5,
         "odds_type": "goblin"},
    ])
    out = props.collapse_alt_lines(df)
    assert len(out) == 1
    assert out.iloc[0]["line"] == 245.5, "must prefer the STANDARD line"


def test_collapse_keeps_different_stats_for_the_same_player():
    df = pd.DataFrame([
        {"name": "Patrick Mahomes", "stat_type": "Pass Yards", "line": 245.5,
         "odds_type": "standard"},
        {"name": "Patrick Mahomes", "stat_type": "Pass Attempts", "line": 31.5,
         "odds_type": "standard"},
    ])
    assert len(props.collapse_alt_lines(df)) == 2


# ── Props we refuse to price ─────────────────────────────────────────────────
@pytest.mark.parametrize("stat", ["Longest Reception", "Longest Rush",
                                  "Longest Completion", "Longest Field Goal"])
def test_longest_anything_is_refused(stat):
    """E[max] is not a function of E[sum]. A mean model cannot price a maximum,
    and feeding it one produces confidently wrong numbers."""
    cols, scale, reason = props._resolve_stat(stat)
    assert cols is None
    assert "maximum" in reason


def test_sequence_conditional_props_are_refused():
    cols, _, reason = props._resolve_stat("Rush Yards in First 5 Attempts")
    assert cols is None
    assert "sequence-conditional" in reason


def test_a_mapped_stat_resolves():
    cols, scale, reason = props._resolve_stat("Receiving Yards")
    assert cols == ("receiving_yards",) and reason is None


def test_combination_stats_sum_their_components():
    """Summing component means is exact for a mean: expectation is linear."""
    cols, _, _ = props._resolve_stat("Rush+Rec Yards")
    assert set(cols) == {"rushing_yards", "receiving_yards"}


# ── Comparison ───────────────────────────────────────────────────────────────
def _proj():
    return pd.DataFrame([{
        "gsis_id": "00-1", "player_display_name": "Patrick Mahomes",
        "position": "QB", "team": "KC", "opponent_team": "DEN",
        "game_id": "g1", "passing_yards": 230.8, "attempts": 33.7,
        "receiving_yards": None, "receptions": None, "rushing_yards": 12.0,
        "passing_tds": 1.5, "rushing_tds": 0.1, "receiving_tds": None,
        "passing_interceptions": 0.6,
    }])


def _lines(stat="Pass Yards", line=245.5):
    return pd.DataFrame([{"name": "Patrick Mahomes", "stat_type": stat,
                          "line": line, "direction": "both",
                          "odds_type": "standard"}])


def test_compare_computes_the_gap_and_the_lean():
    table, meta = props.compare(_lines(), _proj())
    assert meta["matched"] == 1
    r = table.iloc[0]
    assert r["model"] == pytest.approx(230.8)
    assert r["diff"] == pytest.approx(-14.7)
    assert r["lean"] == "Less"


def test_compare_counts_refusals_rather_than_dropping_them_silently():
    table, meta = props.compare(_lines("Longest Completion", 40.5), _proj())
    assert table.empty
    assert meta["refused"]


def test_compare_counts_unmatched_players():
    lines = pd.DataFrame([{"name": "Nobody At All", "stat_type": "Pass Yards",
                           "line": 200.5, "direction": "both",
                           "odds_type": "standard"}])
    table, meta = props.compare(lines, _proj())
    assert table.empty and meta["unmatched_players"] == 1


def test_compare_reports_a_reason_when_there_is_nothing_to_compare():
    """Never let an empty result render as silence."""
    _, meta = props.compare(pd.DataFrame(), _proj())
    assert meta["reason"] == "no lines pasted"
    _, meta2 = props.compare(_lines(), pd.DataFrame())
    assert meta2["reason"] == "no projections for this week"


def test_compare_labels_a_baseline_served_target():
    """A target that failed its gate is served from a baseline and must be
    labelled, or the page presents it with the same confidence as a real one."""
    registry = {"targets": {"passing_yards": {"present_as": "baseline"}}}
    table, meta = props.compare(_lines(), _proj(), registry)
    assert table.iloc[0]["source"] == "baseline"
    assert meta["baseline_served"] == 1


# ── Grading ──────────────────────────────────────────────────────────────────
def _actuals(value=283.0):
    return pd.DataFrame([{"player_display_name": "Patrick Mahomes",
                          "passing_yards": value}])


def test_grade_scores_the_models_lean():
    table, _ = props.compare(_lines(), _proj())
    g = props.grade(table, _actuals(283.0))
    assert g.iloc[0]["result"] == "More"
    assert bool(g.iloc[0]["model_correct"]) is False   # the model leaned Less


def test_an_exact_tie_is_neither_a_win_nor_a_loss():
    """An exact line is NOT a push on PrizePicks, it lowers the payout tier, so
    it must never be counted as a hit."""
    table, _ = props.compare(_lines(), _proj())
    g = props.grade(table, _actuals(245.5))
    assert g.iloc[0]["result"] == "Exact"
    assert pd.isna(g.iloc[0]["model_correct"])


def test_a_player_who_did_not_play_is_not_graded():
    table, _ = props.compare(_lines(), _proj())
    g = props.grade(table, pd.DataFrame([
        {"player_display_name": "Patrick Mahomes", "passing_yards": None}]))
    assert g.empty


def test_scorecard_keeps_ties_out_of_the_hit_rate():
    graded = pd.DataFrame([
        {"model": 1, "actual": 2, "line": 1.5, "result": "More",
         "model_correct": True},
        {"model": 1, "actual": 1.5, "line": 1.5, "result": "Exact",
         "model_correct": None},
    ])
    card = props.week_scorecard(graded)
    assert card["decided"] == 1 and card["hits"] == 1
    assert card["hit_rate"] == 100.0
    assert card["exact_ties"] == 1


# ── The in-sample guard ──────────────────────────────────────────────────────
def test_a_season_inside_the_training_window_is_flagged():
    """Grading a week the model trained on is the model marking its own
    homework; the hit rate is flattering and meaningless."""
    reg = {"trained_through": 2025}
    assert props.is_in_training_window(2025, reg) is True
    assert props.is_in_training_window(2024, reg) is True
    assert props.is_in_training_window(2026, reg) is False


def test_missing_registry_does_not_claim_in_sample():
    assert props.is_in_training_window(2026, None) is False


# ── Storage ──────────────────────────────────────────────────────────────────
def test_lines_round_trip_by_week(tmp_path, monkeypatch):
    monkeypatch.setattr(props, "dc_path", lambda name: tmp_path / name)
    df = props.parse_prizepicks_board(BOARD)
    props.save_lines(2026, 3, df)
    back = props.load_lines(2026, 3)
    assert back is not None and len(back) == len(df)
    assert back.attrs["saved_at"]


def test_load_lines_returns_none_when_nothing_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(props, "dc_path", lambda name: tmp_path / name)
    assert props.load_lines(2026, 9) is None


def test_props_module_cannot_reach_the_network():
    """The paste path is deliberate. PrizePicks' API 403s server-side requests
    AND their terms prohibit automated access, so solving the block would not
    make fetching acceptable.

    Checked by parsing the module's imports rather than grepping its text: the
    docstrings legitimately NAME the endpoint while explaining why nothing calls
    it, and a text scan cannot tell those apart.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(props))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    network = {"requests", "urllib", "urllib3", "httpx", "aiohttp", "socket"}
    assert not (imported & network), (
        f"props.py imported a network library: {imported & network}. The "
        f"ingestion path is paste-only by design."
    )


# ── Demon and Goblin markers ─────────────────────────────────────────────────
def test_demon_and_goblin_get_a_marker():
    """These lines are deliberately SHIFTED off the standard number, so a large
    model-versus-line gap on one is manufactured rather than an edge. Shown
    unmarked next to standard lines it would read as signal that is not there."""
    import props_ui

    assert "Demon" in props_ui._odds_badge("demon")
    assert "gv-odds demon" in props_ui._odds_badge("demon")
    assert "Goblin" in props_ui._odds_badge("goblin")
    assert "gv-odds goblin" in props_ui._odds_badge("goblin")


def test_a_standard_line_gets_no_marker():
    """Marking every row would make the marker meaningless."""
    import props_ui

    assert props_ui._odds_badge("standard") == ""
    assert props_ui._odds_badge(None) == ""
    assert props_ui._odds_badge("") == ""


def test_marker_is_case_insensitive():
    import props_ui

    assert "Demon" in props_ui._odds_badge("DEMON")


def test_the_footnote_appears_only_when_one_is_on_screen():
    import props_ui

    plain = pd.DataFrame([{"odds_type": "standard"}])
    assert props_ui._odds_footnote(plain) == ""

    mixed = pd.DataFrame([{"odds_type": "standard"}, {"odds_type": "demon"}])
    foot = props_ui._odds_footnote(mixed)
    # The HEADLINE names only what is actually on screen. The body explains both
    # concepts either way, which is correct: a reader seeing a Demon still
    # benefits from knowing what a Goblin is when they meet one.
    assert foot.startswith('<div class="gv-note"><b>Demons on this board.</b>')


def test_the_footnote_names_both_when_both_are_present():
    import props_ui

    both = pd.DataFrame([{"odds_type": "demon"}, {"odds_type": "goblin"}])
    foot = props_ui._odds_footnote(both)
    assert foot.startswith(
        '<div class="gv-note"><b>Demons and Goblins on this board.</b>')


def test_the_footnote_dates_the_policy_change_and_flags_the_partial_rollout():
    """The change is real but PARTIAL. Saying "both sides now" flat would be
    wrong for NFL, where the rollout had not landed as of 2026-08-31, and
    saying "More-only" would be wrong the day it does. The footnote has to
    carry the date and the caveat."""
    import props_ui

    foot = props_ui._odds_footnote(pd.DataFrame([{"odds_type": "demon"}]))
    assert "More" in foot and "Less" in foot
    assert "21 August 2026" in foot
    assert "coming soon" in foot
    assert "per line" in foot


def test_the_footnote_says_the_gap_is_manufactured():
    """The whole point of the marker: a gap against a shifted line is not
    evidence the market is wrong."""
    import props_ui

    foot = props_ui._odds_footnote(pd.DataFrame([{"odds_type": "goblin"}]))
    assert "shifted" in foot.lower()
    assert "not evidence" in foot.lower()


def test_footnote_is_safe_on_a_table_without_the_column():
    import props_ui

    assert props_ui._odds_footnote(pd.DataFrame([{"stat": "x"}])) == ""
    assert props_ui._odds_footnote(None) == ""


def test_odds_type_survives_into_the_per_player_payload():
    """The badge on a player's own row needs odds_type to reach it."""
    table = pd.DataFrame([{
        "player": "Patrick Mahomes", "team": "KC", "opponent": "DEN",
        "stat": "Pass Yards", "model": 230.8, "line": 265.5, "diff": -34.7,
        "lean": "Less", "source": "model", "odds_type": "demon",
        "sides": "both",
    }])
    rendered = __import__("props_ui").render_player_lines([{
        "stat": "Pass Yards", "model": 230.8, "line": 265.5, "diff": -34.7,
        "lean": "Less", "source": "model", "odds_type": "demon",
    }])
    assert "gv-odds demon" in rendered


# ── Reading direction from the board's own buttons ───────────────────────────
BOARD_WITH_BUTTONS = """Patrick Mahomes
KC - QB
245.5
Pass Yards
Less
More
Rashee Rice
KC - WR
72.5
Receiving YardsDemon
More
Travis Kelce
KC - TE
34.5
Receiving YardsGoblin
Less
"""


def test_board_buttons_are_read_as_direction():
    """The board renders only the buttons that exist, so reading them beats any
    rule of thumb about how Demons or Goblins behave this season."""
    df = props.parse_prizepicks_board(BOARD_WITH_BUTTONS)
    by = {r["name"]: r for _, r in df.iterrows()}
    assert by["Patrick Mahomes"]["direction"] == "both"
    assert by["Rashee Rice"]["direction"] == "more"
    assert by["Travis Kelce"]["direction"] == "less"


def test_board_without_buttons_leaves_direction_unknown():
    """Some copy modes drop the buttons; that must not be read as one-sided."""
    df = props.parse_prizepicks_board(BOARD)
    assert set(df["direction"]) == {""}


def test_buttons_do_not_swallow_the_next_player():
    """The button scan must stop at the next name, not eat the following prop."""
    df = props.parse_prizepicks_board(BOARD_WITH_BUTTONS)
    assert len(df) == 3


# ── The odds-type fallback policy ────────────────────────────────────────────
def test_explicit_direction_beats_the_odds_type_policy():
    """A board that says 'More only' wins over any default."""
    assert props.offered_sides("more", "standard") == "more"
    assert props.offered_sides("less", "demon") == "less"


def test_unknown_direction_falls_back_to_the_policy_table():
    for odds in ("standard", "demon", "goblin"):
        assert props.offered_sides("", odds) == props.ODDS_TYPE_SIDES[odds]


def test_the_policy_lives_in_exactly_one_place():
    """Changing the Demon rule must be a one-line edit, not a hunt."""
    assert set(props.ODDS_TYPE_SIDES) == {"standard", "demon", "goblin"}
    for v in props.ODDS_TYPE_SIDES.values():
        assert v in ("both", "more", "less")


def test_an_unrecognised_odds_type_stays_permissive():
    """A new line type must not silently hide every line that carries it."""
    assert props.offered_sides("", "some_new_type") == "both"


# ── Pickability ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("lean,sides,expected", [
    ("More", "both", True), ("Less", "both", True),
    ("More", "more", True), ("Less", "more", False),
    ("Less", "less", True), ("More", "less", False),
    ("Even", "both", False), ("", "both", False), (None, "both", False),
])
def test_is_pickable(lean, sides, expected):
    assert props.is_pickable(lean, sides) is expected


def test_filter_pickable_splits_and_keeps_the_hidden_rows():
    """Hidden rows are RETURNED, not dropped: the count has to be reportable or
    the board silently shrinks, which is worse than not filtering."""
    t = pd.DataFrame([
        {"lean": "More", "sides": "more"},
        {"lean": "Less", "sides": "more"},
        {"lean": "Less", "sides": "both"},
        {"lean": "Even", "sides": "both"},
    ])
    keep, hidden = props.filter_pickable(t)
    assert len(keep) == 2
    assert len(hidden) == 2
    assert list(keep["lean"]) == ["More", "Less"]


def test_filter_pickable_is_safe_on_an_empty_table():
    keep, hidden = props.filter_pickable(pd.DataFrame())
    assert keep.empty and hidden.empty


def test_a_less_lean_on_a_more_only_line_is_hidden():
    """The exact case Barrett asked for: a prop the board only sells one way,
    where the projection leans the other way, is not a pick and is removed."""
    board = props.parse_prizepicks_board(BOARD_WITH_BUTTONS)
    rice = board[board["name"] == "Rashee Rice"].iloc[0]
    sides = props.offered_sides(rice["direction"], rice["odds_type"])
    assert sides == "more"
    assert props.is_pickable("Less", sides) is False
    assert props.is_pickable("More", sides) is True


# ── allowed_wager_types: the authoritative per-row signal ────────────────────
@pytest.mark.parametrize("value,expected", [
    (None, "both"),            # absent = unrestricted, the majority of rows
    ("over", "more"),
    (["over"], "more"),
    ("under", "less"),
    (["over", "under"], "both"),
    ([], "both"),
    ("OVER", "more"),          # case-insensitive
])
def test_allowed_wager_types_maps_to_sides(value, expected):
    assert props.sides_from_allowed_wager_types(value) == expected


def test_an_unknown_wager_type_stays_permissive():
    """The field's vocabulary is thinly observed. A value we do not recognise
    must not silently hide every line carrying it; it is logged and treated as
    two-sided."""
    assert props.sides_from_allowed_wager_types("some_new_value") == "both"


def test_feed_reads_direction_per_row_not_from_odds_type():
    """The whole point: a Demon is NOT automatically More-only. The row says."""
    payload = {
        "data": [
            {"type": "projection", "id": "1", "attributes": {
                "stat_type": "Receiving Yards", "line_score": 72.5,
                "odds_type": "demon", "allowed_wager_types": ["over"]},
             "relationships": {"new_player": {"data": {"id": "9"}}}},
            {"type": "projection", "id": "2", "attributes": {
                "stat_type": "Rush Yards", "line_score": 60.5,
                "odds_type": "demon"},
             "relationships": {"new_player": {"data": {"id": "9"}}}},
        ],
        "included": [{"type": "new_player", "id": "9", "attributes": {
            "name": "Rashee Rice", "team": "KC", "position": "WR"}}],
    }
    df = props.parse_prizepicks_json(json.dumps(payload))
    # Same odds_type, different direction, decided by the row's own field.
    assert list(df["odds_type"]) == ["demon", "demon"]
    assert list(df["direction"]) == ["more", "both"]


def test_the_feed_parse_records_the_wager_type_distribution():
    """Logged so a rollout landing shows up here, not in a user complaint."""
    payload = {
        "data": [{"type": "projection", "id": "1", "attributes": {
            "stat_type": "Pass Yards", "line_score": 245.5,
            "odds_type": "standard", "allowed_wager_types": ["over"]},
            "relationships": {"new_player": {"data": {"id": "8"}}}}],
        "included": [{"type": "new_player", "id": "8", "attributes": {
            "name": "Patrick Mahomes", "team": "KC"}}],
    }
    df = props.parse_prizepicks_json(json.dumps(payload))
    assert df.attrs["allowed_wager_types"]


def test_odds_type_is_never_the_thing_that_decides():
    """A demon with no restriction is two-sided; a STANDARD line can be
    restricted. Both directions of the rule matter."""
    assert props.offered_sides("both", "demon") == "both"
    assert props.offered_sides("more", "standard") == "more"


# ── Suppressing edges the model cannot inform ────────────────────────────────
def test_thin_history_rows_are_split_out():
    """An edge computed from an empty feature vector is the model's error, not
    the market's, and ranking by absolute gap puts those rows on top."""
    t = pd.DataFrame([
        {"player": "Veteran", "games_prior": 40.0, "diff": -5.0},
        {"player": "Rookie", "games_prior": 0.0, "diff": -37.6},
        {"player": "Second gamer", "games_prior": 1.0, "diff": 20.0},
    ])
    keep, thin = props.filter_informed(t)
    assert list(keep["player"]) == ["Veteran"]
    assert set(thin["player"]) == {"Rookie", "Second gamer"}


def test_the_biggest_edge_can_be_the_one_suppressed():
    """The whole point: the largest gap on the board was a player with no
    history, and that is a structural property of ranking by gap size."""
    t = pd.DataFrame([
        {"player": "Known", "games_prior": 30.0, "diff": -8.0},
        {"player": "Unknown", "games_prior": 0.0, "diff": -37.6},
    ])
    keep, thin = props.filter_informed(t)
    assert thin["diff"].abs().max() > keep["diff"].abs().max()


def test_filter_informed_is_safe_without_the_column():
    t = pd.DataFrame([{"player": "x", "diff": 1.0}])
    keep, thin = props.filter_informed(t)
    assert len(keep) == 1 and thin.empty


def test_compare_carries_games_prior_through():
    proj = _proj()
    proj["f_games_prior"] = 42.0
    table, _ = props.compare(_lines(), proj)
    assert table.iloc[0]["games_prior"] == 42.0
