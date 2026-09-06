"""Team identity: 32 teams, alias folding, and readable card text."""
from __future__ import annotations

import pytest

from gridlib.teams import (
    all_teams,
    canonical,
    team_color,
    team_division,
    team_name,
    team_nick,
)


def test_exactly_32_teams():
    assert len(all_teams()) == 32


def test_schedule_spellings_are_the_canonical_ones():
    """The schedule file uses LA (not LAR) and JAX (not JAC).

    Getting this backwards silently drops a team from every join.
    """
    assert "LA" in all_teams()
    assert "LAR" not in all_teams()
    assert "JAX" in all_teams()
    assert "JAC" not in all_teams()


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("LAR", "LA"), ("STL", "LA"),           # Rams, twice relocated
        ("OAK", "LV"), ("SD", "LAC"),           # Raiders, Chargers
        ("JAC", "JAX"), ("WSH", "WAS"), ("WFT", "WAS"),
        ("GNB", "GB"), ("KAN", "KC"), ("NWE", "NE"), ("SFO", "SF"),
        ("la", "LA"), ("  kc  ", "KC"),         # case and whitespace
    ],
)
def test_alias_folding(alias, expected):
    assert canonical(alias) == expected


def test_unknown_team_degrades_quietly():
    """A feed typo or an expansion team renders gray, never raises."""
    primary, on = team_color("ZZZ")
    assert primary.startswith("#")
    assert on in ("#fff", "#111")
    assert team_name("ZZZ") == "ZZZ"
    assert team_division("ZZZ") == ""


def test_none_is_safe():
    assert canonical(None) is None
    assert team_color(None) == team_color("ZZZ")
    assert team_name(None) == ""


def test_every_team_has_readable_card_text():
    """Contrast is computed, not guessed, so check the borderline cases land."""
    for abbr in all_teams():
        primary, on = team_color(abbr)
        assert len(primary) == 7 and primary.startswith("#")
        assert on in ("#fff", "#111")


def test_the_borderline_contrast_cases():
    """New Orleans' pale gold needs dark text; the black teams need white."""
    assert team_color("NO")[1] == "#111"      # #D3BC8D
    assert team_color("LV")[1] == "#fff"      # #000000
    assert team_color("PIT")[1] == "#fff"     # #000000
    assert team_color("CLE")[1] == "#111"     # #FF3C00


def test_names_and_divisions():
    assert team_name("KC") == "Kansas City Chiefs"
    assert team_nick("KC") == "Chiefs"
    assert team_division("KC") == "AFC West"
    assert team_division("LA") == "NFC West"


def test_all_eight_divisions_have_four_teams():
    counts: dict[str, int] = {}
    for abbr in all_teams():
        counts[team_division(abbr)] = counts.get(team_division(abbr), 0) + 1
    assert len(counts) == 8
    assert set(counts.values()) == {4}


# ── Matchup color disambiguation ─────────────────────────────────────────────
def test_four_teams_really_do_share_a_primary():
    """The premise of distinguish(): this is not a hypothetical problem."""
    navy = {t for t in all_teams() if team_color(t)[0] == "#002244"}
    assert navy == {"NE", "DEN", "DAL", "SEA"}


def test_distinct_teams_keep_their_own_colors():
    from gridlib.teams import distinguish
    assert distinguish("KC", "LV") == ("#E31837", "#000000")


def test_colliding_teams_are_pulled_apart():
    """NE at SEA is week 1 of 2026 and both primaries are #002244."""
    from gridlib.teams import color_distance, distinguish
    away, home = distinguish("NE", "SEA")
    assert away != home
    assert color_distance(away, home) >= 60.0


def test_home_team_keeps_its_primary_on_a_collision():
    from gridlib.teams import distinguish, team_color
    _, home = distinguish("NE", "SEA")
    assert home == team_color("SEA")[0]


def test_every_matchup_in_the_league_is_distinguishable():
    """All 496 unordered pairs, both orderings. No pair may render as one color."""
    import itertools

    from gridlib.teams import color_distance, distinguish
    for a, b in itertools.permutations(all_teams(), 2):
        away, home = distinguish(a, b)
        assert color_distance(away, home) >= 60.0, f"{a} at {b} renders as one color"


def test_distinguish_never_raises_on_unknowns():
    from gridlib.teams import distinguish
    assert len(distinguish("ZZZ", "QQQ")) == 2
    assert len(distinguish(None, None)) == 2


def test_depth_chart_uses_canonical_team_codes():
    """The depth-chart files and the stats files disagree about relocations.

    depth_charts_2016 spells the Raiders OAK and the Chargers SD;
    stats_player_week_2016 spells them LV and LAC. Everything joins the two on
    `team`, so an uncanonicalized chart matched nothing for those franchises.
    """
    from gridlib import fetch

    dc = fetch.depth_chart_normalized(2016)
    if dc is None or dc.empty:
        pytest.skip("2016 depth chart not cached")
    stale = sorted({"OAK", "SD", "STL"} & set(dc["team"]))
    assert not stale, f"relocated franchises left uncanonical: {stale}"
    assert {"LV", "LAC"} <= set(dc["team"]), (
        "the Raiders and Chargers vanished from the 2016 chart entirely, which "
        "is worse than the mismatch this replaced"
    )


def test_no_franchise_loses_its_depth_rank_entirely():
    """A team-season with a 100% null depth_rank is a join that matched nothing.

    Measured before the fix: LV 2016-2019 and LAC 2016 ran at exactly 1.000
    while every other team ran 0.90 to 0.98. `is_starter` is
    `(depth_rank_capped == 1)` and `NaN == 1` is False, so 127 quarterback games
    with 20 or more pass attempts, Derek Carr and Philip Rivers among them,
    trained as non-starters. Both `depth_rank_capped` and `is_starter` are model
    features.
    """
    from gridlib.cache import dc_path, read_parquet_or_none

    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    if pw is None or pw.empty:
        pytest.skip("backfill not built")
    rate = (pw.groupby(["season", "team"])["depth_rank"]
              .apply(lambda s: float(s.isna().mean())))
    dead = rate[rate > 0.99]
    assert dead.empty, (
        "these team-seasons have no depth rank at all, so the join matched "
        f"nothing for them:\n{dead.to_string()}"
    )
