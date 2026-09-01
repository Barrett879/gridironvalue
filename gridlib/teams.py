"""NFL team identity: colors, names, conference and division.

Colors are the official values from nflverse `load_teams()` (team_color =
primary, team_color2 = secondary), captured 2026-08-31 and frozen here. They are
static data; making the app do a network round trip to learn that the Bengals
are orange would be silly, and it would put a fetch in the render path.

The on-primary text color is COMPUTED from relative luminance rather than hand
guessed, because several of these are genuinely borderline (New Orleans' gold,
Tennessee's mid blue) and a bad guess produces unreadable text on a card.

Canonical abbreviations are the ones `schedules/games.parquet` actually uses,
which matters in two places people get wrong:
  - The Rams are **LA**, not LAR, in the schedule file. nflverse ships both.
  - Jacksonville is **JAX**, not JAC.
Aliases below cover the relocated franchises (OAK/SD/STL) and the abbreviation
variants ESPN and PFR use, so a join never silently drops a team.
"""
from __future__ import annotations

# canonical abbr -> (primary, secondary, full name, nickname, conference, division)
_TEAMS: dict[str, tuple[str, str, str, str, str, str]] = {
    "ARI": ("#97233F", "#000000", "Arizona Cardinals", "Cardinals", "NFC", "West"),
    "ATL": ("#A71930", "#000000", "Atlanta Falcons", "Falcons", "NFC", "South"),
    "BAL": ("#241773", "#9E7C0C", "Baltimore Ravens", "Ravens", "AFC", "North"),
    "BUF": ("#00338D", "#C60C30", "Buffalo Bills", "Bills", "AFC", "East"),
    "CAR": ("#0085CA", "#000000", "Carolina Panthers", "Panthers", "NFC", "South"),
    "CHI": ("#0B162A", "#E64100", "Chicago Bears", "Bears", "NFC", "North"),
    "CIN": ("#FB4F14", "#000000", "Cincinnati Bengals", "Bengals", "AFC", "North"),
    "CLE": ("#FF3C00", "#311D00", "Cleveland Browns", "Browns", "AFC", "North"),
    "DAL": ("#002244", "#B0B7BC", "Dallas Cowboys", "Cowboys", "NFC", "East"),
    "DEN": ("#002244", "#FB4F14", "Denver Broncos", "Broncos", "AFC", "West"),
    "DET": ("#0076B6", "#B0B7BC", "Detroit Lions", "Lions", "NFC", "North"),
    "GB":  ("#203731", "#FFB612", "Green Bay Packers", "Packers", "NFC", "North"),
    "HOU": ("#03202F", "#A71930", "Houston Texans", "Texans", "AFC", "South"),
    "IND": ("#002C5F", "#A5ACAF", "Indianapolis Colts", "Colts", "AFC", "South"),
    "JAX": ("#006778", "#9F792C", "Jacksonville Jaguars", "Jaguars", "AFC", "South"),
    "KC":  ("#E31837", "#FFB612", "Kansas City Chiefs", "Chiefs", "AFC", "West"),
    "LA":  ("#003594", "#FFD100", "Los Angeles Rams", "Rams", "NFC", "West"),
    "LAC": ("#007BC7", "#FFC20E", "Los Angeles Chargers", "Chargers", "AFC", "West"),
    "LV":  ("#000000", "#A5ACAF", "Las Vegas Raiders", "Raiders", "AFC", "West"),
    "MIA": ("#008E97", "#F58220", "Miami Dolphins", "Dolphins", "AFC", "East"),
    "MIN": ("#4F2683", "#FFC62F", "Minnesota Vikings", "Vikings", "NFC", "North"),
    "NE":  ("#002244", "#C60C30", "New England Patriots", "Patriots", "AFC", "East"),
    "NO":  ("#D3BC8D", "#000000", "New Orleans Saints", "Saints", "NFC", "South"),
    "NYG": ("#0B2265", "#A71930", "New York Giants", "Giants", "NFC", "East"),
    "NYJ": ("#003F2D", "#000000", "New York Jets", "Jets", "AFC", "East"),
    "PHI": ("#004C54", "#A5ACAF", "Philadelphia Eagles", "Eagles", "NFC", "East"),
    "PIT": ("#000000", "#FFB612", "Pittsburgh Steelers", "Steelers", "AFC", "North"),
    "SEA": ("#002244", "#69BE28", "Seattle Seahawks", "Seahawks", "NFC", "West"),
    "SF":  ("#AA0000", "#B3995D", "San Francisco 49ers", "49ers", "NFC", "West"),
    "TB":  ("#A71930", "#322F2B", "Tampa Bay Buccaneers", "Buccaneers", "NFC", "South"),
    "TEN": ("#4495D2", "#D50A0A", "Tennessee Titans", "Titans", "AFC", "South"),
    "WAS": ("#5A1414", "#FFB612", "Washington Commanders", "Commanders", "NFC", "East"),
}

# alias -> canonical. Relocations plus the ESPN/PFR spelling variants.
_ALIAS = {
    "LAR": "LA", "STL": "LA", "RAM": "LA",
    "OAK": "LV", "RAI": "LV",
    "SD": "LAC", "SDG": "LAC",
    "JAC": "JAX",
    "WSH": "WAS", "WFT": "WAS",
    "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB", "ARZ": "ARI", "BLT": "BAL",
    "CLV": "CLE", "HST": "HOU", "SL": "LA",
}

_NEUTRAL = ("#8a8a93", "#6f6f78", "Unknown", "Unknown", "", "")


def canonical(abbr: str | None) -> str | None:
    """Fold any known abbreviation variant onto the schedule file's spelling."""
    if not abbr:
        return None
    key = str(abbr).strip().upper()
    return _ALIAS.get(key, key)


def _on_color(hex_color: str) -> str:
    """#fff or #111, whichever reads on `hex_color`.

    Uses WCAG relative luminance, not a naive average, so a saturated mid blue
    and a pale gold are classified correctly.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    # Contrast against white vs against near-black; pick the better one.
    return "#fff" if (1.05 / (lum + 0.05)) >= ((lum + 0.05) / 0.05) else "#111"


def on_color(hex_color: str) -> str:
    """Public form of the contrast picker, for callers that have a raw color
    rather than a team (the matchup panels, which use a disambiguated color)."""
    return _on_color(hex_color)


def team_color(abbr: str | None) -> tuple[str, str]:
    """(primary, on-text) for a team abbreviation.

    Never raises: an unknown abbreviation renders neutral gray rather than
    breaking a card, so a mid-season expansion or a feed typo degrades quietly.
    """
    rec = _TEAMS.get(canonical(abbr) or "", _NEUTRAL)
    return rec[0], _on_color(rec[0])


def team_secondary(abbr: str | None) -> str:
    return _TEAMS.get(canonical(abbr) or "", _NEUTRAL)[1]


def team_name(abbr: str | None) -> str:
    """Full club name, e.g. 'Kansas City Chiefs'. Falls back to the abbr."""
    rec = _TEAMS.get(canonical(abbr) or "")
    return rec[2] if rec else (str(abbr) if abbr else "")


def team_nick(abbr: str | None) -> str:
    """Nickname only, e.g. 'Chiefs'. Falls back to the abbr."""
    rec = _TEAMS.get(canonical(abbr) or "")
    return rec[3] if rec else (str(abbr) if abbr else "")


def team_division(abbr: str | None) -> str:
    """e.g. 'AFC West'. Empty string for unknowns."""
    rec = _TEAMS.get(canonical(abbr) or "")
    return f"{rec[4]} {rec[5]}" if rec else ""


def all_teams() -> list[str]:
    """The 32 canonical abbreviations, sorted."""
    return sorted(_TEAMS)


# ── Telling two teams apart when their colors collide ────────────────────────
# Four clubs share the exact primary #002244: New England, Denver, Dallas and
# Seattle. Any matchup among them renders monochrome, and NE at SEA is week 1
# of 2026, so this is not a hypothetical.
_COLLISION_THRESHOLD = 60.0  # empirical: below this the two read as one color


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def color_distance(a: str, b: str) -> float:
    """Perceptual-ish RGB distance, weighted the way the eye actually works.

    Redmean: cheaper than CIEDE2000 and more honest than plain Euclidean RGB,
    which would call navy and black further apart than they look.
    """
    r1, g1, b1 = _rgb(a)
    r2, g2, b2 = _rgb(b)
    rmean = (r1 + r2) / 2
    dr, dg, db = r1 - r2, g1 - g2, b1 - b2
    return (
        (2 + rmean / 256) * dr * dr
        + 4 * dg * dg
        + (2 + (255 - rmean) / 256) * db * db
    ) ** 0.5


def distinguish(away: str | None, home: str | None) -> tuple[str, str]:
    """Two visually distinct colors for a matchup, one per team.

    Returns the primaries when they are far enough apart. When they collide,
    the HOME team keeps its primary (it is the home team) and the away team
    falls back to its secondary, then to a neutral gray if the secondary
    collides too. Never raises.
    """
    away_c, _ = team_color(away)
    home_c, _ = team_color(home)
    if color_distance(away_c, home_c) >= _COLLISION_THRESHOLD:
        return away_c, home_c

    alt = team_secondary(away)
    if alt and color_distance(alt, home_c) >= _COLLISION_THRESHOLD:
        return alt, home_c

    alt_home = team_secondary(home)
    if alt_home and color_distance(away_c, alt_home) >= _COLLISION_THRESHOLD:
        return away_c, alt_home

    return _NEUTRAL[0], home_c
