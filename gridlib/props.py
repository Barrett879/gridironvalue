"""PrizePicks lines: ingest, match, compare, grade. Week by week.

HOW LINES GET IN, AND WHY THERE IS NO SCRAPER
----------------------------------------------
Barrett pastes them. That is the whole ingestion path, and it is deliberate:

  1. As of 2026-08-31 both `api.prizepicks.com` and `partner-api.prizepicks.com`
     return HTTP 403 to server-side requests, behind Cloudflare bot management.
     That is fingerprint-level blocking, so a browser User-Agent and a proxy do
     not fix it.
  2. Independently of the technical block, PrizePicks' Terms effective
     2026-08-03 section 16(l) prohibit "any robot, spider, or other automatic
     device, process, or means to access the Site or App", and restrict use to
     personal, non-commercial purposes.

Solving (1) would not solve (2). Barrett's own logged-in browser works, so he
opens the feed or the board himself and pastes the result. This module parses
what comes back. **Do not add a fetcher here.**

FIVE THINGS THAT ARE EASY TO GET WRONG
---------------------------------------
1. **Never infer pick direction from `odds_type`.** Demons and Goblins were
   More-only from their Dec 2023 launch until **2026-08-21**, when PrizePicks
   shipped Less picks on them ("Demons and Goblins, Explained: Less Picks Are
   Here", plus the iOS 12.86 note "Take Less on select Demons and Goblins").
   The rollout is PARTIAL: the announcement scopes it to several MLB and WNBA
   stat types with "more sports, **including NFL, coming soon**", so for NFL
   today most of them are still More-only.

   That is exactly why a hardcoded rule is wrong in both directions: assume
   two-sided and the site shows NFL picks that cannot be placed; assume
   More-only and it hides real ones the day the rollout lands. Direction is
   read PER ROW, from `allowed_wager_types` in the feed or the Less/More buttons
   on the board, and only falls back to a default when neither is present.

   Beware even first-party sources: PrizePicks' own /help-center/player-picks
   page still states the retired More-only rule, while
   /help-center/demons-and-goblins states the new one. Every third-party guide
   checked (Stokastic, RotoGrinders, FantasyLife, HeatCheck) is stale.
2. **Route each line to the week of ITS OWN game**, not the week on screen. The
   pre-game board posted on Wednesday is Sunday's games, and a Thursday board
   carries next Monday's too.
3. **Collapse the alt-line ladder**: one line per (player, stat), preferring the
   standard line, or the ladder will triple-count every player.
4. **Reject non-person names.** A feed pasted without its `included[]` player
   list puts the TEAM code in `description`, and thousands of lines keyed to
   team codes match nothing.
5. **Never let a saved-but-unmatched board render as silence.** Say why it
   matched nothing.

WHAT THIS REFUSES TO PRICE, AND WHY
------------------------------------
Longest reception, longest rush, longest completion and longest field goal are
MAXIMA over plays, not means. `E[max]` is not a function of `E[sum]`, so a
mean-projection model literally cannot price them. Feeding a mean projection
into a longest-X comparison produces confidently wrong numbers, so those props
are rejected by name and counted, not silently dropped.

Sequence-conditional props ("rush yards in first 5 attempts") are refused for
the same reason: they are not full-game stats.

EDGE IS SHOWN AS A DIFFERENCE, NEVER AS DOLLARS
------------------------------------------------
The Arena format computes payouts from the entry mix rather than a fixed
multiplier table, so a dollar EV figure would be fiction. This module reports
the model-minus-line gap and nothing else.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import atomic_to_parquet, dc_path, json_load, json_save, logger


def _uncertainty():
    """Imported lazily so props.py keeps no import-time dependency on a built
    artifact, and so the network-isolation test still sees a clean import graph."""
    from . import uncertainty
    return uncertainty

VERSION = "v1"

# ── PrizePicks stat_type -> our projection columns ───────────────────────────
# A tuple of columns is summed, which is exact for a mean because expectation is
# linear. The scale handles props quoted in different units.
STAT_MAP: dict[str, tuple[tuple[str, ...], float]] = {
    "pass yards": (("passing_yards",), 1.0),
    "passing yards": (("passing_yards",), 1.0),
    "pass tds": (("passing_tds",), 1.0),
    "pass touchdowns": (("passing_tds",), 1.0),
    "pass attempts": (("attempts",), 1.0),
    "pass completions": (("completions",), 1.0),
    "completions": (("completions",), 1.0),
    "interceptions thrown": (("passing_interceptions",), 1.0),
    "int": (("passing_interceptions",), 1.0),
    "sacks taken": (("sacks_suffered",), 1.0),
    "rush yards": (("rushing_yards",), 1.0),
    "rushing yards": (("rushing_yards",), 1.0),
    "rush attempts": (("carries",), 1.0),
    "carries": (("carries",), 1.0),
    "receiving yards": (("receiving_yards",), 1.0),
    "rec yards": (("receiving_yards",), 1.0),
    "receptions": (("receptions",), 1.0),
    "targets": (("targets",), 1.0),
    # PrizePicks' own board wording. These were silently unmapped and cost 109
    # lines of stats the model is actually decent at: "Rec Targets" alone was
    # 72 lines on the week 1 board, for a stat with a measured +1.8 edge.
    "rec targets": (("targets",), 1.0),
    "receiving targets": (("targets",), 1.0),
    "rush+rec yds": (("rushing_yards", "receiving_yards"), 1.0),
    "pass+rush yds": (("passing_yards", "rushing_yards"), 1.0),
    "rec+rush yds": (("receiving_yards", "rushing_yards"), 1.0),
    # Anytime touchdown. Routed to a PROBABILITY, never to a mean, and this is
    # a correctness fix rather than a preference.
    #
    # A 0.5 touchdown line asks "does he score at least one", which is
    # P(TD >= 1). E[TD] is a different quantity: it counts two-touchdown games
    # twice, so it sits ABOVE P(>=1) for every player. On the 2026 week 1 board
    # that gap flips the side for most of the players it flags: of 36 players
    # with E[TD] > 0.5, twenty-six have P(>=1) BELOW 0.5. Ja'Marr Chase projects
    # 0.69 expected touchdowns and a 50.0% chance of scoring one. Comparing the
    # mean to the line would lean More on all 36 and be wrong-sided on 26.
    #
    # These 368 lines were the single largest block on the board and were
    # invisible because the stat had no mapping. Showing them required the
    # probability, not a rename.
    "player touchdowns": (("__prob__", "rushing_tds", "receiving_tds"), 1.0),
    "anytime td": (("__prob__", "rushing_tds", "receiving_tds"), 1.0),
    "pass+rush+rec tds": (("__prob__", "rushing_tds", "receiving_tds"), 1.0),
    "rush tds": (("__prob__", "rushing_tds"), 1.0),
    "rec tds": (("__prob__", "receiving_tds"), 1.0),
    "receiving tds": (("__prob__", "receiving_tds"), 1.0),
    # Combination props. Summing component means is exact for the mean.
    "rush+rec yards": (("rushing_yards", "receiving_yards"), 1.0),
    "rush + rec yards": (("rushing_yards", "receiving_yards"), 1.0),
    "pass+rush yards": (("passing_yards", "rushing_yards"), 1.0),
    "pass + rush yards": (("passing_yards", "rushing_yards"), 1.0),
    "rec+rush yards": (("receiving_yards", "rushing_yards"), 1.0),
    "fg made": (("fg_made",), 1.0),
    "field goals made": (("fg_made",), 1.0),
    "kicking points": (("fg_made", "pat_att"), 1.0),  # approximated below
}

# Props this model CANNOT price. Rejected by name, counted, and explained.
REFUSE = {
    "longest reception": "a maximum over plays, not a mean",
    "longest rush": "a maximum over plays, not a mean",
    "longest completion": "a maximum over plays, not a mean",
    "longest passing completion": "a maximum over plays, not a mean",
    "longest field goal": "a maximum over plays, not a mean",
    "rush yards in first 5 attempts": "sequence-conditional, not a full-game stat",
    "receiving yards in first 2 receptions":
        "sequence-conditional, not a full-game stat",
    "first td scorer": "an ordering, not a per-game mean",
    "anytime td": "a probability, not a mean; needs a scoring model",
    "tackles": "defensive; not projected in v1",
    "sacks": "defensive; not projected in v1",
}

# Fantasy Score gets its own handling because it is composed, not modelled.
FANTASY_KEYS = {"fantasy score", "fantasy points"}

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_NONWORD = re.compile(r"[^a-z0-9 ]+")


def normalize_name(name) -> str:
    """Accent- and suffix-tolerant key for matching a pasted name to a roster.

    Strips diacritics, punctuation and generational suffixes. Without this,
    every Jr., III and apostrophe silently fails to match.
    """
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = _NONWORD.sub(" ", s)
    s = _SUFFIX.sub("", s)
    return " ".join(s.split())


def _person_like(name) -> bool:
    """Reject team codes and other non-person strings.

    A feed pasted without its `included[]` player list puts the TEAM code in
    `description`, and thousands of lines keyed to "KC" match nothing.
    """
    s = str(name or "").strip()
    if len(s) < 4 or " " not in s:
        return False
    if s.isupper() and len(s) <= 5:
        return False
    return bool(re.search(r"[A-Za-z]{2,}\s+[A-Za-z]", s))


def parse_prizepicks_json(raw) -> pd.DataFrame:
    """Parse an api.prizepicks.com/projections payload (dict or JSON string).

    Tolerant of the JSON:API shape: `data[]` projections plus `included[]`
    new_player records. Lines whose player cannot be resolved are skipped and
    counted in `df.attrs['skipped_noname']`, so the UI can say WHY a paste
    produced nothing.
    """
    if isinstance(raw, str):
        raw = json.loads(raw)
    data = raw.get("data", []) if isinstance(raw, dict) else []
    included = raw.get("included", []) if isinstance(raw, dict) else []

    players: dict[str, tuple] = {}
    for inc in included:
        if inc.get("type") in ("new_player", "player"):
            a = inc.get("attributes", {}) or {}
            players[str(inc.get("id"))] = (
                a.get("name") or a.get("display_name"),
                a.get("team") or a.get("team_name"),
                a.get("position"),
            )

    rows, skipped = [], 0
    wager_types: dict[str, int] = {}
    for p in data:
        if p.get("type") != "projection":
            continue
        a = p.get("attributes", {}) or {}
        rel = (p.get("relationships", {}) or {}).get("new_player", {}) or {}
        pid = str(((rel.get("data") or {}) or {}).get("id"))
        name, team, pos = players.get(pid, (None, None, None))
        if not name and _person_like(a.get("description")):
            name = a.get("description")
        if a.get("line_score") is None or not name:
            skipped += 1
            continue
        odds = str(a.get("odds_type") or "standard").lower()
        # The authoritative per-row answer. NEVER derived from odds_type: that
        # rule changed on 2026-08-21 and is now partial by sport and stat type,
        # so only the row itself knows. Absent means unrestricted.
        awt = a.get("allowed_wager_types")
        direction = sides_from_allowed_wager_types(awt)
        wager_types[str(awt)] = wager_types.get(str(awt), 0) + 1
        rows.append({
            "name": name, "team": team, "position": pos,
            "stat_type": a.get("stat_type") or a.get("stat_display_name"),
            "line": float(a["line_score"]),
            "start_time": a.get("start_time"),
            "direction": direction,
            "odds_type": odds,
        })
    df = pd.DataFrame(rows)
    df.attrs["skipped_noname"] = skipped
    # Logged so a shift in the field's distribution shows up here rather than in
    # a user complaint. The NFL rollout landing will change these counts.
    df.attrs["allowed_wager_types"] = wager_types
    if wager_types:
        logger.info("allowed_wager_types distribution: %s", wager_types)
    return df


_BOARD_NUM = re.compile(r"^\d+(\.\d+)?$")
_BOARD_TEAMPOS = re.compile(r"^[A-Z]{2,4}\s*-\s*[A-Za-z0-9]{1,3}$")
_BOARD_SUFFIX = re.compile(r"(Demon|Goblin)+$")


def parse_prizepicks_board(text: str) -> pd.DataFrame:
    """Parse text copied from the PrizePicks board itself.

    The board is the RELIABLE source for pick direction, because it renders the
    Less and More BUTTONS the feed does not spell out. Reading those buttons is
    the only way to know which sides a prop actually offers, and it beats any
    rule of thumb about Demons and Goblins: whatever the current policy is, the
    board shows what it shows.

        Ja'Marr Chase
        CIN - WR
        85.5
        Receiving Yards
        Less
        More

    Both buttons present means either side can be taken; one button means only
    that side. When the copied text carries no buttons at all (some copy modes
    drop them), direction stays unknown and falls back to the odds-type policy
    in `ODDS_TYPE_SIDES`.

    Tolerant of a Demon or Goblin suffix glued to the stat name.
    """
    rows: list[dict] = []
    lines = [ln.strip() for ln in str(text).splitlines()]
    lines = [ln for ln in lines if ln]
    i = 0
    while i < len(lines) - 2:
        name = lines[i]
        if not _person_like(name):
            i += 1
            continue
        team = pos = None
        j = i + 1
        if j < len(lines) and _BOARD_TEAMPOS.match(lines[j]):
            team, pos = (x.strip() for x in re.split(r"\s*-\s*", lines[j], 1))
            j += 1
        if j >= len(lines) or not _BOARD_NUM.match(lines[j]):
            i += 1
            continue
        line_val = float(lines[j])
        j += 1
        if j >= len(lines):
            break
        stat = _BOARD_SUFFIX.sub("", lines[j]).strip()
        odds = "demon" if "Demon" in lines[j] else \
               "goblin" if "Goblin" in lines[j] else "standard"
        j += 1

        # The Less / More buttons follow the stat label. This is the ONLY
        # trustworthy source of pick direction: it is what the board actually
        # renders, so it needs no assumption about how Demons or Goblins behave
        # this season. Both buttons means either side; one means that side only;
        # none means the copy dropped them and direction stays unknown.
        btns = set()
        while j < len(lines) and lines[j] in ("Less", "More"):
            btns.add(lines[j])
            j += 1
        direction = ("both" if {"Less", "More"} <= btns
                     else "more" if "More" in btns
                     else "less" if "Less" in btns else "")

        rows.append({"name": name, "team": team, "position": pos,
                     "stat_type": stat, "line": line_val, "start_time": None,
                     "direction": direction, "odds_type": odds})
        i = j
    return pd.DataFrame(rows)


def parse_any(text: str) -> pd.DataFrame:
    """Accept either a pasted JSON feed or pasted board text."""
    t = str(text or "").strip()
    if not t:
        return pd.DataFrame()
    if t.startswith("{") or t.startswith("["):
        try:
            return parse_prizepicks_json(t)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("JSON parse failed (%s); trying board text", e)
    return parse_prizepicks_board(t)


def collapse_alt_lines(df: pd.DataFrame) -> pd.DataFrame:
    """One line per (player, stat), preferring the standard line.

    PrizePicks posts an alt-line ladder for popular props. Left alone it
    triple-counts every player and makes any accuracy number meaningless.
    """
    if df.empty:
        return df
    out = df.copy()
    out["_key"] = out["name"].map(normalize_name) + "|" + \
        out["stat_type"].astype(str).str.lower().str.strip()
    out["_rank"] = (out["odds_type"] != "standard").astype(int)
    out = (out.sort_values(["_key", "_rank"])
              .drop_duplicates("_key", keep="first")
              .drop(columns=["_key", "_rank"]))
    return out.reset_index(drop=True)


# ── Which sides a line actually offers ───────────────────────────────────────
# Researched 2026-08-31 across PrizePicks' own dated sources. The short version,
# because the internet is uniformly wrong about this:
#
#   - Demons and Goblins WERE More-only from their Dec 2023 launch.
#   - That changed on **2026-08-21**: "Demons and Goblins, Explained: Less Picks
#     Are Here" (prizepicks.com playbook, and the matching help-centre page),
#     confirmed by the iOS 12.86 release note "Take Less on select Demons and
#     Goblins".
#   - The rollout is PARTIAL. The announcement scopes it to "several MLB and
#     WNBA stat types, with more sports, **including NFL, coming soon**". So for
#     NFL today most Demons and Goblins are still More-only.
#   - PrizePicks' own /help-center/player-picks page STILL states the old
#     More-only rule, so even a first-party citation is not automatically
#     current. Every third-party guide checked (Stokastic, RotoGrinders,
#     FantasyLife, HeatCheck) is stale.
#
# The consequence for this code: do NOT gate on odds_type. Hardcoding
# "demon means More-only" would hide real placeable picks the moment the NFL
# rollout lands, and hardcoding "both" hides nothing today when it should. The
# feed answers it PER ROW via `allowed_wager_types`, and the board answers it
# per row via its Less/More buttons. Both are read; this table is only the
# last-resort default when neither is present.
ODDS_TYPE_SIDES = {
    "standard": "both",
    "demon": "both",
    "goblin": "both",
}

# `attributes.allowed_wager_types` in the feed. ABSENT means unrestricted; the
# only value observed in the wild is "over". Parsed defensively: an unrecognised
# value is logged and treated as unrestricted rather than silently hiding lines.
_WAGER_TYPE_SIDES = {"over": "more", "under": "less"}


def sides_from_allowed_wager_types(value) -> str:
    """Translate the feed's `allowed_wager_types` into an offered-sides string.

    None or missing -> "both" (the overwhelming majority of rows). A list or a
    bare string is accepted, because the field's vocabulary is thinly observed
    and its shape may vary.
    """
    if value is None:
        return "both"
    items = value if isinstance(value, (list, tuple, set)) else [value]
    sides = set()
    for item in items:
        mapped = _WAGER_TYPE_SIDES.get(str(item).strip().lower())
        if mapped:
            sides.add(mapped)
        else:
            logger.warning(
                "unrecognised allowed_wager_types value %r; treating the line "
                "as two-sided. If this appears often the feed's vocabulary has "
                "changed and props.py needs updating.", item)
            return "both"
    if not sides or sides == {"more", "less"}:
        return "both"
    return sides.pop()


def offered_sides(direction, odds_type=None) -> str:
    """Which side(s) can actually be taken: 'both', 'more' or 'less'.

    An EXPLICIT direction always wins: that comes from the board's own Less and
    More buttons, which is what the market is really offering. Only when the
    paste carried no buttons does this fall back to `ODDS_TYPE_SIDES`.

    The fallback is deliberately permissive. Showing a side the market may not
    offer is a smaller failure than silently hiding a real line, so an unknown
    direction resolves to both rather than to a guess.
    """
    d = str(direction or "").strip().lower()
    if d in ("both", "more", "less"):
        return d
    return ODDS_TYPE_SIDES.get(str(odds_type or "").strip().lower(), "both")


def is_pickable(lean: str, sides: str) -> bool:
    """Can the model's lean actually be played?

    'Even' is never pickable: the model landed exactly on the line, so there is
    no side to take.
    """
    lean = str(lean or "").strip().lower()
    sides = str(sides or "both").strip().lower()
    if lean not in ("more", "less"):
        return False
    return sides == "both" or sides == lean


# A player with fewer prior games than this has an essentially empty feature
# vector, so the model falls back to the base rate of "fringe debutant" and its
# error is at its largest. Ranking a board by absolute gap is an argmax over the
# model's own error, so those rows sort to the TOP by construction.
#
# Measured on 2025 week 6: 61 projected players with 2 or fewer prior games
# averaged 2.97 projected carries against 0.08 actual. That is not an edge, it
# is ignorance being ranked first.
MIN_GAMES_FOR_EDGE = 3

# ── Per-stat reliability ─────────────────────────────────────────────────────
# Measured, not assumed: `scripts/validate_lean_accuracy.py` asks the question a
# prop actually poses, which is not "how close is the number" but "which side of
# ONE number does the outcome land on". Those are different, and the difference
# is not small. For receiving yards the model BEATS a season-to-date average on
# MAE by 5.1% and LOSES to it on side-picking by 5.7 points.
#
# Ranking a board purely by gap size treats every stat as equally informative.
# On the measured numbers they are opposite in sign (+10.1 for pass attempts,
# -9.4 for rushing yards), so the largest gaps on an unweighted board are
# disproportionately the stats where the model is worse than a coin flip.
TIER_ORDER = {"strong": 0, "moderate": 1, "weak": 2, "none": 3, "negative": 4}
TIER_LABEL = {
    "strong": "strong",
    "moderate": "moderate",
    "weak": "weak",
    "none": "no edge",
    "negative": "worse than a coin flip",
}


@lru_cache(maxsize=1)
def reliability_table() -> pd.DataFrame:
    """data/stat_reliability.csv. Missing file degrades to 'unmeasured'."""
    path = Path(__file__).resolve().parent.parent / "data" / "stat_reliability.csv"
    try:
        return pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        logger.warning("stat reliability table unavailable (%s)", e)
        return pd.DataFrame(columns=["stat", "tier", "side_edge_pts", "note"])


def reliability_for(columns) -> dict:
    """{tier, edge, note} for the projection column(s) behind a prop.

    A combination prop is only as trustworthy as its WEAKEST component: a
    rush+rec yards line inherits the rushing-yards problem whole.
    """
    tbl = reliability_table()
    if tbl.empty or not columns:
        return {"tier": "unmeasured", "edge": None, "note": ""}
    rows = tbl[tbl["stat"].isin(list(columns))]
    if rows.empty:
        return {"tier": "unmeasured", "edge": None, "note": ""}
    worst = max(rows["tier"], key=lambda t: TIER_ORDER.get(str(t), 2))
    hit = rows[rows["tier"] == worst].iloc[0]
    edge = hit.get("side_edge_pts")
    return {"tier": str(worst),
            "edge": None if pd.isna(edge) else float(edge),
            "note": str(hit.get("note") or "")}


def rank_by_reliability(table: pd.DataFrame) -> pd.DataFrame:
    """Sort by how much the lean is worth, then by CONFIDENCE in it.

    Tier first, because sorting by size alone puts the least trustworthy stats
    at the top of the board, which is the opposite of useful.

    Confidence second, NOT gap size. The gap was the secondary key until now and
    it is the wrong quantity twice over: it is biased on right-skewed stats (a
    big positive gap can still be under a coin flip), and it is not comparable
    across stats, since seven yards and half a reception are different things
    that sorted as 7.0 against 0.5. Distance of the probability from a coin flip
    is scale-free and is what a reader is actually ranking by.

    Rows with no fitted distribution fall back to the RELATIVE gap, which is at
    least scale-free, rather than the raw one.
    """
    if table is None or table.empty or "tier" not in table.columns:
        return table
    out = table.copy()
    out["_tier_rank"] = out["tier"].map(
        lambda t: TIER_ORDER.get(str(t), 2)).fillna(2)
    if "p_over" in out.columns:
        conf = pd.to_numeric(out["p_over"], errors="coerce").sub(0.5).abs()
    else:
        conf = pd.Series(np.nan, index=out.index)
    # A probability prop's own value IS a probability, so its confidence is its
    # distance from a coin flip too.
    if "kind" in out.columns:
        is_prob = out["kind"] == "probability"
        conf = conf.mask(is_prob,
                         pd.to_numeric(out["model"], errors="coerce")
                         .sub(0.5).abs())
    fallback = (pd.to_numeric(out.get("diff_pct"), errors="coerce").abs() / 400.0
                if "diff_pct" in out.columns
                else pd.Series(0.0, index=out.index))
    out["_gap"] = conf.fillna(fallback).fillna(0.0)
    return (out.sort_values(["_tier_rank", "_gap"], ascending=[True, False])
               .drop(columns=["_tier_rank", "_gap"])
               .reset_index(drop=True))


def filter_informed(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a compared table into (informed, too-thin-to-rank).

    Returns the suppressed rows rather than dropping them, so the count can be
    reported. A board that quietly shrinks is worse than one that explains why.
    """
    if table is None or table.empty or "games_prior" not in table.columns:
        return table, pd.DataFrame()
    keep = table["games_prior"].fillna(0) >= MIN_GAMES_FOR_EDGE
    return table[keep].reset_index(drop=True), table[~keep].reset_index(drop=True)


def filter_pickable(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a compared table into (pickable, hidden).

    A row whose lean is a side the board does not offer is a pick the user
    cannot make. Showing it invites someone to act on a number that leads
    nowhere, so it is removed from the ledger. The hidden rows are RETURNED
    rather than dropped, because the count has to be reported: a board that
    quietly shrinks is worse than one that explains itself.
    """
    if table is None or table.empty or "lean" not in table.columns:
        return table, pd.DataFrame()
    keep = table.apply(
        lambda r: is_pickable(r.get("lean"), r.get("sides", "both")), axis=1)
    return table[keep].reset_index(drop=True), table[~keep].reset_index(drop=True)


# ── Storage, keyed by the week each line's own game belongs to ───────────────
def lines_path(season: int, week: int):
    return dc_path(f"pp_lines_{season}_w{week:02d}_{VERSION}.json")


def save_lines(season: int, week: int, lines: pd.DataFrame) -> None:
    """Persist a pasted board.

    This writes even on a host with no persistent disk, where the file lives
    only until the next restart. That is deliberate and it is what DiamondValue
    does: a temporary board is still worth showing, and losing it later costs
    nothing that was not already going to be re-pasted.

    What must NOT happen on such a host is freezing, which is guarded in
    `freeze_projections`. The board is a view; the record is evidence.
    """
    payload = {
        "season": season, "week": week,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": lines.to_dict(orient="records"),
    }
    json_save(lines_path(season, week), payload)


def load_lines(season: int, week: int) -> pd.DataFrame | None:
    path = lines_path(season, week)
    if not path.exists():
        return None
    try:
        payload = json_load(path)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not read saved lines %s: %s", path, e)
        return None
    df = pd.DataFrame(payload.get("rows", []))
    df.attrs["saved_at"] = payload.get("saved_at")
    return df


def saved_weeks(season: int) -> list[int]:
    """Weeks with a saved board, ascending. Drives the week-by-week view."""
    out = []
    for w in range(1, 19):
        if lines_path(season, w).exists():
            out.append(w)
    return out


# ── Frozen projections: what the model ACTUALLY said, when it said it ───────
# Without this the accuracy record is worthless, and worse than worthless
# because it looks authoritative.
#
# `compare()` joins saved lines to projections computed RIGHT NOW. Grade a week
# from three weeks ago and it is scored against today's model, which has since
# been retrained on data that did not exist when the board was posted. The
# record would show what a future model would have done. It would also silently
# improve every time the models are retrained, which is exactly backwards: a
# record that cannot get worse is not measuring anything.
#
# So the model's number is frozen the moment a line is first seen, and grading
# reads the frozen value. FIRST SEEN WINS: re-pasting a board on Saturday must
# not overwrite Wednesday's projection with one that has since absorbed three
# days of injury news. New lines in that paste are frozen at their own time.
FREEZE_KEYS = ["player", "stat", "line"]


def snapshot_path(season: int, week: int):
    return dc_path(f"pp_frozen_{season}_w{week:02d}_{VERSION}.json")


def _freeze_key(row) -> str:
    return "|".join(str(row.get(k)) for k in FREEZE_KEYS)


def freeze_projections(season: int, week: int, table: pd.DataFrame,
                       model_version: str | None = None) -> int:
    """Record the model's number for every line not already frozen.

    Returns how many NEW rows were frozen, so a caller can say so rather than
    reporting silence. Safe to call repeatedly; it is a no-op for lines that
    already have a frozen value.
    """
    from .cache import READ_ONLY
    if READ_ONLY:
        # Pasting is allowed on a read-only host; FREEZING is not. A snapshot
        # taken here would be stamped with whatever model this container loaded,
        # would be lost at the next restart, and in the meantime would make the
        # published record disagree with the committed one. The record only ever
        # changes through a commit, which is what makes it evidence rather than
        # whatever the last visitor happened to paste.
        return 0
    if table is None or table.empty:
        return 0
    keep = [c for c in ("player", "stat", "line", "model", "diff", "lean",
                        "sides", "odds_type", "source", "tier", "position",
                        "team", "opponent", "gsis_id", "game_id",
                        "games_prior", "p_play",
                        # The claimed probability and its units. Without these
                        # the record can say whether the LEAN was right but not
                        # whether the NUMBER was honest, and "we said 65%" is
                        # only checkable if 65% was written down.
                        "p_over", "kind") if c in table.columns]
    path = snapshot_path(season, week)
    existing = {}
    if path.exists():
        try:
            payload = json_load(path)
            existing = {r["_key"]: r for r in payload.get("rows", [])}
        except Exception as e:  # noqa: BLE001
            logger.warning("could not read snapshot %s: %s", path, e)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    added = 0
    for _, r in table[keep].iterrows():
        k = _freeze_key(r)
        if k in existing:
            continue          # first seen wins
        row = {c: (None if pd.isna(r[c]) else r[c]) for c in keep}
        row["_key"] = k
        row["frozen_at"] = now
        row["model_version"] = model_version
        existing[k] = row
        added += 1

    json_save(path, {"season": season, "week": week,
                     "updated_at": now, "rows": list(existing.values())})
    return added


def load_frozen(season: int, week: int) -> pd.DataFrame | None:
    """The frozen comparison for a week, or None if nothing was ever frozen."""
    path = snapshot_path(season, week)
    if not path.exists():
        return None
    try:
        payload = json_load(path)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not read snapshot %s: %s", path, e)
        return None
    df = pd.DataFrame(payload.get("rows", []))
    if df.empty:
        return None
    df.attrs["updated_at"] = payload.get("updated_at")
    return df


def frozen_weeks(season: int) -> list[int]:
    return [w for w in range(1, 19) if snapshot_path(season, w).exists()]


def route_to_weeks(lines: pd.DataFrame, games: pd.DataFrame,
                   season: int) -> dict[int, pd.DataFrame]:
    """Split a pasted board into the weeks its games actually belong to.

    A board pasted on Wednesday carries Sunday's games; a Thursday board can
    carry next Monday's too. Routing to the week on screen would file half of
    them under the wrong week and quietly corrupt the accuracy history.

    Lines with no usable start time fall back to the caller's week, which is
    reported rather than hidden.
    """
    if lines.empty:
        return {}
    reg = games[(games["season"] == season) & (games["game_type"] == "REG")].copy()
    reg["kickoff"] = fetch_kickoffs(reg)
    starts = pd.to_datetime(lines.get("start_time"), errors="coerce", utc=True)

    out: dict[int, list] = {}
    unrouted = []
    for idx, row in lines.iterrows():
        ts = starts.iloc[lines.index.get_loc(idx)] if "start_time" in lines else pd.NaT
        if pd.isna(ts):
            unrouted.append(idx)
            continue
        # The week whose kickoff window contains this start time.
        same_day = reg[(reg["kickoff"].dt.tz_convert("UTC") - ts).abs()
                       < pd.Timedelta(hours=6)]
        wk = int(same_day["week"].mode().iloc[0]) if len(same_day) else None
        if wk is None:
            unrouted.append(idx)
            continue
        out.setdefault(wk, []).append(row)

    result = {w: pd.DataFrame(rows).reset_index(drop=True)
              for w, rows in out.items()}
    if unrouted:
        result.setdefault("_unrouted", pd.DataFrame(
            lines.loc[unrouted]).reset_index(drop=True))
    return result


def fetch_kickoffs(games: pd.DataFrame):
    from .fetch import kickoff_series
    return kickoff_series(games)


# ── Matching lines to projections ────────────────────────────────────────────
def poisson_at_least(lam: float, k: int) -> float:
    """P(X >= k) for X ~ Poisson(lam). Touchdowns are the canonical count.

    Poisson is the right family here and not merely a convenience: scoring
    chances arrive as a stream of near-independent low-probability events, which
    is exactly the generating story. It is imperfect (goal-line volume clusters,
    so the true distribution is slightly over-dispersed), and over-dispersion
    pushes P(>=1) DOWN for a given mean, so this is the optimistic side of the
    error rather than the flattering one.
    """
    if lam is None or not np.isfinite(lam) or lam < 0:
        return float("nan")
    if k <= 0:
        return 1.0
    # P(X >= k) = 1 - sum_{i<k} pmf(i), summed forward to stay stable at small k.
    term = np.exp(-lam)
    cum = term
    for i in range(1, k):
        term *= lam / i
        cum += term
    return float(min(1.0, max(0.0, 1.0 - cum)))


def threshold_for_line(line: float) -> int:
    """How many touchdowns 'More' actually requires.

    A 0.5 line needs 1, a 1.5 line needs 2. PrizePicks posts half points so the
    outcome cannot land on the number.
    """
    return int(np.floor(float(line))) + 1


def _resolve_stat(stat_type: str):
    """(columns, scale, note) for a stat, or (None, None, reason) if refused."""
    key = str(stat_type or "").strip().lower()
    if key in REFUSE:
        return None, None, REFUSE[key]
    if key in FANTASY_KEYS:
        return ("__fantasy__",), 1.0, None
    hit = STAT_MAP.get(key)
    if hit:
        return hit[0], hit[1], None
    return None, None, "stat not mapped to a projected quantity"


def compare(lines: pd.DataFrame, proj: pd.DataFrame,
            registry: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Join posted lines to our projections. Returns (table, meta).

    `meta` carries the counts that stop an empty result from rendering as
    silence: how many lines were refused and why, how many players did not
    match a roster, how many stats are unmapped.

    Every row is labelled with the source of its projection, because a target
    that failed the ship gate is served from a baseline and the reader has to
    know that. A comparison that presents a failed touchdown model with the same
    confidence as a passing-attempts model is a misleading page.
    """
    from .predict import PP_SCORING, fantasy_score

    meta = {"posted": int(len(lines)), "refused": {}, "unmatched_players": 0,
            "unmapped_stats": {}, "matched": 0, "low_confidence": 0,
            "baseline_served": 0}
    if lines.empty or proj.empty:
        meta["reason"] = ("no lines pasted" if lines.empty
                          else "no projections for this week")
        return pd.DataFrame(), meta

    proj = proj.copy()
    proj["_key"] = proj["player_display_name"].map(normalize_name)
    proj["_fantasy"] = proj.apply(fantasy_score, axis=1)
    by_key = proj.drop_duplicates("_key").set_index("_key")

    sources = {}
    if registry:
        sources = {t: m.get("present_as", "model")
                   for t, m in registry.get("targets", {}).items()}

    rows = []
    for _, ln in lines.iterrows():
        cols, scale, refusal = _resolve_stat(ln.get("stat_type"))
        if cols is None:
            bucket = meta["refused"] if refusal in REFUSE.values() else \
                meta["unmapped_stats"]
            k = str(ln.get("stat_type"))
            bucket[k] = bucket.get(k, 0) + 1
            continue

        key = normalize_name(ln.get("name"))
        if key not in by_key.index:
            meta["unmatched_players"] += 1
            continue
        p = by_key.loc[key]

        line_val = float(ln["line"])
        # Most props compare a projected MEAN to the posted number. Touchdowns
        # compare a PROBABILITY to 0.5, because the question a 0.5 line asks is
        # "at least one", not "how many on average". Carried as `kind` so the
        # reader is never shown a probability formatted as a stat line.
        kind, compare_to = "mean", line_val

        if cols and cols[0] == "__prob__":
            parts = [p.get(c) for c in cols[1:]]
            lam = sum(float(v) for v in parts
                      if v is not None and pd.notna(v))
            if lam <= 0:
                continue
            model_val = poisson_at_least(lam, threshold_for_line(line_val))
            kind, compare_to = "probability", 0.5
            # Both components are served from a baseline; they lose the ship
            # gate to a season-to-date mean. Labelled, never dressed up.
            src = "baseline"
        elif cols == ("__fantasy__",):
            model_val = float(p["_fantasy"])
            src = "composite"
        else:
            vals = [p.get(c) for c in cols]
            if any(v is None or pd.isna(v) for v in vals):
                meta["unmatched_players"] += 0  # matched player, unprojected stat
                continue
            model_val = float(sum(float(v) for v in vals)) * float(scale)
            srcs = {sources.get(c, "model") for c in cols}
            src = ("baseline" if "baseline" in srcs
                   else "model_low_confidence" if "model_low_confidence" in srcs
                   else "model")

        if src == "baseline":
            meta["baseline_served"] += 1
        elif src == "model_low_confidence":
            meta["low_confidence"] += 1

        # A probability prop inherits the reliability of the stats behind it.
        _rel = reliability_for(
            tuple(cols[1:]) if cols and cols[0] == "__prob__"
            else () if cols == ("__fantasy__",) else cols)
        diff = model_val - compare_to
        lean = "More" if diff > 0 else "Less" if diff < 0 else "Even"
        # P(actual > line), from measured residual distributions. Only for
        # single-stat MEAN props: a combination prop has no fitted ratio, and a
        # probability prop already IS one.
        p_over = float("nan")
        if kind == "mean" and len(cols) == 1 and cols[0] not in ("__fantasy__",):
            p_over = _uncertainty().prob_over(cols[0], model_val, line_val)
        # SHOW THE MEDIAN, not the mean, wherever a distribution exists.
        # The lean is decided by the median (More exactly when P(over) > 0.5),
        # so printing the mean beside the line produced rows that contradicted
        # themselves: "MODEL 4.8, LINE 4.5, LEAN Less". Both numbers were right
        # and the row was unreadable. With the median, MODEL > LINE and "More"
        # are the same statement and the contradiction cannot occur.
        #
        # `model_val` stays the MEAN above, because the ratio distribution is
        # defined against it and P(over) must keep being computed from it.
        if kind == "mean" and np.isfinite(p_over):
            _med = _uncertainty().median_projection(cols[0], model_val)
            if np.isfinite(_med):
                model_val = _med
                diff = model_val - compare_to
        if kind == "mean" and np.isfinite(p_over):
            # THE LEAN COMES FROM THE PROBABILITY, not the gap, wherever one
            # exists. Measured on 2024: the gap rule leaned More on 66% of
            # receiving-yards lines when only 41% went Over, because these
            # stats are right-skewed and a line below the projected MEAN can
            # still sit above the median. Switching the rule moves receiving
            # yards from -5.4 to +3.0 and rushing yards from -7.7 to +2.3,
            # while costing the near-symmetric QB props 0.7 to 1.5.
            lean = "More" if p_over > 0.5 else "Less" if p_over < 0.5 else "Even"
        rows.append({
            "player": p["player_display_name"], "team": p["team"],
            "position": p["position"], "opponent": p.get("opponent_team"),
            "stat": ln.get("stat_type"), "line": line_val,
            "model": round(model_val, 2), "diff": round(diff, 2),
            "kind": kind,
            # A ratio against the line is meaningless for a probability, so it
            # is reported in PERCENTAGE POINTS away from a coin flip instead.
            "diff_pct": (round(100 * diff, 1) if kind == "probability"
                         else round(100 * diff / line_val, 1) if line_val
                         else None),
            "lean": lean,
            "p_over": (round(float(p_over), 4)
                       if np.isfinite(p_over) else None),
            "sides": offered_sides(ln.get("direction"), ln.get("odds_type")),
            "odds_type": ln.get("odds_type", "standard"),
            "source": src,
            "gsis_id": p.get("gsis_id"),
            "game_id": p.get("game_id"),
            # How much history the model had for this player. An edge is only
            # as good as the information behind it, and this is the cheapest
            # honest measure of that.
            **{f"_rel_{k}": v for k, v in _rel.items()},
            "games_prior": (float(p.get("f_games_prior"))
                            if p.get("f_games_prior") is not None
                            and pd.notna(p.get("f_games_prior")) else 0.0),
            "p_play": (float(p.get("p_play"))
                       if p.get("p_play") is not None
                       and pd.notna(p.get("p_play")) else np.nan),
        })

    meta["matched"] = len(rows)
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.rename(columns={"_rel_tier": "tier",
                                      "_rel_edge": "tier_edge",
                                      "_rel_note": "tier_note"})
        # Tier first, gap second. Sorting by gap alone puts the least
        # trustworthy stats at the top, which is worse than not sorting.
        table = rank_by_reliability(table)
        meta["by_tier"] = table["tier"].value_counts().to_dict()
    return table, meta


# ── Grading, once the games have been played ─────────────────────────────────
# The actual value of a prop, in the line's units, from the played box score.
ACTUAL_MAP = {
    "passing_yards": "passing_yards", "passing_tds": "passing_tds",
    "attempts": "attempts", "completions": "completions",
    "passing_interceptions": "passing_interceptions",
    "sacks_suffered": "sacks_suffered",
    "rushing_yards": "rushing_yards", "carries": "carries",
    "receiving_yards": "receiving_yards", "receptions": "receptions",
    "targets": "targets", "fg_made": "fg_made", "pat_att": "pat_att",
}


def grade(table: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Score a compared table against what actually happened.

    THREE GRADING CAVEATS, all of which make our number differ from what
    PrizePicks itself would settle:

      1. PrizePicks settles via Sportradar, Genius or Stats Perform, NOT
         nflverse. The stats where providers disagree are disproportionately the
         propped ones, so some disagreements here are provider differences
         rather than model error.
      2. An exact tie on the line is NOT a push on PrizePicks; it lowers the
         payout tier. Ties are reported separately here rather than folded into
         either side.
      3. "Reboot" protection is asymmetric by direction, so grading both
         directions symmetrically against a final box score overstates More-side
         accuracy. This grades the MODEL'S LEAN only, which sidesteps that.
    """
    if table.empty or actuals is None or actuals.empty:
        return pd.DataFrame()

    act = actuals.copy()
    act["_key"] = act["player_display_name"].map(normalize_name)
    by_key = act.drop_duplicates("_key").set_index("_key")

    rows = []
    for _, r in table.iterrows():
        cols, scale, refusal = _resolve_stat(r["stat"])
        if cols is None or cols == ("__fantasy__",):
            continue
        key = normalize_name(r["player"])
        if key not in by_key.index:
            continue
        a = by_key.loc[key]
        vals = [a.get(ACTUAL_MAP.get(c, c)) for c in cols]
        if any(v is None or pd.isna(v) for v in vals):
            continue  # did not play, or not scored
        actual = float(sum(float(v) for v in vals)) * float(scale)
        if actual > r["line"]:
            result = "More"
        elif actual < r["line"]:
            result = "Less"
        else:
            result = "Exact"
        rows.append({**r.to_dict(), "actual": round(actual, 2),
                     "result": result,
                     "model_correct": (None if result == "Exact"
                                       else result == r["lean"])})
    return pd.DataFrame(rows)


def is_in_training_window(season: int, registry: dict | None) -> bool:
    """True when this season was part of the model's training data.

    Grading a week the model trained on is not a backtest, it is the model
    grading its own homework, and the hit rate will be flattering and
    meaningless. The models ship trained through 2025, so any 2025-or-earlier
    week must be labelled in-sample wherever a record is displayed.
    """
    if not registry:
        return False
    through = registry.get("trained_through")
    return through is not None and season <= int(through)


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a hit rate, Wilson score.

    Wilson rather than the normal approximation because the samples here are
    small and the rate sits near 0.5 where the normal interval is at its worst,
    and because Wilson does not produce impossible bounds outside [0, 1].
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def season_record(graded_weeks: list[pd.DataFrame]) -> dict:
    """Aggregate record across weeks, with an interval and a verdict.

    THE INTERVAL IS THE POINT. A prop board grades a few hundred lines a season,
    and at n=300 the 95% interval on a hit rate is roughly plus or minus 5.7
    points. So 53% and 47% are the same measurement, and a bare percentage
    invites reading a coin flip as an edge. Reporting the interval, and refusing
    to call anything an edge until the lower bound clears 50, is the difference
    between a record and a story.
    """
    frames = [g for g in graded_weeks if g is not None and not g.empty]
    if not frames:
        return {"n": 0, "decided": 0}
    allg = pd.concat(frames, ignore_index=True)
    decided = allg[allg["model_correct"].notna()]
    n = int(len(decided))
    if n == 0:
        return {"n": int(len(allg)), "decided": 0}
    hits = int(decided["model_correct"].sum())
    lo, hi = wilson_interval(hits, n)
    return {
        "n": int(len(allg)),
        "decided": n,
        "hits": hits,
        "hit_rate": round(100 * hits / n, 1),
        "ci_low": round(100 * lo, 1),
        "ci_high": round(100 * hi, 1),
        # A coin is 50. An edge exists only if the LOWER bound clears it.
        "beats_coin": bool(lo > 0.5),
        "ties": int(allg["model_correct"].isna().sum()),
    }


def week_scorecard(graded: pd.DataFrame) -> dict:
    """Honest per-week summary. Ties are their own bucket, never a win."""
    if graded.empty:
        return {"n": 0}
    decided = graded[graded["model_correct"].notna()]
    n = len(decided)
    hits = int(decided["model_correct"].sum()) if n else 0
    return {
        "n": int(len(graded)),
        "decided": n,
        "exact_ties": int((graded["result"] == "Exact").sum()),
        "hits": hits,
        "hit_rate": round(100 * hits / n, 1) if n else None,
        "mae": round(float((graded["model"] - graded["actual"]).abs().mean()), 2),
        "line_mae": round(float((graded["line"] - graded["actual"]).abs().mean()), 2),
    }
