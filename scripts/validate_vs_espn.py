"""Spot-check the backfill against ESPN, an INDEPENDENT data provider.

The internal reconciliation in `validate_backfill.py` proves the pipeline is
self-consistent: our player sums equal the nflverse box score, and our pbp
aggregates equal it too. It cannot prove nflverse itself is right, because every
number in it comes from nflverse.

This script closes that loop by comparing season totals against ESPN's public
athlete gamelog API, which is sourced separately from nflverse/GSIS. Agreement
across providers on volume stats is strong evidence the backfill is correct.

Pro-Football-Reference would be the other natural check, and the spec suggests
it, but pro-football-reference.com sits behind Cloudflare bot verification that
blocks automated requests. That is a bot-detection control and is not something
to work around, so ESPN is used instead. A human can still open PFR in a browser
to confirm any line here.

Note the spec's own warning applies in reverse too: providers disagree on some
stats by design (tackles, longest reception, occasionally target counts), so a
mismatch here is a prompt to investigate, not automatic proof of a bug.

Usage:
    python scripts/validate_vs_espn.py                 # the default panel
    python scripts/validate_vs_espn.py --season 2023
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import fetch  # noqa: E402
from gridlib.cache import dc_path, read_parquet_or_none  # noqa: E402

ESPN_GAMELOG = ("https://site.web.api.espn.com/apis/common/v3/sports/football/"
                "nfl/athletes/{athlete_id}/gamelog?season={season}")
UA = {"User-Agent": "GridironValue/0.1 (personal analytics; verification)"}

# One player per position group, chosen for high volume so a mismatch is
# unambiguous rather than a rounding artifact. Keyed by gsis_id, never by name:
# there are two Josh Allens in players.parquet (a QB and a center), which is
# exactly the collision that makes name joins unsafe.
PANEL = {
    2024: [
        ("00-0036900", "WR"),   # Ja'Marr Chase
        ("00-0034857", "QB"),   # Josh Allen, the quarterback
        ("00-0037692", "K"),    # Brandon Aubrey
        ("00-0036223", "RB"),   # Jonathan Taylor
        ("00-0039338", "TE"),   # Brock Bowers, a genuine tight end
    ],
}

# ESPN label -> our column, per position group.
MAPPING = {
    "QB": {
        "completions": "completions", "passingAttempts": "attempts",
        "passingYards": "passing_yards", "passingTouchdowns": "passing_tds",
        "interceptions": "passing_interceptions", "sacks": "sacks_suffered",
        "rushingAttempts": "carries", "rushingYards": "rushing_yards",
        "rushingTouchdowns": "rushing_tds",
    },
    "SKILL": {
        "receptions": "receptions", "receivingTargets": "targets",
        "receivingYards": "receiving_yards",
        "receivingTouchdowns": "receiving_tds",
        "rushingAttempts": "carries", "rushingYards": "rushing_yards",
        "rushingTouchdowns": "rushing_tds",
    },
}


def espn_totals(athlete_id: str, season: int) -> dict[str, str]:
    req = urllib.request.Request(
        ESPN_GAMELOG.format(athlete_id=athlete_id, season=season), headers=UA)
    with urllib.request.urlopen(req, timeout=45) as fh:
        d = json.load(fh)
    labels = d.get("names") or []
    for st in d.get("seasonTypes") or []:
        if "Regular" not in str(st.get("displayName", "")):
            continue
        for cat in st.get("categories") or []:
            totals = cat.get("totals")
            if totals and len(totals) == len(labels):
                return dict(zip(labels, totals))
    return {}


def _num(text: str) -> float | None:
    try:
        return float(str(text).replace(",", ""))
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--table", default="player_week_2016_2025_v1.parquet")
    args = ap.parse_args()

    pw = read_parquet_or_none(dc_path(args.table))
    if pw is None:
        print(f"{args.table} not found. Run scripts/build_player_week.py first.")
        sys.exit(2)
    players = fetch.load_players()
    espn_by_gsis = (players.dropna(subset=["espn_id"])
                    .drop_duplicates("gsis_id")
                    .set_index("gsis_id")["espn_id"].astype(str))
    name_by_gsis = players.drop_duplicates("gsis_id").set_index("gsis_id")["display_name"]

    panel = PANEL.get(args.season)
    if not panel:
        print(f"No panel defined for {args.season}. Add one to PANEL.")
        sys.exit(2)

    checked = mismatched = 0
    print(f"\nSeason {args.season}: GridironValue vs ESPN\n")
    for gsis, group in panel:
        sub = pw[(pw["gsis_id"] == gsis) & (pw["season"] == args.season)]
        name = name_by_gsis.get(gsis, gsis)
        if sub.empty:
            print(f"  {name}: NOT IN TABLE")
            mismatched += 1
            continue
        aid = espn_by_gsis.get(gsis)
        if not aid:
            print(f"  {name}: no ESPN id in the crosswalk")
            continue
        try:
            totals = espn_totals(aid, args.season)
        except Exception as e:  # noqa: BLE001 — a provider outage is not a data bug
            print(f"  {name}: ESPN fetch failed ({e})")
            continue
        if not totals:
            print(f"  {name}: ESPN returned no regular-season totals")
            continue

        mapping = MAPPING["QB" if group == "QB" else "SKILL"]
        print(f"  {name} ({group}, {len(sub)} games)")
        for espn_key, our_col in mapping.items():
            if espn_key not in totals or our_col not in sub.columns:
                continue
            theirs = _num(totals[espn_key])
            ours = float(sub[our_col].sum())
            if theirs is None:
                continue
            checked += 1
            ok = abs(ours - theirs) < 0.5
            if not ok:
                mismatched += 1
            flag = "ok " if ok else "DIFF"
            print(f"    [{flag}] {our_col:24s} ours={ours:8.0f}  espn={theirs:8.0f}")

        if group == "K":
            fg = totals.get("fieldGoalsMade-fieldGoalAttempts", "")
            xp = totals.get("extraPointsMade-extraPointAttempts", "")
            for label, pair, cols in [("field goals", fg, ("fg_made", "fg_att")),
                                      ("extra points", xp, ("pat_made", "pat_att"))]:
                if "-" not in str(pair):
                    continue
                made, att = (_num(x) for x in str(pair).split("-", 1))
                ours_made = float(sub[cols[0]].sum())
                ours_att = float(sub[cols[1]].sum())
                checked += 2
                ok = abs(ours_made - made) < 0.5 and abs(ours_att - att) < 0.5
                if not ok:
                    mismatched += 2
                print(f"    [{'ok ' if ok else 'DIFF'}] {label:24s} "
                      f"ours={ours_made:.0f}-{ours_att:.0f}  espn={made:.0f}-{att:.0f}")
        print()

    print(f"{checked} quantities checked, {mismatched} mismatched.")
    sys.exit(1 if mismatched else 0)


if __name__ == "__main__":
    main()
