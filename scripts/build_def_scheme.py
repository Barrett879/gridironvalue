"""Defensive scheme tendencies per (season, defence), from participation data.

This is section 3 of Barrett's prop-strategy document, and the data has been
sitting unused in `pbp_participation` the whole time: man versus zone, pressure,
blitz rate and how many defenders a defence puts in the box.

WHY THIS IS A SEPARATE ARTIFACT AND NOT PART OF THE BACKFILL
------------------------------------------------------------
Two reasons, both practical.

Raw play-by-play is ~100MB and is deliberately EXCLUDED from the deploy, so
anything that needs pbp must be reduced to a small committed file offline. This
writes ~320 rows.

And participation has an availability property that nothing else in the pipeline
has, described below, which makes it worth isolating so the constraint is
visible in one place rather than spread through the feature builder.

THE AVAILABILITY CONSTRAINT, WHICH IS THE WHOLE DESIGN
-------------------------------------------------------
`pbp_participation` for season S publishes only AFTER S's postseason. There is
no in-season participation feed at all. So a feature of the form "what this
defence has played SO FAR THIS SEASON" is buildable in training, where every
season is complete, and is simply absent when projecting a live week.

That is the exact shape of the three train/serve skews this project has already
shipped: the column exists because history supplies it, every live row is NaN,
and a tree routes NaN to a learned branch and returns a confident wrong number
instead of failing.

So these are STRICTLY PRIOR-SEASON aggregates. Projecting week 1 of 2026 uses
2025 and earlier, all of which published months ago. The feature layer does the
lagging; this script only records what each defence actually did, per season.

WHY THE DENOMINATORS DIFFER PER RATE
-------------------------------------
Man/zone is only charted on pass plays (51% of participation rows carry an empty
string), so `man_rate` is over CLASSIFIED dropbacks, not over all plays. Box
counts are meaningful on runs. Using one denominator for everything would
silently mix "how often this defence plays man" with "how often this defence
faced a pass", which is an offence's choice, not a defence's.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import fetch  # noqa: E402
from gridlib.cache import atomic_to_parquet, dc_path  # noqa: E402

OUT = "def_scheme_2016_2025_v1.parquet"
_MZ = ("MAN_COVERAGE", "ZONE_COVERAGE")


def season_scheme(season: int) -> pd.DataFrame | None:
    """One row per defence for one season, or None if participation is absent."""
    part = fetch.load_participation(season)
    if part is None or part.empty:
        return None
    pbp = fetch.load_pbp(season)
    if pbp is None or pbp.empty:
        return None

    # Participation carries `possession_team` but no defending team, so the
    # defence comes from pbp. Joining on (game_id, play_id) is exact: it
    # matched 45,919 of 45,919 rows for 2024.
    cols = [c for c in ("game_id", "play_id", "defteam", "pass", "rush")
            if c in pbp.columns]
    m = (part.rename(columns={"nflverse_game_id": "game_id"})
             .merge(pbp[cols], on=["game_id", "play_id"], how="inner"))
    if m.empty or "defteam" not in m.columns:
        return None
    m = m[m["defteam"].notna()]

    is_pass = m.get("pass", 0) == 1
    is_rush = m.get("rush", 0) == 1
    classified = m["defense_man_zone_type"].isin(_MZ)

    rows = []
    for team, g in m.groupby("defteam", sort=True):
        gp, gr = g[is_pass.loc[g.index]], g[is_rush.loc[g.index]]
        gc = g[classified.loc[g.index]]
        box = pd.to_numeric(gr.get("defenders_in_box"), errors="coerce")
        rush_n = pd.to_numeric(gp.get("number_of_pass_rushers"), errors="coerce")
        rows.append({
            "season": season,
            "def_team": team,
            # Over CLASSIFIED dropbacks only. See the module docstring.
            "man_rate": (float((gc["defense_man_zone_type"] == "MAN_COVERAGE").mean())
                         if len(gc) else np.nan),
            "pressure_rate": (float(pd.to_numeric(gp["was_pressure"], errors="coerce").mean())
                              if len(gp) and "was_pressure" in gp else np.nan),
            "blitz_rate": float((rush_n >= 5).mean()) if len(gp) else np.nan,
            "box8_rate": float((box >= 8).mean()) if len(gr) else np.nan,
            "box_avg": float(box.mean()) if len(gr) else np.nan,
            # Sample sizes, so the feature layer can see how much evidence
            # each rate rests on rather than treating all rates as equal.
            "n_classified": int(len(gc)),
            "n_pass": int(len(gp)),
            "n_rush": int(len(gr)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2016-2025")
    args = ap.parse_args()
    a, b = (int(x) for x in args.seasons.split("-"))

    frames = []
    for season in range(a, b + 1):
        df = season_scheme(season)
        if df is None:
            print(f"  {season}: participation absent, skipped")
            continue
        frames.append(df)
        print(f"  {season}: {len(df):>2} defences   man {df.man_rate.mean():.3f}"
              f"  pressure {df.pressure_rate.mean():.3f}"
              f"  blitz {df.blitz_rate.mean():.3f}"
              f"  box8 {df.box8_rate.mean():.3f}")

    if not frames:
        print("No participation data at all.")
        sys.exit(2)
    out = pd.concat(frames, ignore_index=True)
    atomic_to_parquet(out, dc_path(OUT))
    print(f"\nwrote {OUT}  ({len(out)} rows, {out.season.nunique()} seasons)")


if __name__ == "__main__":
    main()
