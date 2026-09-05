"""Referee crew tendencies per (season, referee), from pbp penalties.

Item 7 of the feature programme. The claim under test is that some crews call
more than others and that a flag-heavy crew lengthens drives, which would move
passing volume.

THE SOURCE IS THE OFFICIALS FILE, NOT THE SCHEDULE'S `referee` COLUMN
---------------------------------------------------------------------
`schedules.parquet` carries a `referee` column that is 100% populated for every
completed season and 0% populated for the upcoming one. It is a POSTGAME field,
exactly like `temp` and `wind`: perfect in training, absent at inference. A
feature built on it would validate beautifully and be null for every live
projection.

`officials.parquet` is the publishing feed. It reaches ~95% on completed seasons
and, crucially, it populates BEFORE kickoff: on 2026-09-01, with no games played,
it already carried crews for 16 of 272 games, which is week 1. Crews are
announced a few days ahead.

Both sources are used here only to be compared: where both have a referee they
agree on 94.2% of 2,891 games. Training and serving then both read the officials
file, so there is one source and no skew.

THE AVAILABILITY CEILING, STATED PLAINLY
-----------------------------------------
Because crews are announced roughly a week out, this feature exists for the
IMMINENT week and is null for anything further ahead. That is acceptable for a
board that projects the upcoming week and it is not acceptable as a silent NaN,
so `USE_REFEREE` must stay off unless the block earns its place, and if it ever
ships the null-crew case needs an explicit decision rather than a default branch.

Rates are stored raw and centred per season in the feature layer, for the same
reason as the scheme block: officiating points of emphasis change year to year
and a raw penalty count encodes the era as much as the crew.
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

OUT = "referee_2016_2025_v1.parquet"


def season_referee(season: int, ref_map: pd.DataFrame) -> pd.DataFrame | None:
    pbp = fetch.load_pbp(season)
    if pbp is None or pbp.empty:
        return None
    need = [c for c in ("game_id", "old_game_id", "penalty", "penalty_yards",
                        "penalty_type") if c in pbp.columns]
    if "penalty" not in need:
        return None
    p = pbp[need].copy()
    if "old_game_id" not in p.columns:
        return None
    p["old_game_id"] = p["old_game_id"].astype(str)
    p = p.merge(ref_map, on="old_game_id", how="inner")
    if p.empty:
        return None

    p["penalty"] = pd.to_numeric(p["penalty"], errors="coerce").fillna(0)
    p["penalty_yards"] = pd.to_numeric(p.get("penalty_yards"),
                                       errors="coerce").fillna(0)
    ptype = p.get("penalty_type", pd.Series("", index=p.index)).astype(str)
    # Defensive pass interference is the single penalty most able to move a
    # passing line, because it is an untapped-yardage spot foul.
    p["dpi"] = ptype.str.contains("Defensive Pass Interference",
                                  case=False, na=False).astype(float)

    games = p.groupby(["referee", "old_game_id"], as_index=False).agg(
        pen=("penalty", "sum"), yds=("penalty_yards", "sum"),
        dpi=("dpi", "sum"))
    out = games.groupby("referee", as_index=False).agg(
        ref_pen_per_game=("pen", "mean"),
        ref_pen_yards_per_game=("yds", "mean"),
        ref_dpi_per_game=("dpi", "mean"),
        n_games=("old_game_id", "count"))
    out.insert(0, "season", season)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2016-2025")
    args = ap.parse_args()
    a, b = (int(x) for x in args.seasons.split("-"))

    ref_map = fetch.referee_by_game()
    if ref_map.empty:
        print("officials.parquet unavailable; nothing to build.")
        sys.exit(2)

    frames = []
    for season in range(a, b + 1):
        df = season_referee(season, ref_map)
        if df is None or df.empty:
            print(f"  {season}: no pbp/officials overlap, skipped")
            continue
        frames.append(df)
        print(f"  {season}: {len(df):>2} crews   pen/game {df.ref_pen_per_game.mean():5.2f}"
              f"  yds/game {df.ref_pen_yards_per_game.mean():6.2f}"
              f"  DPI/game {df.ref_dpi_per_game.mean():4.2f}")

    if not frames:
        print("Nothing built.")
        sys.exit(2)
    out = pd.concat(frames, ignore_index=True)
    atomic_to_parquet(out, dc_path(OUT))
    print(f"\nwrote {OUT}  ({len(out)} rows, {out.referee.nunique()} crews)")
    spread = out.groupby("season")["ref_pen_per_game"].agg(["min", "max"])
    print("\nper-season crew spread in penalties per game:")
    print(spread.round(2).to_string())


if __name__ == "__main__":
    main()
