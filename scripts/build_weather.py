"""Per-game weather at kickoff, from Open-Meteo.

WHY THIS IS NOT ALREADY IN THE BACKFILL
----------------------------------------
`schedules.parquet` carries `temp` and `wind`, and both are POSTGAME fields:
66.7% populated for 2025's completed games and 0% for 2026's unplayed ones. They
are listed in `columns.POSTGAME_TRAP` for that reason. The spec assumed wind was
available and it is not, so the model has never known whether a game was played
in a gale.

Open-Meteo's archive is free, needs no key, and covers the whole backfill, which
matters more than it sounds: it means this block can go through the same
two-stage gate that rejected twelve other ideas, rather than shipping on the
plausible-sounding story that wind must matter.

ONE CALL PER VENUE, NOT PER GAME
--------------------------------
The archive takes a date range, so a decade of hourly weather for one stadium is
a single request returning ~74k hours in under two seconds. 47 venues, 47 calls.

UTC THROUGHOUT
--------------
`gametime` is US Eastern for EVERY game including London, Munich and Melbourne,
so kickoff is converted to UTC and matched against UTC hours. Working in local
time would need per-venue timezones and would get the international games wrong.

THE GAME WINDOW, NOT THE KICKOFF INSTANT
-----------------------------------------
A three-hour game is not its first hour. Wind at kickoff and the worst wind of
the afternoon are different things, and the second is likelier to matter, so
both are recorded and the ablation decides.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import fetch  # noqa: E402
from gridlib.cache import (atomic_to_parquet, dc_path, logger,  # noqa: E402
                           read_parquet_or_none)

OUT = "weather_2016_2026_v1.parquet"
COORDS = Path(__file__).resolve().parent.parent / "data" / "stadium_coords.csv"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation"
GAME_HOURS = 3


def _hourly(url: str, lat: float, lon: float, start: str, end: str, tries: int = 5):
    """One archive request, with backoff on the rate limiter.

    The first version of this asked for each venue's ENTIRE decade, summers
    included: about 160,000 day-units of data to place 2,900 games. Open-Meteo
    weights by volume, so it started returning 429 after roughly fifteen venues
    and 2,000 games were silently missing. Asking per season, for only the span
    that season's games actually cover, is a fraction of the data and small
    enough not to trip the limiter.
    """
    params = {"latitude": lat, "longitude": lon,
              "start_date": start, "end_date": end, "hourly": HOURLY,
              "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
              "precipitation_unit": "mm", "timezone": "UTC"}
    delay = 8.0
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code == 429:
                if attempt == tries - 1:
                    logger.warning("rate limited after %d tries: %s %s-%s",
                                   tries, lat, start, end)
                    return None
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            h = r.json().get("hourly")
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                logger.warning("weather fetch failed for %s,%s: %s", lat, lon, e)
                return None
            time.sleep(delay)
            delay *= 2
            continue
        if not h or not h.get("time"):
            return None
        df = pd.DataFrame(h)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time").sort_index()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2016-2026")
    args = ap.parse_args()
    a, b = (int(x) for x in args.seasons.split("-"))

    coords = pd.read_csv(COORDS)
    sched = fetch.apply_stadium_corrections(fetch.load_schedules())
    sched = sched[(sched["season"] >= a) & (sched["season"] <= b)].copy()
    sched["kick_utc"] = fetch.kickoff_series(sched).dt.tz_convert("UTC")
    sched = sched[sched["kick_utc"].notna()]

    # RESUME. A partial run must not be thrown away, and a re-run must not
    # refetch what already succeeded: that is what got this rate limited.
    done_pairs = set()
    rows = []
    prev = read_parquet_or_none(dc_path(OUT))
    if prev is not None and not prev.empty:
        rows = prev.to_dict("records")
        have = sched[["game_id", "season"]].drop_duplicates()
        p = prev.merge(have, on="game_id", how="left")
        done_pairs = set(zip(p["stadium_id"], p["season"]))
        print(f"resuming: {len(rows)} games already fetched, "
              f"{len(done_pairs)} venue-seasons complete\n")

    coord_by_id = {c.stadium_id: c for c in coords.itertuples()}
    pairs = (sched[["stadium_id", "season"]].dropna().drop_duplicates()
                  .sort_values(["stadium_id", "season"]))
    pairs = [(sid, int(yr)) for sid, yr in pairs.itertuples(index=False)
             if sid in coord_by_id]
    todo = [p for p in pairs if p not in done_pairs]
    print(f"{len(pairs)} venue-seasons, {len(todo)} still to fetch\n")

    cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ok = fail = 0
    for i, (sid, yr) in enumerate(todo, 1):
        c = coord_by_id[sid]
        g = sched[(sched["stadium_id"] == sid) & (sched["season"] == yr)]
        if g.empty:
            continue
        start = (g["kick_utc"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (g["kick_utc"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = min(end, cutoff)
        if start > end:
            continue                      # entirely in the future
        wx = _hourly(ARCHIVE, c.lat, c.lon, start, end)
        if wx is None:
            fail += 1
            print(f"  {sid} {yr}  FAILED")
            continue
        n_ok = 0
        for row in g.itertuples():
            k = row.kick_utc.floor("h")
            if k not in wx.index:
                continue
            win = wx.loc[k:k + pd.Timedelta(hours=GAME_HOURS - 1)]
            if win.empty:
                continue
            rows.append({
                "game_id": row.game_id, "stadium_id": sid,
                "wx_temp": float(win["temperature_2m"].iloc[0]),
                "wx_wind": float(win["wind_speed_10m"].iloc[0]),
                "wx_gust": float(win["wind_gusts_10m"].iloc[0]),
                "wx_wind_max": float(win["wind_speed_10m"].max()),
                "wx_gust_max": float(win["wind_gusts_10m"].max()),
                "wx_precip": float(win["precipitation"].sum()),
            })
            n_ok += 1
        ok += 1
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} venue-seasons   {len(rows)} games   "
                  f"{fail} failed")
            # Checkpoint, so a later rate limit cannot lose the work.
            atomic_to_parquet(pd.DataFrame(rows).drop_duplicates("game_id"),
                              dc_path(OUT))
        time.sleep(1.5)

    if not rows:
        print("nothing fetched")
        sys.exit(2)
    out = pd.DataFrame(rows).drop_duplicates("game_id")
    atomic_to_parquet(out, dc_path(OUT))
    print(f"\nwrote {OUT}  ({len(out)} games)")
    print(f"  wind mph  mean {out.wx_wind.mean():5.1f}  p90 {out.wx_wind.quantile(.9):5.1f}"
          f"  max {out.wx_wind.max():5.1f}")
    print(f"  temp F    mean {out.wx_temp.mean():5.1f}  min {out.wx_temp.min():5.1f}"
          f"  max {out.wx_temp.max():5.1f}")
    print(f"  games with measurable rain: {(out.wx_precip > 0.5).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
