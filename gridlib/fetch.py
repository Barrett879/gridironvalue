"""nflverse data access for GridironValue.

WHY THIS BYPASSES nflreadpy
---------------------------
`nflreadpy` is the correct package (nfl_data_py is deprecated and archived), and
it is a dependency here. But its loaders **hardcode a season range and the range
is stale**. Verified 2026-08-31 with nflreadpy 0.1.5:

    load_pbp([2026])            -> ValueError: Season must be between 1999 and 2025
    load_injuries([2026])       -> ValueError: Season must be between 2009 and 2025
    load_snap_counts([2026])    -> ValueError: Season must be between 2012 and 2025
    load_rosters_weekly([2026]) -> ValueError: Season must be between 2002 and 2025

That bound is baked into the installed version. When nflverse publishes the 2026
files in September, a pinned nflreadpy will *still refuse to load them* until a
new release ships and we bump the pin. For a live weekly pipeline that is a
guaranteed in-season outage on a schedule nobody controls.

So the loaders below read the release parquets over HTTPS directly. nflreadpy is
kept for the handful of calls whose bounds are not a problem (`load_teams`,
`load_schedules`, `load_depth_charts`, which already accepts 2026) and as a
reference for URL shapes. Reading parquet directly is also what the spec
recommends for a cached pipeline.

STALE BEATS EMPTY
-----------------
Every fetcher reads the possibly-stale disk cache first, retries the network
with bounded exponential backoff, and on total failure serves the stale copy
with a logged warning. A page never blocks on a hung endpoint and never returns
empty when stale exists.
"""
from __future__ import annotations

import datetime as dt
import io
import time
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .cache import (
    atomic_to_parquet,
    dc_fresh,
    dc_path,
    logger,
    read_parquet_or_none,
)

# ── Constants ────────────────────────────────────────────────────────────────
_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
_UA = "GridironValue/0.1 (personal analytics project; nflverse data, CC-BY-SA)"
_TIMEOUT = 45
_RETRIES = 3

# games.parquet `gametime` is US Eastern, always, regardless of stadium.
EASTERN = ZoneInfo("America/New_York")

# nflverse coverage floors, from the release pages (verified 2026-08-31). These
# are OUR bounds, not nflreadpy's, and they have no stale upper bound.
COVERAGE = {
    "pbp": 1999,
    "stats_player": 1999,
    "injuries": 2009,
    "snap_counts": 2012,
    "weekly_rosters": 2002,
    "depth_charts": 2001,
    "participation": 2016,
    "pfr_advstats": 2018,
}


# ── Generic release-parquet fetcher ──────────────────────────────────────────
def _release_url(tag: str, filename: str) -> str:
    return f"{_RELEASE}/{tag}/{filename}"


def _http_parquet(url: str) -> pd.DataFrame | None:
    """GET a parquet with bounded exponential backoff. None on total failure.

    A 404 is NOT retried: for this data source it means "that season has not
    been published yet", which is an expected preseason state, not an outage.
    """
    delay = 1.0
    for attempt in range(1, _RETRIES + 1):
        try:
            r = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": _UA})
            if r.status_code == 404:
                logger.info("not published yet (404): %s", url)
                return None
            r.raise_for_status()
            return pd.read_parquet(io.BytesIO(r.content))
        except Exception as e:  # noqa: BLE001 — any failure falls through to stale
            logger.warning(
                "fetch attempt %d/%d failed for %s: %s", attempt, _RETRIES, url, e
            )
            if attempt < _RETRIES:
                time.sleep(delay)
                delay *= 2
    return None


def load_release(
    tag: str,
    filename: str,
    cache_name: str,
    ttl: int | None = None,
    kickoff: dt.datetime | None = None,
) -> pd.DataFrame | None:
    """Cached read of one nflverse release parquet, stale-beats-empty.

    Returns None only when the file has never been cached AND the network could
    not supply it (including a legitimate 404 for an unpublished season).
    """
    path = dc_path(cache_name)
    if dc_fresh(path, kickoff=kickoff, ttl=ttl):
        cached = read_parquet_or_none(path)
        if cached is not None:
            return cached

    fresh = _http_parquet(_release_url(tag, filename))
    if fresh is not None:
        atomic_to_parquet(fresh, path)
        return fresh

    stale = read_parquet_or_none(path)
    if stale is not None:
        logger.warning("serving STALE %s (network fetch failed)", cache_name)
        return stale
    return None


def _seasonal(tag: str, stem: str, season: int, ttl: int, floor_key: str | None = None):
    """Loader for the `{stem}_{season}.parquet` release layout."""
    if floor_key and season < COVERAGE[floor_key]:
        logger.info("%s not covered before %d (asked %d)", stem, COVERAGE[floor_key], season)
        return None
    return load_release(tag, f"{stem}_{season}.parquet", f"{stem}_{season}.parquet", ttl=ttl)


# ── The one file that is schedule, market, weather and rest ──────────────────
# Refreshed roughly every 5 minutes in season, and it carries spread_line and
# total_line for games that have NOT been played yet. 15-minute TTL: lines do
# move, but not fast enough to justify hammering the endpoint on every rerun.
SCHEDULE_TTL = 900


def load_schedules(ttl: int = SCHEDULE_TTL) -> pd.DataFrame:
    """All seasons 1999-present, one row per game. Never returns None.

    On a cold cache with no network this raises, because there is no meaningful
    app without a schedule and a silent empty frame would render as "no games
    this week", which is a lie.
    """
    df = load_release("schedules", "games.parquet", "games.parquet", ttl=ttl)
    if df is None:
        raise RuntimeError(
            "games.parquet unavailable and never cached. GridironValue cannot "
            "render a slate without it."
        )
    return df


def load_player_week(season: int, ttl: int = 3600):
    """Weekly player box scores (150 columns, includes full kicking detail)."""
    return _seasonal("stats_player", "stats_player_week", season, ttl, "stats_player")


def load_pbp(season: int, ttl: int = 6 * 3600):
    """Play-by-play. Large (48k rows, 372 cols per season); cache hard."""
    return _seasonal("pbp", "play_by_play", season, ttl, "pbp")


def load_injuries(season: int, ttl: int = 1800):
    """Weekly injury reports (2009+).

    Cadence is slate-specific, NOT Wed/Thu/Fri: that is Sunday games only. It is
    Mon/Tue/Wed for Thursday games, Tue/Wed/Thu for Saturday, Thu/Fri/Sat for
    Monday. The Friday report is also not final; Saturday and Sunday-morning
    downgrades are routine.
    """
    return _seasonal("injuries", "injuries", season, ttl, "injuries")


def load_snap_counts(season: int, ttl: int = 3600):
    """PFR snap counts (2012+), updated several times daily in season.

    This is the LIVE exposure proxy. Routes run is the better exposure for
    WR/TE, but participation data only publishes after the postseason, so
    routes are a training-time feature and snaps are the in-season stand-in.
    """
    return _seasonal("snap_counts", "snap_counts", season, ttl, "snap_counts")


def load_rosters_weekly(season: int, ttl: int = 3600):
    return _seasonal("weekly_rosters", "roster_weekly", season, ttl, "weekly_rosters")


def load_participation(season: int, ttl: int = 24 * 3600):
    """Per-play personnel (2016+): who was on the field for each play.

    The release path is `pbp_participation/pbp_participation_{season}`, NOT
    `participation/participation_{season}` as the tag list would suggest. A
    wrong path here fails SILENTLY (404 -> None -> "no data"), which looks
    identical to "this season is not published yet", so it is worth pinning.

    Publishes only AFTER the postseason and does not update in season, so this
    is TRAINING-ONLY. Routes run is derived from it (the correct WR/TE
    exposure); snap counts are the live in-season proxy. Source changed to FTN
    for 2023+ after the NGS-based feed died mid-season.
    """
    return _seasonal(
        "pbp_participation", "pbp_participation", season, ttl, "participation"
    )


def load_players(ttl: int = 24 * 3600):
    """The player master table: gsis_id, name, position, birthdate, ids."""
    return load_release("players", "players.parquet", "players.parquet", ttl=ttl)


# ── Depth charts: the schema trap ────────────────────────────────────────────
def load_depth_charts(season: int, ttl: int = 1800):
    """Depth charts, with the 2025+ schema handled.

    From 2025 onward these files have **no `week` column at all**. They carry a
    `dt` ISO8601 timestamp instead (verified: 2025 and 2026 both ship
    `dt`/`team`/`player_name`/`espn_id`/`gsis_id`/`pos_*`, no `week`). Code that
    assumes `week` breaks silently, so this loader always ADDS a `week` column,
    derived from the timestamp against the schedule when it is missing.
    """
    df = _seasonal("depth_charts", "depth_charts", season, ttl, "depth_charts")
    if df is None:
        return None
    if "week" in df.columns:
        return df
    if "dt" not in df.columns:
        logger.warning("depth_charts %d has neither `week` nor `dt`; leaving as is", season)
        return df
    return _attach_week_from_timestamp(df, season)


def _attach_week_from_timestamp(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Map each depth-chart snapshot timestamp onto the NFL week it describes.

    A snapshot taken at time T describes the roster going into the next game, so
    it belongs to the week whose first kickoff is the earliest one at or after
    T. Snapshots taken after the season's last kickoff are pinned to the final
    week rather than dropped.
    """
    out = df.copy()
    ts = pd.to_datetime(out["dt"], format="ISO8601", utc=True, errors="coerce")

    sched = load_schedules()
    s = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
    if s.empty:
        out["week"] = pd.NA
        return out
    kick = kickoff_series(s).dt.tz_convert("UTC")
    week_start = (
        pd.DataFrame({"week": s["week"].to_numpy(), "kick": kick.to_numpy()})
        .groupby("week", as_index=False)["kick"].min()
        .sort_values("kick")
        .reset_index(drop=True)
    )

    # searchsorted on week-start boundaries: index i means "before week i+1".
    bounds = week_start["kick"].to_numpy()
    idx = bounds.searchsorted(ts.to_numpy(), side="left")
    idx = idx.clip(0, len(week_start) - 1)
    out["week"] = week_start["week"].to_numpy()[idx]
    out.loc[ts.isna(), "week"] = pd.NA
    return out


# ── Kickoff time, and the derived quantities that depend on it ───────────────
def kickoff_series(games: pd.DataFrame) -> pd.Series:
    """Timezone-aware kickoff datetimes for a games frame.

    `gameday` is a date and `gametime` is HH:MM **US Eastern**, for every game,
    including the ones played in London, Munich and Melbourne. Rows with a
    missing gametime fall back to 13:00 ET so they sort into the early window
    rather than becoming NaT and silently vanishing from a slate.
    """
    day = games["gameday"].astype(str)
    tod = games["gametime"].astype(str).where(
        games["gametime"].notna() & (games["gametime"].astype(str) != ""), "13:00"
    )
    naive = pd.to_datetime(day + " " + tod, format="%Y-%m-%d %H:%M", errors="coerce")
    return naive.dt.tz_localize(EASTERN, ambiguous=True, nonexistent="shift_forward")


def implied_team_totals(games: pd.DataFrame) -> pd.DataFrame:
    """Add `home_implied_total` and `away_implied_total` from the market.

    `spread_line` is from the HOME team's perspective: POSITIVE means the home
    team is favored by that many points. So

        home = total/2 + spread/2
        away = total/2 - spread/2

    These belong at the team-volume and TD-rate layer. The market sets the level
    well and says almost nothing about variance: game totals carry RMSE ~13.2
    against a mean of 44.4 while the market's own SD is only 4.8. Do not use
    them as a direct multiplier on an individual stat line.
    """
    out = games.copy()
    half_total = out["total_line"] / 2.0
    half_spread = out["spread_line"] / 2.0
    out["home_implied_total"] = half_total + half_spread
    out["away_implied_total"] = half_total - half_spread
    return out


# ── Which season and week are we actually in? ────────────────────────────────
# nflreadpy.get_current_season() returned 2025 on 2026-08-31, ten days before
# the 2026 opener, so it is not usable for "which board do we show". The
# schedule file is authoritative; ask it.
_PRESEASON_LEAD_DAYS = 45


def current_season(games: pd.DataFrame | None = None, now: dt.datetime | None = None) -> int:
    """The season whose board a visitor should see.

    Rule: the latest season whose first regular-season kickoff is within
    `_PRESEASON_LEAD_DAYS` of now or already past. Six weeks of lead means the
    upcoming season takes over in late July, once schedules and rosters are
    real, rather than in February.
    """
    g = games if games is not None else load_schedules()
    _now = now or dt.datetime.now(EASTERN)
    if _now.tzinfo is None:
        _now = _now.replace(tzinfo=EASTERN)

    reg = g[g["game_type"] == "REG"]
    firsts = (
        pd.DataFrame({"season": reg["season"].to_numpy(), "kick": kickoff_series(reg).to_numpy()})
        .groupby("season", as_index=False)["kick"].min()
    )
    cutoff = _now + dt.timedelta(days=_PRESEASON_LEAD_DAYS)
    eligible = firsts[firsts["kick"] <= cutoff]
    if eligible.empty:
        return int(firsts["season"].min())
    return int(eligible["season"].max())


def current_week(
    season: int | None = None,
    games: pd.DataFrame | None = None,
    now: dt.datetime | None = None,
) -> int:
    """The week a visitor should land on: the earliest week with a game still to
    kick off. Once every game in a season has kicked off, the last week.

    Weeks are NOT calendar weeks. Every team has a bye, so a player plays 17
    games across 18 weeks, and any week-indexed logic must respect that.
    """
    g = games if games is not None else load_schedules()
    _now = now or dt.datetime.now(EASTERN)
    if _now.tzinfo is None:
        _now = _now.replace(tzinfo=EASTERN)
    szn = season if season is not None else current_season(g, _now)

    s = g[(g["season"] == szn) & (g["game_type"] == "REG")].copy()
    if s.empty:
        return 1
    s["kick"] = kickoff_series(s)
    upcoming = s[s["kick"] >= _now]
    if upcoming.empty:
        return int(s["week"].max())
    return int(upcoming["week"].min())


# ── The hand-curated corrections layer ───────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def stadium_corrections() -> pd.DataFrame:
    """data/stadium_corrections.csv, keyed on the stable `stadium_id`.

    The feed's `stadium` NAME drifts between seasons (the 2026 file calls NRG
    Stadium "Reliant Stadium", retired in 2014, while the 2025 file has it
    right), so display names are curated. Missing file is not fatal: the app
    falls back to the feed.
    """
    path = _REPO / "data" / "stadium_corrections.csv"
    try:
        return pd.read_csv(path, dtype=str)
    except Exception as e:  # noqa: BLE001
        logger.warning("stadium corrections unavailable (%s); using feed values", e)
        return pd.DataFrame(columns=["stadium_id", "display_name", "roof_type", "note"])


def apply_stadium_corrections(games: pd.DataFrame) -> pd.DataFrame:
    """Overlay curated venue name and structural roof type onto a games frame.

    Adds `roof_type`, the STRUCTURAL fact (outdoors / dome / retractable), which
    the feed's `roof` column conflates with per-game state: for a retractable
    stadium `roof` is EMPTY until the game is played and only then becomes
    `closed` or `open`. Structure is knowable at prediction time; state is not.
    """
    out = games.copy()
    fix = stadium_corrections()
    if fix.empty or "stadium_id" not in out.columns:
        out["roof_type"] = out.get("roof", pd.Series(index=out.index, dtype=object))
        return out

    name_map = dict(zip(fix["stadium_id"], fix["display_name"]))
    roof_map = dict(zip(fix["stadium_id"], fix["roof_type"]))
    sid = out["stadium_id"].astype(str)
    out["stadium"] = sid.map(name_map).fillna(out["stadium"])
    # Fall back to the feed's roof where we have no curated structure, mapping
    # the per-game states onto the structural label they imply.
    feed = out.get("roof", pd.Series(index=out.index, dtype=object)).astype(str)
    feed_struct = feed.replace(
        {"closed": "retractable", "open": "retractable", "": pd.NA, "nan": pd.NA}
    )
    out["roof_type"] = sid.map(roof_map).fillna(feed_struct)
    return out


def week_games(season: int, week: int, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """One week's games, kickoff-ordered, with market and timing columns added.

    Adds: `kickoff` (tz-aware), `lock_at` (kickoff minus 90 minutes, when
    inactives publish), `home_implied_total`, `away_implied_total`, `roof_type`
    (curated structural roof), and `window` (the human slate label: Thursday
    night, Sunday early, and so on).
    """
    g = games if games is not None else load_schedules()
    s = g[(g["season"] == season) & (g["week"] == week)].copy()
    if s.empty:
        return s
    s = implied_team_totals(s)
    s = apply_stadium_corrections(s)
    s["kickoff"] = kickoff_series(s)
    s["lock_at"] = s["kickoff"] - pd.Timedelta(minutes=90)
    s["window"] = [_window_label(k) for k in s["kickoff"]]
    return s.sort_values(["kickoff", "home_team"]).reset_index(drop=True)


def _window_label(kick: pd.Timestamp) -> str:
    """The broadcast window a kickoff belongs to, in plain language.

    The NFL board is one week with staggered locks, and users think in windows
    ("the early games"), not in timestamps.
    """
    if pd.isna(kick):
        return "Time to be confirmed"
    day = kick.strftime("%A")
    hour = kick.hour + kick.minute / 60.0
    if day == "Thursday":
        return "Thursday night"
    if day == "Friday":
        return "Friday"
    if day == "Saturday":
        return "Saturday"
    if day == "Monday":
        return "Monday night"
    if day == "Sunday":
        if hour < 12:
            return "Sunday morning"
        if hour < 15.5:
            return "Sunday early"
        if hour < 19:
            return "Sunday late"
        return "Sunday night"
    return day


# ── Resolving a shareable link back to a game ────────────────────────────────
def find_game(
    season: int, week: int, away: str, home: str,
    games: pd.DataFrame | None = None,
) -> pd.Series | None:
    """Resolve the readable deep-link params back to one game row.

    Links are minted as ?season=&week=&away=&home= rather than with an opaque
    id, because Streamlit's iframe means the address bar never follows in-app
    navigation and a user has nothing correct to copy unless we hand it to them.
    That readability is only worth anything if the resolution is tolerant, so
    team abbreviations are folded through the alias table (a link written with
    LAR, OAK or JAC still resolves).

    Returns None when nothing matches, so the caller can say why rather than
    rendering an empty page.
    """
    from .teams import canonical

    g = games if games is not None else load_schedules()
    a, h = canonical(away), canonical(home)
    if not a or not h:
        return None
    wk = week_games(season, week, g)
    if wk.empty:
        return None
    hit = wk[(wk["away_team"].map(canonical) == a) & (wk["home_team"].map(canonical) == h)]
    if hit.empty:
        # Tolerate a link with the teams the wrong way round rather than
        # dead-ending: the matchup is unambiguous either way.
        hit = wk[(wk["away_team"].map(canonical) == h) & (wk["home_team"].map(canonical) == a)]
    return hit.iloc[0] if len(hit) else None


# Skill positions in scope for v1. Kickers are included (Barrett asked for all
# five), fullbacks come along because the feed groups them with the backs.
SKILL_POSITIONS = ["QB", "RB", "FB", "WR", "TE", "PK"]

# Display order and label for each, so a depth chart reads like a depth chart
# rather than alphabetically.
POSITION_ORDER = [
    ("QB", "Quarterback"),
    ("RB", "Running back"),
    ("FB", "Fullback"),
    ("WR", "Wide receiver"),
    ("TE", "Tight end"),
    ("PK", "Kicker"),
]


def latest_depth_chart(season: int, week: int, team: str) -> pd.DataFrame:
    """The most recent depth-chart snapshot for one team in one week.

    The file holds MANY snapshots per team per week (164 for Seattle in week 1
    of 2026), because it is re-scraped continuously. Taking all of them would
    show each player several times, so this filters to the single latest `dt`.

    Returns an empty frame rather than raising when the season is not published
    or the team has no rows, so the page can explain itself.
    """
    from .teams import canonical

    dc = load_depth_charts(season)
    if dc is None or dc.empty:
        return pd.DataFrame()
    tm = canonical(team)
    sub = dc[(dc["week"] == week) & (dc["team"].map(canonical) == tm)]
    if sub.empty:
        return pd.DataFrame()
    sub = sub[sub["dt"] == sub["dt"].max()]
    sub = sub[sub["pos_abb"].isin(SKILL_POSITIONS)]
    # The feed carries occasional rows with an espn_id but no name or gsis_id.
    # They cannot be joined to stats or displayed, so drop them here rather
    # than letting a blank row reach the table.
    sub = sub[sub["player_name"].notna() & (sub["player_name"].astype(str) != "None")]
    return sub.sort_values(["pos_abb", "pos_rank"]).reset_index(drop=True)


def injury_report(season: int, week: int, teams: list[str] | None = None) -> pd.DataFrame:
    """Injury rows for a week, or an empty frame when the season is unpublished.

    Empty is a normal preseason state (2026 injuries did not exist as of
    2026-08-31), NOT an error. Callers must say which it is rather than
    rendering silence.
    """
    from .teams import canonical

    inj = load_injuries(season)
    if inj is None or inj.empty:
        return pd.DataFrame()
    sub = inj[inj["week"] == week]
    if teams:
        want = {canonical(t) for t in teams}
        sub = sub[sub["team"].map(canonical).isin(want)]
    return sub.reset_index(drop=True)


# ── Depth charts, normalized across two incompatible eras ────────────────────
# The file changed shape entirely in 2025, and the two schemas share almost
# nothing. Anything that wants a depth rank has to go through here.
#
#   2001-2024:  season, club_code, week, game_type, depth_team, gsis_id,
#               position, depth_position, full_name, formation
#               -> rank is `depth_team` (1 = starter), team is `club_code`
#   2025+:      dt, team, player_name, espn_id, gsis_id, pos_grp, pos_abb,
#               pos_slot, pos_rank
#               -> rank is `pos_rank`, team is `team`, and there is NO week
#                  column (load_depth_charts derives one from `dt`)
_DEPTH_POS_MAP = {"QB": "QB", "RB": "RB", "FB": "FB", "WR": "WR", "TE": "TE",
                  "PK": "K", "K": "K"}


def depth_chart_normalized(season: int) -> pd.DataFrame:
    """(season, week, team, gsis_id, position, depth_rank) for skill positions.

    One row per (team, week, player): the LATEST snapshot at or before that
    week. Returns an empty frame rather than raising when the season has no
    published chart.

    POINT-IN-TIME NOTE for 2025+: those files are timestamped, not week-numbered,
    and `load_depth_charts` pins any snapshot taken after the final kickoff to
    the last week. That means offseason snapshots pile onto week 18. For
    training, a snapshot taken in March knows how the season ended, so callers
    building features must use the per-week rank as an approximation of what was
    knowable then, not as gospel. The rank itself is a roster ordering, not an
    outcome, so the leak is mild, but it is real and worth naming.
    """
    dc = load_depth_charts(season)
    if dc is None or dc.empty:
        logger.warning("no depth chart for %d", season)
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id",
                                     "position", "depth_rank"])

    cols = set(dc.columns)
    if {"club_code", "depth_team", "position"} <= cols:
        # The 2001-2024 shape.
        out = dc.rename(columns={"club_code": "team", "depth_team": "depth_rank"})
        if "game_type" in out.columns:
            out = out[out["game_type"] == "REG"]
        out["position"] = out["position"].map(_DEPTH_POS_MAP)
        keep_dt = None
    elif {"team", "pos_rank", "pos_abb"} <= cols:
        # The 2025+ shape.
        out = dc.rename(columns={"pos_rank": "depth_rank"})
        out["position"] = out["pos_abb"].map(_DEPTH_POS_MAP)
        keep_dt = "dt"
    else:
        logger.warning("depth chart %d has an unrecognized schema: %s",
                       season, sorted(cols))
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id",
                                     "position", "depth_rank"])

    out = out[out["position"].notna() & out["gsis_id"].notna()]
    if out.empty:
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id",
                                     "position", "depth_rank"])
    out["season"] = season
    out["depth_rank"] = pd.to_numeric(out["depth_rank"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["week"].notna()]

    # Keep the LATEST snapshot per (team, week, player). The 2025+ files carry
    # many snapshots a week; the older ones carry one.
    if keep_dt and keep_dt in out.columns:
        out = out.sort_values(keep_dt)
    out = (out.drop_duplicates(["team", "week", "gsis_id"], keep="last")
              [["season", "week", "team", "gsis_id", "position", "depth_rank"]])
    out["week"] = out["week"].astype(int)
    return out.reset_index(drop=True)
