"""LOCAL ONLY. Grade the model against historical PrizePicks NFL lines.

READ THIS BEFORE RUNNING IT, AND BEFORE SHOWING ANYONE THE OUTPUT
------------------------------------------------------------------
PrizePicks publishes no line history and blocks automated access, and this
project's hardest rule is that nothing here ever fetches them: `props.py` may
not import a symbol that reaches the network, and a test walks the imports
transitively to enforce it. This script does not break that rule. It reads a
git clone that YOU already have on disk, and it never touches the network.

The clone in question is a third-party mirror of the PrizePicks API. On the
sibling MLB site that is `github.com/enkday/prizepicks-data-mirror`, which has
NO LICENCE and is openly automated collection of an API whose terms prohibit
exactly that. So the data is fine for private curiosity and is NOT safe to
republish, and DiamondValue drew the line in two places that are copied here:

  - everything this writes lands under `cache/local_only/`, which .gitignore
    excludes wholesale, and the script REFUSES to run if that is not true
  - no derived number from it goes on the public site, not even an aggregate

DiamondValue went further and deleted the Accuracy panel that had displayed the
mirror-derived summary. Nothing here is wired into a page at all.

WHERE THE PROJECTIONS COME FROM, AND WHY THEY ARE STRICTER THAN MLB'S
----------------------------------------------------------------------
DiamondValue recovered the ORIGINAL point-in-time prediction file for each date
from its own git history. GridironValue cannot: `cache/predictions_*.parquet`
has never been committed (.gitignore), so there is no history to recover.

Instead the projections come from `cache/accuracy_history_v1.parquet`, where
every row was produced by a model trained on seasons STRICTLY BEFORE the one it
is scoring. That is a harder test than "whatever the model happened to be that
day", because it can never have seen any part of the season the line is from.

SNAPSHOT CHOICE
---------------
For each (player, stat, game) this keeps the LATEST snapshot taken strictly
BEFORE kickoff. That is the closest thing to a closing line the mirror offers
and it is the harder, more honest test: an earlier snapshot is softer, because
the market has had less time to absorb injury news and inactives.

Usage:
    python scripts/import_mirror_lines.py /path/to/prizepicks-data-mirror
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import fetch, predict, props  # noqa: E402
from gridlib.cache import CACHE_DIR, dc_path, logger, read_parquet_or_none  # noqa: E402

# Everything this writes goes here, and .gitignore excludes the whole directory.
LOCAL_DIR = CACHE_DIR / "local_only"
HISTORY = "mirror_props_history.parquet"
SUMMARY = "mirror_props_summary.json"


def _assert_fenced(repo: Path) -> None:
    """Refuse to run unless the output directory is genuinely ignored by git.

    The fence is the only thing separating private curiosity from republishing
    data the source's terms prohibit, so it is checked rather than assumed. A
    .gitignore edit that silently un-ignores this directory must stop the
    script, not be discovered later in a diff.
    """
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    probe = LOCAL_DIR / ".fence_probe"
    probe.write_text("probe")
    try:
        out = subprocess.run(["git", "check-ignore", "-q", str(probe)],
                             cwd=repo, capture_output=True)
        ignored = out.returncode == 0
    finally:
        probe.unlink(missing_ok=True)
    if not ignored:
        raise SystemExit(
            f"REFUSING TO RUN: {LOCAL_DIR} is not git-ignored.\n"
            "Everything this script writes derives from a mirror of the "
            "PrizePicks API whose terms prohibit that collection. It is for "
            "private curiosity and must never be committed. Add "
            "'cache/local_only/' to .gitignore and try again.")


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True).stdout


def _nfl_files(clone: Path) -> list[str]:
    """Snapshot files in the clone's history that look like NFL.

    DISCOVERED, not hardcoded. The MLB importer names two files outright, but
    the NFL ones may be called anything, and a hardcoded name that silently
    matches nothing would report "0 lines" as though the mirror had no football
    rather than as though this script had the wrong filename.
    """
    names = set()
    for path in _git(["log", "--all", "--pretty=format:", "--name-only"],
                     clone).split("\n"):
        p = path.strip()
        if p and "nfl" in p.lower() and p.lower().endswith(".json"):
            names.add(p)
    return sorted(names)


def extract_mirror_lines(clone: Path) -> pd.DataFrame:
    """Latest pre-kickoff snapshot per (player, stat, game), from git history."""
    files = _nfl_files(clone)
    if not files:
        raise SystemExit(
            f"no NFL snapshot files found in {clone}.\n"
            "Files whose history was searched: any *.json with 'nfl' in the "
            "path. If this mirror only carries MLB, there is nothing to import "
            "and that is the answer, not a bug.")
    logger.warning("NFL snapshot files in the mirror: %s", ", ".join(files))

    best: dict[tuple, tuple] = {}
    for f in files:
        shas = _git(["log", "--all", "--format=%H", "--", f], clone).split()
        logger.warning("  %s: %d revisions", f, len(shas))
        for sha in shas:
            blob = _git(["show", f"{sha}:{f}"], clone)
            if not blob.strip():
                continue
            try:
                payload = json.loads(blob)
            except ValueError:
                continue
            scraped = payload.get("scrapedAt") or payload.get("scraped_at")
            if not scraped:
                continue
            try:
                s_at = datetime.fromisoformat(str(scraped).replace("Z", "+00:00"))
            except ValueError:
                continue
            for p in payload.get("props", []) or []:
                iso = p.get("startTimeIso") or p.get("start_time")
                if not iso:
                    continue
                try:
                    start = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
                except ValueError:
                    continue
                # Strictly BEFORE kickoff. A snapshot taken after the game
                # started is not a line anyone could have taken.
                if s_at >= start:
                    continue
                key = (p.get("player"), p.get("stat"), start.date().isoformat())
                prev = best.get(key)
                if prev is None or s_at > prev[0]:
                    best[key] = (s_at, p, start)

    rows = []
    for (player, stat, _d), (s_at, p, start) in best.items():
        try:
            line = float(p.get("line"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(line):
            continue
        rows.append({
            "name": player, "team": p.get("teamCode") or p.get("team"),
            "stat_type": stat, "line": line,
            "kickoff": start.isoformat(),
            # The mirror carries no Less/More buttons, so the offered sides are
            # unknown. Permissive, which props._offered_sides already handles.
            "direction": "", "odds_type": (p.get("oddsType") or "standard"),
            "scraped_at": s_at.isoformat(),
        })
    return pd.DataFrame(rows)


def _projection_for(season: int, week: int, acc: pd.DataFrame,
                    reg: dict) -> pd.DataFrame:
    """A projection frame for one week, rebuilt from the accuracy history.

    Long-form (one row per target) pivoted back to the wide shape
    `props.compare` expects. Every value was produced by a model trained on
    seasons strictly before this one.
    """
    sub = acc[(acc["season"] == season) & (acc["week"] == week)]
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot_table(index="gsis_id", columns="target", values="pred",
                           aggfunc="first").reset_index()
    ident = (sub[["gsis_id", "player_display_name", "position", "team",
                  "opponent_team"]].drop_duplicates("gsis_id"))
    out = wide.merge(ident, on="gsis_id", how="left")
    out["season"], out["week"] = season, week
    out["game_id"] = out["team"].astype(str) + "_" + out["opponent_team"].astype(str)
    # These rows are IN the box score, so the player appeared. Availability is
    # known after the fact and is not the thing being graded here.
    out["p_play"] = 1.0
    for target, meta in (reg or {}).get("targets", {}).items():
        out[f"{target}_source"] = meta.get("present_as", "model")
    return out


def _actuals_for(season: int, week: int):
    """The real box score for one week.

    Same source and shape props_ui uses, rebuilt here so a script does not have
    to import the Streamlit page module to grade anything.
    """
    df = fetch.load_player_week(season)
    if df is None or df.empty:
        return None
    sub = df[(df["week"] == week) & (df["season_type"] == "REG")]
    return sub.rename(columns={"player_id": "gsis_id"}) if not sub.empty else None


def _games_prior(pw: pd.DataFrame) -> pd.DataFrame:
    """Prior games played per (gsis_id, season, week), for filter_informed."""
    s = pw.sort_values(["gsis_id", "season", "week"])
    s = s.assign(f_games_prior=s.groupby("gsis_id").cumcount().astype(float))
    return s[["gsis_id", "season", "week", "f_games_prior"]]


def main(argv):
    repo = Path(__file__).resolve().parent.parent
    if not argv:
        raise SystemExit(
            "usage: import_mirror_lines.py /path/to/prizepicks-data-mirror\n\n"
            "This reads a clone YOU already have. It does not fetch anything, "
            "and everything it writes is git-ignored and must stay that way.")
    clone = Path(argv[0]).expanduser().resolve()
    if not (clone / ".git").is_dir():
        raise SystemExit(f"not a git clone: {clone}")
    _assert_fenced(repo)

    acc = read_parquet_or_none(dc_path("accuracy_history_v1.parquet"))
    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    if acc is None or acc.empty or pw is None:
        raise SystemExit(
            "accuracy history missing. Run "
            "`python scripts/build_accuracy_tracker.py --seasons 2023-2025` "
            "first; the projections graded here come from it.")
    reg = predict.load_registry()

    logger.warning("walking mirror history at %s", clone)
    lines = extract_mirror_lines(clone)
    if lines.empty:
        raise SystemExit("no usable pre-kickoff NFL lines in that mirror.")
    logger.warning("mirror: %d pre-kickoff lines", len(lines))

    # Map each line's kickoff onto the (season, week) it belongs to, so it can
    # be graded against the projection for that week.
    sched = fetch.load_schedules()
    sched = sched[sched["game_type"] == "REG"]
    kick = fetch.kickoff_series(sched).dt.tz_convert("UTC")
    games = pd.DataFrame({"season": sched["season"].to_numpy(),
                          "week": sched["week"].to_numpy(),
                          "kick": kick.to_numpy()}).dropna()
    lines["_k"] = pd.to_datetime(lines["kickoff"], utc=True, errors="coerce")
    lines = lines.dropna(subset=["_k"])
    merged = pd.merge_asof(
        lines.sort_values("_k"), games.sort_values("kick"),
        left_on="_k", right_on="kick", direction="nearest",
        tolerance=pd.Timedelta("36h"))
    merged = merged.dropna(subset=["season", "week"])
    merged["season"] = merged["season"].astype(int)
    merged["week"] = merged["week"].astype(int)
    logger.warning("routed to %d (season, week) slates",
                   merged.groupby(["season", "week"]).ngroups)

    prior = _games_prior(pw)
    frames, skipped = [], []
    for (season, week), batch in merged.groupby(["season", "week"]):
        proj = _projection_for(int(season), int(week), acc, reg)
        if proj.empty:
            skipped.append((int(season), int(week), "no out-of-sample projection"))
            continue
        proj = proj.merge(prior[(prior.season == season) & (prior.week == week)]
                          [["gsis_id", "f_games_prior"]], on="gsis_id", how="left")
        table, meta = props.compare(batch, proj, reg)
        if table.empty:
            skipped.append((int(season), int(week), "nothing matched"))
            continue
        actual = _actuals_for(int(season), int(week))
        graded = props.grade(table, actual)
        if graded is None or graded.empty:
            skipped.append((int(season), int(week), "nothing gradeable"))
            continue
        graded["season"], graded["week"] = int(season), int(week)
        frames.append(graded)
        logger.warning("  %d wk%02d: %d posted, %d matched, %d graded",
                       season, week, meta["posted"], meta["matched"], len(graded))

    if not frames:
        raise SystemExit("nothing gradeable. Skipped: %s" % skipped[:8])
    out = pd.concat(frames, ignore_index=True)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LOCAL_DIR / HISTORY, index=False)

    decided = out[out["result"].isin(["More", "Less"])] if "result" in out else out
    rec = props.season_record([decided]) if len(decided) else {}
    summary = {
        "graded": int(len(decided)),
        "slates": int(out.groupby(["season", "week"]).ngroups),
        "seasons": sorted(int(s) for s in out["season"].unique()),
        "record": rec,
        "source": "third-party mirror of the PrizePicks API; no licence; "
                  "LOCAL ONLY, never publish",
    }
    (LOCAL_DIR / SUMMARY).write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 66}")
    print(f"LOCAL ONLY. Wrote {LOCAL_DIR / HISTORY} ({len(out):,} rows)")
    print(f"{'=' * 66}")
    print(f"slates graded : {summary['slates']}  seasons {summary['seasons']}")
    if rec.get("decided"):
        print(f"record        : {rec['hits']} of {rec['decided']} "
              f"({rec['hit_rate']}%), 95% CI {rec['ci_low']}-{rec['ci_high']}%")
        print(f"do-nothing bar: {rec.get('baseline')}%   "
              f"break-even needs 54.7-58.5% per leg")
    if skipped:
        print(f"skipped       : {len(skipped)} slates, e.g. {skipped[:3]}")
    print("\nThis derives from a mirror of an API whose terms prohibit that "
          "collection.\nIt stays on this machine. Nothing here goes on the site.")


if __name__ == "__main__":
    main(sys.argv[1:])
