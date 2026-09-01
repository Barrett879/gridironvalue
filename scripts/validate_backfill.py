"""Validate the training backfill. Prints PASS/FAIL per check and exits nonzero
if any hard check fails.

The checks are chosen to catch the failure modes that are silent. A backfill
that is wrong by two percent in a denominator looks completely normal in a
histogram, trains without complaint, and quietly degrades every projection.

Hard checks (must pass):
  1. No duplicate (game_id, gsis_id) rows.
  2. Player stats SUM to the team totals in the team-week table, per team-game.
  3. Every player-game joins to a team-week row and a schedule row.
  4. One kicker per team-game, roughly: kicker rows are within 2% of team-games.
  5. Team-games per season match the schedule exactly.
  6. No negative counts anywhere.
  7. Our computed target share matches the feed's own within tolerance.

Soft checks (reported, not fatal):
  - Exposure coverage by position.
  - Routes-to-snaps ratio, which should sit near the league pass rate.
  - League rate drift across seasons.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import fetch  # noqa: E402
from gridlib.cache import dc_path, read_parquet_or_none  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def soft(name: str, detail: str) -> None:
    print(f"  [info] {name}  {detail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2016-2025")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seasons.split("-")) if "-" in args.seasons \
        else (int(args.seasons), int(args.seasons))
    tag = f"{lo}_{hi}_{args.version}"

    pw = read_parquet_or_none(dc_path(f"player_week_{tag}.parquet"))
    tw = read_parquet_or_none(dc_path(f"team_week_{tag}.parquet"))
    if pw is None or tw is None:
        print("Backfill artifacts missing. Run build_team_week.py then "
              "build_player_week.py first.")
        sys.exit(2)

    print(f"\nplayer_week: {len(pw):,} rows x {len(pw.columns)} cols")
    print(f"team_week:   {len(tw):,} rows x {len(tw.columns)} cols\n")

    # ── 1. Uniqueness ────────────────────────────────────────────────────────
    print("Structure")
    dupes = int(pw.duplicated(["game_id", "gsis_id"]).sum())
    check("no duplicate player-games", dupes == 0, f"{dupes} dupes")
    tdupes = int(tw.duplicated(["game_id", "team"]).sum())
    check("no duplicate team-games", tdupes == 0, f"{tdupes} dupes")

    # ── 2. Team-games match the schedule ─────────────────────────────────────
    sched = fetch.load_schedules()
    reg = sched[(sched["game_type"] == "REG") & sched["season"].between(lo, hi)]
    played = reg[reg["home_score"].notna()]
    expected = len(played) * 2
    check("team-games match the played schedule", len(tw) == expected,
          f"{len(tw)} vs {expected} expected ({len(played)} games x 2)")

    # ── 3. Joins ─────────────────────────────────────────────────────────────
    print("\nJoins")
    orphan_tw = int(pw["off_plays"].isna().sum())
    check("every player-game joins a team-week row", orphan_tw == 0,
          f"{orphan_tw} orphans")
    orphan_g = int(pw["gameday"].isna().sum())
    check("every player-game joins a schedule row", orphan_g == 0,
          f"{orphan_g} orphans")

    # ── 4. THE reconciliation, as TWO separate invariants ────────────────────
    # Comparing our skill-position player sums straight to the pbp team totals
    # conflates two different things and fails for a reason that is not a bug:
    # punters throw on fake punts and defenders throw on trick plays, so the
    # team total legitimately exceeds the sum over skill positions. Across
    # 2016-2025 that accounts for 92 attempts in 94 team-games, 84 of them by
    # punters. Splitting the check tests each link on its own.
    print("\nInvariant A: our player-week sums == the box score, skill positions only")
    print("            (tests OUR pipeline; must be exact)")
    ours = pw.groupby(["game_id", "team"], as_index=False).agg(
        p_att=("attempts", "sum"), p_car=("carries", "sum"),
        p_sack=("sacks_suffered", "sum"), p_tgt=("targets", "sum"),
        p_rec=("receptions", "sum"), p_pass_td=("passing_tds", "sum"),
        p_rush_td=("rushing_tds", "sum"), p_fg=("fg_att", "sum"),
    )
    src = []
    for season in range(lo, hi + 1):
        st = fetch.load_player_week(season)
        if st is None:
            continue
        st = st[(st["season_type"] == "REG")
                & st["position"].isin(["QB", "RB", "FB", "WR", "TE", "K"])]
        src.append(st.groupby(["game_id", "team"], as_index=False).agg(
            b_att=("attempts", "sum"), b_car=("carries", "sum"),
            b_sack=("sacks_suffered", "sum"), b_tgt=("targets", "sum"),
            b_rec=("receptions", "sum"), b_pass_td=("passing_tds", "sum"),
            b_rush_td=("rushing_tds", "sum"), b_fg=("fg_att", "sum"),
        ))
    skill_box = pd.concat(src, ignore_index=True)
    a = ours.merge(skill_box, on=["game_id", "team"], how="inner")
    check("player-week covers every team-game in the box score",
          len(a) == len(ours) == len(skill_box),
          f"ours={len(ours)} box={len(skill_box)} joined={len(a)}")
    for oc, bc, label in [
        ("p_att", "b_att", "pass attempts"), ("p_car", "b_car", "rush attempts"),
        ("p_sack", "b_sack", "sacks taken"), ("p_tgt", "b_tgt", "targets"),
        ("p_rec", "b_rec", "receptions"),
        ("p_pass_td", "b_pass_td", "passing TDs"),
        ("p_rush_td", "b_rush_td", "rushing TDs"),
        ("p_fg", "b_fg", "field goal attempts"),
    ]:
        d = (a[oc] - a[bc]).abs()
        check(f"{label} preserved exactly", int((d == 0).sum()) == len(a),
              f"{int((d == 0).sum()):,}/{len(a):,}  worst={d.max():.0f}")

    print("\nInvariant B: the full box score == our pbp team totals")
    print("            (tests our PBP DEFINITIONS; tiny residual is official scoring)")
    allsrc = []
    for season in range(lo, hi + 1):
        st = fetch.load_player_week(season)
        if st is None:
            continue
        st = st[st["season_type"] == "REG"]
        allsrc.append(st.groupby(["game_id", "team"], as_index=False).agg(
            all_att=("attempts", "sum"), all_car=("carries", "sum"),
            all_sack=("sacks_suffered", "sum"),
            all_pass_td=("passing_tds", "sum"), all_rush_td=("rushing_tds", "sum"),
        ))
    full_box = pd.concat(allsrc, ignore_index=True)
    b = full_box.merge(
        tw[["game_id", "team", "off_pass_att", "off_rush_att", "off_sacks",
            "off_pass_td", "off_rush_td"]], on=["game_id", "team"], how="inner")
    for bc, tc, label, min_rate in [
        ("all_att", "off_pass_att", "pass attempts", 0.999),
        ("all_car", "off_rush_att", "rush attempts", 0.999),
        ("all_sack", "off_sacks", "sacks taken", 1.0),
        ("all_pass_td", "off_pass_td", "passing TDs", 1.0),
        ("all_rush_td", "off_rush_td", "rushing TDs", 1.0),
    ]:
        d = (b[bc] - b[tc]).abs()
        rate = float((d == 0).mean())
        check(f"{label} match pbp", rate >= min_rate,
              f"{int((d == 0).sum()):,}/{len(b):,} = {100*rate:.2f}%  worst={d.max():.0f}")

    # The gap between A and B is trick plays, and it is worth naming.
    gap = a.merge(b[["game_id", "team", "all_att"]], on=["game_id", "team"], how="inner")
    trick = (gap["all_att"] - gap["p_att"])
    soft("non-skill pass attempts (fake punts, trick plays)",
         f"{int(trick.sum())} attempts across {int((trick != 0).sum())} team-games")

    m = ours.merge(tw[["game_id", "team", "off_pass_att"]], on=["game_id", "team"])
    over = int((m["p_tgt"] > m["off_pass_att"]).sum())
    check("targets never exceed pass attempts", over == 0, f"{over} violations")

    # ── 5. Kicker coverage ───────────────────────────────────────────────────
    print("\nCoverage")
    kick = int((pw["position"] == "K").sum())
    ratio = kick / len(tw)
    check("about one kicker per team-game", 0.95 <= ratio <= 1.02,
          f"{kick} kicker-games / {len(tw)} team-games = {ratio:.3f}")

    # ── 6. Sanity of values ──────────────────────────────────────────────────
    count_cols = ["attempts", "carries", "targets", "receptions", "completions",
                  "fg_att", "fg_made", "off_snaps", "routes_run_proxy"]
    neg = {c: int((pw[c] < 0).sum()) for c in count_cols if c in pw.columns}
    check("no negative counts", sum(neg.values()) == 0,
          str({k: v for k, v in neg.items() if v}) or "clean")

    made_gt_att = int((pw["fg_made"].fillna(0) > pw["fg_att"].fillna(0)).sum())
    check("field goals made never exceed attempts", made_gt_att == 0,
          f"{made_gt_att} violations")
    rec_gt_tgt = int((pw["receptions"].fillna(0) > pw["targets"].fillna(0)).sum())
    check("receptions never exceed targets", rec_gt_tgt == 0,
          f"{rec_gt_tgt} violations")
    comp_gt_att = int((pw["completions"].fillna(0) > pw["attempts"].fillna(0)).sum())
    check("completions never exceed attempts", comp_gt_att == 0,
          f"{comp_gt_att} violations")

    # ── 7. Our share vs the feed's own ───────────────────────────────────────
    w = pw[pw["position"].isin(["WR", "TE", "RB"])
           & pw["target_share"].notna() & (pw["off_pass_att"] > 0)].copy()
    w["ours"] = w["targets"] / w["off_pass_att"]
    err = (w["ours"] - w["target_share"]).abs()
    check("computed target share matches the feed", err.mean() < 0.01,
          f"n={len(w):,} mean|diff|={err.mean():.4f} corr={w['ours'].corr(w['target_share']):.4f}")

    # ── Soft reporting ───────────────────────────────────────────────────────
    print("\nExposure coverage by position (soft)")
    cov = pw.groupby("position").agg(
        n=("gsis_id", "size"),
        snaps_pct=("off_snaps", lambda x: 100 * x.notna().mean()),
        routes_pct=("routes_run_proxy", lambda x: 100 * x.notna().mean()),
    )
    for pos, r in cov.iterrows():
        soft(f"{pos:3s}", f"n={int(r.n):6,d}  snaps {r.snaps_pct:5.1f}%  "
                          f"routes {r.routes_pct:5.1f}%")

    wr = pw[(pw["position"] == "WR") & pw["routes_run_proxy"].notna()
            & pw["off_snaps"].notna() & (pw["off_snaps"] > 0)]
    ratio = (wr["routes_run_proxy"] / wr["off_snaps"]).mean()
    soft("WR routes/snaps", f"{ratio:.3f} (should sit near the league pass rate)")

    print("\nLeague drift (soft): the environment is non-stationary")
    per = pw.groupby("season").agg(
        wr_tgt_per_g=("targets", lambda x: x[pw.loc[x.index, "position"] == "WR"].mean()),
    )
    tw_per = tw.groupby("season").agg(
        plays=("off_plays", "mean"), pass_rate=("off_pass_rate", "mean"),
        proe=("off_proe", "mean"))
    for s, r in tw_per.iterrows():
        soft(f"{s}", f"plays/g {r.plays:5.2f}  pass rate {r.pass_rate:.4f}  "
                     f"PROE {r.proe:+.3f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("All hard checks passed.")


if __name__ == "__main__":
    main()
