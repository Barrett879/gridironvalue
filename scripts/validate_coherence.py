"""Do the projections COMPOSE? Team-level coherence checks.

WHY THIS EXISTS
---------------
Per-target MAE hid a serious defect for an entire build. Every model beat its
baselines by 5 to 13% out of sample, and the projections still did not add up:
summing a team's players and comparing to reality on a played week gave

    rush yards   1.49x    carries    1.40x
    pass yards   1.69x    attempts   1.73x    targets  1.41x

MAE cannot see this. It is computed per player, over players who actually
played, so it never asks whether a team's projections sum to a plausible team
total and never sees the players who were projected but never took a snap.

THE CAUSE, MEASURED
-------------------
It is NOT that players fail to compose. Restricting the model's own output to
the players who actually appeared, the team sums are already right:

    carries 0.96   targets 1.02   attempts 1.03   passing yards 1.02

100% of the failure is in the rows that never played. The backfill contains
only players who recorded a stat line, so every model estimates
E[Y | the player appeared], and inference applies that to a 19-man depth chart
where P(appear) runs from about 0.93 for an RB1 to 0.00 for an RB5. The missing
factor is P(appear), not a compositional constraint.

That is why every total below is reported TWICE, over the full projected roster
and restricted to players who appeared. The pair separates an availability
failure from a scaling failure: 1.40 alongside 0.96 names the cause on sight,
where either number alone would be ambiguous.

A NOTE ON CONCENTRATION AND JENSEN
-----------------------------------
Comparing a conditional MEAN to a single week's REALISED concentration is not a
valid test. E[max_i Y_i / sum Y] > max_i E[Y_i] / E[sum Y], so a perfectly
calibrated projection is ALWAYS less concentrated than what happened. Measured
oracle floors: top-1 carry share 0.528 projected against 0.577 realised, a
built-in gap of +0.049; attempts +0.075; targets +0.068.

So concentration is banded against that oracle reference, not against the raw
actual. Gating on the raw actual would condemn a correct model permanently.
Likewise "a real team gives a carry to 4.5 players" is a REALISED count; the
number of players with a nonzero EXPECTED carry is legitimately about 8.

CONCENTRATION MATTERS AS MUCH AS THE TOTAL
-------------------------------------------
A team total can be right while the distribution is wrong. Real offences
concentrate: about 4.5 players take a carry in a game. A model that spreads the
same total across 12 players is wrong in the way that actually reaches a user,
because no individual gets a starter's workload and every line looks low.

So this checks the totals, the SPREAD, and the top-share, not just the sum.

Usage:
    python scripts/validate_coherence.py --season 2025 --week 6
    python scripts/validate_coherence.py --season 2025 --weeks 1-18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import predict  # noqa: E402
from gridlib.cache import dc_path, read_parquet_or_none  # noqa: E402

FAILURES: list[str] = []

# Tolerances. A projection system that is within 10% on team volume is doing
# well; beyond 20% the per-player numbers cannot all be right at once.
TOTAL_TOL = 0.20
# Concentration is compared against an ORACLE reference, not the realised
# actual, so the band can be tight without being unfair. See the Jensen note.
CONCENTRATION_TOL = 0.25
# Empirically measured ratio of a correctly-specified conditional mean's
# concentration to the realised concentration. A model matching these IS right.
ORACLE_TOP1 = {"carries": 0.528 / 0.577, "targets": 0.228 / 0.296}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def soft(name: str, detail: str) -> None:
    print(f"  [info] {name}  {detail}")


def _effective_n(values: np.ndarray) -> float:
    """How many players actually share a quantity, via the inverse Simpson index.

    A cleaner measure than "count above zero", because it is not sensitive to a
    long tail of near-zero projections: ten players at 0.1 carries each barely
    move it, which is exactly the behaviour wanted.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return 0.0
    p = v / v.sum()
    return float(1.0 / np.sum(p ** 2))


def compare_week(season: int, week: int) -> pd.DataFrame | None:
    """One row per team: model totals, actual totals, and concentration."""
    proj = predict.project_week_cached(season, week)
    hist = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    if proj is None or proj.empty or hist is None:
        return None
    act = hist[(hist["season"] == season) & (hist["week"] == week)]
    if act.empty:
        return None

    played = set(act["gsis_id"])
    rows = []
    for team, g in proj.groupby("team"):
        a = act[act["team"] == team]
        if a.empty:
            continue
        # The same projection, restricted to players who actually appeared.
        # This is the diagnostic half: if the FULL sum is inflated and this one
        # is not, the defect is availability and nothing else.
        gp = g[g["gsis_id"].isin(played)]
        rec = {"team": team, "m_players": int(len(g)), "a_players": int(len(a)),
               "m_players_played": int(len(gp))}
        for col in ("rushing_yards", "carries", "passing_yards", "attempts",
                    "targets", "receiving_yards"):
            rec[f"m_{col}"] = g[col].sum()
            rec[f"p_{col}"] = gp[col].sum()       # conditional, played-only
            rec[f"a_{col}"] = a[col].sum()
            # The UNCONDITIONAL twin. This is the only column that should ever
            # be summed across a roster, and the only one the totals gate reads.
            exp = f"{col}_expected"
            rec[f"e_{col}"] = g[exp].sum() if exp in g.columns else np.nan
        rec["sum_p_play"] = float(g["p_play"].sum()) if "p_play" in g.columns else np.nan
        # Concentration is measured on the EXPECTED column, not the conditional
        # one. The conditional number asks "if he plays", which is uniform
        # across the roster and therefore looks far flatter than reality; the
        # expected number already discounts the players who will not appear,
        # which is the distribution that should be compared to a realised one.
        for col, short in (("carries", "carry"), ("targets", "target")):
            exp = f"{col}_expected"
            src = g[exp] if exp in g.columns else g[col]
            rec[f"m_top_{short}_share"] = (src.max() / src.sum()
                                           if src.sum() else np.nan)
            rec[f"a_top_{short}_share"] = (a[col].max() / a[col].sum()
                                           if a[col].sum() else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--week", type=int)
    ap.add_argument("--weeks", help="e.g. 1-6")
    args = ap.parse_args()

    if args.weeks:
        lo, hi = (int(x) for x in args.weeks.split("-"))
        weeks = list(range(lo, hi + 1))
    else:
        weeks = [args.week or 6]

    frames = []
    for w in weeks:
        df = compare_week(args.season, w)
        if df is not None and not df.empty:
            df["week"] = w
            frames.append(df)
    if not frames:
        print("No weeks could be compared (need projections and a played week).")
        sys.exit(2)
    allw = pd.concat(frames, ignore_index=True)
    n_team_games = len(allw)

    print(f"\nCoherence: {args.season} week(s) {weeks}, {n_team_games} team-games\n")

    print("Team totals, reported TWICE. The pair is the diagnosis:")
    print("  full    = every player the site would publish")
    print("  played  = the same projection, restricted to players who appeared")
    print("  A full ratio well above 1 with a played ratio near 1 means the")
    print("  defect is AVAILABILITY, not scaling.\n")
    print(f"  {'target':<18}{'cond':>7}{'exp':>8}{'played':>8}{'actual':>8}"
          f"{'cond/a':>8}{'exp/a':>8}{'plyd/a':>8}")
    for col, label in [("carries", "carries"), ("rushing_yards", "rushing yards"),
                       ("attempts", "pass attempts"), ("passing_yards", "passing yards"),
                       ("targets", "targets"), ("receiving_yards", "receiving yards")]:
        ac = allw[f"a_{col}"].sum()
        r_cond = allw[f"m_{col}"].sum() / ac if ac else np.nan
        r_exp = allw[f"e_{col}"].sum() / ac if ac else np.nan
        r_played = allw[f"p_{col}"].sum() / ac if ac else np.nan
        print(f"  {label:<18}{allw[f'm_{col}'].mean():7.1f}"
              f"{allw[f'e_{col}'].mean():8.1f}{allw[f'p_{col}'].mean():8.1f}"
              f"{allw[f'a_{col}'].mean():8.1f}"
              f"{r_cond:8.2f}{r_exp:8.2f}{r_played:8.2f}")

    print("\nHard gates, on the EXPECTED column (p_play x conditional).")
    print("That is the only column that may be summed across a roster.")
    for col, label in [("carries", "carries"), ("attempts", "pass attempts"),
                       ("targets", "targets"), ("passing_yards", "passing yards"),
                       ("rushing_yards", "rushing yards"),
                       ("receiving_yards", "receiving yards")]:
        ac = allw[f"a_{col}"].sum()
        r = allw[f"e_{col}"].sum() / ac if ac else np.nan
        check(f"{label} within {int(TOTAL_TOL*100)}%",
              abs(r - 1.0) <= TOTAL_TOL, f"ratio {r:.2f}")

    if "sum_p_play" in allw.columns and allw["sum_p_play"].notna().any():
        m = allw["sum_p_play"].mean()
        check("expected appearances per team in [10, 14]", 10.0 <= m <= 14.0,
              f"sum(p_play) {m:.1f}  actual {allw['a_players'].mean():.1f}")

    print("\nDiagnostic: is the conditional model itself sound?")
    for col, label in [("carries", "carries"), ("attempts", "pass attempts"),
                       ("targets", "targets"), ("passing_yards", "passing yards")]:
        ac = allw[f"a_{col}"].sum()
        r = allw[f"p_{col}"].sum() / ac if ac else np.nan
        soft(f"{label} played-only", f"ratio {r:.2f}"
             + ("  (conditional model is fine)" if abs(r - 1) <= 0.10
                else "  (a real scaling problem too)"))

    print("\nRoster (soft: the chart legitimately lists more than will play)")
    soft("published rows vs stat lines",
         f"model {allw['m_players'].mean():.1f}  actual "
         f"{allw['a_players'].mean():.1f}")

    print("\nConcentration, banded against an ORACLE mean (never the realised")
    print("actual: a conditional mean is inherently less concentrated)")
    for col, key, label in [("carry", "carries", "lead rusher share"),
                            ("target", "targets", "lead receiver share")]:
        m = allw[f"m_top_{col}_share"].mean()
        a = allw[f"a_top_{col}_share"].mean()
        expected = a * ORACLE_TOP1[key]      # what a CORRECT model should show
        ratio = m / expected if expected else np.nan
        check(f"{label} matches the oracle band",
              abs(ratio - 1.0) <= CONCENTRATION_TOL,
              f"model {m:.3f}  oracle-expected {expected:.3f}  "
              f"realised {a:.3f}  ratio {ratio:.2f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} COHERENCE CHECK(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("All coherence checks passed.")


if __name__ == "__main__":
    main()
