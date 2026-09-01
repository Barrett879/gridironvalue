"""Build the team-week table: one row per (game, team), from play-by-play.

This is level 1 and 2 of the four-level decomposition the spec calls for:

    team plays -> pass/run split -> player share -> player rate

A player's projection is a share of a team quantity, so the team quantity has to
exist first and has to be defined exactly. Every count here is reconciled
against the official weekly box scores by `validate_backfill.py`.

TWO DEFINITIONS THAT WERE VERIFIED, NOT ASSUMED
-----------------------------------------------
Reconciling pbp against `stats_player_week` for 2024 (544 team-games):

  - **Pass attempts** must be `complete_pass + incomplete_pass + interception`.
    Using pbp's `pass_attempt` column overcounts by 2.47 per team-game
    (75/544 exact). The formula above is exact: 544/544.
  - **Rush attempts** must EXCLUDE two-point conversions. Raw `rush_attempt`
    overcounts by 0.066 per team-game (509/544 exact); dropping two-point
    attempts is exact: 544/544. Kneels are NOT the cause and are kept, because
    the official box score counts a kneel as a carry.

Getting either wrong shifts every player's share denominator by a percent or
two, which is small enough to never look like a bug and big enough to matter.

PROE CARRIES A SEASON-LEVEL OFFSET, SO RAW VALUES ARE STORED AND NOT CENTERED
----------------------------------------------------------------------------
nflfastR's `xpass` model has drifted. Its mean prediction is flat at ~0.627 in
every season 2016-2025, while the actual league pass rate fell from 0.608 to
0.591 over the same span. League-mean `pass_oe` is therefore NOT zero, and it
slides monotonically:

    2016 +0.18   2018 +0.02   2020 -0.16   2022 -2.00   2024 -1.72   2025 -1.67

Left raw, a model can learn that offset as "later seasons pass less", which is
true but entangles league drift with team tendency. Centering is the fix, but
centering with the CURRENT season's league mean would leak games that have not
been played at inference time. So this builder stores raw PROE only and writes
the per-season league means to a separate `league_season` table; the feature
layer decides how to center in a point-in-time-safe way (prior seasons, or
season-to-date).

GARBAGE TIME IS NOT FILTERED AWAY
---------------------------------
The widely-copied 20-80% win-probability filter discards far too much data for a
17-game sport. This builder computes BOTH the full-game aggregate and a
neutral-script one (win probability 5-95%) and stores both, so the feature layer
chooses rather than having the choice baked in irreversibly at backfill time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import fetch  # noqa: E402
from gridlib.cache import atomic_to_parquet, dc_path, logger  # noqa: E402

VERSION = "v1"

# Neutral script: the spec's 5-95% window, not the copied 20-80% one.
NEUTRAL_WP_LO, NEUTRAL_WP_HI = 0.05, 0.95


def _offensive_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Scrimmage plays only, with the verified attempt definitions attached."""
    p = pbp[pbp["posteam"].notna() & pbp["play_type"].notna()].copy()
    p = p[p["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike"])]

    two_pt = p["two_point_attempt"].fillna(0) == 0
    p["off_pass_att"] = (
        p["complete_pass"].fillna(0)
        + p["incomplete_pass"].fillna(0)
        + p["interception"].fillna(0)
    ) * two_pt
    p["off_rush_att"] = p["rush_attempt"].fillna(0) * two_pt
    p["off_sack"] = p["sack"].fillna(0)
    p["off_scramble"] = p["qb_scramble"].fillna(0)
    # A designed run is a carry that was not a scramble. Never merge the two:
    # scrambles are a pass-play outcome and respond to script like a dropback.
    p["off_designed_rush"] = (p["off_rush_att"] - p["off_scramble"]).clip(lower=0)
    p["is_dropback"] = ((p["off_pass_att"] > 0) | (p["off_sack"] > 0)
                        | (p["off_scramble"] > 0)).astype(float)
    p["is_scrimmage"] = ((p["off_pass_att"] > 0) | (p["off_rush_att"] > 0)
                         | (p["off_sack"] > 0)).astype(float)
    p["neutral"] = p["wp"].between(NEUTRAL_WP_LO, NEUTRAL_WP_HI, inclusive="both")
    return p


def _weighted_mean(values: pd.Series, mask: pd.Series) -> float:
    sub = values[mask]
    return float(sub.mean()) if len(sub) and sub.notna().any() else np.nan


def _team_side(p: pd.DataFrame, season: int) -> pd.DataFrame:
    """Aggregate the offensive side, one row per (game_id, posteam)."""
    rows = []
    for (game_id, team), g in p.groupby(["game_id", "posteam"], sort=False):
        scrim = g["is_scrimmage"] > 0
        neutral = g["neutral"] & scrim
        drop = g["is_dropback"] > 0

        pass_att = g["off_pass_att"].sum()
        rush_att = g["off_rush_att"].sum()
        designed = g["off_designed_rush"].sum()
        sacks = g["off_sack"].sum()
        scrambles = g["off_scramble"].sum()
        dropbacks = drop.sum()
        plays = float(scrim.sum())

        n_pass = g.loc[neutral, "is_dropback"].sum()
        n_plays = float(neutral.sum())

        rz = scrim & (g["yardline_100"] <= 20)
        i10 = scrim & (g["yardline_100"] <= 10)
        i5 = scrim & (g["yardline_100"] <= 5)

        rows.append({
            "game_id": game_id, "season": season, "week": int(g["week"].iloc[0]),
            "team": team,
            "opponent": g["defteam"].mode().iloc[0] if g["defteam"].notna().any() else None,
            "is_home": int(g["home_team"].iloc[0] == team),
            # ── Exposure: the denominators every player share divides by ──
            "off_plays": plays,
            "off_dropbacks": float(dropbacks),
            "off_pass_att": float(pass_att),
            "off_rush_att": float(rush_att),
            "off_designed_rush": float(designed),
            "off_scrambles": float(scrambles),
            "off_sacks": float(sacks),
            # ── Tendency: coach intent vs score-driven script ──
            "off_pass_rate": float(dropbacks / plays) if plays else np.nan,
            "off_neutral_pass_rate": float(n_pass / n_plays) if n_plays else np.nan,
            "off_proe": _weighted_mean(g["pass_oe"], scrim),
            "off_neutral_proe": _weighted_mean(g["pass_oe"], neutral),
            "off_xpass": _weighted_mean(g["xpass"], scrim),
            "off_neutral_plays": n_plays,
            # ── Efficiency ──
            "off_epa_play": _weighted_mean(g["epa"], scrim),
            "off_pass_epa": _weighted_mean(g["epa"], scrim & drop),
            "off_rush_epa": _weighted_mean(g["epa"], scrim & (g["off_designed_rush"] > 0)),
            "off_success_rate": _weighted_mean(g["success"], scrim),
            "off_yards": float(g.loc[scrim, "yards_gained"].sum()),
            # ── Pace ──
            "off_sec_per_play": _weighted_mean(g["play_clock"].astype(float), neutral)
            if "play_clock" in g.columns else np.nan,
            "off_drives": float(g.loc[scrim, "drive"].nunique()),
            # ── Scoring exposure: the raw ingredients for expected TDs ──
            "off_rz_plays": float(rz.sum()),
            "off_i10_plays": float(i10.sum()),
            "off_i5_plays": float(i5.sum()),
            "off_rz_rush": float(g.loc[rz, "off_rush_att"].sum()),
            "off_rz_pass": float(g.loc[rz, "off_pass_att"].sum()),
            "off_pass_td": float(g.loc[scrim, "pass_touchdown"].fillna(0).sum()),
            "off_rush_td": float(g.loc[scrim, "rush_touchdown"].fillna(0).sum()),
            # ── Realized script. Model-time features must use the MARKET's
            #    implied script instead; these are here for validation and for
            #    building shrunk priors, never as an inference feature. ──
            "realized_mean_wp": _weighted_mean(g["wp"], scrim),
            "realized_mean_score_diff": _weighted_mean(g["score_differential"], scrim),
            "realized_frac_leading": float(
                (g.loc[scrim, "score_differential"] > 0).mean()) if plays else np.nan,
            "realized_frac_trailing": float(
                (g.loc[scrim, "score_differential"] < 0).mean()) if plays else np.nan,
        })
    return pd.DataFrame(rows)


def build_season(season: int) -> pd.DataFrame | None:
    """Team-week for one season, offense and the defense it faced."""
    pbp = fetch.load_pbp(season)
    if pbp is None or pbp.empty:
        logger.warning("no pbp for %d", season)
        return None
    pbp = pbp[pbp["season_type"] == "REG"]
    if pbp.empty:
        return None

    p = _offensive_plays(pbp)
    off = _team_side(p, season)

    # The defensive side of a game is the OTHER team's offensive row. Joining it
    # in gives "what this defense allowed" without a second aggregation pass,
    # and guarantees the two sides are defined identically.
    def_cols = [c for c in off.columns if c.startswith("off_")]
    dfn = off[["game_id", "team"] + def_cols].rename(
        columns={"team": "opponent", **{c: c.replace("off_", "def_allowed_")
                                        for c in def_cols}}
    )
    out = off.merge(dfn, on=["game_id", "opponent"], how="left")
    return out.sort_values(["week", "team"]).reset_index(drop=True)


def build_league_context(team_week: pd.DataFrame) -> pd.DataFrame:
    """Per-season league means, so the feature layer can center features
    point-in-time safely rather than having centering baked in here.

    Keeping this OUT of the team-week table is deliberate. Any centering that
    uses the current season's league mean is leakage at inference time, because
    it depends on games that have not been played yet.
    """
    cols = ["off_plays", "off_dropbacks", "off_pass_att", "off_designed_rush",
            "off_pass_rate", "off_neutral_pass_rate", "off_proe",
            "off_neutral_proe", "off_epa_play", "off_rz_plays", "off_drives",
            "off_pass_td", "off_rush_td"]
    out = team_week.groupby("season")[cols].mean().reset_index()
    out.columns = ["season"] + [f"league_{c}" for c in cols]
    out["league_team_games"] = team_week.groupby("season").size().to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2016-2025",
                    help="e.g. 2016-2025 or 2024 (default: 2016-2025)")
    args = ap.parse_args()

    if "-" in args.seasons:
        lo, hi = (int(x) for x in args.seasons.split("-"))
        seasons = list(range(lo, hi + 1))
    else:
        seasons = [int(args.seasons)]

    frames = []
    for season in seasons:
        df = build_season(season)
        if df is None:
            print(f"  {season}: SKIPPED (no pbp)")
            continue
        path = dc_path(f"team_week_{season}_{VERSION}.parquet")
        atomic_to_parquet(df, path)
        print(f"  {season}: {len(df):4d} team-games -> {path.name}")
        frames.append(df)

    if frames:
        allf = pd.concat(frames, ignore_index=True)
        path = dc_path(f"team_week_{seasons[0]}_{seasons[-1]}_{VERSION}.parquet")
        atomic_to_parquet(allf, path)
        print(f"\ncombined: {len(allf)} team-games, {len(allf.columns)} cols -> {path.name}")

        league = build_league_context(allf)
        lpath = dc_path(f"league_season_{seasons[0]}_{seasons[-1]}_{VERSION}.parquet")
        atomic_to_parquet(league, lpath)
        print(f"league context: {len(league)} seasons -> {lpath.name}")
        print(league[["season", "league_off_pass_rate", "league_off_proe",
                      "league_off_plays"]].to_string(index=False))


if __name__ == "__main__":
    main()
