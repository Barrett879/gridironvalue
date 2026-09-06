"""Projections for a week that has not been played.

THE HARD PART IS THAT THE ROWS DO NOT EXIST YET
------------------------------------------------
Every other table in this project has one row per thing that happened. A
projection needs a row per thing that is ABOUT to happen, and nothing in
nflverse supplies that. So the rows are constructed: take each team's latest
depth chart for the week, keep the skill positions, attach the game and market
context from the schedule, append those rows to the real history, and run the
SAME feature builder used for training. The features for the constructed rows
then look back over real prior games exactly as a training row does.

Running the identical `features.build` on both is the point. Two code paths
drift, and the drift shows up as a model that validates well and projects badly.

TWO ESTIMANDS, AND THEY MUST NEVER BE CONFUSED
-----------------------------------------------
Every model here trains on the backfill, which contains only players who
recorded a stat line. So each estimates

    E[Y | the player appeared]

That is the RIGHT quantity for a prop line: a prop voids if the player does not
play, so the posted number is itself conditional on playing. It is the WRONG
quantity to sum across a roster, because a 19-man depth chart contains about
seven players who will not appear at all. Summing the conditional number over
the full chart inflated team totals to 1.40x actual carries and 1.73x actual
pass attempts, while the SAME projection restricted to players who did appear
summed to 0.96x and 1.02x. The conditional model was never the problem; the
missing factor was P(appears).

So this module serves BOTH, in separate columns, never mixed:

    {target}            E[Y | plays]. The board and the props comparison use
                        this, unmultiplied. Do not "fix" it with p_play.
    p_play              P(the player appears), from the availability grid.
    {target}_expected   p_play x E[Y | plays]. Aggregates and the coherence
                        checks use this, and nothing else does.

Multiplying the DISPLAYED number by p_play would deflate every questionable
player against a line that does not share the assumption, manufacturing a fake
Under across the whole props surface. The separation is enforced here rather
than left to each caller.

FAILED TARGETS ARE NOT SILENTLY SERVED
---------------------------------------
`models/registry_*.json` records the ship-gate verdict per target. Receiving and
rushing touchdowns FAIL: both lose to a plain season-to-date average out of
sample. For those, this module returns the BASELINE, not the model, and labels
it. A projection that is measurably worse than an average is not worth shipping
because it came out of a gradient booster.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import features as F
from . import fetch
from .cache import atomic_to_parquet, dc_path, logger, read_parquet_or_none

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VERSION = "m1"
BACKFILL = "player_week_2016_2025_v1.parquet"
# The current season, rebuilt weekly by CI. DATA version v1, not the model
# version m1: `VERSION` here is the model stamp and using it would have looked
# for player_week_2026_2026_m1.parquet, which nothing writes, so the file would
# never be found and the staleness this fixes would have persisted silently.
CURRENT_TMPL = "player_week_{s}_{s}_v1.parquet"
LEAGUE = "league_season_2016_2025_v1.parquet"

# Fit on the 2025+ depth-chart schema, which is the one the site serves. The
# pre-2025 files encode rank so differently (charts stop at 3, so a "rank 3" is
# the last man rather than a rotation player) that pooling the eras biased the
# grid low: 9.8 expected appearances per team against 11.9 observed, which
# passed straight through into 0.84x team totals.
AVAILABILITY = "availability_2025_2025_v1.parquet"
MAX_RANK = 6
# Injury status -> multiplicative adjustment on the grid. The report is pregame
# (Wed-Fri for a Sunday game), so this is legitimately knowable.
INJURY_ADJ = {"Out": 0.02, "Doubtful": 0.25, "Questionable": 0.92}


def availability_grid():
    """P(appears) by (position, capped depth rank). None when not yet built."""
    return read_parquet_or_none(dc_path(AVAILABILITY))


def attach_p_play(df: pd.DataFrame) -> pd.DataFrame:
    """Add `p_play`, the probability the player appears in the box score.

    Falls back to 1.0 when the grid is missing, which keeps the site running
    and is loudly wrong rather than quietly wrong: the coherence checks will
    fail immediately and say so.
    """
    out = df.copy()
    grid = availability_grid()
    if grid is None or grid.empty:
        logger.warning(
            "no availability grid (%s); p_play defaults to 1.0 and team "
            "aggregates will be inflated. Run scripts/build_availability.py.",
            AVAILABILITY)
        out["p_play"] = 1.0
        return out
    rank = pd.to_numeric(out.get("depth_rank"), errors="coerce")
    out["_rank_capped"] = rank.clip(upper=MAX_RANK).fillna(MAX_RANK).astype(int)
    key = grid.set_index(["position", "rank_capped"])["p_play"]
    out["p_play"] = [
        float(key.get((pos, rk), 0.5))
        for pos, rk in zip(out["position"], out["_rank_capped"])
    ]
    # The injury report is the one case position and rank get badly wrong.
    if "report_status" in out.columns:
        adj = out["report_status"].map(INJURY_ADJ).fillna(1.0)
        out["p_play"] = (out["p_play"] * adj).clip(0.0, 1.0)
    return out.drop(columns=["_rank_capped"])


# Depth-chart position -> the position label the models were trained on.
_POS_MAP = {"QB": "QB", "RB": "RB", "FB": "FB", "WR": "WR", "TE": "TE", "PK": "K"}


def load_registry() -> dict | None:
    path = MODELS_DIR / f"registry_{VERSION}.json"
    if not path.exists():
        logger.warning("no model registry at %s; run scripts/train_models.py", path)
        return None
    reg = json.loads(path.read_text())
    try:
        import sklearn
        if sklearn.__version__ != reg.get("sklearn"):
            logger.warning(
                "scikit-learn %s but artifacts were saved with %s. joblib.load "
                "can break silently across versions; retrain before trusting "
                "these numbers.", sklearn.__version__, reg.get("sklearn"))
    except ImportError:  # pragma: no cover
        pass
    return reg


def build_inference_rows(season: int, week: int,
                         games: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (projected player, upcoming game) for a week.

    Players come from each team's LATEST depth-chart snapshot for the week, so
    the roster reflects the most recent information rather than a season-opening
    guess. Returns an empty frame when the week has no games or no depth chart,
    so callers can say why rather than rendering silence.
    """
    g = games if games is not None else fetch.load_schedules()
    wk = fetch.week_games(season, week, g)
    if wk.empty:
        return pd.DataFrame()

    rows = []
    for _, game in wk.iterrows():
        for team, opp in ((game["away_team"], game["home_team"]),
                          (game["home_team"], game["away_team"])):
            depth = fetch.latest_depth_chart(season, week, team)
            if depth.empty:
                logger.warning("no depth chart for %s in %d week %d",
                               team, season, week)
                continue
            for _, p in depth.iterrows():
                pos = _POS_MAP.get(str(p.get("pos_abb")))
                if not pos or not p.get("gsis_id"):
                    continue
                is_home = int(team == game["home_team"])
                rows.append({
                    "gsis_id": p["gsis_id"],
                    "player_display_name": p.get("player_name"),
                    "position": pos,
                    "position_group": pos,
                    "season": season, "week": week,
                    "game_id": game["game_id"],
                    "team": team, "opponent_team": opp,
                    "is_home": is_home,
                    "depth_rank": p.get("pos_rank"),
                    "spread_line": game.get("spread_line"),
                    "total_line": game.get("total_line"),
                    "team_implied_total": (game.get("home_implied_total") if is_home
                                           else game.get("away_implied_total")),
                    "opp_implied_total": (game.get("away_implied_total") if is_home
                                          else game.get("home_implied_total")),
                    "team_spread": (-game["spread_line"] if is_home
                                    else game["spread_line"])
                    if pd.notna(game.get("spread_line")) else np.nan,
                    "rest_days": (game.get("home_rest") if is_home
                                  else game.get("away_rest")),
                    "roof_type": game.get("roof_type"),
                    "surface": game.get("surface"),
                    "div_game": game.get("div_game"),
                    "gameday": game.get("gameday"),
                    "weekday": game.get("weekday"),
                    "kickoff": game.get("kickoff"),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # The backfill derives these from depth_rank; inference MUST derive them the
    # same way or they arrive as NaN and the model silently loses the feature.
    # Capped at 3 for the same reason as in the backfill: the two depth-chart
    # eras are not comparable beyond that.
    out["depth_rank_capped"] = (pd.to_numeric(out["depth_rank"], errors="coerce")
                                .clip(upper=3))
    out["is_starter"] = (out["depth_rank_capped"] == 1).astype(float)

    # ── The injury report, derived EXACTLY as the backfill derives it ──
    # These are trained-on features. Leaving them absent at inference does not
    # error: the concat with history supplies the column and every live row gets
    # NaN, so the model quietly loses the feature. Zero is the correct default
    # and matches the backfill, where "not on the report" is the healthy
    # baseline rather than missing data.
    inj = fetch.injury_report(season, week)
    if inj is not None and not inj.empty:
        rep = (inj[["gsis_id", "report_status", "practice_status"]]
               .dropna(subset=["gsis_id"]).drop_duplicates("gsis_id"))
        out = out.merge(rep, on="gsis_id", how="left")
    else:
        out["report_status"] = pd.NA
        out["practice_status"] = pd.NA
        logger.info(
            "no injury report for %d week %d; injury features fall back to the "
            "healthy baseline. Before the season starts the DEPTH CHART is the "
            "only availability signal, which is why rank is a feature.",
            season, week)
    # ── Bio, derived EXACTLY as the backfill derives it ──
    # Same trap as the injury block above, and it was live: bio shipped into
    # the feature set and this path was never taught to build it, so every
    # projection would have carried NaN age, experience, draft round, height
    # and weight. A tree routes NaN down a learned branch and returns a
    # confident wrong number instead of failing. Caught by
    # test_predict_refuses_to_serve_with_missing_features once the models were
    # retrained; the test could not see it before, because the artifacts then
    # in `models/` predated the bio block.
    bio = fetch.player_bio()
    if not bio.empty:
        out = out.merge(bio, on="gsis_id", how="left")
        out["experience"] = (out["season"] - pd.to_numeric(
            out.get("rookie_season"), errors="coerce")).clip(lower=0)
        out["draft_round"] = pd.to_numeric(
            out.get("draft_round"), errors="coerce").fillna(8.0)
        _born = pd.to_datetime(out.get("birth_date"), errors="coerce")
        _gd = pd.to_datetime(out.get("gameday"), errors="coerce")
        out["age"] = (_gd - _born).dt.days / 365.25
    else:
        for c in ("age", "experience", "draft_round", "height", "weight"):
            out[c] = np.nan

    _SEV = {"Questionable": 1.0, "Doubtful": 2.0, "Out": 3.0}
    _PRAC = {"Full Participation in Practice": 0.0,
             "Limited Participation in Practice": 1.0,
             "Did Not Participate In Practice": 2.0}
    out["injury_severity"] = out["report_status"].map(_SEV).fillna(0.0)
    out["practice_limitation"] = out["practice_status"].map(_PRAC).fillna(0.0)
    out["on_injury_report"] = out["report_status"].notna().astype(float)
    # One row per player per week. A player listed at two positions on the depth
    # chart (a WR taking snaps at RB) would otherwise be projected twice.
    return out.drop_duplicates(["gsis_id", "game_id"]).reset_index(drop=True)


def projections_path(season: int, week: int):
    """Version-stamped in the filename, so a model change invalidates by bump
    rather than by editing a file in place."""
    return dc_path(f"predictions_{season}_w{week:02d}_{VERSION}.parquet")


def project_week_cached(season: int, week: int, ttl: int = 3600,
                        games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Disk-cached projections. Building from scratch takes ~25 seconds because
    the feature builder runs over ten seasons of history, which is far too slow
    to sit in a page render."""
    from .cache import dc_fresh
    path = projections_path(season, week)
    if dc_fresh(path, ttl=ttl):
        cached = read_parquet_or_none(path)
        if cached is not None:
            return cached
    fresh = project_week(season, week, games)
    if not fresh.empty:
        # kickoff is tz-aware; parquet round-trips it fine, but drop the column
        # if it ever causes trouble rather than failing the whole write.
        try:
            atomic_to_parquet(fresh, path)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not cache projections: %s", e)
    return fresh


def project_week(season: int, week: int,
                 games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-player projections for one week. Empty frame if it cannot be built.

    The returned frame carries one column per target plus `{target}_source`,
    which is "model", "model_low_confidence" or "baseline". The UI must show
    that distinction; it is the difference between a number worth reading and a
    number we know is worse than an average.
    """
    reg = load_registry()
    if reg is None:
        return pd.DataFrame()

    hist = read_parquet_or_none(dc_path(BACKFILL))
    lg = read_parquet_or_none(dc_path(LEAGUE))
    if hist is None or lg is None:
        logger.warning("backfill missing; cannot project")
        return pd.DataFrame()

    # THE CURRENT SEASON, appended to the frozen history.
    #
    # Without this the site projects mid-season as if the season had not
    # started. `BACKFILL` is a fixed ten-year file that ends in 2025, so in week
    # 5 of 2026 a player's own weeks 1 to 4 would be invisible to his features:
    # every season-to-date value empty, every rolling window reaching back into
    # last year, and `f_games_prior_season` zero for the entire league. The
    # projections would degrade a little more every week and nothing would fail.
    #
    # The current season lives in its own file so the ten years of history are
    # never rewritten by an automated job, and so the weekly rebuild is one
    # season rather than ten. `.github/workflows/weekly.yml` regenerates and
    # commits it after each slate; Streamlit Cloud redeploys on the push.
    cur = read_parquet_or_none(dc_path(CURRENT_TMPL.format(s=season)))
    if cur is not None and not cur.empty:
        hist = pd.concat([hist, cur], ignore_index=True, sort=False)
        logger.info("appended %d current-season rows to history", len(cur))

    inf = build_inference_rows(season, week, games)
    if inf.empty:
        return inf

    # Append the constructed rows to real history and build features over the
    # whole thing, so the point-in-time helpers see genuine prior games.
    hist = hist[(hist["season"] < season)
                | ((hist["season"] == season) & (hist["week"] < week))]
    combined = pd.concat([hist, inf], ignore_index=True, sort=False)
    priors = reg.get("league_priors") or F.league_priors(hist, season)
    feat = F.build(combined, lg, priors)

    mask = (feat["season"] == season) & (feat["week"] == week) & \
           feat["gsis_id"].isin(set(inf["gsis_id"]))
    live = feat[mask].copy()
    if live.empty:
        return live

    cols = reg["feature_columns"]
    missing = [c for c in cols if c not in live.columns]
    if missing:
        # Do NOT fail open. A tree model turns a NaN into a learned default
        # branch, which is a silent wrong answer rather than an abstention, and
        # the feature simply disappears with no error. This exact skew already
        # cost one silent regression: depth_rank_capped and is_starter existed
        # in training and were never built at inference.
        raise RuntimeError(
            f"{len(missing)} feature column(s) the models were trained on are "
            f"missing at inference: {missing[:8]}. Serving would silently drop "
            f"them. Fix build_inference_rows rather than filling with NaN."
        )

    # NO shared X. Each model gets the exact column list it was fitted on,
    # because targets no longer share one feature set: the QB models are
    # trained without the positional-defence block. Handing every model the
    # union would feed the QB models four columns they never saw, in the wrong
    # positions, which is a silent wrong answer rather than an error.
    for target, meta in reg["targets"].items():
        live[f"{target}_source"] = meta["present_as"]
        if meta["present_as"] == "baseline":
            # The model is measurably worse than a season-to-date average for
            # this target, so serve the average and say so.
            col = f"f_std_{target}"
            fallback = f"f_career_{target}"
            base = live[col] if col in live.columns else None
            if base is None or base.isna().all():
                base = live[fallback] if fallback in live.columns else np.nan
            live[target] = pd.to_numeric(base, errors="coerce").fillna(0.0)
            continue
        path = MODELS_DIR / meta["file"]
        if not path.exists():
            logger.warning("artifact %s missing", path)
            live[target] = np.nan
            continue
        mdl = joblib.load(path)
        # Fall back to the union for a registry written before per-target
        # columns existed; those artifacts were fitted on the union.
        mcols = meta.get("feature_columns") or cols
        missing_t = [c for c in mcols if c not in live.columns]
        if missing_t:
            raise RuntimeError(
                f"{target}: {len(missing_t)} trained feature column(s) missing "
                f"at inference: {missing_t[:8]}."
            )
        pred = np.clip(mdl.predict(live[mcols]), 0, None)
        # Measured LEVEL correction, for the three targets where one survived
        # the two-stage gate. The lowest projection quintile is over-projected
        # by 10-25% because it is mostly zeros and nothing separates a receiver
        # who dressed and was never targeted from one who saw a little work.
        # Monotone, so it cannot reorder players. A no-op for every other target.
        from . import uncertainty as _u
        raw_mean = pred.copy()
        pred = _u.apply_level(target, pred)
        # Keep the UNCALIBRATED mean for the aggregate column. See below.
        live[f"__raw_{target}"] = np.where(
            live["position"].isin(meta["positions"]).to_numpy(), raw_mean, np.nan)
        # A model only applies to the positions it was trained on.
        applies = live["position"].isin(meta["positions"]).to_numpy()
        live[target] = np.where(applies, pred, np.nan)

    live = attach_p_play(live)

    keep = ["gsis_id", "player_display_name", "position", "team",
            "opponent_team", "game_id", "season", "week", "depth_rank",
            "is_home", "team_implied_total", "team_spread", "kickoff", "p_play"]
    keep = [c for c in keep if c in live.columns]
    tcols = [c for c in reg["targets"] if c in live.columns]
    scols = [f"{t}_source" for t in tcols if f"{t}_source" in live.columns]
    # Carry the uncalibrated means through; they feed the aggregate column below
    # and are dropped again before returning.
    rcols = [f"__raw_{t}" for t in tcols if f"__raw_{t}" in live.columns]
    out = live[keep + tcols + scols + rcols].copy()

    # The unconditional twin of every projection. ONLY aggregates and the
    # coherence checks may use these; the board and the props comparison use
    # the conditional column above, because a prop voids on a DNP.
    for t in tcols:
        # AGGREGATES USE THE UNCALIBRATED MEAN, not the level-corrected value.
        #
        # The level calibration is fitted and gated on per-player MAE, which is
        # minimised by the median, so it deliberately pulls projections down
        # toward it. That is right for a number shown next to a line and wrong
        # for a number summed across a roster: on 2025 week 6, restricted to
        # players who actually appeared, the calibrated receiving projections
        # summed to 85% of actual while rushing yards summed to 1.00. Two
        # coherence gates failed as a result, and the MAE gate could not see it
        # because MAE is per-player and insensitive to a systematic shift that
        # reduces absolute error on the many near-zero rows.
        #
        # This is the same two-estimand split the module already enforces
        # between {target} and {target}_expected, applied one level deeper.
        _raw = out.get(f"__raw_{t}")
        _base = _raw if _raw is not None and _raw.notna().any() else out[t]
        out[f"{t}_expected"] = _base * out["p_play"]

    out = out.drop(columns=[c for c in out.columns if c.startswith("__raw_")])
    out["f_games_prior"] = live.get("f_games_prior", np.nan)
    return out.sort_values(["team", "position", "depth_rank"]).reset_index(drop=True)


# ── The PrizePicks fantasy composite ─────────────────────────────────────────
# PrizePicks' own scoring, not a generic fantasy scale.
PP_SCORING = {
    "passing_yards": 0.04, "rushing_yards": 0.10, "receiving_yards": 0.10,
    "receptions": 1.0, "passing_tds": 4.0, "rushing_tds": 6.0,
    "receiving_tds": 6.0, "passing_interceptions": -1.0, "fumbles_lost": -1.0,
}


def fantasy_score(row) -> float:
    """PrizePicks Fantasy Score, composed from the component projections.

    COMPOSING A MEAN FROM COMPONENT MEANS IS EXACT for the mean, because
    expectation is linear. It is NOT exact for the spread: the components are
    correlated (a big passing game and a passing touchdown arrive together), so
    the composite's variance is not the sum of the parts' variances. Any
    uncertainty band on this number would be too narrow, which is why none is
    shown.

    It also inherits the touchdown problem: two of the three touchdown terms are
    served from a baseline rather than a model, and they carry the largest
    per-unit weight in the whole formula.
    """
    total = 0.0
    for col, weight in PP_SCORING.items():
        v = row.get(col)
        if v is not None and pd.notna(v):
            total += float(v) * weight
    return round(total, 2)
