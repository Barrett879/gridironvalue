"""Score past projections against what actually happened, and commit the result.

This is the GridironValue port of DiamondValue's accuracy tracker. It answers
the only question a reader of a projection site actually has: over a real
season, is this model better than doing nothing?

WHAT "DOING NOTHING" MEANS HERE
--------------------------------
The comparator is the same one the ship gate uses, so the page cannot quietly
grade on an easier curve than the models were admitted under:

  baseline 2  the player's own season-to-date per-game mean, this game excluded
  baseline 3  a shrunk multi-season rate times expected opportunity

Beating a league constant proves nothing and is not reported here.

STRICTLY OUT OF SAMPLE, WHICH IS THE WHOLE POINT
-------------------------------------------------
The SHIPPED models are trained through 2025, so asking them about 2025 is
asking a student to mark their own paper. Every row here comes from a model
trained on seasons strictly BEFORE the one being scored, refit per fold, with
league priors recomputed from seasons before the fold as well. The site already
refuses to count an in-sample week toward its props record; this holds the
projection record to the same rule.

ONLY PLAYERS WHO APPEARED
-------------------------
The training table contains one row per player-game that produced a stat line,
so a player who did not appear has no row and cannot be scored. That is
deliberate: these models estimate E[Y | appeared], and scoring a projection
against a phantom zero for a player who was inactive measures availability, not
accuracy. Availability is a separate factor (`p_play`) and has its own gates.

PER ROW, NOT PER TARGET
------------------------
`validate_models.py` answers "does this target pass the gate" and prints
aggregates. This writes one row per (season, week, player, target) so the page
can slice by week and by player, and so the paired comparison the page depends
on is possible at all.

Usage:
    python scripts/build_accuracy_tracker.py --seasons 2023-2025
    python scripts/build_accuracy_tracker.py --seasons 2025-2025 --rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import features as F  # noqa: E402
from gridlib.cache import atomic_to_parquet, dc_path, logger, read_parquet_or_none  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_models import (  # noqa: E402
    OPP_FEATURE, TARGET_SPECS, YARD_RATE, _model,
)

VERSION = "v1"
HISTORY = f"accuracy_history_{VERSION}.parquet"

# The identity columns carried onto every scored row, so the page can show a
# name rather than an id and can group by week without a second join.
KEYS = ["season", "week", "gsis_id", "player_display_name", "position", "team",
        "opponent_team"]


def score_fold(feat: pd.DataFrame, fold_season: int, train_max: int) -> pd.DataFrame:
    """One row per (player-game, target) for `fold_season`, model and baselines.

    Mirrors `validate_models.evaluate_target` exactly, including the compound
    versus direct choice for yards, so the numbers this commits are the numbers
    the gate was decided on rather than a second, subtly different model.
    """
    out = []
    for target, spec in TARGET_SPECS.items():
        if target not in feat.columns:
            continue
        positions, kind, exposure = spec
        cols = F.feature_columns(feat, target)
        sub = feat[feat["position"].isin(positions) & feat[target].notna()]
        tr = sub[sub["season"] <= train_max]
        te = sub[sub["season"] == fold_season]
        if len(tr) < 500 or len(te) < 200:
            logger.info("%s %d: too few rows (train %d, test %d), skipping",
                        target, fold_season, len(tr), len(te))
            continue

        X_tr, y_tr = tr[cols], tr[target].astype(float)
        y_te = te[target].astype(float).to_numpy()

        if kind == "yards" and exposure in OPP_FEATURE:
            opp_model = _model("count")
            opp_model.fit(X_tr, tr[exposure].astype(float))
            opp_hat = np.clip(opp_model.predict(te[cols]), 0, None)
            rate = te[YARD_RATE[target]].astype(float).to_numpy()
            pred_compound = opp_hat * rate
            direct = _model("yards")
            direct.fit(X_tr, y_tr)
            pred_direct = np.clip(direct.predict(te[cols]), 0, None)
            # Whichever wins on THIS fold, which is what validate_models ships.
            mae_c = float(np.mean(np.abs(y_te - pred_compound)))
            mae_d = float(np.mean(np.abs(y_te - pred_direct)))
            pred, form = ((pred_compound, "compound") if mae_c <= mae_d
                          else (pred_direct, "direct"))
        else:
            model = _model(kind)
            model.fit(X_tr, y_tr)
            pred = np.clip(model.predict(te[cols]), 0, None)
            form = kind

        # Baseline 2: season-to-date per-game mean, this game excluded. A
        # player with no season-to-date history falls back to the training
        # mean, exactly as the gate does.
        std_col = f"f_std_{target}"
        b2 = (te[std_col].astype(float).to_numpy() if std_col in te.columns
              else np.full(len(te), np.nan))
        b2 = np.where(np.isnan(b2), float(y_tr.mean()), b2)

        # Baseline 3: shrunk multi-season rate times expected opportunity.
        if target in YARD_RATE:
            opp_col = OPP_FEATURE.get(exposure, f"f_career_{exposure}")
            opp = (te[opp_col].astype(float).to_numpy()
                   if opp_col in te.columns else None)
            rate = te[YARD_RATE[target]].astype(float).to_numpy()
            b3 = (np.where(np.isnan(opp), np.nanmean(opp), opp) * rate
                  if opp is not None else np.full(len(te), float(y_tr.mean())))
        else:
            career = te.get(f"f_career_{target}")
            b3 = (career.astype(float).to_numpy() if career is not None
                  else np.full(len(te), float(y_tr.mean())))
            b3 = np.where(np.isnan(b3), float(y_tr.mean()), b3)

        rows = te[[c for c in KEYS if c in te.columns]].copy()
        rows["target"] = target
        rows["form"] = form
        rows["trained_through"] = train_max
        rows["pred"] = pred
        rows["actual"] = y_te
        rows["abs_err_model"] = np.abs(pred - y_te)
        rows["abs_err_b2"] = np.abs(b2 - y_te)
        rows["abs_err_b3"] = np.abs(b3 - y_te)
        out.append(rows)
        logger.warning("  %-22s %5d rows  model MAE %.3f  season-avg MAE %.3f",
                       target, len(rows), float(rows.abs_err_model.mean()),
                       float(rows.abs_err_b2.mean()))
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2023-2025",
                    help="seasons to SCORE; each is predicted by a model "
                         "trained on everything strictly before it")
    ap.add_argument("--rebuild", action="store_true",
                    help="discard the existing history instead of appending")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seasons.split("-"))

    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    if pw is None or lg is None:
        raise SystemExit("backfill missing; run build_player_week.py first")

    path = dc_path(HISTORY)
    existing = None if args.rebuild else read_parquet_or_none(path)
    frames = [existing] if existing is not None and not existing.empty else []

    for season in range(lo, hi + 1):
        train_max = season - 1
        # Features are built over everything up to and including the fold, so
        # the point-in-time helpers see real prior games; the TRAIN/TEST split
        # below is what keeps the fold season out of the model's fitting set.
        print(f"\n{'=' * 66}\nscoring {season}  (models trained through {train_max})\n{'=' * 66}")
        priors = F.league_priors(pw, season)
        feat = F.build(pw[pw["season"] <= season], lg, priors)
        scored = score_fold(feat, season, train_max)
        if scored.empty:
            logger.warning("%d: nothing scored", season)
            continue
        frames.append(scored)

    if not frames:
        raise SystemExit("nothing scored; no history written")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["season", "week", "gsis_id", "target"], keep="last")
    atomic_to_parquet(combined, path)

    print(f"\nwrote {path.name}  ({len(combined):,} rows, "
          f"{combined.season.nunique()} season(s), "
          f"{combined.target.nunique()} targets)")
    # The headline, computed the way the page computes it.
    #
    # PAIRED, so a row carrying a model error but no baseline error cannot
    # inflate one column alone. On DiamondValue that exact artifact turned a
    # true -0.07% into a published +4.60%.
    #
    # And PER TARGET, then averaged, rather than pooling absolute errors across
    # targets. Pooling is not scale-free: passing yards carries an MAE near 61
    # and receiving touchdowns near 0.23, so a pooled ratio is very nearly a
    # report on passing yards alone. The per-target percentage puts a count and
    # a rate stat on one comparable axis.
    paired = combined.dropna(subset=["abs_err_model", "abs_err_b2"])
    per = (paired.groupby("target")
                 .agg(model=("abs_err_model", "mean"),
                      base=("abs_err_b2", "mean")))
    per["edge_pct"] = 100 * (1 - per.model / per.base)
    print(f"\nper target, against that player's own season average:")
    for t, r in per.sort_values("edge_pct", ascending=False).iterrows():
        print(f"  {t:<24} {r.edge_pct:+6.1f}%")
    print(f"\nmean across {len(per)} targets: {per.edge_pct.mean():+.1f}%   "
          f"(pooled, which passing yards dominates: "
          f"{100 * (1 - paired.abs_err_model.mean() / paired.abs_err_b2.mean()):+.1f}%)")
    print(f"targets better than a season average: "
          f"{int((per.edge_pct > 0).sum())} of {len(per)}")


if __name__ == "__main__":
    main()
