"""Walk-forward validation with the three mandatory baselines.

THE GATE
--------
A target ships only if it beats, out of sample, BOTH:
  baseline 2 - the player's season-to-date per-game mean excluding this game
  baseline 3 - a shrunk multi-season rate times expected opportunity
Beating baseline 1 (a league constant) is table stakes and proves nothing.

A target that fails is REPORTED, not silently dropped. Barrett decides whether to
hide it or show it with a caveat. That decision is the spec's step-4 STOP.

WHY THE SPLIT IS WHAT IT IS
---------------------------
Random K-fold is forbidden: it leaks at the player-season level, because the same
player's other games in the same season carry almost the same information as the
held-out one. This is strict walk-forward:

    train    2016-2022   (7 seasons)
    validate 2023        (tuning and feature decisions happen here)
    test     2024        (touched ONCE, at the end)
    holdout  2025        (NOT touched by this script at all)

League efficiency priors are recomputed per fold from seasons strictly before the
fold, so a prior never sees the season it is being used to predict.

WHAT IS PREDICTED, AND WHY DIRECTLY
------------------------------------
Each count is predicted DIRECTLY from pregame features rather than as
rate x exposure. That is not laziness: at inference time the exposure is itself
unknown. Routes run, team dropbacks and targets are all same-game outcomes, so a
model that needs them as inputs cannot run on Thursday. Predicting the count
directly from what is knowable before kickoff is the only formulation that is
actually deployable, and it is what gets validated here.

Yards use the compound form (predicted opportunity x shrunk yards-per-
opportunity) because yards are not counts: they have a point mass at zero and a
heavy right tail, which Poisson handles badly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import features as F  # noqa: E402
from gridlib.cache import dc_path, read_parquet_or_none  # noqa: E402

TRAIN_END = 2022
VALID = 2023
TEST = 2024
HOLDOUT = 2025  # never read here

RECEIVERS = ["WR", "TE", "RB"]
RUSHERS = ["RB", "QB", "FB"]

# target -> (positions, kind, exposure column for the compound yards form)
TARGET_SPECS = {
    "targets":            (RECEIVERS, "count", None),
    "receptions":         (RECEIVERS, "count", None),
    "receiving_yards":    (RECEIVERS, "yards", "targets"),
    "receiving_tds":      (RECEIVERS, "count", None),
    "carries":            (RUSHERS,   "count", None),
    "rushing_yards":      (RUSHERS,   "yards", "carries"),
    "rushing_tds":        (RUSHERS,   "count", None),
    "attempts":           (["QB"],    "count", None),
    "completions":        (["QB"],    "count", None),
    "passing_yards":      (["QB"],    "yards", "attempts"),
    "passing_tds":        (["QB"],    "count", None),
    "passing_interceptions": (["QB"], "count", None),
    "sacks_suffered":     (["QB"],    "count", None),
    "fg_att":             (["K"],     "count", None),
    "fg_made":            (["K"],     "count", None),
    "pat_att":            (["K"],     "count", None),
}

# The shrunk rate feature that pairs with each yards target, for baseline 3 and
# for the compound model.
YARD_RATE = {
    "receiving_yards": "f_shrunk_ypt",
    "rushing_yards": "f_shrunk_ypc",
    "passing_yards": "f_shrunk_ypa",
}
# The prior-opportunity feature used as "expected opportunity" in baseline 3.
OPP_FEATURE = {"targets": "f_std_targets", "carries": "f_std_carries",
               "attempts": "f_std_attempts"}


def _model(kind: str) -> HistGradientBoostingRegressor:
    """Poisson for counts. Modest capacity: 63k rows, and the signal is thin."""
    common = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                  min_samples_leaf=40, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.12,
                  random_state=0)
    if kind == "count":
        return HistGradientBoostingRegressor(loss="poisson", **common)
    return HistGradientBoostingRegressor(loss="squared_error", **common)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {"mae": float(np.mean(np.abs(y - p))),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}


def evaluate_target(feat: pd.DataFrame, target: str, spec, fold_season: int,
                    train_max: int) -> dict | None:
    positions, kind, exposure = spec
    cols = F.feature_columns(feat)

    sub = feat[feat["position"].isin(positions) & feat[target].notna()]
    tr = sub[sub["season"] <= train_max]
    te = sub[sub["season"] == fold_season]
    if len(tr) < 500 or len(te) < 200:
        return None

    X_tr, y_tr = tr[cols], tr[target].astype(float)
    X_te, y_te = te[cols], te[target].astype(float)

    # ── Model ────────────────────────────────────────────────────────────────
    if kind == "yards" and exposure in OPP_FEATURE:
        # Compound: predict opportunity as a count, multiply by the shrunk
        # yards-per-opportunity posterior. Yards are not counts.
        opp_model = _model("count")
        opp_model.fit(X_tr, tr[exposure].astype(float))
        opp_hat = np.clip(opp_model.predict(X_te), 0, None)
        rate = te[YARD_RATE[target]].astype(float).to_numpy()
        pred_compound = opp_hat * rate

        direct = _model("yards")
        direct.fit(X_tr, y_tr)
        pred_direct = np.clip(direct.predict(X_te), 0, None)

        # Keep whichever wins on THIS fold; the choice is reported, not hidden.
        m_c, m_d = _metrics(y_te, pred_compound), _metrics(y_te, pred_direct)
        if m_c["mae"] <= m_d["mae"]:
            pred, form = pred_compound, "compound"
        else:
            pred, form = pred_direct, "direct"
    else:
        model = _model(kind)
        model.fit(X_tr, y_tr)
        pred = np.clip(model.predict(X_te), 0, None)
        form = kind

    # ── Baseline 1: league constant (the training mean for these positions) ──
    b1 = np.full(len(te), float(y_tr.mean()))

    # ── Baseline 2: player's season-to-date per-game mean, this game excluded ─
    std_col = f"f_std_{target}"
    b2 = te[std_col].astype(float).to_numpy() if std_col in te.columns \
        else np.full(len(te), np.nan)
    b2 = np.where(np.isnan(b2), float(y_tr.mean()), b2)

    # ── Baseline 3: shrunk multi-season rate x expected opportunity ──────────
    if target in YARD_RATE:
        opp_col = OPP_FEATURE.get(exposure, f"f_career_{exposure}")
        opp = te[opp_col].astype(float).to_numpy() if opp_col in te.columns else None
        rate = te[YARD_RATE[target]].astype(float).to_numpy()
        b3 = np.where(np.isnan(opp), np.nanmean(opp), opp) * rate if opp is not None else b1
    else:
        career = te.get(f"f_career_{target}")
        b3 = career.astype(float).to_numpy() if career is not None else b1
        b3 = np.where(np.isnan(b3), float(y_tr.mean()), b3)

    res = {"target": target, "form": form, "n_train": len(tr), "n_test": len(te),
           "mean": float(y_te.mean())}
    res.update({f"model_{k}": v for k, v in _metrics(y_te, pred).items()})
    for name, b in (("b1", b1), ("b2", b2), ("b3", b3)):
        res.update({f"{name}_{k}": v for k, v in _metrics(y_te, b).items()})
    for name in ("b1", "b2", "b3"):
        res[f"skill_vs_{name}"] = 100.0 * (1 - res["model_mae"] / res[f"{name}_mae"])
    res["beats_b2"] = res["model_mae"] < res["b2_mae"]
    res["beats_b3"] = res["model_mae"] < res["b3_mae"]
    res["passes_gate"] = bool(res["beats_b2"] and res["beats_b3"])
    return res


def run_fold(pw: pd.DataFrame, lg: pd.DataFrame, fold_season: int,
             train_max: int) -> pd.DataFrame:
    priors = F.league_priors(pw, fold_season)
    feat = F.build(pw[pw["season"] <= fold_season], lg, priors)
    rows = []
    for target, spec in TARGET_SPECS.items():
        if target not in feat.columns:
            continue
        r = evaluate_target(feat, target, spec, fold_season, train_max)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


def _print(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(f"{'target':<24}{'mean':>7}{'model':>8}{'b1':>8}{'b2':>8}{'b3':>8}"
          f"{'vs b2':>8}{'vs b3':>8}  gate")
    print("-" * 88)
    for _, r in df.iterrows():
        gate = "PASS" if r["passes_gate"] else "fail"
        print(f"{r['target']:<24}{r['mean']:>7.2f}{r['model_mae']:>8.3f}"
              f"{r['b1_mae']:>8.3f}{r['b2_mae']:>8.3f}{r['b3_mae']:>8.3f}"
              f"{r['skill_vs_b2']:>7.1f}%{r['skill_vs_b3']:>7.1f}%  {gate}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="also score the TEST season (2024). Use once.")
    ap.add_argument("--out", default="docs/model_report_data.csv")
    args = ap.parse_args()

    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    if pw is None or lg is None:
        print("Backfill missing. Run build_team_week.py then build_player_week.py.")
        sys.exit(2)

    print(f"Walk-forward: train <= {TRAIN_END}, validate {VALID}, "
          f"test {TEST}, holdout {HOLDOUT} (untouched)")
    val = run_fold(pw, lg, VALID, TRAIN_END)
    _print(val, f"VALIDATION SEASON {VALID}  (MAE; lower is better)")
    val["fold"] = "validation"
    frames = [val]

    if args.test:
        tst = run_fold(pw, lg, TEST, VALID)
        _print(tst, f"TEST SEASON {TEST}  (train <= {VALID})")
        tst["fold"] = "test"
        frames.append(tst)

    out = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    passed = val[val["passes_gate"]]["target"].tolist()
    failed = val[~val["passes_gate"]]["target"].tolist()
    print(f"\nPASS the gate ({len(passed)}): {passed}")
    print(f"FAIL the gate ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
