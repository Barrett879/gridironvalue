"""Experiment: can the shipped model configuration be beaten?

Two-stage gating, the most valuable practice carried over from the MLB build. A
variant that wins on the VALIDATION season must then REPLICATE on a separate
held-out season before it ships. The MLB build caught multiple mirages that way:
features that won by 0.6% on one year and reversed sign on the next.

This script is `exp_*` and stays in the repo whether or not anything ships. A
rejected experiment is a result.

Variants under test:
  base      the shipped configuration
  recency   exponential sample weighting by season age (half-life 3 seasons),
            motivated by the league being non-stationary: plays per game fell
            63.9 -> 61.3 and pass rate 0.613 -> 0.594 across the window
  bypos     one model per position instead of one pooled across WR/TE/RB
  deeper    more capacity (63 leaves, 800 iterations, lower learning rate)
  longwin   adds 16-game rolling windows to the feature set (via --extra-window)

Run:
    python scripts/exp_model_variants.py            # validation only
    python scripts/exp_model_variants.py --confirm  # + replicate on test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import features as F  # noqa: E402
from gridlib.cache import dc_path, read_parquet_or_none  # noqa: E402

VALID, TEST = 2023, 2024
TRAIN_END_VALID, TRAIN_END_TEST = 2022, 2023

# Representative targets: two volume counts, one yards, one that failed the
# gate. Enough to see a real effect without a 40-minute run.
PROBE_TARGETS = {
    "targets": ["WR", "TE", "RB"],
    "carries": ["RB", "QB", "FB"],
    "attempts": ["QB"],
    "receiving_yards": ["WR", "TE", "RB"],
    "receiving_tds": ["WR", "TE", "RB"],
}

BASE_PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                   min_samples_leaf=40, l2_regularization=1.0,
                   early_stopping=True, validation_fraction=0.12, random_state=0)
DEEP_PARAMS = dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=63,
                   min_samples_leaf=25, l2_regularization=1.0,
                   early_stopping=True, validation_fraction=0.12, random_state=0)

RECENCY_HALF_LIFE = 3.0  # seasons


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _loss_for(target: str) -> str:
    return "squared_error" if target.endswith("_yards") else "poisson"


def run_variant(feat: pd.DataFrame, target: str, positions: list[str],
                fold: int, train_max: int, variant: str) -> float | None:
    cols = F.feature_columns(feat)
    sub = feat[feat["position"].isin(positions) & feat[target].notna()]
    tr = sub[sub["season"] <= train_max]
    te = sub[sub["season"] == fold]
    if len(tr) < 500 or len(te) < 200:
        return None

    loss = _loss_for(target)
    params = DEEP_PARAMS if variant == "deeper" else BASE_PARAMS

    if variant == "bypos" and len(positions) > 1:
        preds = np.zeros(len(te))
        for pos in positions:
            m_tr = tr[tr["position"] == pos]
            m_te_mask = (te["position"] == pos).to_numpy()
            if m_tr.empty or not m_te_mask.any():
                continue
            if len(m_tr) < 300:
                # Too thin to fit alone; fall back to the pooled model.
                mdl = HistGradientBoostingRegressor(loss=loss, **params)
                mdl.fit(tr[cols], tr[target].astype(float))
            else:
                mdl = HistGradientBoostingRegressor(loss=loss, **params)
                mdl.fit(m_tr[cols], m_tr[target].astype(float))
            preds[m_te_mask] = np.clip(mdl.predict(te[cols][m_te_mask]), 0, None)
        return _mae(te[target].astype(float), preds)

    mdl = HistGradientBoostingRegressor(loss=loss, **params)
    if variant == "recency":
        age = train_max - tr["season"].to_numpy()
        w = 0.5 ** (age / RECENCY_HALF_LIFE)
        mdl.fit(tr[cols], tr[target].astype(float), sample_weight=w)
    else:
        mdl.fit(tr[cols], tr[target].astype(float))
    return _mae(te[target].astype(float), np.clip(mdl.predict(te[cols]), 0, None))


def build_features(pw: pd.DataFrame, lg: pd.DataFrame, fold: int,
                   extra_window: bool) -> pd.DataFrame:
    if extra_window:
        F.ROLL_WINDOWS = (3, 8, 16)
    else:
        F.ROLL_WINDOWS = (3, 8)
    priors = F.league_priors(pw, fold)
    return F.build(pw[pw["season"] <= fold], lg, priors)


def run_fold(pw, lg, fold: int, train_max: int, variants: list[str]) -> pd.DataFrame:
    rows = []
    feat_std = build_features(pw, lg, fold, extra_window=False)
    feat_long = None
    for target, positions in PROBE_TARGETS.items():
        base = run_variant(feat_std, target, positions, fold, train_max, "base")
        if base is None:
            continue
        rec = {"target": target, "base": base}
        for v in variants:
            if v == "base":
                continue
            if v == "longwin":
                if feat_long is None:
                    feat_long = build_features(pw, lg, fold, extra_window=True)
                m = run_variant(feat_long, target, positions, fold, train_max, "base")
            else:
                m = run_variant(feat_std, target, positions, fold, train_max, v)
            rec[v] = m
            rec[f"delta_{v}"] = 100.0 * (1 - m / base) if m else np.nan
        rows.append(rec)
        print(f"  {target:<20} " + "  ".join(
            f"{v}:{rec.get(f'delta_{v}', float('nan')):+5.2f}%"
            for v in variants if v != "base"), flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true",
                    help="replicate on the TEST season after validation")
    ap.add_argument("--variants", default="recency,bypos,deeper,longwin")
    args = ap.parse_args()
    variants = ["base"] + [v.strip() for v in args.variants.split(",")]

    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    if pw is None or lg is None:
        print("Backfill missing.")
        sys.exit(2)

    t0 = time.time()
    print(f"STAGE 1: validation season {VALID} (train <= {TRAIN_END_VALID})")
    print("  positive delta = variant improves on base\n")
    val = run_fold(pw, lg, VALID, TRAIN_END_VALID, variants)
    val["fold"] = "validation"

    print(f"\nvalidation mean delta by variant:")
    winners = []
    for v in variants:
        if v == "base":
            continue
        col = f"delta_{v}"
        if col in val.columns:
            mean = val[col].mean()
            print(f"  {v:<10} {mean:+.2f}%")
            if mean > 0:
                winners.append(v)

    frames = [val]
    if args.confirm and winners:
        print(f"\nSTAGE 2: CONFIRMATION on test season {TEST} "
              f"(train <= {TRAIN_END_TEST})")
        print(f"  confirming: {winners}\n")
        tst = run_fold(pw, lg, TEST, TRAIN_END_TEST, ["base"] + winners)
        tst["fold"] = "test"
        frames.append(tst)
        print("\ntest mean delta by variant:")
        for v in winners:
            col = f"delta_{v}"
            if col in tst.columns:
                v_mean, t_mean = val[col].mean(), tst[col].mean()
                verdict = ("REPLICATES" if t_mean > 0 else "MIRAGE, does not replicate")
                print(f"  {v:<10} validation {v_mean:+.2f}%  test {t_mean:+.2f}%"
                      f"  -> {verdict}")
    elif args.confirm:
        print("\nNo variant won on validation; nothing to confirm.")

    out = pd.concat(frames, ignore_index=True)
    Path("docs").mkdir(exist_ok=True)
    out.to_csv("docs/exp_model_variants.csv", index=False)
    print(f"\nwrote docs/exp_model_variants.csv   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
