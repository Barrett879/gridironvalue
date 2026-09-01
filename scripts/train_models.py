"""Fit and persist the shipped model artifacts.

Trains on everything up to and including `--through` (default 2025, the last
completed season) and writes one joblib per target to `models/`, plus a
`registry.json` carrying the validation verdict for each target.

THE REGISTRY IS NOT DECORATION
------------------------------
Every artifact records whether its target PASSED or FAILED the ship gate in
`docs/model_report.md`, and what its measured skill was against the
season-to-date baseline. The UI reads this and refuses to present a failed
target as a projection. Two targets fail: receiving_tds and rushing_tds both
lose to a plain season-to-date average, so for those the registry marks the
BASELINE as the thing to show, not the model.

Field goals are a third case: they beat the player baselines but lose to a
league constant, so they are marked low-confidence.

PINS
----
scikit-learn and joblib versions are stamped into the registry. Loading an
artifact under a different scikit-learn is the silent-breakage path the spec
warns about, so `predict.py` checks the stamp and warns loudly rather than
returning quietly wrong numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import features as F  # noqa: E402
from gridlib.cache import dc_path, logger, read_parquet_or_none  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VERSION = "m1"

RECEIVERS = ["WR", "TE", "RB"]
RUSHERS = ["RB", "QB", "FB"]

# target -> (positions, loss, gate verdict, measured skill vs baseline 2 on the
# 2024 test season). Verdicts come from docs/model_report.md and are stamped
# into the registry so the UI cannot present a failed target as a projection.
SPECS = {
    "targets":               (RECEIVERS, "poisson", "pass", 6.4),
    "receptions":            (RECEIVERS, "poisson", "pass", 5.9),
    "receiving_yards":       (RECEIVERS, "squared_error", "pass", 5.1),
    "receiving_tds":         (RECEIVERS, "poisson", "FAIL", -4.6),
    "carries":               (RUSHERS,   "poisson", "pass", 9.5),
    "rushing_yards":         (RUSHERS,   "squared_error", "pass", 5.6),
    "rushing_tds":           (RUSHERS,   "poisson", "FAIL", -0.9),
    "attempts":              (["QB"],    "poisson", "pass", 12.5),
    "completions":           (["QB"],    "poisson", "pass", 11.0),
    "passing_yards":         (["QB"],    "squared_error", "pass", 13.2),
    "passing_tds":           (["QB"],    "poisson", "pass", 4.1),
    "passing_interceptions": (["QB"],    "poisson", "weak", 1.8),
    "sacks_suffered":        (["QB"],    "poisson", "pass", 4.7),
    "fg_att":                (["K"],     "poisson", "weak", 11.3),
    "fg_made":               (["K"],     "poisson", "weak", 9.4),
    "pat_att":               (["K"],     "poisson", "pass", 6.2),
}

PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, random_state=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", type=int, default=2025,
                    help="last season to train on (default 2025)")
    args = ap.parse_args()

    pw = read_parquet_or_none(dc_path("player_week_2016_2025_v1.parquet"))
    lg = read_parquet_or_none(dc_path("league_season_2016_2025_v1.parquet"))
    if pw is None or lg is None:
        print("Backfill missing. Run build_team_week.py then build_player_week.py.")
        sys.exit(2)

    # Priors from seasons strictly before the first season we would predict.
    priors = F.league_priors(pw, args.through + 1)
    feat = F.build(pw[pw["season"] <= args.through], lg, priors)
    cols = F.feature_columns(feat)
    MODELS_DIR.mkdir(exist_ok=True)

    registry = {
        "version": VERSION,
        "trained_through": args.through,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "feature_columns": cols,
        "league_priors": priors,
        "targets": {},
    }

    for target, (positions, loss, verdict, skill) in SPECS.items():
        if target not in feat.columns:
            logger.warning("target %s not in the table; skipped", target)
            continue
        sub = feat[feat["position"].isin(positions) & feat[target].notna()]
        if len(sub) < 500:
            print(f"  {target:<24} SKIPPED (only {len(sub)} rows)")
            continue
        mdl = HistGradientBoostingRegressor(loss=loss, **PARAMS)
        mdl.fit(sub[cols], sub[target].astype(float))
        path = MODELS_DIR / f"{target}_{VERSION}.joblib"
        joblib.dump(mdl, path)
        registry["targets"][target] = {
            "file": path.name, "positions": positions, "loss": loss,
            "gate": verdict, "skill_vs_baseline_pct": skill,
            "n_train": int(len(sub)),
            # What the UI should present for this target.
            "present_as": ("model" if verdict == "pass"
                           else "baseline" if verdict == "FAIL"
                           else "model_low_confidence"),
        }
        flag = {"pass": "", "weak": "  (low confidence)",
                "FAIL": "  (FAILS the gate; show the baseline instead)"}[verdict]
        print(f"  {target:<24} {len(sub):>6} rows -> {path.name}{flag}")

    (MODELS_DIR / f"registry_{VERSION}.json").write_text(
        json.dumps(registry, indent=2))
    print(f"\nwrote models/registry_{VERSION}.json "
          f"({len(registry['targets'])} targets, {len(cols)} features)")
    print(f"sklearn {sklearn.__version__}, joblib {joblib.__version__} stamped in.")


if __name__ == "__main__":
    main()
