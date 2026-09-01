# scripts/

Naming discipline, enforced from day one so the directory stays readable at
thirty files instead of becoming a junk drawer.

| prefix | means | kept when it fails? |
|---|---|---|
| `build_*` | Produces a derived artifact on disk: a training table, a feature table, a predictions parquet, a cache. Idempotent, safe to re-run. | n/a |
| `train_*` | Fits model artifacts and writes them to `models/`. | n/a |
| `validate_*` / `backtest_*` | Scores an existing artifact against held-out data. Prints a verdict, writes no models. | n/a |
| `exp_*` | A research probe: does feature block X help? **Kept in the repo even when the answer is no.** | YES |

## Current scripts

| script | what it does |
|---|---|
| `build_team_week.py` | team-week aggregates from pbp (levels 1-2 of the decomposition) plus the per-season league-context table. Run FIRST. |
| `build_player_week.py` | the player-week training table (levels 3-4). Joins team-week, snaps, routes, per-player pbp opportunity, and game context. |
| `validate_backfill.py` | PASS/FAIL structural and reconciliation checks. Exits nonzero on failure. |
| `validate_vs_espn.py` | cross-provider spot-check of season totals against ESPN, one player per position. |
| `validate_uncertainty.py` | how wrong a projected stat line actually is: MAE as a share of the mean, p50/p90 absolute error, and how often the actual lands within 20% of the projection. Skill vs a baseline says the model beats an average; this says whether one number is worth acting on. |
| `validate_coherence.py` | do the projections COMPOSE? Team sums reported twice (full roster vs played-only) and concentration banded against an oracle. |
| `build_availability.py` | the shrunk P(appears \| position, depth rank) grid. |
| `validate_models.py` | walk-forward validation with the three mandatory baselines. Writes `docs/model_report_data.csv`. `--test` scores the test season; use once. |
| `exp_model_variants.py` | REJECTED: recency weighting, per-position models, more capacity, longer windows. All noise or worse. |
| `exp_redzone_block.py` | REJECTED: prior red-zone opportunity. Its first +13% was a missingness leak, documented in `docs/model_report.md`. |

## Why `exp_*` scripts stay after they fail

A rejected experiment is a result, and deleting it means the next person
re-runs it. DiamondValue has several documented rounds that ended "tested four
features, shipped none," and those write-ups are load-bearing. Every `exp_*`
script should leave a short verdict in `docs/` saying what was tried, what the
numbers were, and whether it shipped.

## Version-stamp derived files

Every derived cache file carries a version in its FILENAME
(`predictions_2026_w01_m1.parquet`). To invalidate, bump the version. Never edit
a derived file in place: on a host with a persistent disk, seeding is gap-fill
only, so a changed file with the same name never reaches production.

## Validation rules that apply to every script here

- **Random K-fold is forbidden.** It leaks at the player-season level. Use
  walk-forward: train on earlier seasons, tune on one validation season, test on
  a held-out season exactly once.
- **Two-stage feature gating.** A feature block that wins on the validation
  season must REPLICATE on a separate held-out season before it ships. The
  confirmation run must load identical history to the validator; a thin-history
  harness shipped two false confirmations in the MLB build.
- **Grant-wins column policy.** If a policy layer decides which columns a target
  trains on, a column is kept if ANY block grants it. Any-block-can-veto
  semantics silently stripped granted columns in the MLB build and produced
  byte-identical models with 0.00% deltas everywhere. A confirmation showing
  exactly zero change is a plumbing bug until proven otherwise.
- **Every target reports against three baselines**: league-average constant,
  the player's season-to-date per-game mean excluding the target game, and a
  shrunk multi-season rate times expected opportunities. A target that cannot
  beat baselines 2 and 3 out of sample is reported honestly, not quietly
  dropped.
