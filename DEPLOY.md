# Deploying GridironValue

## What the running app needs

Committed, about 11MB total:

- `models/` — 16 joblib artifacts plus `registry_m1.json`
- `cache/player_week_2016_2025_v1.parquet` — the 63k-row training table. The
  feature builder needs a player's real history to project him, and a fresh host
  cannot rebuild that inside a web request.
- `cache/league_season_2016_2025_v1.parquet`, `availability_2025_2025_v1.parquet`
- `data/*.csv` — corrections and the per-stat reliability tiers

NOT committed, about 100MB, rebuildable from nflverse in minutes:

- raw play-by-play, participation, weekly stats, snap counts

Everything else (schedule, depth charts, injuries) is fetched live and cached,
with stale-beats-empty on every fetcher.

## The choice that actually matters: persistence

**Pasted PrizePicks boards are written to disk.** There is no other copy.

- **Render with a disk** — boards survive restarts. `cache.py` switches
  `CACHE_DIR` to `/data/cache` automatically when `/data` exists, so this needs
  no code change. `render.yaml` declares a 1GB disk.
- **Streamlit Community Cloud** — no persistent storage. Every restart loses
  every pasted board, and the week-by-week record resets with them.

Since the whole model-versus-market feature depends on those pastes accumulating
week over week, this is the deciding factor rather than a detail.

## Two known traps

**Streamlit Community Cloud ignores `runtime.txt`** and builds on the newest
Python. That forced two coupled dependency bumps in the DiamondValue build. If
deploying there, resolve `requirements.txt` in a clean venv on their Python
first, and expect `pyarrow` and `streamlit` to need moving.

**Every push to Render is a brief hard-down** (about 45 seconds), because a
service with a disk cannot do a zero-downtime swap. Batch pushes; do not
deploy-spam.

## scikit-learn is pinned for a reason

`registry_m1.json` stamps the exact scikit-learn and joblib versions the
artifacts were saved with, and `predict.load_registry()` warns loudly on a
mismatch. `joblib.load` can break silently across versions, so do not bump
either without retraining and re-committing the artifacts.

## Rebuilding the artifacts

```bash
python scripts/build_team_week.py    --seasons 2016-2025
python scripts/build_player_week.py  --seasons 2016-2025
python scripts/build_availability.py --seasons 2025-2025
python scripts/train_models.py       --through 2025
python scripts/validate_backfill.py  --seasons 2016-2025
python scripts/validate_coherence.py --season 2025 --weeks 4-8
```

Then commit the four cache artifacts and `models/`. Note the `/data` seeding
rule: on a host with a persistent disk, seeding is gap-fill only, so a CHANGED
file with the SAME name never reaches production. Bump the filename version.
