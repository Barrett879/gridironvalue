# Deploying GridironValue

## Host: Streamlit Community Cloud (the shareable link)

Free, and the URL looks like `gridironvalue.streamlit.app`, the same setup as
DiamondValue.

**It has no persistent disk, and that is the one thing that shapes the design.**
Anything the app writes at runtime survives only until the next restart. For most
of the cache that is harmless, because projections regenerate on demand. It is
NOT harmless for the accuracy record: a board pasted on the live site would save,
freeze a snapshot stamped with whatever model that container loaded, show for a
while, and then vanish, leaving the published record silently different from the
one in this repository. Freezing exists precisely to stop the record drifting, so
a host that quietly loses it is worse than one that refuses to write.

So on this host the record is a COMMITTED artifact, not runtime state:

1. Paste the board LOCALLY. `save_lines` and `freeze_projections` run and write
   `cache/pp_lines_*.json` and `cache/pp_frozen_*.json`.
2. Commit both and push.
3. The deployment serves them read-only.

A side benefit worth having on a public link: a visitor cannot overwrite the
record from the site.

### Setting it up (needs a GitHub account; there is no gh CLI on this machine)

1. Create a **private** repo on github.com, e.g. `gridironvalue`.
2. `git remote add origin git@github.com:<you>/gridironvalue.git` (or the HTTPS
   URL), then `git push -u origin main`.
3. Go to share.streamlit.io, **New app**, pick the repo, branch `main`, main file
   `app.py`.
4. Under **Advanced settings**, set **Python 3.12**. Streamlit Cloud IGNORES
   `runtime.txt`, so this must be chosen in the UI. It matters: `scikit-learn`
   and `joblib` are pinned to the exact versions the artifacts in `models/` were
   saved with, and `predict.py` warns loudly on a mismatch.
5. In **Secrets**, add:

   ```
   GRIDIRONVALUE_READONLY = "1"
   ```

   Without this the site will accept pastes it cannot keep. `gridlib/cache.py`
   reads it; `props.save_lines` and `props.freeze_projections` become no-ops and
   the paste panel explains why instead of looking broken.

### Adding a week

```
# locally, with the app running
paste the board -> it saves and freezes
git add cache/pp_lines_*.json cache/pp_frozen_*.json
git commit -m "week N board"
git push
```

Streamlit Cloud redeploys on push. About 800KB per week.

## Alternative host: Render with a disk

`render.yaml` is kept and still correct. Choose it only if you want to paste
boards ON the live site rather than locally. It costs about $7/month because a
persistent disk requires a paid instance, and every push takes the site down for
roughly 45 seconds, so batch pushes. `cache.py` switches `CACHE_DIR` to
`/data/cache` when the disk is mounted, and `GRIDIRONVALUE_READONLY` is left
unset there so writes work normally.

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
