"""Disk-cache toolkit for GridironValue.

Ported from DiamondValue's mlblib/cache.py, which was itself ported from
HoopsValue. The invariants are unchanged and non-negotiable:

  - CACHE_DIR uses the host's persistent disk (/data/cache) when mounted, else a
    repo-local ./cache that is wiped on ephemeral restarts.
  - seed_disk_cache_from_repo() gap-fills the persistent disk from the committed
    repo snapshot ONCE per process at import, never clobbering fresher files.
    Shipping a CHANGED committed cache file therefore requires a filename
    version bump (_vN) or the disk copy wins forever.
  - Every write is atomic (tmp sibling + os.replace) so a process killed
    mid-write (deploy SIGTERM, a one-shot script exiting) can never leave a
    truncated file whose fresh mtime fools the stale-beats-empty logic.

What is NEW here is `dc_fresh`, rewritten a second time for football. MLB keys
freshness on a game DATE; the NFL board is one week with staggered locks, so a
"date" is the wrong unit. This version keys on KICKOFF TIME, which is what
actually determines whether a row can still change.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pickle
import shutil
import threading
import time
from pathlib import Path

import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
# One named logger so cache misses and fetch failures never disappear silently.
# Set GRIDIRONVALUE_LOG=DEBUG for verbose output.
logger = logging.getLogger("gridironvalue")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[gridironvalue] %(levelname)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(os.environ.get("GRIDIRONVALUE_LOG", "WARNING").upper())

# ── CACHE_DIR ────────────────────────────────────────────────────────────────
_RENDER_DISK = Path("/data/cache")
_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _RENDER_DISK if _RENDER_DISK.parent.exists() else _REPO_ROOT / "cache"


# ── Read-only deployments ────────────────────────────────────────────────────
# On a host with no persistent disk (Streamlit Community Cloud), anything the
# app writes at runtime survives only until the next restart. For most of the
# cache that is harmless: projections regenerate on demand.
#
# It is NOT harmless for the accuracy record. A board pasted on the live site
# would save, freeze a snapshot stamped with whatever model was loaded, show for
# a while, and then vanish, leaving the deployed record silently different from
# the canonical one in the repo. The whole value of freezing is that the record
# cannot drift.
#
# So a deployment declares itself read-only and the record becomes a COMMITTED
# artifact: boards are pasted locally, frozen locally, and pushed. The live site
# serves them and refuses to write. Set GRIDIRONVALUE_READONLY=1 in the host's
# secrets. Unset locally, which is where pasting is meant to happen.
READ_ONLY = os.environ.get("GRIDIRONVALUE_READONLY", "").strip().lower() in (
    "1", "true", "yes", "on")


def seed_disk_cache_from_repo() -> None:
    """Copy the committed repo cache into the persistent disk for any files the
    disk does not already have. Gap-fill only, best-effort, never fatal.

    Because it never overwrites, a changed committed file with the same name
    never reaches an already-seeded disk: bump the filename version (_vN).
    """
    repo_cache = _REPO_ROOT / "cache"
    if CACHE_DIR == repo_cache or not repo_cache.is_dir():
        return  # local/dev already reads the repo cache directly
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        have = set(os.listdir(CACHE_DIR))
        copied = 0
        for src in repo_cache.iterdir():
            if src.is_file() and src.name not in have:
                shutil.copy2(src, CACHE_DIR / src.name)
                copied += 1
        if copied:
            logger.info("seeded %d cache files from repo -> %s", copied, CACHE_DIR)
    except Exception as e:  # noqa: BLE001 — seeding is best-effort, never fatal
        logger.warning("disk-cache seed skipped: %s", e)


seed_disk_cache_from_repo()


# ── Path + freshness ─────────────────────────────────────────────────────────
def dc_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


# Inactives drop at kickoff minus 90 minutes. Anything keyed to a game is
# volatile right up to that moment and effectively frozen a few hours after the
# final whistle. These constants are the whole freshness policy.
LOCK_MINUTES_BEFORE_KICKOFF = 90
_TTL_PREGAME_FAR = 6 * 3600     # kickoff is days away: news moves slowly
_TTL_PREGAME_NEAR = 900         # inside 24h: practice reports and downgrades
_TTL_PREGAME_LOCK = 300         # inside the 90-minute window: inactives land
_TTL_IN_PROGRESS = 120          # live box score
_TTL_POSTGAME_GRACE = 3600      # stat corrections land for a while after
_GRACE_HOURS_AFTER_KICKOFF = 12  # after this a completed game is immutable
_TTL_DEFAULT = 3600


def dc_fresh(
    path: Path,
    kickoff: dt.datetime | None = None,
    now: dt.datetime | None = None,
    ttl: int | None = None,
) -> bool:
    """Freshness rule for a kickoff-keyed sports site.

    Priority:
      1. Explicit `ttl` (seconds) always wins.
      2. If `kickoff` is given, the TTL tightens as kickoff approaches and the
         file becomes IMMUTABLE once the game is comfortably over. The ramp is
         6h -> 15min (inside a day) -> 5min (inside the 90-minute inactives
         window) -> 2min (in progress) -> 1h (post-game grace) -> forever.
      3. No kickoff context: a flat 1h TTL.

    `kickoff` and `now` are timezone-aware datetimes; a naive `kickoff` is
    assumed to already be in the same frame as `now`. Filename version bumps
    (_vN) are how we invalidate immutable past files, never edits in place.
    """
    if not path.exists():
        return False
    if ttl is not None:
        return (time.time() - path.stat().st_mtime) < ttl
    if kickoff is None:
        return (time.time() - path.stat().st_mtime) < _TTL_DEFAULT

    _now = now or dt.datetime.now(dt.timezone.utc)
    if kickoff.tzinfo is None and _now.tzinfo is not None:
        kickoff = kickoff.replace(tzinfo=_now.tzinfo)
    elif kickoff.tzinfo is not None and _now.tzinfo is None:
        _now = _now.replace(tzinfo=kickoff.tzinfo)

    age = time.time() - path.stat().st_mtime
    hours_to_kick = (kickoff - _now).total_seconds() / 3600.0

    if hours_to_kick > 24:
        return age < _TTL_PREGAME_FAR
    if hours_to_kick > (LOCK_MINUTES_BEFORE_KICKOFF / 60.0):
        return age < _TTL_PREGAME_NEAR
    if hours_to_kick > 0:
        return age < _TTL_PREGAME_LOCK
    # Kickoff has passed. An NFL game runs a bit over 3 hours.
    hours_since_kick = -hours_to_kick
    if hours_since_kick < 3.5:
        return age < _TTL_IN_PROGRESS
    if hours_since_kick < _GRACE_HOURS_AFTER_KICKOFF:
        return age < _TTL_POSTGAME_GRACE
    return True  # immutable


def lock_time(kickoff: dt.datetime) -> dt.datetime:
    """The moment inactives are published for a game: kickoff minus 90 minutes.

    This is computed PER GAME on purpose. A wall-clock Sunday-morning cutoff is
    wrong: it is ~11:30am ET for the 1pm window, ~6:50pm for SNF, and lands on
    Thursday, Saturday or Monday for those slates.
    """
    return kickoff - dt.timedelta(minutes=LOCK_MINUTES_BEFORE_KICKOFF)


# ── Atomic read/write helpers ────────────────────────────────────────────────
def _tmp_sibling(path: Path) -> Path:
    return path.with_suffix(path.suffix + f".tmp{os.getpid()}-{threading.get_ident()}")


def atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet atomically (tmp file + os.replace on the same fs)."""
    tmp = _tmp_sibling(path)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        logger.warning("parquet write failed for %s: %s", path, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def read_parquet_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001 — a truncated/corrupt file is not fatal
        logger.warning("parquet read failed for %s: %s", path, e)
        return None


def pkl_load(path: Path):
    return pickle.loads(path.read_bytes())


def pkl_save(path: Path, obj) -> None:
    tmp = _tmp_sibling(path)
    try:
        tmp.write_bytes(pickle.dumps(obj))
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        logger.warning("cache write failed for %s: %s", path, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def json_load(path: Path):
    return json.loads(path.read_text())


def json_save(path: Path, obj) -> None:
    tmp = _tmp_sibling(path)
    try:
        tmp.write_text(json.dumps(obj))
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        logger.warning("cache write failed for %s: %s", path, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
