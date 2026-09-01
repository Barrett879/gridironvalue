# Data layer: what was verified, and what the spec got wrong

Everything below was checked live on **2026-08-31** against nflreadpy 0.1.5 and
the nflverse release parquets. Re-verify before relying on it.

The spec's Section 2 was accurate on the big things. Four items need correcting
or qualifying, and two of them change the model design.

---

## 1. `nflreadpy` hardcodes a season range, and the range is stale

This is the one that would have bitten us in week 1.

```
load_pbp([2026])            -> ValueError: Season must be between 1999 and 2025
load_injuries([2026])       -> ValueError: Season must be between 2009 and 2025
load_snap_counts([2026])    -> ValueError: Season must be between 2012 and 2025
load_rosters_weekly([2026]) -> ValueError: Season must be between 2002 and 2025
```

The bound is baked into the installed version, not read from the server. When
nflverse publishes the 2026 files in September, a pinned `nflreadpy` will still
refuse to load them until a new release ships and we bump the pin. For a live
weekly pipeline that is a guaranteed in-season outage on a schedule nobody
controls, and the failure mode is an exception on the main path.

`load_schedules` and `load_depth_charts` do NOT have the problem: depth charts
already returned 487,912 rows of 2026 data.

**What we do:** `gridlib/fetch.py` reads the release parquets directly over
HTTPS and keeps `nflreadpy` only for the calls whose bounds are harmless. Our
own coverage floors live in `fetch.COVERAGE` and have no upper bound. The spec
already suggested direct parquet reads for caching reasons; this is the hard
reason.

## 2. Wind and temperature are POST-GAME fields, not forecast fields

The spec says "Wind is the dominant effect on passing and is already in
`games.parquet`." That is true for training and false for prediction.

| season | rows | `wind` populated | `temp` populated |
|---|---|---|---|
| 2025 | 285 | 190 | 190 |
| 2026 | 272 | **0** | **0** |

The 190 of 285 in 2025 is the outdoor games; domes are null by design. The zero
in 2026 is the point: weather is written after the game is played. **There is no
wind forecast in `games.parquet` for an upcoming game.**

**Consequence:** wind can be a training feature but cannot be an inference
feature without an external forecast source. Either drop it from the shipped
model, or add a weather API and accept a second live dependency. This has to be
decided before feature work, because a feature that exists at training time and
not at prediction time is the classic silent-degradation bug.

## 3. `roof` conflates structure with per-game state

For the five retractable-roof stadiums, `roof` is an **empty string** until the
game is played, then becomes `closed` or `open`.

| season | outdoors | dome | closed | open | empty |
|---|---|---|---|---|---|
| 2023 | 188 | 53 | 33 | 11 | 0 |
| 2024 | 187 | 53 | 45 | 0 | 0 |
| 2025 | 193 | 50 | 42 | 0 | 0 |
| 2026 | 177 | 52 | 0 | 0 | **43** |

So roof state is also unknown at prediction time. The base rate is lopsided
enough that "assume closed" is defensible (0 open in each of the last two
seasons, 11 in 2023), but it must be a stated assumption, not a silent NaN.

**What we do:** `data/stadium_corrections.csv` carries `roof_type`, the
STRUCTURAL fact (`outdoors` / `dome` / `retractable`), keyed on `stadium_id`.
`fetch.apply_stadium_corrections` overlays it.

## 4. Stadium names drift; `stadium_id` does not

The `stadium` string is re-sponsored season to season, and the 2026 file
actually regressed one: it calls Houston's venue **"Reliant Stadium"**, a name
retired in 2014, while the 2025 file correctly says "NRG Stadium".

Renames seen between the 2025 and 2026 files: New Era Field to Highmark,
FirstEnergy to Huntington Bank Field, TIAA Bank to EverBank, Mercedes-Benz
Superdome to Caesars Superdome, FedExField to Northwest Stadium, and NRG
backwards to Reliant.

**Never join or key on the stadium name.** `stadium_id` (`HOU00`, `BUF00`) is
stable. Display names are curated in `data/stadium_corrections.csv`.

Three international venues also carry a wrong `roof`: the feed calls the
open-air Melbourne Cricket Ground, Stade de France, and Allianz Arena "dome".
All three are corrected.

## 5. Depth charts have no `week` column from 2025 onward

Confirmed for both 2025 and 2026: the columns are `dt`, `team`, `player_name`,
`espn_id`, `gsis_id`, `pos_grp_id`, `pos_grp`, `pos_id`, `pos_name`, `pos_abb`,
`pos_slot`, `pos_rank`. There is no `week` at all; `dt` is an ISO8601 timestamp
(`2026-08-31T14:44:50Z`).

`fetch.load_depth_charts` always ADDS a `week` column, derived by mapping each
snapshot timestamp to the week whose first kickoff is the next one at or after
it. Code downstream can assume `week` exists.

## 6. `get_current_season()` is not usable for "which board do we show"

`nflreadpy.get_current_season()` returned **2025** on 2026-08-31, ten days
before the 2026 opener, and `get_current_week()` returned 22. The schedule file
is authoritative, so `fetch.current_season` / `fetch.current_week` resolve it
from the games themselves. The rollover rule is "the latest season whose first
regular-season kickoff is within 45 days," which flips to the new season in late
July when rosters and schedules are real.

---

## Confirmed as the spec described

- **Free Vegas lines for upcoming games.** `schedules/games.parquet` holds all
  seasons 1999-2026 in one 7,548-row file. For 2026: 272 rows, **112 already
  carrying both `spread_line` and `total_line`**, zero scores recorded. It is
  simultaneously the schedule, the market, the rest situation, and the
  cross-reference id table (`espn`, `pfr`, `pff`, `ftn`, `gsis`).
- **`spread_line` is from the home perspective**, positive means home favored.
  Verified against the implied totals it produces and pinned in
  `tests/test_schedule.py`.
- **Play-by-play carries every column the model design needs**, including
  `xpass` and `pass_oe` (PROE), `vegas_wp`, `yardline_100`, and the per-play
  actor ids. All 29 columns checked were present in the 2025 file.
- **Participation exists for completed seasons only.** 2025 has 45,184 rows with
  `offense_players` per play, so routes run is derivable for training. It does
  not update in season, so snap counts remain the live proxy, exactly as the
  spec warned.
- **Weekly player stats carry full kicking detail**: `fg_made` by distance band
  (0-19 through 60+), `fg_att`, `pat_made`, `gwfg_att`, plus the made and missed
  distance lists. Kickers are well supported by the source data.
- **2026 injuries do not exist yet**, as expected in the preseason window.

## Open question for the model design

The 2026 Week 1 opener is listed as **Wednesday 2026-09-09, NE at SEA**, with
the Melbourne game on Thursday. A Wednesday opener is unusual enough to be worth
a sanity check against a second source before anything depends on it. The
`window` label renders it as "Wednesday" rather than forcing it into the
Thursday-night bucket, so a data fix would not require a code change.
