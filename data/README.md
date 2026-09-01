# data/

Hand-curated CSVs that OVERRIDE the scraped feeds. They are created early, with
headers, even while empty, so there is never a moment where the only way to fix
a bad upstream value is to edit code.

Rule: a correction file is authoritative over the feed for the fields it names,
and every row carries `verified_on` plus a `note` saying how it was checked. A
correction with no note is not a correction, it is a guess.

| file | keyed on | what it overrides |
|---|---|---|
| `stadium_corrections.csv` | `stadium_id` | venue display name and roof type. Populated: the feed's stadium NAME drifts between seasons (the 2026 file calls NRG Stadium "Reliant Stadium", a name retired in 2014) while `stadium_id` is stable, and several roof values are wrong or blank. |
| `game_corrections.csv` | `game_id` + `field` | any single field on one game (kickoff time, market line, a swapped home/away). Empty. |
| `player_corrections.csv` | `gsis_id` + `field` | player attributes: position, team, name spelling. Empty. |
| `availability_overrides.csv` | `season` + `week` + `gsis_id` | forces a player's availability when the injury feed is wrong or has not updated. This is the manual escape hatch for the Saturday and Sunday-morning Questionable-to-Out downgrades that the Friday report does not carry. Empty. |

## Why roof matters enough to curate

`games.parquet` encodes a retractable roof as an EMPTY string until the game has
been played, then writes `closed` or `open`. So for the five retractable-roof
stadiums the roof state is a post-game field and is unavailable at prediction
time. The base rate is lopsided (2024: 45 closed and 0 open; 2025: 42 closed and
0 open; 2023: 33 closed and 11 open), so "assume closed" is a defensible default,
but it has to be a stated assumption rather than a silent NaN.

`roof_type` here is the STRUCTURAL fact (outdoors, dome, retractable), which
never changes game to game, as opposed to the feed's `roof`, which mixes
structure and per-game state in one column.
