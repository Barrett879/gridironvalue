# The training backfill

Built 2026-08-31. Spec step 2.

Three artifacts, all version-stamped in the filename so they are invalidated by
a version bump and never edited in place:

| artifact | grain | rows | cols |
|---|---|---|---|
| `team_week_2016_2025_v1.parquet` | one row per (game, team) | 5,278 | 64 |
| `player_week_2016_2025_v1.parquet` | one row per (game, player) | 63,042 | 125 |
| `league_season_2016_2025_v1.parquet` | one row per season | 10 | 15 |

Rebuild:

```bash
.venv/bin/python scripts/build_team_week.py   --seasons 2016-2025
.venv/bin/python scripts/build_player_week.py --seasons 2016-2025
.venv/bin/python scripts/validate_backfill.py --seasons 2016-2025
.venv/bin/python scripts/validate_vs_espn.py  --season 2024
```

`build_team_week.py` must run first; the player builder joins its output.

## Why 2016 to 2025

2016 is the participation floor, and participation is the only source for routes
run. Ten seasons is also enough for walk-forward validation with room to spare:
train on 2016-2022, tune on 2023, test on 2024, and keep 2025 untouched as a
final hold-out.

Going back further would buy rows at the cost of era mismatch. The league is
non-stationary in exactly the direction that matters here: plays per game fell
from 63.9 to 61.3 and the pass rate from 0.613 to 0.594 across this window
alone.

## Row counts, and why they are right

- **Team-games: 5,278.** Not a round number, and it should not be. 2016-2020
  were 16-game seasons (512 team-games each); 2021 onward are 17 (544). 2022 has
  542 because the Bills-Bengals game was abandoned and never made up. The
  validator asserts this equals `2 x (played regular-season games)` from the
  schedule, so a drifting count fails loudly.
- **Player-games: 63,042** across QB/RB/FB/WR/TE/K.
- **Kicker-games: 5,273 against 5,278 team-games**, a ratio of 0.999. There is
  one kicker per team per game, so this is a strong independent check that no
  team-game is missing or duplicated.

## Two definitions that were verified, not assumed

Reconciling play-by-play against the official weekly box score:

| quantity | naive pbp column | exact formula | naive accuracy | exact accuracy |
|---|---|---|---|---|
| pass attempts | `pass_attempt` | `complete_pass + incomplete_pass + interception` | 75/544 | **544/544** |
| rush attempts | `rush_attempt` | same, **excluding two-point conversions** | 509/544 | **544/544** |

The naive versions are wrong by 2.47 and 0.07 per team-game. That is small enough
to never look like a bug and large enough to shift every player's share
denominator. Kneels are deliberately KEPT: the official box score counts a kneel
as a carry.

## Validation: two invariants, kept separate

Comparing our skill-position player sums directly against pbp team totals fails,
and not because of a bug: **punters throw on fake punts**. Across 2016-2025 that
is 93 pass attempts in 93 team-games, 84 of them by punters and the rest by
defenders on trick plays. Splitting the check tests each link on its own.

**Invariant A** (our pipeline): our player-week sums equal the nflverse box score
restricted to skill positions. Must be exact.

    pass attempts  5,278/5,278      targets       5,278/5,278
    rush attempts  5,278/5,278      receptions    5,278/5,278
    sacks taken    5,278/5,278      passing TDs   5,278/5,278
    rushing TDs    5,278/5,278      FG attempts   5,278/5,278

**Invariant B** (our pbp definitions): the full box score equals our pbp team
totals.

    sacks taken   5,278/5,278 = 100.00%      passing TDs  5,278/5,278 = 100.00%
    rushing TDs   5,278/5,278 = 100.00%
    pass attempts 5,275/5,278 =  99.94%      rush attempts 5,277/5,278 = 99.98%

The four residual team-games are official scoring adjustments and laterals, a
0.06% rate that is inherent to the source rather than to the pipeline.

## Cross-provider spot-check

`validate_backfill.py` can only prove the pipeline is self-consistent, because
every number in it comes from nflverse. `validate_vs_espn.py` closes that loop
against ESPN, which is sourced separately from nflverse/GSIS. One player per
position, 2024 season totals: **34 quantities checked, 0 mismatched.**

    Ja'Marr Chase   175 tgt / 127 rec / 1,708 yds / 17 TD
    Josh Allen      483 att / 307 cmp / 3,731 yds / 28 TD / 6 INT / 14 sacks
                    102 car / 531 yds / 12 TD
    Brandon Aubrey  40-47 FG, 30-30 XP
    Jonathan Taylor 303 car / 1,431 yds / 11 TD
    Brock Bowers    153 tgt / 112 rec / 1,194 yds / 5 TD

Pro-Football-Reference would be the other natural check and the spec suggests
it, but pro-football-reference.com sits behind Cloudflare bot verification.
That is a bot-detection control, so it is not worked around; a human can open
any of these lines in a browser to confirm.

## Exposure: three measures, never merged

| measure | source | coverage | available live? |
|---|---|---|---|
| `off_snaps`, `off_pct` | PFR snap counts | 99.8-100% | **yes**, updated several times daily |
| `routes_run_proxy` | participation | 93-99% (K 1%) | **no**, publishes after the postseason |
| `carries`, `targets` | box score | 100% | after the game |

`routes_run_proxy` counts plays the player was on the field for a team dropback.
Two checks say the derivation is right: the WR routes-to-snaps ratio is **0.608**,
which is the league pass rate, and kicker coverage is 1.2%, which is the rounding
error it should be.

**Rush attempts and routes are stored separately and must never be merged into
"touches."** They respond to game script with opposite signs, so merging cancels
the script signal and yields a model that looks fine on aggregate error while
being useless in exactly the blowout and comeback games where projections matter.

Our computed target share matches the feed's own at **corr 0.9983, mean absolute
difference 0.0037** across 50,272 player-games. That is an independent
confirmation the team pass-attempt denominator is right.

## The column contract, and why most of this table is unusable as features

`gridlib/columns.py` classifies **every** column into one of five buckets, and
`classify()` RAISES on an unknown column so a newly added field must be
classified before anything can train on the table.

| bucket | meaning |
|---|---|
| `key` | identifiers and the join spine |
| `target` | what the model predicts, from the box score |
| `same_game_exposure` | snaps, routes, shares for THIS game: outcomes, not inputs |
| `same_game_team` | the whole `off_*` / `def_allowed_*` / `realized_*` block |
| `pregame` | the only columns usable directly as features |
| `postgame_trap` | looks pregame, is not: `temp` and `wind` |

The `same_game_team` block is the dangerous one. `off_dropbacks` is how many
dropbacks the team actually had **in the game being projected**. It is
indispensable for deriving realized shares and for building lagged features, and
it is catastrophic as an inference feature because at prediction time it does not
exist. Nothing in the name says so, and a gradient booster handed it will post a
beautiful validation score and be useless in production.

Only 12 columns are directly safe: the market (`team_implied_total`,
`opp_implied_total`, `team_spread`, `spread_line`, `total_line`), the venue
(`roof_type`, `surface`), and the schedule (`is_home`, `rest_days`, `div_game`,
`gameday`, `weekday`). Everything else needs a lag or a rolling window applied
first, which is step 3's job.

## Findings from this step

**nflfastR's `xpass` model has drifted.** Its mean prediction is flat at ~0.627
in every season while the actual league pass rate fell from 0.613 to 0.594.
League-mean PROE is therefore not zero and slides monotonically:

    2016 -0.32   2018 -0.52   2020 -0.72   2022 -2.57   2024 -2.33   2025 -2.28

Left raw, a model learns that offset as "later seasons pass less", which is true
but entangles league drift with team tendency. Centering is the fix, but
centering on the current season's league mean leaks games that have not been
played. So the backfill stores raw PROE and writes per-season league means to
`league_season`; the feature layer decides how to center point-in-time safely.

**The participation release path is `pbp_participation/pbp_participation_{season}`,
not `participation/participation_{season}`.** The wrong path 404s, and the
fetch layer correctly treats a 404 as "not published yet", so it failed silently
and looked exactly like a preseason absence. All ten seasons load now.

**Participation did not degrade in 2023** despite the mid-stream source change
from NGS to FTN that the spec flagged: routes coverage is 88.8% that season
against an 87.0-88.8% range across the window.

## Known gaps, carried forward

- **Weather is unusable as a feature.** `temp` and `wind` are written only after
  a game is played. Classified as `postgame_trap`. Still an open decision.
- **Rolling windows must count games played, not weeks elapsed.** Every team has
  a bye, so 17 games span 18 weeks and "the last 4 weeks" silently becomes 3
  games. `tests/test_leakage.py` asserts the condition exists in the data.
- **Regular season only.** Postseason is excluded: week numbers 19-22 would
  corrupt week-indexed logic, and playoff fields are a selected sample.
- **Defensive players, offensive linemen and punters are not in the table**, by
  design. They are why invariant A and B differ.
