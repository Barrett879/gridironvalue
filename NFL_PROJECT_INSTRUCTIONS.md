# Build instructions: an NFL per-game player projection site

You are building this for Barrett. Read this whole document before writing code.

This is the third site in a family. Two already exist and work:
- **HoopsValue** (NBA contract values) at `Desktop/Claude/nba-value-app`
- **DiamondValue** (MLB per-game player projections) at `Desktop/Claude/MLB Predict`, live at diamondvalue.streamlit.app

DiamondValue is the direct template: same idea, different sport. Its spec is at
`Desktop/Claude/MLB Predict/MLB_PROJECT_INSTRUCTIONS.md` and its code is ~8,900 lines of working
Python. **Read that spec and skim that codebase before starting.** Most of Sections 1, 4.3-4.5, and
6-7 below are lifted from it because they were paid for in real debugging time.

Working title: **GridironValue** (parallel to HoopsValue). Barrett has not settled on a name and was
mid-discussion about renaming the MLB site too, so ask him before stamping a name into the UI. Keep
the name in ONE constant so a rename is a one-line change.

---

## 0. What you are building

A site where you pick an NFL week, see every game, click a game, and get per-game expected stat
lines for every projected-to-play player on both teams, plus an accuracy page that scores past
predictions honestly, plus an optional model-vs-market comparison against posted prop lines.

**Every number displayed is a distribution mean, not a prediction of what will happen.** That framing
is non-negotiable and appears in the UI. This is an informational analytics site. It is not a
sportsbook, it accepts no wagers, and it gives no personalized betting advice.

### The canonical stat table (single source of truth; every other section defers to this)

Predict per player per game. Group by position because the exposure differs (see Section 3).

| Position | Exposure (model first) | Rate/volume targets | Derived |
|---|---|---|---|
| QB | team dropbacks; designed rush attempts | pass attempts, completions, pass yards, pass TD, INT, sacks taken, rush attempts, rush yards, rush TD | completion %, yards/attempt |
| RB | rush attempts AND routes run (**separate, never merged**) | rush yards, rush TD, targets, receptions, receiving yards, receiving TD | yards/carry, catch rate |
| WR/TE | routes run | targets, receptions, receiving yards, receiving TD, air yards | catch rate, yards/target, target share |
| K | FG attempts | FG made, XP made, kicking points | FG% |

Fantasy points are a derived composite, computed from the components, never modeled directly.
If you add PrizePicks' Fantasy Score, use THEIR scoring: 0.04/pass yd, 0.1/rush+rec yd, 1.0 PPR,
4 pass TD, 6 rush/rec TD, -1 INT, -1 fumble lost. Note that composing it from component means
ignores covariance between components, so the composite's variance is not the sum of the parts'.

### What v1 deliberately does NOT predict, and why (show Barrett this list before starting)

- **Longest reception / longest rush / longest completion / longest FG.** These are *maxima over
  plays*, not means. `E[max]` is not a function of `E[sum]`; a mean-projection model literally cannot
  price them. Doing them right needs extreme-value fitting or play-level simulation. Scope them out
  of v1 explicitly rather than silently feeding a mean projection into a longest-X comparison, which
  produces confidently wrong numbers.
- **"Rush yards in first 5 attempts" / "receiving yards in first 2 receptions."** Sequence-conditional,
  not full-game stats. They need their own model and grading logic and will break a generic pipeline.
- **Defensive player stats (tackles, sacks, INTs).** Possible later; the opportunity model is
  completely different and v1 has enough surface area.
- **Individual offensive linemen.** No meaningful per-game stat line.

### Honest expectations (put a version of this on the About page, in Barrett's voice, no hype)

Football projection is *much* harder than baseball projection, and you should tell Barrett so before
he judges the output. A full 17-game NFL season has fewer player-events than a quarter of an MLB
season. Volume carries nearly all the signal; efficiency is close to noise. Expect the honest result
to be "volume stats beat the baselines clearly, TDs and efficiency barely beat them at all."

---

## 1. Football realities that break the MLB playbook (read this twice)

These are the five things that make this NOT a find-and-replace of DiamondValue.

**1. Exposure is endogenous.** In baseball, plate appearances are nearly deterministic given a lineup
slot. In football, a player's opportunity depends on game script (score state), which depends on the
game outcome you are also trying to project. This is circular. Resolve it explicitly: use the market's
implied team total and spread as an **exogenous** instrument for expected script. Never train on
realized in-game win probability, which leaks the outcome.

**2. The decomposition is four levels, not two.**
`team plays -> pass/run split -> player share -> player rate`
Use nflfastR's `xpass`/PROE to separate coach tendency from score-driven script. Position notes:
- **QB has no player-share term.** A healthy starter is at ~100% of dropbacks. QB exposure is a pure
  *team* quantity, so QB projections inherit team-level uncertainty directly and need a separate
  P(starts) model driven by depth chart and injury status.
- **Never merge RB "touches."** Rush attempts and routes/targets respond to game script with
  *opposite* signs: leading teams run more and throw less. Merging them cancels the game-script
  signal and yields a model that looks fine on aggregate MAE while being useless in exactly the
  blowout and comeback games where projections matter.
- **RB and TE target shares are negatively correlated within an offense.** Shares are compositional;
  they should sum sensibly across an offense rather than being modeled fully independently.

**3. Availability is a first-class model, not an afterthought.** NFL rosters have far more binary
availability events than MLB. Build `P(plays) x E[production | plays]`, not a single conditional
mean. A model that quietly assumes everyone plays will produce confidently wrong lines every week.
Inactives drop **90 minutes before kickoff**, computed per game as kickoff minus 90 minutes (NOT a
wall-clock Sunday-morning cutoff: that is ~11:30am ET for the 1pm window, ~6:50pm for SNF, and falls
on Thursday/Saturday/Monday for those slates). Teams dress at most 48 of 53.

**4. Efficiency metrics are traps.** A gradient booster given yards-per-attempt will select it and
will not generalize. Published year-over-year R^2 (Sharp Football): WR targets/game 0.585 vs WR
yards/target 0.0395 and TD rate 0.0155; RB touches/game 0.567 vs rush TD/game 0.229; QB rush yards
0.553 vs Y/A 0.092. **Yards per carry is the single biggest trap**: it needs ~1,978 same-team carries
to stabilize, which no career supplies; an 8-game 5.00 YPC regresses to ~4.37. Either exclude
efficiency features or shrink them hard toward league/scheme baselines before they reach the model.
Treat published stability numbers as optimistic upper bounds, since nearly every such study filters
to "10+ games in both seasons," which is survivorship selection on exactly the fragile,
committee, and team-changing players a production system must project.

**5. Use per-game forms, never season totals.** Seasons through 2020 were 16 games and 2021+ are 17,
and totals conflate rate with games played. The evidence is direct: WR targets/game R^2 = 0.585 vs
targets/season 0.478. Also: each team has a bye, so a player has 17 games across 18 weeks. Any
week-indexed pipeline needs explicit bye handling, and every rolling window must count **games
played**, not weeks elapsed.

Two more worth knowing: league-wide passing environment is non-stationary and directional (WR
targets per game fell ~8.7% from 2017 to 2025 while QB rushing rose), so pooling many seasons without
era terms or recency weighting bakes in a stale environment. And the widely-copied 20-80%
win-probability garbage-time filter discards far too much data for a 17-game sport; use 5-95%, or
better, keep the plays and include win probability as a feature.

---

## 2. Data layer

**Everything below was verified live on 2026-08-31. Re-verify before relying on it.** The NFL season
kicks off ~Sept 10, 2026, so you are building in the preseason window: 2025 is complete, 2026 files
mostly do not exist yet. That is expected, not a bug.

### The package: `nflreadpy`, NOT `nfl_data_py`

`nfl_data_py` is **deprecated and archived** (repo archived 2025-09-25, PyPI frozen at 0.3.3 from
2024-09-20). Its own README says users "are encouraged to switch immediately." Nearly every tutorial
and LLM answer still uses it. Use **`nflreadpy`** (PyPI 0.1.5, requires Python >= 3.10).

**It returns Polars DataFrames, not pandas.** Call `.to_pandas()` at the boundary and work in pandas
everywhere else, so the rest of the codebase matches DiamondValue.

```python
import nflreadpy as nfl
pbp = nfl.load_pbp([2021, 2022]).to_pandas()
```

Verified load functions and coverage: `load_pbp` (1999+), `load_player_stats`, `load_team_stats`,
`load_schedules`, `load_teams`, `load_players`, `load_rosters` (1920+), `load_rosters_weekly` (2002+),
`load_snap_counts` (2012+), `load_nextgen_stats` (2016+), `load_ftn_charting` (2022+),
`load_participation` (2016+), `load_injuries` (2009+), `load_depth_charts` (2001+),
`load_pfr_advstats` (2018+), `load_officials` (2015+), `load_ff_opportunity` (2006+), `load_contracts`.

You can bypass the package and read parquet directly, which is often better for a cached pipeline:
`https://github.com/nflverse/nflverse-data/releases/download/{tag}/{file}.parquet`
Verified paths: `pbp/play_by_play_{season}.parquet`, `stats_player/stats_player_week_{season}.parquet`,
`injuries/injuries_{season}.parquet`, `depth_charts/depth_charts_{season}.parquet`,
`snap_counts/snap_counts_{season}.parquet`, `nextgen_stats/ngs_{passing|receiving|rushing}.parquet`,
`pfr_advstats/advstats_week_{pass|rush|rec|def}_{season}.parquet`,
`weekly_rosters/roster_weekly_{season}.parquet`, `players/players.parquet`, `schedules/games.parquet`.

### The headline finding: free Vegas lines for UPCOMING games

`schedules/games.parquet` (one file, all seasons 1999-2026, ~7,500 rows) carries **`spread_line` and
`total_line` for games that have not been played yet**, refreshed roughly every 5 minutes in season.
Verified 2026-08-31: 272 rows for season 2026, 112 already carrying both lines, zero scores recorded.

It also carries `away_moneyline`, `home_moneyline`, over/under odds, `temp`, `wind`, `roof`, `surface`,
`div_game`, `away_rest`/`home_rest`, `referee`, `stadium`, and cross-reference IDs (`espn`, `pfr`,
`pff`, `ftn`, `gsis`). This one file is your schedule, your market data, your weather, and your rest
situation. **`spread_line` is from the HOME team's perspective; positive means home favored.**

Derive implied team totals as `total/2 ± spread/2`. But calibrate expectations: game totals have
RMSE ~13.2 points against a mean of 44.4 while the market's own SD is only 4.8. The market sets the
*level* accurately and says almost nothing about the *variance*. Implied totals belong at the
team-volume and TD-rate layer, not as a direct multiplier on individual stat lines.

### Availability data, and its holes

- `injuries/injuries_{season}.parquet` (2009+) gives one row per player-week with `report_status`
  (Out/Doubtful/Questionable), `report_primary_injury`, and `practice_status` (Full/Limited/DNP).
  **Verified: 2024 and 2025 exist; 2026 does not exist yet** (season has not started). Have a
  fallback and do not assume the current season's file is present.
- **There is no inactives dataset anywhere in nflverse.** You must scrape ESPN on gameday.
- Injury report cadence is **Wed/Thu/Fri only for Sunday games**. It shifts entirely for other
  slates: Mon/Tue/Wed for Thursday games, Tue/Wed/Thu for Saturday, Thu/Fri/Sat for Monday. A
  scraper hardcoded to Wed/Thu/Fri silently misses every TNF and MNF game.
- The Friday Game Status Report is **not final**. The policy requires updates when a player's
  condition changes, and Saturday/Sunday-morning downgrades from Questionable to Out are routine and
  are among the largest single line movers of the week.
- **Depth charts from 2025 onward carry ISO8601 timestamps instead of week numbers.** Code that
  assumes a `week` column breaks silently. Verify the schema of the current season's file at build
  time.

### ESPN unofficial API (no key) for live box scores and gameday injuries

Base: `site.api.espn.com/apis/site/v2/sports/football/nfl/` with `scoreboard`
(`?dates=YYYYMMDD` or `?dates=YYYY&seasontype=2&week=N`), `summary?event={id}`,
`teams/{ABBR}/roster`, `injuries`, `teams`. Player gamelogs are on a *different* host:
`site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{id}/gamelog`.

`summary?event={id}` returns per-player stat blocks (passing/rushing/receiving/kicking/defensive)
and is the practical free in-progress box score. Join to nflverse via `games.parquet.espn`.

### Data licensing

nflverse data is CC-BY-SA 4.0 with **mandatory attribution to FTN Data via nflverse**. Put visible
attribution in the footer from day one, the way DiamondValue attributes the MLB Stats API. Check
whether share-alike propagates to any derived dataset you publish.

### Known participation gap (design around this from day one)

**Routes run is the correct exposure for WR/TE** (the true plate-appearance analog; targets are then
routes x target-per-route-run). But nflverse participation data is released **only after the
postseason ends and does not update in season**. A live weekly system cannot use routes run for the
current season. Plan around PFR snap counts (`load_snap_counts`, updated several times daily in
season) as the live proxy, with routes run used only for training on completed seasons. Discovering
this after building the pipeline costs weeks. Also note participation changed sources mid-stream:
2023+ comes from FTN after the previous NGS-based source died during the 2023 season.

---

## 3. Models

Follow DiamondValue Section 4 for artifact structure, then adapt:

**Losses.** Poisson for counts: targets, carries, receptions, TDs, attempts, completions. **Yards are
not counts** and Poisson is wrong for them: rushing and receiving yards have a point mass at zero plus
a heavy right tail, which is Tweedie territory (XGBoost `reg:tweedie` with a tuned
`tweedie_variance_power`, or an explicit compound Poisson touches x Gamma yards-per-touch). Model
receptions as Binomial(targets, catch probability) rather than a marginal Poisson.

**The sklearn exposure trick carries over** from DiamondValue: `HistGradientBoostingRegressor` has no
offset parameter, so rate models train on `y = count / opportunities` with
`sample_weight = opportunities` and `loss="poisson"`. Zero-opportunity rows carry zero weight and drop
out naturally.

**TDs.** Do not fit on historical TD rate. Build a yard-line-weighted expected-TD exposure. Even so,
be honest: expected TDs are stickier year over year than raw TDs (r^2 0.382 vs 0.276) but barely
better at *predicting* next-season TDs (0.283 vs 0.275). TDs are near-irreducibly noisy. Regress hard
toward opportunity-implied rates and let the uncertainty band show it rather than chasing a better TD
model.

**Defense adjustments.** Raw defense-vs-position multipliers are mostly opponent-schedule artifacts in
a 17-game season and are worst for WR/TE; unshrunk DVP actively degrades projections. Use team-level
defensive efficiency (EPA/play allowed) with empirical-Bayes shrinkage.

**Weather.** Wind is the dominant effect on passing and is already in `games.parquet`. Expect a real
but modest effect concentrated in the high-wind tail; verify on your own data rather than trusting
published band tables.

### Validation and the ship gate (port this discipline exactly, it is the most valuable thing here)

Random K-fold is **forbidden**; it leaks player-season level. Use walk-forward: train on earlier
seasons, tune on one validation season, and test on a held-out season exactly once.

**Mandatory baselines**, per target: (1) league-average constant, (2) player's season-to-date
per-game mean excluding the target game, (3) a shrunk multi-season rate x expected opportunities.
A target that cannot beat baselines 2 and 3 out of sample is reported honestly at the Section 5 STOP,
not silently dropped, and Barrett decides whether to hide it or show it with a caveat.

**Two-stage feature gating, the single most valuable practice from DiamondValue.** Every candidate
feature block is ablated on a validation season, and any block that wins there must then *replicate*
on a separate held-out season before it ships. This caught multiple "mirages" in the MLB build,
features that won by 0.6% on one year and reversed sign on the next. Expect to reject more features
than you ship; that is the process working. Two hard-won corollaries:
- The confirmation run must load **identical history** to the validator. A thin-history experiment
  harness shipped two false confirmations in the MLB build before an audit caught them.
- If a policy layer decides which columns each target trains on, make it **grant-wins** (a column is
  kept if ANY block grants it). The MLB build had any-block-can-veto semantics, and blocks sharing
  column names silently stripped granted columns, producing byte-identical models and 0.00% deltas
  everywhere. If a "confirmation" shows exactly zero change, suspect the plumbing, not the feature.

**Calibrate your expectations for the sport.** 272 regular-season games a year versus MLB's 2,430.
Backtest confidence intervals are far wider. Any "beat the market by X%" claim on half a season of
NFL is noise. Do not port DiamondValue's validation thresholds.

**Leakage tests are mandatory**, mirroring `MLB Predict/tests/`. The MLB build shipped two bugs that
tests would have caught: history must be filtered to strictly before the target date, or a target row
sees its own game as a prior appearance.

---

## 4. Weekly pipeline (not daily)

The NFL board is **one weekly board with staggered locks**, not a daily cycle: Thursday night, the
Sunday 1pm/4pm/SNF windows, then Monday night. A daily cron modeled on MLB will either miss the
Thursday lock or refresh pointlessly.

Cadence should track the injury calendar rather than a fixed morning job:
- Tuesday: rebuild features from completed games; regenerate the week's projections.
- Wed/Thu/Fri: refresh injury practice reports (per the slate-specific cadence above), regenerate.
- Saturday and each gameday morning: refresh status changes.
- Kickoff minus ~90 minutes per game: pull inactives, regenerate that game.
- After each game: score predictions into the accuracy tracker.

Automate with GitHub Actions exactly as DiamondValue does, including the "generate next slate too"
step so links are never empty. **Note the workflow-file gotcha**: Barrett's GitHub token historically
lacked `workflow` scope, and both git push and the contents API reject workflow-file writes while
masking the failure as a 404. He may need to paste `.github/workflows/*.yml` through the GitHub web
UI, or verify his current token has that scope.

**Stale beats empty, everywhere.** Every network fetcher reads the possibly-stale disk cache first,
retries with bounded exponential backoff, and on total failure serves the stale copy with a logged
warning. Never block a page behind a hung endpoint; never return empty when stale exists.

---

## 5. Order of work, with STOP points

1. **Data layer + slate viewer.** Pick a week, see the games, with team colors. Barrett's first
   visible milestone; get it in front of him early.
2. **Training backfill.** Build the player-week training table from pbp and weekly stats. Verify row
   counts and spot-check against Pro-Football-Reference by hand.
3. **Features**, point-in-time, shared by training and inference.
4. **Models + validation.** **STOP HERE.** Write `docs/model_report.md` opening with a plain-English
   per-target verdict Barrett can act on ("target volume has real signal; receiving TDs barely beat a
   season average; recommendation: show with a caveat"), then the full tables. Get his column
   decision before building UI around numbers that may not survive.
5. **Weekly pipeline + accuracy tracker.**
6. **UI**, porting DiamondValue's theme and components.
7. **Deploy.** Barrett is weighing Render vs staying on Streamlit Community Cloud (see Section 7).
8. **Market comparison** (Section 6), only if he still wants it after reading that section.

---

## 6. Market comparison: read this before promising it

DiamondValue has a PrizePicks model-vs-market feature. Porting it to NFL is **not** straightforward,
and Barrett should hear these three things before you build it.

**1. Automated access is blocked, and separately prohibited.** As of 2026-08-31, both
`api.prizepicks.com` and `partner-api.prizepicks.com` return HTTP 403 behind Cloudflare bot
management to server-side requests. This is fingerprint-level blocking, so a browser User-Agent and a
proxy will not fix it. **Barrett's own logged-in browser works** (that is how DiamondValue gets lines:
the user opens the feed URL themselves and pastes the result). Independently, PrizePicks' Terms
effective 2026-08-03 §16(l) prohibit "any robot, spider, or other automatic device, process, or
means to access the Site or App," and restrict use to personal, non-commercial purposes. Solving the
technical block does not solve the terms question. The paste-it-yourself path is the one to keep;
do not build a scraper.

**2. Demons and Goblins are no longer More-only.** PrizePicks' own help center and product page both
now read: "You can now pick More or Less on Demons and Goblins." Nearly every third-party guide
(Stokastic, RotoGrinders, HeatCheck) still says More-only, and so did PrizePicks' own older posts.
**Never infer pick direction from `odds_type`.** Read the available directions from the feed or the
board text; if direction is unknown, assume both sides are available rather than hiding lines.
(DiamondValue currently infers More-only from odds_type and needs the same correction.)

**3. Grading will not match your data.** PrizePicks settles via Sportradar/Genius/Stats Perform, not
nflverse or PFR, and the stats where providers disagree (tackles, longest reception, target counts)
are exactly the propped ones. Backtesting against nflverse will surface grading disagreements you
will mistake for model error. Two more grading subtleties: an exact-line tie is not a push (it
lowers the payout tier), and "Reboot" protection is asymmetric by direction, so grading both
directions symmetrically against final box scores overstates More-side accuracy.

Also: league ids drift and the NFL family is several boards (full-game, 1H/1Q/2H, season-long).
Resolve ids from `/leagues` at runtime rather than hardcoding. And present edge as
probability-versus-implied-line, never as dollar EV, since the peer-to-peer Arena format computes
payouts from the entry mix rather than a fixed multiplier table.

Port these DiamondValue lessons directly:
- Route each pasted line to the **date/week of its own game**, not the week on screen. The pre-game
  board posted Wednesday is Sunday's games.
- Collapse the alt-line ladder: one line per (player, stat), preferring the standard line.
- Reject non-person player names. A feed pasted without its `included[]` player list puts the TEAM
  code in `description`, and thousands of lines keyed to team codes will match nothing.
- Never let a saved-but-unmatched board render as silence. Say why it matched nothing.

---

## 7. Ground rules ported from HoopsValue and DiamondValue (non-negotiable)

1. **Writing style**: no em dashes in UI text or explanatory prose, no emojis. Barrett finds both
   "AI-feeling." Escape dollar signs in Streamlit markdown. Missing values render as an em-dash
   sentinel defined ONCE as `SENTINEL` in the theme module, the only em dash allowed in a literal.
2. **Theme**: port DiamondValue's token system (`var(--panel)`, `var(--fg-1..6)`, accent tokens),
   `render_page_chrome()`, `render_nav()`, and the `?theme=` URL param. Support light and dark and
   verify both before calling anything done.
3. **Module boundaries**: do NOT create a monolith. Package layout mirroring `mlblib/`:
   `cache.py` (disk cache, atomic writes, stale-beats-empty), `fetch.py`, `features.py`, `model.py`,
   `store.py` (HTML table rendering), `teams.py` (team colors), `theme.py`, `parse.py`.
4. **`scripts/` naming discipline** from day one with a README: `build_*`, `train_*`,
   `validate_*`/`backtest_*`, `exp_*` (research probes, kept even when rejected).
5. **Keys**: join on nflverse `game_id` + `gsis_id` (the canonical player id). Never join on
   date + player name. Keep a crosswalk to ESPN athlete ids (`load_ff_playerids` or
   `depth_charts.espn_id`) for the live box-score path. Name matching needs an accent- and
   suffix-tolerant normalizer (Jr./III/apostrophes); DiamondValue's `NameIndex` is the reference.
6. **Pins**: Python 3.12.8 in `runtime.txt`, `.python-version`, and the host's env var. Pin
   scikit-learn and joblib to the exact versions the model artifacts were saved with, with a comment
   saying so; version drift silently breaks `joblib.load` on a rebuild. **Streamlit Community Cloud
   ignores `runtime.txt` and builds on the newest Python**, which forced two dependency bumps in the
   MLB build; if deploying there, verify wheels exist for their Python in a clean venv before pushing.
7. **Data corrections layer**: hand-curated CSVs in `data/` that override scraped feeds, created
   early with headers even while empty.
8. **Version-stamp derived cache files in the filename** (`predictions_2026_w01_m1.parquet`); bump to
   invalidate, never edit in place. On a host with a persistent disk, seeding is gap-fill only, so a
   changed file with the same name never reaches production.
9. **Single named logger**, env-var level, every fetch failure logs a warning rather than failing
   silently.
10. **Tests are mandatory**, including a leakage suite. DiamondValue has 42 tests; they caught real
    bugs repeatedly.

### Streamlit gotchas that cost real debugging time

- **Never reorder a keyed selectbox's options between reruns.** Streamlit hashes `str(options)` into
  the element ID; a reorder orphans the selection and the user sees "my pick doesn't register."
- **Avoid `components.html` iframes for critical UI.** On a cold start the iframe request can be
  answered by the app homepage itself.
- **Streamlit Community Cloud serves the app in an iframe.** Two consequences: HTML anchor nav must
  use `target="_self"`, and **the browser address bar never follows in-app navigation**, so there is
  nothing correct for a user to copy. If shareable per-game links matter, mint them explicitly with a
  Share control (DiamondValue does this) using human-readable params like
  `?week=1&away=NE&home=SEA`, and resolve those back to a game id server-side.
- A keyed `st.container` IS the `stVerticalBlock` element itself, so `flex-direction` CSS goes
  directly on `.st-key-{name}`; a `> [data-testid="stVerticalBlock"]` child selector matches nothing.
- Popovers forbid nested expanders.

---

## 8. How Barrett likes to work

- **Every change gets a written receipt** (what, why, verification) in `Desktop/Claude/receipts/`.
  This is a standing rule across his projects.
- He wants **honest negative results**, not hype. The MLB build has multiple documented rounds where
  the answer was "tested four features, shipped none." Write those up the same way; he values them.
- Show him a visible milestone early and often rather than disappearing into the model for days.
- Verify in a real browser before saying something works. Screenshot the result.
- He is not a professional developer but is deeply engaged with the domain and will catch a wrong
  baseball or football fact instantly. Do not bluff sport knowledge.
- Batch pushes. Each deploy causes a brief outage on some hosts, and deploy-spam is painful.

## 9. First message to Barrett

Before writing code, confirm with him: the **name**, whether he wants the **market-comparison feature
at all** given Section 6, and which **positions** matter most to him for v1 (there is a real scope
difference between "QB/RB/WR/TE" and "everyone including kickers"). Then start with the slate viewer
so he sees something real within the first session.
