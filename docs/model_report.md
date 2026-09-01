# Model report

Spec step 4. **This is the STOP point: the per-target column decision is yours.**

---

## The short version, in plain English

**Volume works. Efficiency and touchdowns do not.** That is the result the spec
predicted, and it is what the numbers say.

| what | verdict | recommendation |
|---|---|---|
| Pass attempts, completions, carries, targets, receptions | **Real signal.** 6-13% better than a season-to-date average, out of sample, on two independent seasons. | **Show them.** |
| Passing, rushing and receiving yards | **Real but smaller.** 5-13% better. | **Show them.** |
| Passing TDs, interceptions, sacks taken | **Marginal.** 2-8% better, and the smaller edges are within noise for a 17-game sport. | Show with a caveat. |
| **Receiving TDs, rushing TDs** | **FAIL.** Both LOSE to a plain season-to-date average, on both seasons. | **Hide, or show the baseline instead of the model.** |
| Field goals, extra points | **Beat the player baselines but lose to a league constant.** | Show with a strong caveat, or hide. |

If you want one sentence for the About page: *the volume numbers are worth
reading, the touchdown numbers are close to guesses, and the site says which is
which.*

---

## How this was measured

Strict walk-forward. Random K-fold is forbidden here because it leaks at the
player-season level: a player's other games in the same season carry nearly the
same information as the held-out one.

    train    2016-2022   (7 seasons, ~44k player-games)
    validate 2023        every tuning and feature decision was made here
    test     2024        touched once, at the end
    holdout  2025        NOT touched by anything in this report

League efficiency priors are recomputed per fold from seasons strictly before
the fold, so a prior never sees the season it helps predict.

**The three mandatory baselines**, per target:
1. a league-average constant
2. the player's season-to-date per-game mean, excluding the game being scored
3. a shrunk multi-season rate times expected opportunity

The gate is beating **both 2 and 3**. Beating 1 proves nothing.

---

## Results

MAE, lower is better. `vs b2` and `vs b3` are percentage improvements over
baselines 2 and 3.

### Test season 2024 (the honest number)

| target | mean | model | b1 | b2 | b3 | vs b2 | vs b3 | gate |
|---|---|---|---|---|---|---|---|---|
| attempts | 26.77 | 8.003 | 9.975 | 9.149 | 9.072 | **+12.5%** | +11.8% | PASS |
| passing_yards | 190.92 | 63.969 | 81.977 | 73.680 | 71.366 | **+13.2%** | +10.4% | PASS |
| completions | 17.49 | 5.457 | 6.960 | 6.129 | 6.137 | +11.0% | +11.1% | PASS |
| carries | 6.25 | 2.893 | 5.064 | 3.197 | 3.262 | +9.5% | +11.3% | PASS |
| targets | 3.25 | 1.643 | 2.580 | 1.755 | 1.814 | +6.4% | +9.4% | PASS |
| receptions | 2.22 | 1.282 | 1.845 | 1.362 | 1.391 | +5.9% | +7.8% | PASS |
| rushing_yards | 27.58 | 17.240 | 25.295 | 18.264 | 18.312 | +5.6% | +5.9% | PASS |
| receiving_yards | 24.32 | 16.471 | 23.613 | 17.355 | 16.835 | +5.1% | +2.2% | PASS |
| passing_tds | 1.21 | 0.849 | 0.981 | 0.885 | 0.893 | +4.1% | +5.0% | PASS |
| sacks_suffered | 1.97 | 1.285 | 1.380 | 1.348 | 1.334 | +4.7% | +3.6% | PASS |
| passing_interceptions | 0.58 | 0.667 | 0.707 | 0.680 | 0.690 | +1.8% | +3.3% | PASS |
| fg_att | 2.05 | 1.017 | **1.004** | 1.147 | 1.038 | +11.3% | +2.0% | PASS* |
| fg_made | 1.72 | 1.013 | 1.021 | 1.118 | 1.031 | +9.4% | +1.8% | PASS* |
| pat_att | 2.28 | 1.076 | 1.183 | 1.147 | 1.135 | +6.2% | +5.2% | PASS |
| **receiving_tds** | 0.15 | 0.236 | 0.269 | **0.226** | 0.239 | **-4.6%** | +1.1% | **FAIL** |
| **rushing_tds** | 0.22 | 0.292 | 0.343 | **0.290** | 0.295 | **-0.9%** | +0.8% | **FAIL** |

The validation season agrees on every verdict; nothing flipped between folds.
Full numbers in `docs/model_report_data.csv`.

### The two asterisks

**Kickers pass the gate but lose to a league constant.** For `fg_att` the model
posts 1.017 against a constant's 1.004 on test, and 1.043 against 1.041 on
validation. It beats the player-specific baselines only because those are worse
still. The honest reading is that field-goal volume is close to unpredictable:
it depends on where drives stall, which is a property of the game, not the
kicker. Recommend hiding, or showing with a blunt caveat.

**Touchdowns fail, and this was expected.** Published year-over-year stability
for WR TD rate is 0.0155. Expected TDs are stickier than raw TDs (r-squared
0.382 against 0.276) but barely better at predicting next-season TDs (0.283
against 0.275). This is near-irreducible noise, not a modelling failure to fix
with a better booster. If you want TDs on the page, the recommendation is to
display the season-to-date baseline, which is measurably better than the model,
and band it widely.

---

## What was tried to improve accuracy, and rejected

Five ideas were tested with two-stage gating: win on the validation season, then
**replicate** on a separate held-out season. Four were rejected. The scripts stay
in the repo (`scripts/exp_*.py`) because a rejected experiment is a result.

| variant | validation | test | verdict |
|---|---|---|---|
| recency weighting (half-life 3 seasons) | -0.01% | not run | **rejected**, no effect |
| one model per position | **-0.72%** | not run | **rejected**, clearly worse |
| more capacity (63 leaves, 800 iters) | -0.01% | not run | **rejected**, no effect |
| 16-game rolling window added | +0.38% | **+0.09%** | **rejected as noise** |
| prior red-zone opportunity block | +0.00 to -1.4% | -0.9% | **rejected** |

Position-specific models being *worse* is the interesting one: pooling WR, TE and
RB gives the model more data than the position distinction costs it.

The 16-game window technically replicated, but +0.09% on a 17-game sport is
indistinguishable from noise, and the spec is explicit that any small edge
claimed on half a season of NFL is noise. Rejected on that basis rather than
banked.

---

## A leak this found, and how

The red-zone block first measured **+13.1% on targets and +9.5% on carries** on
the test season, replicating cleanly across both folds. That is far too large for
a block of prior red-zone counts, and the size is what prompted the check rather
than the celebration.

It was a missingness leak, with two compounding causes:

1. In the backfill, a player-game with no targets got **NaN**, not 0, for the
   red-zone target columns.
2. In the feature builder, `cumsum() - current` propagates a NaN in the
   **current** row straight into the feature.

So the feature was NaN exactly when the player had no opportunity in the game
being predicted. Measured directly:

    P(f_career_targets_rz is NaN | this game's targets == 0) = 1.00
    P(f_career_targets_rz is NaN | this game's targets  > 0) = 0.00

The model was reading missingness as the answer. With both causes fixed, the
block does nothing at all and is rejected.

**The lesson worth keeping: two-stage gating catches mirages, not leaks.** A leak
is present in every fold, so it replicates beautifully. The thing that caught
this was the effect size being implausible for the mechanism claimed. Four
regression tests now pin it (`tests/test_leakage.py`), including a general one
that scans every feature-source column for missingness correlated with the
target, and a second that pins `adot`, `target_epa` and `rush_epa` OUT of the
feature builder, since all three are legitimately NaN exactly when the
opportunity count is zero.

---

## The biggest remaining lever, and it is not a hyperparameter

**The model answers the wrong question.** It predicts production *given the
player plays*. The training table only contains players with a stat line, so a
player who was inactive has no row at all.

Quantified over players with at least 8 appearances in a season:

| | |
|---|---|
| mean appearance rate | 0.807 |
| mean games missed | 3.19 of 16.5 |
| miss at least one game | **74.2%** |
| miss three or more | **50.0%** |
| missing player-game slots | 12,825, about **17%** of what a projection system is asked about |

By position, the share missing at least one game: FB 96.7%, TE 85.9%, RB 76.6%,
WR 76.5%, QB 65.0%, K 31.2%.

The spec is explicit about this: build `P(plays) x E[production | plays]`, not a
single conditional mean. What exists today is only the second factor.

The injury report cannot patch it from inside this table. Only **4.2%** of
player-games carry a report entry, and there is exactly **one "Out" row in
63,042** rows, because a player ruled out never appears. Worse, players listed
Questionable who still played average *more* targets than unlisted players
(5.01 against 4.29 for WRs), because teams list starters. The signal is
present but the sample is selected.

Building it properly needs rows for players who did NOT play, constructed from
weekly rosters and depth charts, plus an ESPN inactives scrape for gameday.
That is a real piece of work and the single largest accuracy gain available.

---

## Decisions for you

1. **Touchdowns.** Hide them, or show the season-to-date baseline with a wide
   band and a caveat? The model is worse than the baseline, so shipping the
   model would be the wrong call either way.
2. **Kickers.** They lose to a league constant. Hide, or show with a caveat?
3. **The availability model.** This is the big one. Worth building next, before
   any further feature work on the conditional model?
4. **Wind.** Still open from two sessions ago and still classified a postgame
   trap. Drop it, or add a forecast dependency?

Nothing here has touched 2025. It stays clean for a final honest read once the
column decisions are made.
