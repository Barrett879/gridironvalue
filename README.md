# GridironValue

Per-game expected stat lines for every projected NFL skill player, with the
market as context and an honest account of which numbers are worth reading.

**Every number is the median of a modelled distribution, not a forecast of what
will happen.** This is an informational analytics site. It is not a sportsbook,
it accepts no wagers, and it offers no betting advice.

```bash
python -m streamlit run app.py --server.port 8555
```

## What it does

- **Week board** — every game grouped by broadcast window, with the market line
  and the time inactives publish (kickoff minus 90 minutes, computed per game).
- **Game pages** — both teams' skill-position depth charts with injury status,
  the market's implied team totals, and each player's posted prop lines opening
  under his own row.
- **Model versus market** — paste a PrizePicks board and see where the
  projection disagrees, ranked by how much that disagreement is actually worth.

## How much to trust it

Side-picking edge over always taking the more common side, in percentage points
of hit rate. Measured out of sample: models trained through 2022, scored on
2024, using the rule the site actually serves.

| stat | edge | read as |
|---|---|---|
| QB pass attempts | +11.5 | the strongest thing this model does |
| Passing yards | +8.4 | strong |
| QB completions | +7.9 | strong |
| Targets | +6.0 | moderate |
| WR/TE receptions | +5.3 | moderate |
| RB rush attempts | +4.3 | moderate |
| Receiving yards | +2.9 | slight |
| Rushing yards | +2.4 | slight |

**These numbers replace an earlier table that had the sign wrong on two props.**
It reported receiving yards at -5.7 and rushing yards at -9.4, described both as
worse than a coin flip, and put targets and rush attempts at zero. Those were
measured under the retired rule that leaned on the gap between the model and the
line. The site leans on P(Over) now, which is what fixed the yards props: on the
same rows, receiving yards go -4.8 to +2.9 and rushing yards -7.4 to +2.4. The
mechanism is that these stats are right-skewed, so the mean sits above the
median, and a line below the projected mean can still be above the median. The
gap rule then leaned More on an outcome that was under a coin flip.

**Read these as a ranking, not as a promise about a real board.** The line in
this measurement is a PROXY: the player's season-to-date mean, because a
historical archive of actual PrizePicks lines does not exist to test against. A
real line is set by a market that has already priced most of what the model
knows, so it is a harder target than the proxy. What the table supports is which
props this model is better and worse at, not a hit rate you should expect.

The board's own out-of-sample record, on real posted lines, is on the site
underneath the slate. That one is the honest number, and it is frozen at the
moment each line was first seen so that retraining the models cannot improve it
after the fact.

**It also assumes the player plays.** Projections are `E[Y | appeared]`, which is
the right quantity for a prop (a prop voids on a DNP) and the wrong one to sum
across a roster. `p_play` carries the availability factor in a separate column.

## What it deliberately will not price

Longest reception, longest rush, longest completion, longest field goal. Those
are maxima over plays, and `E[max]` is not a function of `E[sum]`, so a
mean-projection model cannot price them. They are refused by name and counted.

## Layout

    gridlib/     cache, fetch, features, columns, model, predict, props, store, theme
    scripts/     build_* / train_* / validate_* / exp_*  (see scripts/README.md)
    pages/       Game, About
    data/        hand-curated corrections and the per-stat reliability table
    docs/        data notes, backfill, model report, experiment results

Deployment: see `DEPLOY.md`. Data provenance and the traps found while building:
`docs/data_notes.md`. Model verdicts per target: `docs/model_report.md`.

Play-by-play, schedule, market and roster data from
[nflverse](https://github.com/nflverse/nflverse-data), sourced in part from FTN
Data and used under CC BY-SA 4.0.
