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

Measured out of sample, side-picking edge against a naive line:

| stat | edge | read as |
|---|---|---|
| QB pass attempts | +10.1 | the strongest thing this model does |
| Passing yards | +8.6 | strong |
| QB completions | +8.1 | strong |
| WR/TE receptions | +3.1 | moderate |
| Targets, RB rush attempts | ~0 | no measurable edge |
| Receiving yards | **-5.7** | **worse than a coin flip** |
| Rushing yards | **-9.4** | **worse than a coin flip** |
| Receiving TDs | -0.3 | 86% hit rate, no edge; served as a baseline |

Yards are volume times efficiency, and the efficiency half is close to noise, so
a yards projection ends up a noisier estimate of what a season average already
tells you. The board sorts those to the bottom and says so.

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
