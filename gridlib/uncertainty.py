"""P(actual > line) for a projected stat, from measured residuals.

A projection of 6.2 receptions against a line of 5.5 tells a reader almost
nothing on its own. The gap is 0.7, but receptions carry a median absolute error
near 46% of the mean, so that gap is well inside the noise. Expressed as a
probability the same row reads "56% Over", which is honest about being close to
a coin flip in a way that "+0.7" is not.

This module turns a point projection into that probability using the empirical
ratio distributions built by `scripts/build_prob_calibration.py`. Nothing here
assumes a distributional family; see that script for why Poisson and Normal are
both wrong at the line.

WHAT THIS IS NOT
----------------
It is not a claim of edge. A well-calibrated 56% against a naive line can still
be a losing bet against a sharp one, and for the yards props the measured
side-picking edge is NEGATIVE. Calibration means the number means what it says,
not that acting on it wins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cache import dc_path, read_parquet_or_none, logger

ARTIFACT = "prob_calibration_v1.parquet"
# Below this projection the ratio is meaningless: a player projected for 0.2
# targets who catches one has a ratio of 5, which says nothing about anyone else.
MIN_PRED = 0.25
# Never return certainty. A model that says 0% is claiming an outcome is
# impossible, and nothing measured from a few thousand rows supports that.
FLOOR, CEIL = 0.01, 0.99
# Pull every probability this fraction of the way toward a coin flip.
#
# Even with four fit seasons and ten bins the high end still promises more than
# it delivers: on the 2023 development fold, bins claiming 60%+ came in 3.73
# points optimistic. The residual cause is heterogeneity WITHIN a bin, which no
# amount of binning removes, because two players with the same projection can
# still have different spreads.
#
# A shrink is the standard correction and it has a property that matters here:
# pulling toward 0.5 CANNOT move a probability across 0.5, so it cannot change a
# single lean. Side-picking is unaffected by construction; only the confidence
# attached to it moves. Chosen on 2023 from {5%, 10%, 15%}: 10% gave the best
# calibration error (0.77 against 1.60 unshrunk) while 15% over-corrected the
# middle of the range to fix the tail.
SHRINK = 0.10

# The highest confidence the calibration data actually VALIDATES.
#
# The reliability curve is built against a proxy line (the player's season-to-
# date mean), which rarely sits far from the projection, so its top bin claims
# about 77% and there is nothing above that to check. A real board does contain
# far-from-line props: on the 2026 week 1 board, 11 of 307 rows come out above
# 80% and one at 91%, all of them low-side calls on players projected far under
# a line (a backup back at 17 rushing yards against 51.5).
#
# Those calls are plausible, but nothing here measured them. The display caps at
# this value and says "80%+" rather than printing a precise number the
# calibration cannot stand behind. Ranking still uses the full precision, so the
# order of the board is unaffected.
VALIDATED_MAX = 0.80

_CACHE: dict[str, pd.DataFrame] = {}
_LEVEL_CACHE: dict[str, pd.DataFrame] = {}

LEVEL_ARTIFACT = "mean_calibration_v1.parquet"


def level_table() -> pd.DataFrame | None:
    """The isotonic level maps, cached. None when not built."""
    if "df" not in _LEVEL_CACHE:
        t = read_parquet_or_none(dc_path(LEVEL_ARTIFACT))
        _LEVEL_CACHE["df"] = pd.DataFrame() if t is None or t.empty else t
    df = _LEVEL_CACHE["df"]
    return None if df.empty else df


def has_level(target: str) -> bool:
    t = level_table()
    return t is not None and bool((t["target"] == target).any())


def apply_level(target: str, pred):
    """Correct the LEVEL of a projection for the three targets where a measured
    correction survived the two-stage gate.

    The lowest projection quintile is over-projected by 10-25% because it is
    59-68% zeros and nothing in the data separates a receiver who dressed and
    was never targeted from one who saw a little work. This is a measured map
    from `scripts/build_mean_calibration.py`, fitted on walk-forward predictions
    so it describes an out-of-sample model, which is the situation at serve time.

    Isotonic, therefore MONOTONE: it changes the level, never the order. Two
    players cannot swap places because of it.

    Returns `pred` unchanged for any target without a map, which is five of the
    eight tested. Silently passing through is correct here and not a fallback
    hiding a bug: those targets were explicitly rejected by the gate.
    """
    t = level_table()
    arr = np.asarray(pred, dtype=float)
    if t is None:
        return arr
    row = t[t["target"] == target]
    if row.empty:
        return arr
    r = row.iloc[0]
    grid = np.asarray(r["grid"], dtype=float)
    mapped = np.asarray(r["mapped"], dtype=float)
    if grid.size == 0 or grid.size != mapped.size:
        return arr
    # Stored as a lookup on a grid rather than a pickled estimator, so a
    # scikit-learn upgrade cannot change what it returns.
    out = np.interp(arr, grid, mapped, left=mapped[0], right=mapped[-1])
    # ABOVE the grid, carry the top-of-grid RATIO forward instead of clamping.
    # The grid ends at the 99.9th percentile of fitted projections, so clamping
    # would put a hard ceiling on exactly the best players: a 110-yard
    # projection would be served as 93.7 because that is where the grid stops.
    # No live 2026 week 1 projection exceeded the grid, but that is this week's
    # luck, not a guarantee.
    top = grid[-1]
    if top > 0:
        ratio = mapped[-1] / top
        above = arr > top
        if np.any(above):
            out = np.where(above, arr * ratio, out)
    return np.clip(out, 0, None)




def table() -> pd.DataFrame | None:
    """The calibration artifact, cached. None when it has not been built."""
    if "df" not in _CACHE:
        t = read_parquet_or_none(dc_path(ARTIFACT))
        if t is None or t.empty:
            logger.warning("%s missing; probabilities unavailable", ARTIFACT)
            _CACHE["df"] = pd.DataFrame()
        else:
            _CACHE["df"] = t
    df = _CACHE["df"]
    return None if df.empty else df


def _row_for(target: str, pred: float):
    t = table()
    if t is None:
        return None
    sub = t[t["target"] == target]
    if sub.empty:
        return None
    # The bin whose projection range contains pred, else the nearest edge bin.
    # Clamping rather than returning None is deliberate: a projection above
    # every bin is a star player, and the top bin's spread is the best
    # available description of him.
    hit = sub[(sub["lo"] <= pred) & (pred <= sub["hi"])]
    if not hit.empty:
        return hit.iloc[0]
    return (sub.nsmallest(1, "lo").iloc[0] if pred < sub["lo"].min()
            else sub.nlargest(1, "hi").iloc[0])


def prob_over(target: str, pred: float, line: float) -> float:
    """P(actual > line) given the model projected `pred`.

    Returns NaN when it cannot be answered honestly: no calibration for this
    stat, or a projection too small for the ratio to mean anything.
    """
    if pred is None or line is None:
        return float("nan")
    try:
        pred, line = float(pred), float(line)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(pred) or not np.isfinite(line) or pred < MIN_PRED:
        return float("nan")

    row = _row_for(target, pred)
    if row is None:
        return float("nan")
    qs = np.asarray(row["quantiles"], dtype=float)
    grid = np.asarray(row["q_grid"], dtype=float)
    if qs.size == 0 or qs.size != grid.size:
        return float("nan")

    # The actual beats the line when actual/pred exceeds line/pred, so the
    # probability is the mass of the ratio distribution above that point.
    needed = line / pred
    # `qs` is non-decreasing by construction, which is what interp requires.
    cdf = float(np.interp(needed, qs, grid, left=0.0, right=1.0))
    p = 1.0 - cdf
    p = 0.5 + (p - 0.5) * (1.0 - SHRINK)
    return float(min(CEIL, max(FLOOR, p)))


def median_projection(target: str, pred: float) -> float:
    """The MEDIAN outcome for a projection whose mean is `pred`.

    WHY THE BOARD SHOWS THIS AND NOT THE MEAN
    ------------------------------------------
    A prop line is a 50/50 question, so the lean is decided by the median: the
    model says More exactly when P(actual > line) > 0.5, which is exactly when
    the median exceeds the line.

    Printing the MEAN beside the line broke that. These stats are right-skewed,
    so the mean sits above the median (receiving yards: the median is 0.68 to
    0.90 of the mean), and any line falling between the two produced a row that
    contradicted itself: "MODEL 4.8, LINE 4.5, LEAN Less". The numbers were each
    correct and the row was unreadable.

    Showing the median makes MODEL > LINE and "More" the same statement, so the
    contradiction cannot occur rather than merely being explainable. Passing
    yards is nearly symmetric (0.97 to 1.04), which is why QB rows rarely showed
    the problem and skill rows showed it constantly.

    The MEAN is still what the models estimate and still what aggregates and the
    coherence checks use. This is a display choice for the one place a
    projection is compared to a posted number.
    """
    if pred is None:
        return float("nan")
    try:
        pred = float(pred)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(pred) or pred < MIN_PRED:
        return float("nan")
    row = _row_for(target, pred)
    if row is None:
        return float("nan")
    qs = np.asarray(row["quantiles"], dtype=float)
    grid = np.asarray(row["q_grid"], dtype=float)
    if qs.size == 0 or qs.size != grid.size:
        return float("nan")
    ratio = float(np.interp(0.5, grid, qs))
    return float(max(0.0, pred * ratio))


def prob_series(targets, preds, lines) -> np.ndarray:
    """Vectorised convenience wrapper. Same semantics, row by row."""
    return np.array([prob_over(t, p, l)
                     for t, p, l in zip(targets, preds, lines)], dtype=float)


def available() -> bool:
    return table() is not None
