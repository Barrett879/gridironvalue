"""Accuracy: how the projections have actually done, against doing nothing.

Reads only `cache/accuracy_history_v1.parquet`, which
`scripts/build_accuracy_tracker.py` writes by scoring past projections against
real box scores. The projection stays pure; scoring happens against it, never
the other way round.

Every row on this page comes from a model trained on seasons strictly BEFORE
the one being scored. The shipped models are trained through 2025, so asking
them about 2025 would be asking a student to mark their own paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib.cache import dc_path, read_parquet_or_none  # noqa: E402
from gridlib.theme import (  # noqa: E402
    SITE_NAME,
    render_footer,
    render_nav,
    render_page_chrome,
    theme_fig,
)

st.set_page_config(page_title=f"Accuracy · {SITE_NAME}",
                   page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
render_nav("Accuracy")

# Readable names for the table and the chart. The parquet stores the model's
# own target names, which are not what a reader calls them.
DISPLAY = {
    "attempts": "Pass attempts", "completions": "Completions",
    "passing_yards": "Passing yards", "passing_tds": "Passing TDs",
    "passing_interceptions": "Interceptions", "sacks_suffered": "Sacks taken",
    "carries": "Rush attempts", "rushing_yards": "Rushing yards",
    "rushing_tds": "Rushing TDs", "targets": "Targets",
    "receptions": "Receptions", "receiving_yards": "Receiving yards",
    "receiving_tds": "Receiving TDs", "fg_att": "Field goals attempted",
    "fg_made": "Field goals made", "pat_att": "Extra points",
}

st.markdown(
    '<div class="gv-masthead">'
    '<div class="gv-kicker"><span class="gv-pip"></span>Accuracy</div>'
    '<div class="gv-mast-rule"></div>'
    '<div class="gv-mast-title">Accuracy</div>'
    '<div class="gv-mast-sub">Projections scored against what actually '
    'happened, next to the thing they have to beat.</div>'
    "</div>",
    unsafe_allow_html=True)

acc = read_parquet_or_none(dc_path("accuracy_history_v1.parquet"))
if acc is None or acc.empty:
    st.info(
        "The accuracy history has not been built yet. Run "
        "`python scripts/build_accuracy_tracker.py --seasons 2023-2025` and "
        "this page will show model-versus-baseline error per stat.")
    render_footer()
    st.stop()

seasons = sorted(int(s) for s in acc["season"].dropna().unique())
span = (f"{seasons[0]}" if len(seasons) == 1
        else f"{seasons[0]} to {seasons[-1]}")

st.caption(
    f"Every projection below was made by a model trained only on seasons "
    f"BEFORE the one it is scoring, then compared against the real box score. "
    f"{span}, {len(acc):,} player-games. The comparator is the same one the "
    "models had to beat to ship: that player's own season-to-date average, "
    "this game excluded. Lower mean absolute error is better.")

# ── PAIRED, and this is not a detail ─────────────────────────────────────────
# pandas averages each column independently, so a row that has a model error
# but no baseline error (a player with no season-to-date history yet) would
# lift the model column alone. On the sibling MLB site that exact artifact
# turned a true -0.07% into a published +4.60%.
paired = acc.dropna(subset=["abs_err_model", "abs_err_b2"])
dropped = len(acc) - len(paired)

summary = (paired.groupby("target")
           .agg(n=("abs_err_model", "size"),
                model_mae=("abs_err_model", "mean"),
                base_mae=("abs_err_b2", "mean"))
           .reset_index())
# PER TARGET, then compared. A pooled ratio across targets is not scale-free:
# passing yards carries an MAE near 61 and receiving touchdowns near 0.23, so
# pooling is very nearly a report on passing yards alone.
summary["edge_pct"] = 100 * (1 - summary.model_mae / summary.base_mae)
summary["stat"] = summary["target"].map(lambda t: DISPLAY.get(t, t))

better = int((summary.edge_pct > 0).sum())
st.markdown(
    f'<div class="gv-window"><span class="lab">Against a season average</span>'
    f'<span class="meta">{better} of {len(summary)} stats better, '
    f'{summary.edge_pct.mean():+.1f}% on average</span>'
    f'<span class="lock">out of sample</span></div>',
    unsafe_allow_html=True)

import plotly.graph_objects as go  # noqa: E402

chart = summary.sort_values("edge_pct")
dark = st.session_state.get("theme_dark", False)
pos = "#16d4c1" if dark else "#0fae9d"
neg = "#e74c3c" if dark else "#dc3a2c"
fig = go.Figure()
fig.add_bar(orientation="h", y=chart["stat"], x=chart["edge_pct"],
            marker_color=[pos if v >= 0 else neg for v in chart["edge_pct"]],
            text=[f"{v:+.1f}%" for v in chart["edge_pct"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:+.1f}% better than a season average"
                          "<extra></extra>")
fig.update_layout(height=470, margin=dict(l=10, r=48, t=10, b=10),
                  showlegend=False,
                  xaxis_title="% better than that player's season average "
                              "(higher is better)")
st.plotly_chart(theme_fig(fig), width="stretch")

tbl = summary.sort_values("edge_pct", ascending=False)[
    ["stat", "n", "model_mae", "base_mae", "edge_pct"]].copy()
tbl["model_mae"] = tbl["model_mae"].round(3)
tbl["base_mae"] = tbl["base_mae"].round(3)
tbl["edge_pct"] = tbl["edge_pct"].map(lambda v: f"{v:+.1f}%")
st.dataframe(
    tbl.rename(columns={"stat": "Stat", "n": "Player-games",
                        "model_mae": "Model MAE",
                        "base_mae": "Season-average MAE",
                        "edge_pct": "Better by"}),
    width="stretch", hide_index=True)

notes = [
    "<b>This measures the number, not the call.</b> Mean absolute error says "
    "how close a projection lands, which is a different question from whether "
    "a lean against a posted line was right. The board's own record, on real "
    "posted lines, is under the slate on the home page.",
    "<b>A season average is a genuinely hard baseline.</b> It already knows "
    "who the player is, how his year is going and how often he plays. Beating "
    "it by a few percent over thousands of games is the realistic shape of an "
    "edge here, not a sign that something is broken.",
]
if dropped:
    notes.append(
        f"<b>{dropped:,} of {len(acc):,} scored rows are excluded.</b> They "
        "have a model error but no baseline error, almost always a player with "
        "no season-to-date history yet, and averaging the two columns "
        "separately over different row sets would flatter the model for free.")
for n in notes:
    st.markdown(f'<div class="gv-note">{n}</div>', unsafe_allow_html=True)

# ── By season, so a single good year cannot carry the headline ───────────────
if len(seasons) > 1:
    st.markdown(
        '<div class="gv-window"><span class="lab">By season</span>'
        '<span class="meta">each scored by a model trained only on what came '
        'before it</span></div>', unsafe_allow_html=True)
    per_season = (paired.groupby(["season", "target"])
                  .agg(model=("abs_err_model", "mean"),
                       base=("abs_err_b2", "mean"))
                  .reset_index())
    per_season["edge_pct"] = 100 * (1 - per_season.model / per_season.base)
    wide = (per_season.groupby("season")
            .agg(stats=("target", "nunique"),
                 better=("edge_pct", lambda s: int((s > 0).sum())),
                 mean_edge=("edge_pct", "mean"),
                 rows=("target", "size"))
            .reset_index())
    wide["mean_edge"] = wide["mean_edge"].map(lambda v: f"{v:+.1f}%")
    st.dataframe(
        wide.rename(columns={"season": "Season", "stats": "Stats scored",
                             "better": "Better than average",
                             "mean_edge": "Mean edge"})[
            ["Season", "Stats scored", "Better than average", "Mean edge"]],
        width="stretch", hide_index=True)

render_footer()
