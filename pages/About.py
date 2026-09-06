"""About: what this is, what it is not, and how much to trust each number."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridlib import predict  # noqa: E402
from gridlib.theme import (  # noqa: E402
    SITE_NAME,
    render_footer,
    render_nav,
    render_page_chrome,
)

st.set_page_config(page_title=f"About · {SITE_NAME}",
                   page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
render_nav("About")

reg = predict.load_registry()
through = reg.get("trained_through") if reg else None

st.markdown(
    '<div class="gv-masthead">'
    '<div class="gv-kicker"><span class="gv-pip"></span>About</div>'
    '<div class="gv-mast-rule"></div>'
    f'<div class="gv-brand">{SITE_NAME}</div>'
    '<div class="gv-deck">Per-game expected stat lines for every projected NFL '
    'skill player, with the market as context and an honest account of which '
    'numbers are worth reading.</div>'
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("""
<div class="gv-note" style="font-size:0.95rem;max-width:74ch;line-height:1.65">

<b>Every number here is a median, not a prediction.</b> A projection of 68
receiving yards does not say a player will gain 68 yards. It says that across
the distribution of ways the game could go, the average is about 68. Individual
games scatter widely around that, and football scatters more than most sports.

<br><br><b>Football projection is much harder than baseball projection, and the
honest result reflects it.</b> A full 17-game NFL season has fewer player-events
than a quarter of a baseball season. Volume carries nearly all the signal;
efficiency is close to noise. That is exactly what the validation found.

</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="gv-window"><span class="lab">How much to trust each number'
    '</span></div>', unsafe_allow_html=True)

st.markdown("""
| what | how it was measured | verdict |
|---|---|---|
| Pass attempts, completions, carries, targets, receptions | 6 to 13% better than a season-to-date average, out of sample, on two independent seasons | **Worth reading** |
| Passing, rushing and receiving yards | 5 to 13% better | **Worth reading** |
| Passing TDs, interceptions, sacks taken | 2 to 8% better, and the smaller edges are within noise | Read with caution |
| Receiving and rushing TDs | **worse** than a plain season average | **Shown as a baseline, not a model** |
| Field goals and extra points | beat the player baselines but lose to a league constant | Close to a guess |
""")

st.markdown("""
<div class="gv-note" style="font-size:0.95rem;max-width:74ch;line-height:1.65">

Touchdowns are not a modelling failure to be fixed with a better algorithm. Year
over year, receiving touchdown rate has a stability of about 0.0155, which is
close to nothing. Where the model loses to a simple average, the site shows the
average and says so.

<br><br><b>One thing this does not yet do, and it matters.</b> Every projection
assumes the player plays. Among players with eight or more appearances in a
season, 74% miss at least one game and half miss three or more, so roughly 17%
of the slots a weekly board is asked about are players who will not be on the
field. There is now an availability model, and team totals are multiplied
through it, but a player prop is still a conditional number: read every line as
"if he plays".

<br><br><b>What it deliberately will not price.</b> Longest reception, longest
rush, longest completion and longest field goal are maxima over plays, not
central values, and a model that projects a per-game central value cannot price
a maximum. Those are refused by name rather than filled with a confidently
wrong number.

</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="gv-window"><span class="lab">What this is not</span></div>',
    unsafe_allow_html=True)

st.markdown(f"""
<div class="gv-note" style="font-size:0.95rem;max-width:74ch;line-height:1.65">

{SITE_NAME} is an informational analytics site. It is not a sportsbook, it
accepts no wagers, and it gives no betting advice. The model-versus-market page
compares a projection against a posted line and reports the difference. It never
computes a dollar expected value, because the Arena format sets payouts from the
entry mix rather than a fixed multiplier table, so any such figure would be
fiction.

<br><br>Nothing here fetches from PrizePicks. Their API blocks server-side
requests, and independently their terms prohibit automated access, so lines only
enter this site when you paste them yourself.

</div>
""", unsafe_allow_html=True)

if reg:
    st.markdown(
        '<div class="gv-window"><span class="lab">Model version</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="gv-facts">'
        f'<div class="gv-fact"><div class="gv-fact-lab">Version</div>'
        f'<div class="gv-fact-val">{reg.get("version")}</div></div>'
        f'<div class="gv-fact"><div class="gv-fact-lab">Trained through</div>'
        f'<div class="gv-fact-val">{through}</div></div>'
        f'<div class="gv-fact"><div class="gv-fact-lab">Targets</div>'
        f'<div class="gv-fact-val">{len(reg.get("targets", {}))}</div></div>'
        f'<div class="gv-fact"><div class="gv-fact-lab">Features</div>'
        f'<div class="gv-fact-val">{len(reg.get("feature_columns", []))}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

render_footer()
