"""Theme tokens, page chrome, nav, and footer for GridironValue.

The visual system is HoopsValue's by way of DiamondValue: the same surface,
text-ramp and accent tokens in dark and light, the same Space Grotesk + Manrope
self-hosted webfonts, the fixed pill nav with the red active state, and the
pinned theme toggle. Only the wordmark and the football-specific components
(week bar, window group headers, slate cards carrying the market line) are new.

Two deliberate departures from DiamondValue:
  - The fonts are declared ONCE per family with a variable weight range.
    DiamondValue ships three byte-identical copies of each file and declares
    three @font-face rules pointing at them, so a browser fetches the same
    bytes three times per family. Both files are variable fonts (Manrope wght
    200-800, Space Grotesk wght 300-700), so one declaration covers every
    weight the site uses.
  - The class prefix is `gv-`, not `dv-`.

SITE_NAME is the ONE place the product name lives (spec: a rename is a one-line
change). SENTINEL is defined here ONCE and is the only em dash permitted in a
string literal anywhere in the codebase; it renders in table cells for missing
values.
"""
from __future__ import annotations

import datetime as _dt

import streamlit as st

# ── The one-line rename ──────────────────────────────────────────────────────
SITE_NAME = "GridironValue"
SITE_TAGLINE = "Expected stat lines, not predictions"

# The single permitted em dash: the missing-value table sentinel.
SENTINEL = "—"

# Ship light by default (config.toml base="light" matches); dark is opt-in via
# the pinned toggle, persisted across page navigations via ?theme=.
THEME_DEFAULT_DARK = False

# Nav pages: (label, url). Home is rendered separately.
# Only pages that EXIST. A nav link to a page with no file renders as the app
# shell and then a client-side "page not found", which reads as a broken site
# rather than a missing feature. Accuracy (spec step 5) is not built yet, so it
# is not linked; the week-by-week table on /Props carries that role for now.
_NAV_PAGES = [
    ("About", "/About"),
]

# ── Tokens: HoopsValue's exact palette ───────────────────────────────────────
THEME_BASE_CSS = """
<style>
    :root {
        /* surfaces */
        --app-bg:      #0a0a14;
        --bg-base:     #0a0a14;
        --bg-nav:      #0a0a0a;
        --panel:       rgba(20, 20, 42, 0.55);
        --panel-solid: #15171d;
        --panel-2:     #1a1a2e;
        --panel-hover: rgba(30, 30, 56, 0.85);
        --panel-line:  rgba(80, 80, 110, 0.35);
        --hairline:    rgba(255, 255, 255, 0.08);
        --hairline-soft: rgba(255, 255, 255, 0.04);
        --nav-border:  #222;
        --nav-divider: #333;
        /* tinted surfaces */
        --tint-good:   #1a2e1a;
        --tint-bad:    #2e1a1a;
        --tint-even:   #1a1a2e;
        /* text ramp */
        --fg-1: #ffffff;
        --fg-2: #cdcdd5;
        --fg-3: #aaaaaa;
        --fg-4: #8a8a93;
        --fg-5: #777777;
        --fg-6: #666666;
        /* brand accents */
        --accent-red:  #e63946;
        --accent-teal: #16d4c1;
        --value-good:  #2ecc71;
        --value-bad:   #e74c3c;
        --gold:        #f1c40f;
        --blue:        #3498db;
        --orange:      #f39c12;
        --purple:      #9b59b6;
        --sky:         #7ec8e8;
        --amber:       #f0b35b;
        /* elevation + table polish */
        --shadow-card: 0 4px 16px rgba(0, 0, 0, 0.35);
        --shadow-hover: 0 10px 30px -8px rgba(0, 0, 0, 0.55);
        --rail-lift: 22%;   /* lifts dark team navies off the near-black bg */
        --row-tint: rgba(255, 255, 255, 0.025);
        --bar-tint: rgba(22, 212, 193, 0.16);
    }
    html, body, .stApp { background: var(--app-bg) !important; }
    .stApp, body { color: var(--fg-2); }
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 { color: var(--fg-1) !important; }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label,
    [data-baseweb="form-control-label"] { color: var(--fg-2) !important; }
    /* ── Widget surfaces ─────────────────────────────────────────────────
       STREAMLIT 1.59 REPLACED BASEWEB WITH REACT ARIA. On 1.59.1 there are
       ZERO `data-baseweb` elements in the DOM, so every `data-baseweb`
       selector (the ones DiamondValue and HoopsValue still use) silently
       matches nothing and selectboxes render in Streamlit's own config.toml
       theme. In dark mode that means a white dropdown in a dark app.

       The selectbox is now `.react-aria-ComboBox` wrapping a `[role="group"]`,
       and its dropdown is PORTALED TO document.body as
       `[data-testid="stSelectboxVirtualDropdown"]` with virtualized
       `[role="option"]` rows. Tokens live on :root, so the portal inherits
       them fine.

       Selectors are anchored on ARIA roles and Streamlit testids, never on the
       emotion hash classes (st-emotion-cache-*), which change between builds.
       The data-baseweb rules are kept below as a fallback so this styling
       survives a downgrade to a pre-1.59 Streamlit. */
    .react-aria-ComboBox [role="group"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
        color: var(--fg-1) !important;
    }
    .react-aria-ComboBox input {
        background: transparent !important;
        color: var(--fg-1) !important;
    }
    .react-aria-ComboBox [role="group"]:focus-within {
        border-color: var(--accent-teal) !important;
    }
    .react-aria-ComboBox button { background: transparent !important; }
    .react-aria-ComboBox svg,
    div[data-baseweb="select"] svg { fill: var(--fg-3) !important; color: var(--fg-3) !important; }

    [data-testid="stSelectboxVirtualDropdown"],
    ul[data-baseweb="menu"], div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] [role="listbox"] {
        background: var(--panel-solid) !important;
        border: 1px solid var(--panel-line) !important;
        box-shadow: var(--shadow-hover) !important;
    }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"],
    [data-testid="stSelectboxVirtualDropdown"] [role="option"] *,
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="option"] *,
    ul[data-baseweb="menu"] li { color: var(--fg-2) !important; }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused],
    [data-baseweb="popover"] [role="option"]:hover,
    ul[data-baseweb="menu"] li:hover { background: var(--panel-hover) !important; }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] * {
        background: var(--bar-tint) !important;
        color: var(--fg-1) !important;
        font-weight: 700 !important;
    }

    .react-aria-TextField input,
    [data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-baseweb="input"] {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
    }
    [data-testid="stTextInput"] input,
    div[data-baseweb="input"] input { color: var(--fg-1) !important; }
    [data-testid="stExpander"] details {
        background: var(--panel) !important;
        border-color: var(--panel-line) !important;
    }
    [data-testid="stButton"] button, .stButton button {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
        color: var(--fg-2) !important;
    }
    [data-testid="stButton"] button:hover, .stButton button:hover {
        background: var(--panel-hover) !important;
        border-color: var(--fg-5) !important;
        color: var(--fg-1) !important;
    }
</style>
"""

THEME_LIGHT_CSS = """
<style>
    :root {
        --app-bg:      linear-gradient(180deg, #fbfcfd 0%, #eef1f4 100%);
        --bg-base:     #f4f6f8;
        --bg-nav:      #ffffff;
        --panel:       #ffffff;
        --panel-solid: #ffffff;
        --panel-2:     #eef1f4;
        --panel-hover: #f1f3f6;
        --panel-line:  #e3e6eb;
        --hairline:    rgba(20, 22, 40, 0.10);
        --hairline-soft: rgba(20, 22, 40, 0.05);
        --nav-border:  #e3e6eb;
        --nav-divider: #c9ccd3;
        --tint-good:   #eafaf1;
        --tint-bad:    #fdeceb;
        --tint-even:   #eef3f8;
        --fg-1: #14142a;
        --fg-2: #3a3d48;
        --fg-3: #585c68;
        --fg-4: #71757f;
        --fg-5: #9aa0ab;
        --fg-6: #b3b8c2;
        --accent-teal: #0fae9d;
        --value-good:  #16a34a;
        --value-bad:   #dc3a2c;
        --gold:        #9a6a00;
        --amber:       #a8730a;
        --orange:      #b45f06;
        --purple:      #7d3fa8;
        --blue:        #2471a3;
        --sky:         #146c94;
        --shadow-card: 0 1px 2px rgba(20,22,40,.05);
        --shadow-hover: 0 8px 24px rgba(20,22,40,.12);
        --rail-lift: 0%;   /* true team hue on white */
        --row-tint: rgba(20, 22, 40, 0.028);
        --bar-tint: rgba(15, 174, 157, 0.15);
    }
</style>
"""

# ── Fonts + shared chrome ────────────────────────────────────────────────────
COMMON_CSS = """
<style>
    /* One declaration per family. Both files are VARIABLE fonts, so a weight
       range covers every weight the site uses from a single download. */
    @font-face{font-family:'Space Grotesk';font-style:normal;font-weight:300 700;
      font-display:swap;src:url('/app/static/fonts/space-grotesk-var.woff2') format('woff2');}
    @font-face{font-family:'Manrope';font-style:normal;font-weight:200 800;
      font-display:swap;src:url('/app/static/fonts/manrope-var.woff2') format('woff2');}

    html, body, .stApp, [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"], button, input {
        font-family: 'Manrope', -apple-system, sans-serif;
    }
    [data-testid="stHeading"] h1, [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3, .gv-brand, .gv-slate-away, .gv-slate-home {
        font-family: 'Space Grotesk', 'Manrope', sans-serif;
    }

    /* Clear the fixed nav bar */
    .block-container { padding-top: 3.6rem; max-width: 1360px;
        padding-left: 2.5rem !important; padding-right: 2.5rem !important; }
    /* Collapse the empty spacer rows Streamlit leaves around injected <style>
       blocks (the CSS still applies from a display:none subtree). */
    .stElementContainer:has([data-testid="stMarkdownContainer"] > style:only-child) {
        display: none !important;
    }
    #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
    [data-testid="stToolbar"]        { display: none !important; }
    [data-testid="stDecoration"]     { display: none !important; }
    [data-testid="stStatusWidget"]   { display: none !important; }
    [data-testid="stAppViewerBadge"] { display: none !important; }
    [data-testid="stSidebarNav"], [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }

    /* Fixed top nav bar: the pill nav, red active state */
    .top-nav {
        position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
        display: flex; align-items: center; gap: 0.25rem;
        padding: 0 1.5rem; padding-right: 3.5rem; height: 3rem;
        background: var(--bg-nav); border-bottom: 1px solid var(--nav-border);
        flex-wrap: nowrap;
    }
    .top-nav a {
        text-decoration: none; padding: 0.3rem 0.85rem; border-radius: 20px;
        font-size: 0.82rem; font-weight: 600; color: var(--fg-3);
        border: 1px solid transparent; transition: all 0.15s; white-space: nowrap;
    }
    .top-nav a:hover { border-color: var(--accent-red); color: var(--fg-1); text-decoration: none; }
    .top-nav a.active { background: var(--accent-red); border-color: var(--accent-red); color: #fff; }
    .top-nav .home-link {
        color: var(--fg-6); font-size: 0.82rem; font-weight: 500;
        padding: 0.3rem 0.7rem; margin-right: 0.25rem; border: none;
    }
    .top-nav .home-link:hover { color: var(--fg-1); border: none; }
    .top-nav .divider { color: var(--nav-divider); font-size: 0.75rem;
        margin: 0 0.1rem; user-select: none; }
    @media (max-width: 760px) {
        .top-nav { overflow-x: auto; overflow-y: hidden; scrollbar-width: none; padding-right: 4rem; }
        .top-nav::-webkit-scrollbar { display: none; }
        .top-nav::after { content: ""; flex: 0 0 8rem; }
    }

    /* Header action cluster: "Update lines", pinned into the nav bar just left
       of the theme toggle, same pill look. The keyed container IS the
       stVerticalBlock, so flex-direction goes straight on it; a
       `> [data-testid="stVerticalBlock"]` child selector matches nothing.
       On narrow screens it falls back into the page flow so it cannot overlap
       the nav links. */
    .st-key-gv_nav_actions {
        position: fixed;
        top: 1.5rem; right: 5.4rem;
        transform: translateY(-50%);
        z-index: 10000; width: auto !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.4rem !important;
    }
    .st-key-gv_nav_actions [data-testid="stElementContainer"],
    .st-key-gv_nav_actions [data-testid="stLayoutWrapper"] { width: auto !important; }
    .st-key-gv_nav_actions button {
        padding: 0.24rem 0.85rem !important; font-size: 0.82rem !important;
        font-weight: 600 !important; font-family: inherit !important;
        border-radius: 20px !important; background: transparent !important;
        border: 1px solid var(--panel-line) !important; color: var(--fg-2) !important;
        white-space: nowrap !important; min-height: 0 !important;
        line-height: 1.25 !important; transition: all 0.15s !important;
    }
    .st-key-gv_nav_actions button:hover {
        border-color: var(--accent-teal) !important;
        color: var(--accent-teal) !important; background: transparent !important;
    }
    .st-key-gv_nav_actions button svg { color: inherit !important; }
    @media (max-width: 900px) {
        .st-key-gv_nav_actions { position: static; transform: none;
                                 margin-top: 0.3rem; }
    }

    /* Pinned theme toggle (top-right, inside the nav bar row) */
    .st-key-gv_theme_toggle {
        position: fixed;
        top: 1.5rem; right: 0.9rem;      /* bar is 3rem tall: pin to its */
        transform: translateY(-50%);     /* midline, truly centered      */
        z-index: 10000; width: auto !important;
    }
    .st-key-gv_theme_toggle button {
        padding: 0.24rem 0.85rem !important; font-size: 0.82rem !important;
        font-weight: 600 !important; font-family: inherit !important;
        border-radius: 20px !important; background: transparent !important;
        border: 1px solid var(--panel-line) !important; color: var(--fg-2) !important;
        white-space: nowrap !important; min-height: 0 !important;
        line-height: 1.25 !important; transition: all 0.15s !important;
    }
    .st-key-gv_theme_toggle button:hover {
        border-color: var(--accent-teal) !important;
        color: var(--accent-teal) !important; background: transparent !important;
    }

    /* Brand + masthead */
    .gv-brand {
        font-size: clamp(2rem, 5vw, 2.7rem); font-weight: 700;
        letter-spacing: -0.03em; line-height: 1.0;
        color: var(--fg-1); margin: 0 0 0.1rem;
    }
    .gv-brand .accent { color: var(--accent-teal); }
    .gv-masthead { margin: 0.2rem 0 0.2rem; }
    .gv-mast-rule { position: relative; height: 1px; background: var(--panel-line);
        margin-bottom: 0.9rem; }
    .gv-mast-rule::before { content: ""; position: absolute; left: 0; top: -1px;
        width: 44px; height: 3px; background: var(--accent-teal); }
    .gv-mast-row { display: flex; justify-content: space-between;
        align-items: flex-end; gap: 1.25rem; flex-wrap: wrap; }
    .gv-kicker { display: flex; align-items: center;
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--fg-4); margin-bottom: 0.45rem; }
    /* The football: a squashed diamond. Same visual weight as DiamondValue's
       rotated square, shaped for the sport. */
    .gv-pip { width: 9px; height: 6px; background: var(--accent-teal);
        border-radius: 50% / 60%; margin-right: 0.5rem; flex: none;
        transform: rotate(-12deg); }
    .gv-mast-summary { display: flex; align-items: flex-end; margin-left: auto; }
    .gv-sum { padding: 0 0.95rem; text-align: right; }
    .gv-sum + .gv-sum { border-left: 1px solid var(--panel-line); }
    .gv-sum:last-child { padding-right: 0; }
    .gv-sum-num { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 1.2rem; color: var(--fg-1); font-variant-numeric: tabular-nums;
        line-height: 1.1; white-space: nowrap; }
    .gv-sum-lab { font-size: 0.57rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--fg-5); margin-top: 0.2rem; }
    .gv-deck { color: var(--fg-3); font-size: 0.95rem; line-height: 1.5;
        max-width: 62ch; margin: 0.75rem 0 0; }
    @media (max-width: 640px) { .gv-mast-summary { margin: 0.8rem 0 0; }
        .gv-sum:first-child { padding-left: 0; } }

    /* Week bar (keyed container): chevrons + week selector + This week */
    .st-key-gv_weekbar [data-testid="stHorizontalBlock"] { align-items: flex-end; }
    .st-key-gv_weekbar [data-testid="stWidgetLabel"] p {
        font-family: 'Space Grotesk', sans-serif; font-size: 0.62rem;
        font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
        color: var(--fg-5); }
    .st-key-gv_weekbar [data-testid="stButton"] button {
        height: 2.35rem; font-size: 0.82rem; font-weight: 600;
        border-radius: 10px !important; }
    .st-key-gv_week_prev button, .st-key-gv_week_next button {
        padding: 0 0.5rem !important; font-size: 1.1rem !important;
        font-weight: 700 !important; line-height: 1 !important; }
    .st-key-gv_week_now button {
        border-color: var(--accent-teal) !important; color: var(--accent-teal) !important; }
    .gv-bar-rule { height: 1px; background: var(--panel-line); margin: 0.5rem 0 0.2rem; }

    /* Window group header: "Sunday early - 8 games" */
    .gv-window {
        display: flex; align-items: baseline; gap: 0.6rem;
        margin: 1.15rem 0 0.55rem; padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--hairline);
    }
    .gv-window .lab {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--fg-3);
    }
    .gv-window .meta { font-size: 0.7rem; color: var(--fg-5);
        font-variant-numeric: tabular-nums; }
    .gv-window .lock { margin-left: auto; font-size: 0.66rem; color: var(--fg-5); }

    /* Slate grid: square cards, 4 across, team-color split stripe + duotone */
    .gv-slate-grid {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 0.8rem; margin: 0.2rem 0 1.2rem;
    }
    @media (max-width: 1080px) { .gv-slate-grid { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 760px)  { .gv-slate-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 460px)  { .gv-slate-grid { grid-template-columns: 1fr; } }
    a.gv-slate-card {
        position: relative; overflow: hidden;
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; min-height: 9.5rem;
        padding: 1.15rem 0.6rem 0.75rem;
        background: var(--panel); border: 1px solid var(--panel-line);
        border-radius: 12px; text-decoration: none; text-align: center;
        box-shadow: var(--shadow-card);
        transition: border-color .14s ease, transform .14s ease, box-shadow .14s ease;
    }
    /* Top split stripe: away | 1px seam | home. Plain fallback, lifted when color-mix. */
    a.gv-slate-card::before {
        content: ""; position: absolute; inset: 0 0 auto 0; height: 4px;
        transition: height .14s ease;
        background: linear-gradient(90deg,
            var(--away, var(--fg-5)) 0 calc(50% - .5px),
            var(--panel-line) calc(50% - .5px) calc(50% + .5px),
            var(--home, var(--fg-5)) calc(50% + .5px) 100%);
    }
    @supports (color: color-mix(in srgb, red, white)) {
        a.gv-slate-card::before {
            background: linear-gradient(90deg,
                color-mix(in srgb, var(--away, var(--fg-5)), #fff var(--rail-lift)) 0 calc(50% - .5px),
                var(--panel-line) calc(50% - .5px) calc(50% + .5px),
                color-mix(in srgb, var(--home, var(--fg-5)), #fff var(--rail-lift)) calc(50% + .5px) 100%);
        }
        a.gv-slate-card {
            background: linear-gradient(150deg,
                color-mix(in srgb, var(--away, transparent) 6%, var(--panel)),
                color-mix(in srgb, var(--home, transparent) 6%, var(--panel)));
        }
    }
    a.gv-slate-card:hover {
        border-color: var(--accent-teal); transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }
    a.gv-slate-card:hover::before { height: 5px; }
    a.gv-slate-card:focus-visible { outline: 2px solid var(--accent-teal); outline-offset: 2px; }
    @media (prefers-reduced-motion: reduce) {
        a.gv-slate-card { transition: none; }
        a.gv-slate-card:hover { transform: none; }
    }
    .gv-slate-teams { display: flex; align-items: center; justify-content: center;
        gap: 0.55rem; }
    .gv-slate-away, .gv-slate-home {
        font-size: 1.45rem; font-weight: 700; color: var(--fg-1);
        line-height: 1.1; letter-spacing: -0.01em;
    }
    .gv-slate-at { color: var(--fg-5); font-weight: 600; font-size: 0.7rem; }
    .gv-slate-time {
        color: var(--fg-3); font-size: 0.72rem; font-weight: 600;
        margin-top: 0.55rem; letter-spacing: 0.01em; white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }
    /* The market readout: spread on the favorite, then the total. Informational
       context for the projection, never a wager prompt. */
    .gv-slate-mkt {
        display: inline-flex; align-items: baseline; gap: 0.45rem;
        margin-top: 0.4rem; font-size: 0.7rem; color: var(--fg-4);
        font-variant-numeric: tabular-nums; white-space: nowrap;
    }
    .gv-slate-mkt b { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        color: var(--fg-2); }
    .gv-slate-mkt .sep { color: var(--fg-6); }
    .gv-slate-final { margin-top: 0.4rem; font-size: 0.78rem; font-weight: 700;
        color: var(--fg-2); font-variant-numeric: tabular-nums; }
    .gv-slate-final .w { color: var(--fg-1); }
    /* Top-left micro-pill: the game's status word (Final, Live, or the day). */
    .gv-slate-tag {
        position: absolute; top: 7px; left: 7px; z-index: 1; pointer-events: none;
        padding: 0.1rem 0.4rem; border-radius: 999px;
        background: var(--hairline); border: 1px solid var(--panel-line);
        font-family: 'Space Grotesk', sans-serif; font-size: 0.52rem;
        font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
        color: var(--fg-5); line-height: 1.35; white-space: nowrap;
    }
    .gv-slate-tag.live { color: var(--value-bad); border-color: var(--value-bad); }
    .gv-slate-tag.final { color: var(--fg-4); }

    /* ── Game page ───────────────────────────────────────────────────────── */
    .gv-mh { margin: 0.2rem 0 1.1rem; }
    .gv-mh-tag {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--fg-4); margin-bottom: 0.5rem;
    }
    .gv-mh-row { display: flex; align-items: center; gap: 1.4rem; flex-wrap: wrap; }
    .gv-mh-team { display: flex; flex-direction: column; gap: 0.1rem; }
    .gv-mh-abbr {
        font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem;
        font-weight: 700; line-height: 1; letter-spacing: -0.02em;
        color: var(--c);
    }
    /* On white the true team hue reads; on near-black the dark navies need a
       lift, the same --rail-lift trick the slate cards use. */
    @supports (color: color-mix(in srgb, red, white)) {
        .gv-mh-abbr { color: color-mix(in srgb, var(--c), #fff var(--rail-lift)); }
    }
    .gv-mh-name { font-size: 0.76rem; color: var(--fg-4); font-weight: 600; }
    .gv-mh-at { color: var(--fg-5); font-size: 0.85rem; font-weight: 600; }
    .gv-mh-score {
        display: flex; align-items: baseline; gap: 0.5rem;
        font-family: 'Space Grotesk', sans-serif; font-size: 2rem;
        font-weight: 700; color: var(--fg-4); font-variant-numeric: tabular-nums;
    }
    .gv-mh-score .w { color: var(--fg-1); }
    .gv-mh-score .dash { color: var(--fg-6); font-size: 1.2rem; }
    .gv-mh-when {
        margin-top: 0.7rem; color: var(--fg-3); font-size: 0.85rem;
        font-variant-numeric: tabular-nums;
    }
    .gv-mh-when .sep { color: var(--fg-6); margin: 0 0.5rem; }

    /* Fact strip: the exogenous inputs to a game script */
    .gv-facts {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.1rem 0; background: var(--panel); border: 1px solid var(--panel-line);
        border-radius: 12px; overflow: hidden; margin-bottom: 1.4rem;
    }
    .gv-fact { padding: 0.75rem 1rem; border-right: 1px solid var(--hairline); }
    .gv-fact:last-child { border-right: none; }
    .gv-fact-lab {
        font-size: 0.57rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--fg-5); margin-bottom: 0.28rem;
    }
    .gv-fact-val {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.98rem; color: var(--fg-1); font-variant-numeric: tabular-nums;
        line-height: 1.2;
    }
    .gv-fact-val .sep { color: var(--fg-5); font-weight: 500; }
    .gv-fact-hint {
        font-size: 0.62rem; color: var(--fg-5); margin-top: 0.28rem;
        line-height: 1.35;
    }

    /* Depth charts, one panel per team */
    .gv-depth {
        background: var(--panel); border: 1px solid var(--panel-line);
        border-radius: 12px; overflow: hidden; height: 100%;
    }
    .gv-depth-head {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 0.9rem; padding: 0.7rem 1rem; color: var(--on);
        background: var(--c); letter-spacing: -0.01em;
    }
    .gv-depth-grp { padding: 0.6rem 1rem 0.7rem; border-top: 1px solid var(--hairline); }
    .gv-depth-grp:first-of-type { border-top: none; }
    .gv-depth-pos {
        font-size: 0.57rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--fg-5); margin-bottom: 0.35rem;
    }
    .gv-depth-row {
        display: grid; grid-template-columns: 1.4rem 1fr; align-items: baseline;
        gap: 0.5rem; padding: 0.16rem 0; font-size: 0.86rem;
    }
    .gv-depth-row .rk {
        color: var(--fg-5); font-size: 0.7rem; font-weight: 700;
        font-variant-numeric: tabular-nums; text-align: right;
    }
    .gv-depth-row .nm { color: var(--fg-2); }
    .gv-depth-row:first-of-type .nm { color: var(--fg-1); font-weight: 600; }
    .gv-depth-empty { padding: 1rem; color: var(--fg-4); font-size: 0.85rem; }
    .gv-inj {
        display: inline-block; margin-left: 0.4rem; padding: 0.02rem 0.34rem;
        border-radius: 999px; font-size: 0.56rem; font-weight: 700;
        letter-spacing: 0.05em; text-transform: uppercase; vertical-align: middle;
    }
    .gv-inj-out { background: var(--tint-bad); color: var(--value-bad); }
    .gv-inj-doubtful { background: var(--tint-bad); color: var(--value-bad); }
    .gv-inj-questionable { background: var(--tint-even); color: var(--amber); }

    /* Per-game count of posted PrizePicks lines that map to a stat we project,
       summed across BOTH teams. Informational model-vs-market readout, never a
       wager prompt. A neutral top-LEFT micro-pill: top-right would read as a
       promotional notification badge. Absolutely positioned so it cannot shift
       the centred abbreviations, pointer-events:none so the card stays one
       click target, and shown only when the count is above zero. */
    .gv-slate-lines {
        position: absolute; top: 7px; left: 7px; z-index: 1;
        pointer-events: none;
        display: inline-flex; align-items: baseline; gap: 0.2rem;
        padding: 0.1rem 0.38rem; border-radius: 999px;
        background: var(--hairline); border: 1px solid var(--panel-line);
        font-family: 'Space Grotesk', sans-serif;
        line-height: 1; white-space: nowrap; font-variant-numeric: tabular-nums;
        transition: color .14s ease, border-color .14s ease, background .14s ease;
    }
    .gv-slate-lines b { font-weight: 700; font-size: 0.74rem; color: var(--fg-1); }
    .gv-slate-lines .u {
        font-size: 0.5rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; color: var(--fg-5);
    }
    a.gv-slate-card:hover .gv-slate-lines { border-color: var(--accent-teal); }
    a.gv-slate-card:hover .gv-slate-lines b,
    a.gv-slate-card:hover .gv-slate-lines .u { color: var(--accent-teal); }
    @supports (color: color-mix(in srgb, red, white)) {
        a.gv-slate-card:hover .gv-slate-lines {
            border-color: color-mix(in srgb, var(--accent-teal) 45%, var(--panel-line));
            background: color-mix(in srgb, var(--accent-teal) 12%, var(--panel));
        }
    }
    @media (prefers-reduced-motion: reduce) { .gv-slate-lines { transition: none; } }

    /* The model-vs-market ledger */
    .gv-ledger { background: var(--panel); border: 1px solid var(--panel-line);
        border-radius: 12px; overflow: hidden; margin: 0.2rem 0 0.6rem;
        font-variant-numeric: tabular-nums; }
    .gv-ledger-head, .gv-ledger-row {
        display: grid;
        grid-template-columns: minmax(9rem,1.6fr) minmax(7rem,1.2fr)
                               4.2rem 4.2rem 4.4rem 3.4rem;
        gap: 0.5rem; align-items: center; padding: 0.5rem 0.9rem;
    }
    .gv-ledger-head {
        font-family: 'Space Grotesk', sans-serif; font-weight: 600;
        font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--fg-4); background: var(--panel-2);
        border-bottom: 1px solid var(--panel-line);
    }
    .gv-ledger-row { border-bottom: 1px solid var(--hairline-soft);
        font-size: 0.85rem; }
    .gv-ledger-row:last-child { border-bottom: none; }
    .gv-ledger-row:nth-child(even) { background: var(--row-tint); }
    .gv-ledger .nm { color: var(--fg-1); font-weight: 600; display: flex;
        flex-direction: column; line-height: 1.25; overflow: hidden; }
    .gv-ledger .nm i { font-style: normal; font-size: 0.66rem;
        color: var(--fg-5); font-weight: 500; }
    .gv-ledger .st { color: var(--fg-3); font-size: 0.78rem; }
    .gv-ledger .num, .gv-ledger-head .num { text-align: right; }
    .gv-ledger .num { color: var(--fg-2); }
    .gv-ledger .num.line { color: var(--fg-4); }
    .gv-ledger .lean, .gv-ledger-head .lean { text-align: right;
        font-size: 0.72rem; font-weight: 700; }
    .gv-ledger .diff.over, .gv-ledger .lean.over { color: var(--value-good); }
    .gv-ledger .diff.under, .gv-ledger .lean.under { color: var(--value-bad); }

    /* Provenance badge: which numbers came from a model that passed its gate */
    .gv-src { display: inline-block; margin-left: 0.35rem;
        padding: 0.02rem 0.32rem; border-radius: 999px; font-size: 0.54rem;
        font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
        vertical-align: middle; }
    .gv-src.base { background: var(--tint-bad); color: var(--value-bad); }
    .gv-src.low  { background: var(--tint-even); color: var(--amber); }
    .gv-src.comp { background: var(--hairline); color: var(--fg-4); }

    /* Demon and Goblin markers. These lines are deliberately shifted off the
       standard number, so a big gap against one is manufactured rather than an
       edge, and it has to be visible at a glance. Coloured to match how
       PrizePicks itself signals them: the Demon red, the Goblin green. Outlined
       rather than filled so they read as an annotation on the stat and not as a
       verdict on the row, which is what the Gap and Lean columns are for. */
    .gv-odds {
        display: inline-block; margin-left: 0.35rem;
        padding: 0.02rem 0.34rem; border-radius: 999px;
        font-size: 0.54rem; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; vertical-align: middle;
        border: 1px solid currentColor; background: transparent;
        cursor: help;
    }
    .gv-odds.demon  { color: var(--value-bad); }
    .gv-odds.goblin { color: var(--value-good); }

    /* Measured per-stat reliability. Filled rather than outlined, because
       unlike the Demon marker this IS a verdict on the row: it says how much
       the lean beside it is worth. A pass-attempts gap and a receiving-yards
       gap of the same size are opposite in sign as evidence. */
    .gv-tier {
        display: inline-block; margin-left: 0.35rem;
        padding: 0.02rem 0.34rem; border-radius: 999px;
        font-size: 0.52rem; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; vertical-align: middle;
        white-space: nowrap; cursor: help;
    }
    .gv-tier.t-strong { background: var(--tint-good); color: var(--value-good); }
    .gv-tier.t-mod    { background: var(--bar-tint);  color: var(--accent-teal); }
    .gv-tier.t-weak   { background: var(--hairline);  color: var(--fg-4); }
    .gv-tier.t-none   { background: var(--hairline);  color: var(--fg-5); }
    .gv-tier.t-neg    { background: var(--tint-bad);  color: var(--value-bad); }
    .gv-tier.t-unk    { background: transparent; color: var(--fg-6);
                        border: 1px solid var(--panel-line); }

    /* A single player's lines, opened under his own roster row */
    .gv-pl { margin: 0.25rem 0 0.5rem 1.9rem; border-left: 2px solid var(--panel-line);
        padding-left: 0.7rem; font-variant-numeric: tabular-nums; }
    .gv-pl-head, .gv-pl-row {
        display: grid; grid-template-columns: minmax(6rem,1fr) 3.6rem 3.6rem 3.8rem 3rem;
        gap: 0.4rem; align-items: center; padding: 0.16rem 0;
    }
    .gv-pl-head { font-family: 'Space Grotesk', sans-serif; font-weight: 600;
        font-size: 0.56rem; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--fg-5); }
    .gv-pl-row { font-size: 0.78rem; }
    .gv-pl .st { color: var(--fg-3); }
    .gv-pl .num { text-align: right; color: var(--fg-2); }
    .gv-pl .num.line { color: var(--fg-4); }
    .gv-pl .lean { text-align: right; font-weight: 700; font-size: 0.68rem; }
    .gv-pl .diff.over, .gv-pl .lean.over { color: var(--value-good); }
    .gv-pl .diff.under, .gv-pl .lean.under { color: var(--value-bad); }
    /* Depth rows that carry lines get a quiet marker so a reader can see which
       players are worth opening without expanding every one. */
    .gv-depth-row.has-lines .nm::after {
        content: ""; display: inline-block; width: 5px; height: 5px;
        border-radius: 50%; background: var(--accent-teal);
        margin-left: 0.4rem; vertical-align: middle;
    }

    /* Notices: a quiet panel for "no games" / "data not published yet" states.
       Never let an empty result render as silence; say why it is empty. */
    .gv-notice {
        background: var(--panel); border: 1px solid var(--panel-line);
        border-left: 3px solid var(--amber); border-radius: 10px;
        padding: 0.85rem 1.1rem; margin: 0.9rem 0;
        color: var(--fg-3); font-size: 0.88rem; line-height: 1.55;
    }
    .gv-notice b { color: var(--fg-1); font-weight: 700; }

    .gv-note { color: var(--fg-5); font-size: 0.78rem; line-height: 1.55;
        max-width: 72ch; margin-top: 0.4rem; }
    a.gv-back {
        display: inline-block; margin: 1.4rem 0 0.2rem;
        color: var(--fg-3); font-size: 0.84rem; font-weight: 600;
        text-decoration: none; padding: 0.35rem 0.9rem; border-radius: 20px;
        border: 1px solid var(--panel-line); transition: all 0.15s;
    }
    a.gv-back:hover { color: var(--accent-teal); border-color: var(--accent-teal); }

    /* Footer */
    .gv-footer { margin-top: 3rem; padding-top: 1.1rem; font-size: 0.76rem; }
    .gv-foot-disc { color: var(--fg-5); line-height: 1.6; max-width: 78ch; }
    .gv-foot-disc a { color: var(--fg-4); text-decoration: underline; }
    .gv-foot-disc a:hover { color: var(--fg-2); }
    .gv-foot-rule { height: 1px; background: var(--panel-line); margin: 0.9rem 0 0.7rem; }
    .gv-foot-bottom { display: flex; justify-content: space-between;
        flex-wrap: wrap; gap: 0.5rem 1.1rem; color: var(--fg-5); }
    .gv-foot-bottom a { color: var(--fg-3); text-decoration: none; }
    .gv-foot-bottom a:hover { color: var(--fg-1); }
</style>
"""


def inject_theme() -> None:
    """Emit theme tokens (dark base + light override when active). Call once per
    page before any token-referencing CSS. Theme persists across navigations via
    ?theme=.
    """
    if "theme_dark" not in st.session_state:
        qp = st.query_params.get("theme")
        st.session_state["theme_dark"] = (
            (qp == "dark") if qp in ("dark", "light") else THEME_DEFAULT_DARK
        )
    st.markdown(THEME_BASE_CSS, unsafe_allow_html=True)
    if not st.session_state.get("theme_dark", THEME_DEFAULT_DARK):
        st.markdown(THEME_LIGHT_CSS, unsafe_allow_html=True)


def render_theme_toggle() -> bool:
    """Light/dark toggle backed by st.session_state['theme_dark']. Mirrors the
    choice into ?theme= so a full-reload navigation carries it. Returns dark.
    """
    st.session_state.setdefault("theme_dark", THEME_DEFAULT_DARK)

    def _flip():
        new_dark = not st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
        st.session_state["theme_dark"] = new_dark
        st.query_params["theme"] = "dark" if new_dark else "light"

    dark = st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    st.button("Light" if dark else "Dark", key="gv_theme_toggle_btn",
              on_click=_flip, help="Toggle light/dark")
    return st.session_state.get("theme_dark", THEME_DEFAULT_DARK)


def render_page_chrome() -> None:
    """One-call chrome: theme tokens + shared CSS. Call once after
    st.set_page_config on every page.
    """
    inject_theme()
    st.markdown(COMMON_CSS, unsafe_allow_html=True)


def render_nav(current: str, keep: dict[str, str] | None = None) -> None:
    """Fixed top nav bar plus the pinned theme toggle. `current` matches a label
    in _NAV_PAGES (or "Home").

    Links use target="_self", NOT "_top": Streamlit Community Cloud serves the
    app inside an iframe, and "_top" would navigate the outer wrapper (wrong
    origin) instead of the app frame, so the links appear dead. "_self"
    navigates the app's own frame, which works iframed or served directly.

    `keep` carries view state (season, week) into every destination, because
    each Streamlit page navigation is a fresh session and a bare href would
    reset it.
    """
    parts = [f"{k}={v}" for k, v in (keep or {}).items() if v not in (None, "")]
    parts.append("theme=" + ("dark" if st.session_state.get("theme_dark") else "light"))
    qs = "?" + "&".join(parts)

    home_cls = "active" if current == "Home" else ""
    links = (f'<a class="home-link {home_cls}" href="/{qs}" '
             f'target="_self">{SITE_NAME}</a>')
    links += '<span class="divider">|</span>'
    for label, url in _NAV_PAGES:
        cls = "active" if label == current else ""
        links += (f'<a class="{cls}" href="{url}{qs}" target="_self">{label}</a>')
    st.markdown(f'<div class="top-nav">{links}</div>', unsafe_allow_html=True)

    with st.container(key="gv_theme_toggle"):
        render_theme_toggle()


def render_footer() -> None:
    """Site-wide footer with the required nflverse / FTN Data attribution.

    nflverse data is CC-BY-SA 4.0 and attribution to FTN Data via nflverse is
    MANDATORY, so this is a licence obligation, not decoration. It ships from
    day one for the same reason DiamondValue attributes the MLB Stats API.
    """
    year = _dt.date.today().year
    html = (
        '<div class="gv-footer">'
        '<div class="gv-foot-disc">'
        "Play-by-play, schedule, market and roster data from "
        '<a href="https://github.com/nflverse/nflverse-data" target="_blank" '
        'rel="noopener">nflverse</a>, sourced in part from FTN Data and used '
        'under <a href="https://creativecommons.org/licenses/by-sa/4.0/" '
        'target="_blank" rel="noopener">CC BY-SA 4.0</a>. '
        f"{SITE_NAME} is an informational analytics site. It is not a "
        "sportsbook, it accepts no wagers, and it offers no betting advice. "
        "Every number shown is the mean of a modelled distribution, not a "
        "forecast of what will happen."
        "</div>"
        '<div class="gv-foot-rule"></div>'
        '<div class="gv-foot-bottom">'
        f"<div>&copy; {year} {SITE_NAME}. Every number is a modelled outcome, not a forecast.</div>"
        '<div><a href="/About" target="_self">About</a></div>'
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def theme_fig(fig):
    """Make a Plotly figure follow the active theme (charts cannot read CSS
    vars). Call inline at the plot site.
    """
    dark = st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    axis = "#cdcdd5" if dark else "#3a3d48"
    grid = "rgba(255,255,255,0.08)" if dark else "rgba(20,22,40,0.10)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=axis, family="Manrope, sans-serif"),
        hoverlabel=dict(
            bgcolor="#1a1a2e" if dark else "#ffffff",
            font=dict(color="#e8e8f0" if dark else "#14142a"),
        ),
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    return fig
