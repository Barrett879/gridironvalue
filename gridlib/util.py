"""Small shared helpers: URL minting, time formatting, safe numeric display."""
from __future__ import annotations

import datetime as dt
import html
from urllib.parse import urlencode

import pandas as pd

from .theme import SENTINEL


def game_url(season: int, week: int, away: str, home: str, theme_dark: bool = False) -> str:
    """A shareable, human-readable deep link to one game's page.

    Streamlit Community Cloud serves the app in an iframe and the browser
    address bar never follows in-app navigation, so there is nothing correct for
    a user to copy. Links have to be MINTED explicitly, and they are minted with
    readable params (week/away/home) rather than an opaque id, so the URL says
    what it points at. The page resolves these back to a game_id server-side.
    """
    q = urlencode(
        {
            "season": season,
            "week": week,
            "away": away,
            "home": home,
            "theme": "dark" if theme_dark else "light",
        }
    )
    return f"/Game?{q}"


def fmt_kick(kick: pd.Timestamp | dt.datetime | None) -> str:
    """'Sun 1:00 PM ET'. Empty input renders the sentinel, never a crash."""
    if kick is None or pd.isna(kick):
        return SENTINEL
    hour = kick.strftime("%I").lstrip("0") or "12"
    return f"{kick.strftime('%a')} {hour}:{kick.strftime('%M %p')} ET"


def fmt_num(value, places: int = 1) -> str:
    """Round for display; missing renders as the sentinel."""
    if value is None or pd.isna(value):
        return SENTINEL
    return f"{float(value):.{places}f}"


def fmt_signed(value, places: int = 1) -> str:
    """'+3.5' / '-2.5'. Used for spreads, where the sign carries the meaning."""
    if value is None or pd.isna(value):
        return SENTINEL
    return f"{float(value):+.{places}f}"


def esc(value) -> str:
    """HTML-escape a value for interpolation into a rendered fragment.

    Every string that reaches an unsafe_allow_html block goes through this.
    Team names and stadium names come from a third-party feed, so they are not
    trusted input even though they are not user input.
    """
    return html.escape("" if value is None else str(value), quote=True)
