"""GridironValue: NFL per-game player projections.

Module boundaries mirror DiamondValue's mlblib/ on purpose (spec rule 3):
    cache.py    disk cache, atomic writes, stale-beats-empty, kickoff freshness
    fetch.py    nflverse data access, schedule/market, season and week resolution
    features.py point-in-time features shared by training and inference
    model.py    model artifacts and prediction
    store.py    HTML rendering (tables, cards)
    teams.py    team identity and colors
    theme.py    design tokens, page chrome, nav
    parse.py    input parsing and normalization
"""
