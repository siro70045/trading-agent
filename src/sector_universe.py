"""
sector_universe.py — universe and rules for the sector-rotation system (v2).

Design goal: remove the look-ahead selection bias of v1, where 20 single stocks
were picked in 2026 knowing which ones had won since 2011.

Universe rule: ALL nine original Select Sector SPDR ETFs, chosen by coverage
(every GICS sector that existed at launch), not by past performance.
  - all launched December 1998  ->  ~27 years of history, same start date
  - ETFs do not go bankrupt      ->  no survivorship bias
  - no stock picking             ->  no hindsight in the universe
XLRE (2015) and XLC (2018) are excluded for now: adding them later would
change the universe mid-test.
"""

SECTOR_UNIVERSE = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

RULES = {
    "signal":        "momentum 12-1",   # 12-month return, skipping the last month
    "relative":      "hold top 3 sectors by signal",
    "absolute":      "a selected sector enters only if its signal > 0, else that slot is cash",
    "weighting":     "equal weight (1/3 each)",
    "rebalance":     "monthly",
    "costs_bps":     5,
    "benchmark":     "SPY buy-and-hold",
}

# Fixed BEFORE seeing any result. Only robustness checks allowed, no tuning:
#   top-N in {2, 3, 4} and weekly vs monthly rebalance -> look for a plateau.
# Final hold-out (last 3 years) is run once, at the end.
