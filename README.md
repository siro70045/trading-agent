# Trading Agent

A rule-based, trend-following trading system for US equities, built and backtested in Python. It is part of my MIT Maker Portfolio (2026); the portfolio PDF explains the mathematics and the results in detail.

## Status

- **v1, complete.** 20 large-cap US stocks plus a few ETFs, one decision a week. A 200-day trend filter, 12-1 momentum ranking (top 5), inverse-volatility weights capped at 35%, and a market regime (SPY above or below its 200-day average) that moves to gold and long Treasuries, or to cash. A ±2% dead band around the average cuts the share of regime switches reversed within four weeks from 46% to 25%.
- **v2, in progress.** The same idea on the nine original sector ETFs, to remove the selection bias I found in v1: I picked the 20 stocks in 2026, knowing which ones had won. The universe and the rules are fixed in `src/sector_universe.py` before any result; the backtest is being written.

## v1 results (Jan 2012 – Apr 2026, weekly, 5 bps per trade)

| | Strategy (band ±2%) | SPY buy-and-hold |
| --- | --- | --- |
| CAGR | 16.0% | 12.8% |
| Sharpe ratio (risk-free rate 0) | 1.00 | 0.83 |
| Max drawdown | −21.4% | −32.2% |

These numbers are **not evidence of an edge**: the stock universe was chosen with hindsight, which is why v2 exists. The SPY column is price-only, without dividends (about 1.5–2% a year).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.refresh             # download ~15 years of daily prices into data/trading.duckdb
python -m src.backtest            # v1 backtest from 2012, saved in the database
python -m src.compare_variants    # original rule vs ±2% band vs 2-week confirmation
python -m src.diagnose            # drawdowns, time in each regime, whipsaws, costs
python -m src.buffer_sweep        # band width from 0 to 4%: robustness check
python -m src.plot_equity_vs_spy  # equity curve and drawdowns vs SPY
python -m src.signals --no-save   # this week's target portfolio
```

The price database is not included, because Yahoo Finance data cannot be redistributed; `src.refresh` rebuilds it. Prices downloaded on a different date can differ slightly (dividend and split adjustments), so results may differ in the decimals.

`.env` is optional: `src/verify_setup.py` checks an Anthropic API key left over from an early experiment. The trading system itself does not use any language model.

## Tests

Run `python -m src.refresh` first, then:

```bash
python -m src.tests_indicators
python -m src.tests_metrics
python -m src.tests_strategy
python -m src.tests_backtest
python -m src.tests_diagnose
```

## Layout

- `src/indicators.py`: moving average, 12-1 momentum, realised volatility
- `src/strategy.py`: filter, ranking, capped inverse-volatility weights, regime detection
- `src/backtest.py`: weekly backtest (decision at Friday's close, execution at Monday's open, costs on turnover)
- `src/metrics.py`: CAGR, Sharpe, Sortino, drawdown, written from scratch
- `src/diagnose.py`, `src/compare_variants.py`, `src/buffer_sweep.py`: diagnostics and anti-whipsaw experiments
- `src/sector_universe.py`: v2 universe and rules, fixed before any result
- `outputs/buffer_sweep/`: results of the band-width sweep
- `docs/`: working notes (in Italian)

## How this was built

I designed the system, derived the mathematics, chose the data and the tests, and judged the results. Most of the Python code was written by Claude Code, an AI coding assistant, working from my specifications; I reviewed and tested it. Code comments and docstrings are partly in Italian.
