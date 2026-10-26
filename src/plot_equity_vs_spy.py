"""
plot_equity_vs_spy.py — grafico equity curve della strategia vs SPY buy-and-hold.

Legge un backtest salvato (tabelle backtest_runs / backtest_equity) e lo confronta con
SPY comprato e tenuto, campionato sulle stesse date di esecuzione (open del lunedì),
stessa convenzione di src/benchmarks.py. Entrambe le curve sono normalizzate a 1.0.

Uso (dalla cartella del progetto):
  .venv/bin/python -m src.plot_equity_vs_spy
  .venv/bin/python -m src.plot_equity_vs_spy --run-id bt_20260430_155416_255140
  .venv/bin/python -m src.plot_equity_vs_spy --no-footnote

Output: outputs/equity_vs_spy.png (300 dpi) e outputs/equity_vs_spy.pdf (vettoriale).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .config import DB_PATH

DEFAULT_RUN = "variant_buffer_20260501_002825_253662"  # variante BUFFER ±2%
ROOT = Path(DB_PATH).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

STRAT_COLOR = "#185FA5"
SPY_COLOR = "#888780"


def load_curves(run_id: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        exists = con.execute(
            "SELECT count(*) FROM backtest_runs WHERE run_id = ?", [run_id]
        ).fetchone()[0]
        if not exists:
            runs = con.execute(
                "SELECT run_id, start_date, end_date, final_equity FROM backtest_runs ORDER BY run_at"
            ).df()
            raise SystemExit(f"run_id '{run_id}' non trovato. Run disponibili:\n{runs.to_string(index=False)}")
        df = con.execute(
            """
            SELECT e.date, e.equity, p.open AS spy_open
            FROM backtest_equity e
            JOIN prices p ON p.ticker = 'SPY' AND p.date = e.date
            WHERE e.run_id = ?
            ORDER BY e.date
            """,
            [run_id],
        ).df()
    finally:
        con.close()
    if df.empty:
        raise SystemExit("Nessuna riga di equity (o nessun prezzo SPY alle stesse date).")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    out = pd.DataFrame({
        "strategy": df["equity"] / df["equity"].iloc[0],
        "spy": df["spy_open"] / df["spy_open"].iloc[0],
    })
    return out


def cagr(s: pd.Series) -> float:
    years = (s.index[-1] - s.index[0]).days / 365.25
    return s.iloc[-1] ** (1 / years) - 1


def drawdown(s: pd.Series) -> pd.Series:
    return s / s.cummax() - 1


def plot(curves: pd.DataFrame, run_id: str, footnote: bool) -> list[Path]:
    strat, spy = curves["strategy"], curves["spy"]
    dd_s, dd_b = drawdown(strat), drawdown(spy)
    lab_s = f"Strategy  (CAGR {cagr(strat):.1%}, max DD {dd_s.min():.1%})"
    lab_b = f"SPY buy-and-hold  (CAGR {cagr(spy):.1%}, max DD {dd_b.min():.1%})"

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#888780", "axes.grid": True,
        "grid.color": "#E6E4DC", "grid.linewidth": 0.7, "axes.axisbelow": True,
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08},
    )

    ax1.plot(strat.index, strat, color=STRAT_COLOR, lw=1.6, label=lab_s)
    ax1.plot(spy.index, spy, color=SPY_COLOR, lw=1.3, label=lab_b)
    ax1.set_yscale("log", base=2)
    ax1.yaxis.set_major_locator(mticker.LogLocator(base=2))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax1.set_ylabel("Growth of 1 (log scale)")
    ax1.set_title("Trend-following agent vs SPY — weekly backtest", loc="left",
                  fontsize=12, color="#2C2C2A")
    ax1.legend(frameon=False, loc="upper left")

    ax2.fill_between(dd_s.index, dd_s, 0, color=STRAT_COLOR, alpha=0.25, lw=0)
    ax2.plot(dd_s.index, dd_s, color=STRAT_COLOR, lw=0.9)
    ax2.plot(dd_b.index, dd_b, color=SPY_COLOR, lw=0.9)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax2.set_ylabel("Drawdown")

    if footnote:
        start, end = curves.index[0].strftime("%b %Y"), curves.index[-1].strftime("%b %Y")
        fig.text(
            0.01, -0.02,
            f"{start} – {end}. Weekly rebalancing, 5 bps transaction costs, price returns "
            f"(dividends excluded for both). In-sample backtest; the stock universe was chosen "
            f"ex post and is subject to survivorship bias.",
            fontsize=7.5, color="#5F5E5A", wrap=True,
        )

    OUT_DIR.mkdir(exist_ok=True)
    paths = [OUT_DIR / "equity_vs_spy.png", OUT_DIR / "equity_vs_spy.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=DEFAULT_RUN)
    ap.add_argument("--no-footnote", action="store_true")
    args = ap.parse_args()

    curves = load_curves(args.run_id)
    paths = plot(curves, args.run_id, footnote=not args.no_footnote)
    s, b = curves["strategy"], curves["spy"]
    print(f"run: {args.run_id}  ({curves.index[0].date()} → {curves.index[-1].date()}, {len(curves)} settimane)")
    print(f"strategia: x{s.iloc[-1]:.2f}  CAGR {cagr(s):.2%}  maxDD {drawdown(s).min():.2%}")
    print(f"SPY:       x{b.iloc[-1]:.2f}  CAGR {cagr(b):.2%}  maxDD {drawdown(b).min():.2%}")
    for p in paths:
        print("salvato:", p)


if __name__ == "__main__":
    main()
