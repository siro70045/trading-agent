"""CLI: genera, mostra, persiste i segnali di portafoglio target.

Uso:
  python -m src.signals               # data = oggi
  python -m src.signals --date 2025-06-13
  python -m src.signals --no-save     # solo stampa, non persiste
  python -m src.signals --history     # mostra ultimi 5 run salvati
"""
from __future__ import annotations

import argparse
from datetime import date, datetime

import pandas as pd

from .db import get_conn, init_db
from .strategy import Portfolio, build_portfolio


# -------------------- Persistenza --------------------

def save_signals(portfolio: Portfolio) -> datetime:
    """Persiste il portafoglio in tabella signals. Ritorna run_at usato."""
    run_at = datetime.now()
    with get_conn() as con:
        if not portfolio.positions:
            # Anche RISK_OFF_CASH va loggato come riga "CASH" per tracciabilità
            con.execute(
                """
                INSERT INTO signals (run_at, target_date, regime, ticker, weight,
                                     momentum, volatility, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [run_at, portfolio.target_date, portfolio.regime, "CASH",
                 1.0, None, None, "regime → all cash"],
            )
        else:
            rows = [
                (run_at, portfolio.target_date, portfolio.regime, p.ticker,
                 p.weight, p.momentum, p.volatility, p.reason)
                for p in portfolio.positions
            ]
            con.executemany(
                """
                INSERT INTO signals (run_at, target_date, regime, ticker, weight,
                                     momentum, volatility, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    return run_at


def load_last_signals() -> pd.DataFrame:
    """Carica il run più recente. DataFrame vuoto se non ci sono mai stati run."""
    with get_conn() as con:
        return con.execute(
            """
            WITH last AS (SELECT MAX(run_at) AS run_at FROM signals)
            SELECT s.*
            FROM signals s
            JOIN last ON last.run_at = s.run_at
            ORDER BY s.weight DESC
            """
        ).df()


def load_history(limit: int = 5) -> pd.DataFrame:
    """Riepilogo ultimi N run."""
    with get_conn() as con:
        return con.execute(
            """
            SELECT run_at, target_date, regime,
                   COUNT(*) FILTER (WHERE ticker != 'CASH')        AS n_positions,
                   STRING_AGG(ticker, ', ' ORDER BY weight DESC)   AS tickers
            FROM signals
            GROUP BY run_at, target_date, regime
            ORDER BY run_at DESC
            LIMIT ?
            """,
            [limit],
        ).df()


# -------------------- Diff con run precedente --------------------

def compute_turnover(prev: pd.DataFrame, new: Portfolio) -> dict:
    """Confronta il nuovo portafoglio con l'ultimo salvato."""
    new_dict = {p.ticker: p.weight for p in new.positions}
    if not new_dict:
        new_dict = {"CASH": 1.0}

    prev_dict = (
        dict(zip(prev["ticker"], prev["weight"])) if not prev.empty else {}
    )

    all_tickers = set(prev_dict) | set(new_dict)
    enters, exits, changes = [], [], []
    for t in sorted(all_tickers):
        old_w = prev_dict.get(t, 0.0)
        new_w = new_dict.get(t, 0.0)
        if old_w == 0 and new_w > 0:
            enters.append((t, new_w))
        elif old_w > 0 and new_w == 0:
            exits.append((t, old_w))
        elif abs(old_w - new_w) > 0.005:
            changes.append((t, old_w, new_w))

    # Turnover one-way: somma degli aumenti di peso (= somma diminuzioni)
    turnover = sum(max(0.0, new_dict.get(t, 0) - prev_dict.get(t, 0)) for t in all_tickers)
    return {
        "enters": enters, "exits": exits, "changes": changes, "turnover": turnover,
    }


def print_diff(diff: dict) -> None:
    if not diff["enters"] and not diff["exits"] and not diff["changes"]:
        print("\nNessuna variazione rispetto all'ultimo run.")
        return
    print(f"\nVariazioni (turnover one-way: {diff['turnover']*100:.1f}%):")
    for t, w in diff["enters"]:
        print(f"  ENTRA  {t:<6} → {w*100:5.1f}%")
    for t, w in diff["exits"]:
        print(f"  ESCE   {t:<6} (era {w*100:.1f}%)")
    for t, old, new in diff["changes"]:
        delta = (new - old) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  AGG.   {t:<6}   {old*100:5.1f}% → {new*100:5.1f}%  ({sign}{delta:.1f}%)")


# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera segnali di portafoglio target.")
    p.add_argument("--date", type=str, default=None,
                   help="Data target (YYYY-MM-DD), default oggi.")
    p.add_argument("--no-save", action="store_true",
                   help="Stampa il segnale senza persisterlo nel DB.")
    p.add_argument("--history", action="store_true",
                   help="Mostra cronologia degli ultimi run e basta.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_db()  # idempotente — assicura che la tabella signals esista

    if args.history:
        hist = load_history(limit=10)
        if hist.empty:
            print("Nessuno storico segnali.")
            return
        print("Cronologia segnali:\n")
        for _, row in hist.iterrows():
            ts = pd.to_datetime(row["run_at"]).strftime("%Y-%m-%d %H:%M")
            print(f"  {ts}  target={row['target_date']}  "
                  f"regime={row['regime']:<20}  n={row['n_positions']}")
            print(f"     {row['tickers']}")
        return

    target = (
        datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    )

    print(f"Generazione segnali per target_date = {target}\n")
    portfolio = build_portfolio(target_date=target)
    print(portfolio.summary())

    # Diff vs ultimo run (se esiste)
    prev = load_last_signals()
    diff = compute_turnover(prev, portfolio)
    print_diff(diff)

    if args.no_save:
        print("\n[--no-save] segnale NON persistito.")
        return

    run_at = save_signals(portfolio)
    print(f"\nSalvato run_at = {run_at.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()