"""Benchmark passivi per confronto col backtest principale.

Convenzioni (per coerenza con backtest.py):
- Frequenza settimanale: ogni venerdì (decisione) → lunedì successivo (esecuzione open).
- L'equity curve è indicizzata per `exec_date` (lunedì o primo trading day successivo
  al venerdì), così è confrontabile riga-per-riga con la curva del backtest.
- Equity normalizzata a 1.0 al primo punto.

Benchmark implementati:
- SPY buy-and-hold: 100% SPY, nessun rebalance.
- 60/40: 60% SPY + 40% TLT, ribilanciato al primo trading day di ogni anno solare.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .db import get_conn


# -------------------- Helpers condivisi --------------------

def trading_days(start: date, end: date, calendar_ticker: str = "SPY") -> list[date]:
    """Lista ordinata di trading day in [start, end] usando un ticker come calendario."""
    with get_conn() as con:
        rows = con.execute(
            """
            SELECT date FROM prices
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            [calendar_ticker, start, end],
        ).fetchall()
    return [r[0] for r in rows]


def weekly_exec_dates(start: date, end: date, calendar_ticker: str = "SPY") -> list[date]:
    """Per ogni venerdì in [start, end], il primo trading day strettamente successivo.

    Restituisce solo exec_date <= end e tali da avere ancora un periodo successivo
    valido (cioè un altro trading day dopo). Coerente con la cadenza del backtest.
    """
    days = trading_days(start, end, calendar_ticker)
    if not days:
        return []
    days_set = set(days)
    # Trova ogni venerdì in [start, end]
    fridays: list[date] = []
    cur = start
    # Allinea cur al primo venerdì >= start
    cur += pd.Timedelta(days=(4 - cur.weekday()) % 7)
    while cur <= end:
        fridays.append(cur)
        cur += pd.Timedelta(days=7)
    # Mappa ogni venerdì → primo trading day strettamente successivo
    exec_dates: list[date] = []
    for fri in fridays:
        # Cerca il primo trading day > fri
        nxt = next((d for d in days if d > fri), None)
        if nxt is not None and nxt not in exec_dates:
            exec_dates.append(nxt)
    return exec_dates


def load_open_series(ticker: str, dates: list[date]) -> pd.Series:
    """Open price del ticker alle date richieste, indicizzato per data.

    Le date assenti nel DB vengono droppate (non forward-fillate): chi chiama
    deve gestire il subset coerente.
    """
    if not dates:
        return pd.Series(dtype=float)
    with get_conn() as con:
        df = con.execute(
            """
            SELECT date, open FROM prices
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            [ticker, dates[0], dates[-1]],
        ).df()
    df = df.set_index("date")
    # Allinea sul subset richiesto
    s = df["open"].reindex([pd.Timestamp(d) for d in dates])
    s.index = pd.DatetimeIndex(s.index)
    return s


# -------------------- Buy & hold --------------------

def build_spy_buy_and_hold(start: date, end: date) -> pd.Series:
    """Equity curve di un buy-and-hold 100% SPY, normalizzata a 1.0 al primo punto.

    Indicizzata sui medesimi exec_date del backtest principale.
    """
    exec_dates = weekly_exec_dates(start, end)
    spy = load_open_series("SPY", exec_dates).dropna()
    if spy.empty:
        return pd.Series(dtype=float)
    return spy / spy.iloc[0]


# -------------------- 60/40 --------------------

def build_60_40(
    start: date, end: date, weight_spy: float = 0.60, weight_tlt: float = 0.40,
) -> pd.Series:
    """Equity curve 60/40 SPY/TLT, ribilanciato al primo trading day di ogni anno.

    Logica: tra due rebalance, ogni asset cresce indipendentemente al proprio
    rendimento (drift dei pesi). Al rebalance, i pesi tornano a 60/40 e
    la nuova equity riparte come somma pesata.
    """
    exec_dates = weekly_exec_dates(start, end)
    spy = load_open_series("SPY", exec_dates)
    tlt = load_open_series("TLT", exec_dates)
    common = spy.dropna().index.intersection(tlt.dropna().index)
    spy = spy.loc[common]
    tlt = tlt.loc[common]
    if spy.empty:
        return pd.Series(dtype=float)

    spy_ret = spy.pct_change().fillna(0.0)
    tlt_ret = tlt.pct_change().fillna(0.0)

    # Anni rappresentati (l'inizio è il primo punto, poi ribilanciamo a ogni cambio anno)
    years = pd.DatetimeIndex(common).year
    rebalance_mask = pd.Series(False, index=common)
    rebalance_mask.iloc[0] = True
    for i in range(1, len(common)):
        if years[i] != years[i - 1]:
            rebalance_mask.iloc[i] = True

    equity = pd.Series(index=common, dtype=float)
    w_spy = weight_spy
    w_tlt = weight_tlt
    eq = 1.0
    equity.iloc[0] = eq

    for i in range(1, len(common)):
        # Crescita indipendente: ciascuna sleeve si muove col proprio ritorno
        eq_spy = eq * w_spy * (1.0 + spy_ret.iloc[i])
        eq_tlt = eq * w_tlt * (1.0 + tlt_ret.iloc[i])
        eq = eq_spy + eq_tlt
        if rebalance_mask.iloc[i]:
            w_spy, w_tlt = weight_spy, weight_tlt
        else:
            w_spy = eq_spy / eq if eq > 0 else weight_spy
            w_tlt = eq_tlt / eq if eq > 0 else weight_tlt
        equity.iloc[i] = eq

    return equity
