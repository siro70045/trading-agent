"""Indicatori tecnici. Funzioni pure: input pandas, output pandas, niente side effects.

Convenzioni:
- Tutti gli input arrivano già ordinati per data crescente.
- I rendimenti sono giornalieri logaritmici quando serve la stabilità numerica
  (es. somma di rendimenti = rendimento totale), aritmetici altrove.
- Le finestre rolling usano `min_periods=window` (richiedono dati pieni):
  preferiamo NaN a valori instabili calcolati su finestre incomplete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def moving_average(close: pd.Series, window: int) -> pd.Series:
    """Media mobile semplice."""
    return close.rolling(window=window, min_periods=window).mean()


def momentum_12_1(close: pd.Series) -> pd.Series:
    """Rendimento da 252 a 21 giorni fa (12 mesi escluso ultimo mese).

    Per ogni data t: ritorno tra close[t-21] e close[t-252].
    Espresso in forma decimale (0.15 = +15%).
    """
    past_12m = close.shift(252)
    past_1m = close.shift(21)
    return past_1m / past_12m - 1.0


def realized_volatility(close: pd.Series, window: int = 60) -> pd.Series:
    """Volatilità annualizzata dei rendimenti log giornalieri.

    sigma = std(log_returns) * sqrt(252)
    Espressa in forma decimale (0.20 = 20% annuo).
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )


def dollar_volume(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Media mobile del volume in dollari."""
    dv = close * volume
    return dv.rolling(window=window, min_periods=window).mean()


def trailing_max(close: pd.Series, window: int | None = None) -> pd.Series:
    """Massimo trailing del prezzo. Se window=None, è il max cumulato."""
    if window is None:
        return close.cummax()
    return close.rolling(window=window, min_periods=1).max()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Arricchisce un DataFrame di prezzi con tutti gli indicatori della strategia.

    Input richiesto: colonne `date`, `close`, `volume`, ordinato per `date` crescente,
    relativo a un singolo ticker.

    Output: stesso DataFrame con colonne aggiunte:
      - ma200, mom_12_1, vol_60, dollar_volume_20
    """
    required = {"date", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {missing}")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("DataFrame deve essere ordinato per date crescente")

    out = df.copy()
    return out.assign(
        ma200=moving_average(out["close"], 200),
        mom_12_1=momentum_12_1(out["close"]),
        vol_60=realized_volatility(out["close"], 60),
        dollar_volume_20=dollar_volume(out["close"], out["volume"], 20),
    )
    return out
