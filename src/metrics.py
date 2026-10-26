"""Metriche di performance per backtest. Funzioni pure: input pandas, output scalari/Series.

Convenzioni:
- I rendimenti sono espressi in forma decimale (0.01 = 1%).
- La frequenza di osservazione è dedotta dall'indice o passata esplicitamente
  (TRADING_DAYS_PER_YEAR per giornaliera, 52 per settimanale, 12 per mensile).
- Risk-free rate = 0 per default. Per confronti relativi tra strategie sullo stesso
  periodo questa scelta non altera l'ordinamento.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# -------------------- Conversioni base --------------------

def equity_to_returns(equity: pd.Series) -> pd.Series:
    """Equity curve → rendimenti periodici (pct_change), prima riga droppata."""
    return equity.pct_change().dropna()


def returns_to_equity(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Rendimenti → equity curve cumulativa."""
    return initial * (1.0 + returns).cumprod()


# -------------------- Metriche di rendimento --------------------

def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound Annual Growth Rate.

    Formula: (final/initial)^(periods_per_year/n_periods) - 1
    """
    if len(equity) < 2:
        return float("nan")
    n = len(equity) - 1  # numero di periodi (intervalli)
    total_return = equity.iloc[-1] / equity.iloc[0]
    if total_return <= 0:
        return float("nan")
    return total_return ** (periods_per_year / n) - 1.0


def total_return(equity: pd.Series) -> float:
    """Rendimento totale dall'inizio alla fine."""
    if len(equity) < 2:
        return float("nan")
    return equity.iloc[-1] / equity.iloc[0] - 1.0


# -------------------- Metriche risk-adjusted --------------------

def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free: float = 0.0,
) -> float:
    """Sharpe annualizzato. Robusto a std numericamente nulla."""
    if len(returns) < 2:
        return float("nan")
    rf_per_period = risk_free / periods_per_year
    excess = returns - rf_per_period
    std = excess.std()
    if not np.isfinite(std) or std < 1e-15:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free: float = 0.0,
) -> float:
    """Sortino annualizzato: penalizza solo deviazioni negative."""
    if len(returns) < 2:
        return float("nan")
    rf_per_period = risk_free / periods_per_year
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    dd_std = downside.std(ddof=0)
    if not np.isfinite(dd_std) or dd_std < 1e-15:
        return float("nan")
    return float(excess.mean() / dd_std * np.sqrt(periods_per_year))
# -------------------- Drawdown --------------------

def drawdown_series(equity: pd.Series) -> pd.Series:
    """Serie del drawdown punto per punto: (equity / running_max) - 1.

    Sempre <= 0. Il minimo è il max drawdown.
    """
    running_max = equity.cummax()
    return equity / running_max - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Massimo drawdown (sempre <= 0). NaN se equity vuota."""
    if len(equity) == 0:
        return float("nan")
    return float(drawdown_series(equity).min())


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """CAGR / |max drawdown|. Inf se DD = 0, NaN se equity insufficiente."""
    mdd = max_drawdown(equity)
    if pd.isna(mdd) or mdd == 0:
        return float("nan") if pd.isna(mdd) else float("inf")
    cagr_val = cagr(equity, periods_per_year)
    if pd.isna(cagr_val):
        return float("nan")
    return cagr_val / abs(mdd)


# -------------------- Win rate & profit factor --------------------

def win_rate(returns: pd.Series) -> float:
    """% di periodi con rendimento > 0."""
    if len(returns) == 0:
        return float("nan")
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Somma rendimenti positivi / |somma rendimenti negativi|.

    >1 = sistema profittevole. inf se non ci sono perdite.
    """
    if len(returns) == 0:
        return float("nan")
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()  # negativo o zero
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / abs(losses))


# -------------------- Riepilogo --------------------

@dataclass
class PerformanceSummary:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    n_periods: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def pretty(self, label: str = "") -> str:
        header = f"=== {label} ===\n" if label else ""
        return (
            f"{header}"
            f"  Total return:    {self.total_return*100:>8.1f}%\n"
            f"  CAGR:            {self.cagr*100:>8.2f}%\n"
            f"  Volatility:      {self.volatility*100:>8.2f}%\n"
            f"  Sharpe:          {self.sharpe:>8.2f}\n"
            f"  Sortino:         {self.sortino:>8.2f}\n"
            f"  Max drawdown:    {self.max_drawdown*100:>8.1f}%\n"
            f"  Calmar:          {self.calmar:>8.2f}\n"
            f"  Win rate:        {self.win_rate*100:>8.1f}%\n"
            f"  Profit factor:   {self.profit_factor:>8.2f}\n"
            f"  N periodi:       {self.n_periods:>8d}"
        )


def summarize(
    equity: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free: float = 0.0,
) -> PerformanceSummary:
    """Calcola tutte le metriche da una equity curve."""
    returns = equity_to_returns(equity)
    return PerformanceSummary(
        total_return=total_return(equity),
        cagr=cagr(equity, periods_per_year),
        volatility=float(returns.std() * np.sqrt(periods_per_year)) if len(returns) > 1 else float("nan"),
        sharpe=sharpe_ratio(returns, periods_per_year, risk_free),
        sortino=sortino_ratio(returns, periods_per_year, risk_free),
        max_drawdown=max_drawdown(equity),
        calmar=calmar_ratio(equity, periods_per_year),
        win_rate=win_rate(returns),
        profit_factor=profit_factor(returns),
        n_periods=len(returns),
    )
