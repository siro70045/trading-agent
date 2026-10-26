"""Test metriche su valori controllati. Lancia: python -m src.tests_metrics

Filosofia: ogni metrica testata su input dove conosciamo la risposta esatta a mano,
poi un test di coerenza fra metriche su equity reale.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import metrics as m


def assert_close(actual: float, expected: float, tol: float = 1e-9, label: str = "") -> None:
    if pd.isna(actual) and pd.isna(expected):
        return
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: atteso {expected}, ottenuto {actual}")


# -------------------- CAGR --------------------

def test_cagr_doubling_in_one_year_daily() -> None:
    """Equity raddoppia in 252 giorni → CAGR = 100%."""
    eq = pd.Series([1.0, 2.0], index=pd.RangeIndex(2))
    # Con n_periods=1 e periods_per_year=252 → 2^(252/1) - 1 = enorme.
    # Per testare il caso standard usiamo 252 step giornalieri di crescita uniforme.
    daily = 2 ** (1 / 252)
    eq2 = pd.Series([daily ** i for i in range(253)])
    assert_close(m.cagr(eq2, 252), 1.0, tol=1e-9, label="CAGR 100% in 1 anno daily")
    print("  OK  cagr: raddoppio in un anno daily")


def test_cagr_quadrupling_in_two_years() -> None:
    """4x in 2 anni daily → CAGR = 100% (perché 4 = 2^2 = (1+1)^2)."""
    daily = 4 ** (1 / (252 * 2))
    eq = pd.Series([daily ** i for i in range(252 * 2 + 1)])
    assert_close(m.cagr(eq, 252), 1.0, tol=1e-9, label="CAGR 100% in 2 anni")
    print("  OK  cagr: 4x in 2 anni")


def test_cagr_negative() -> None:
    """Equity scende del 50% in un anno → CAGR = -50%."""
    daily = 0.5 ** (1 / 252)
    eq = pd.Series([daily ** i for i in range(253)])
    assert_close(m.cagr(eq, 252), -0.5, tol=1e-9, label="CAGR -50%")
    print("  OK  cagr: -50% in un anno")


# -------------------- Sharpe --------------------

def test_sharpe_constant_returns() -> None:
    """Rendimenti costanti positivi → vol=0 → Sharpe = NaN."""
    returns = pd.Series([0.001] * 252)
    assert pd.isna(m.sharpe_ratio(returns, 252)), "Sharpe deve essere NaN se vol=0"
    print("  OK  sharpe: NaN su rendimenti costanti")


def test_sharpe_known_value() -> None:
    """
    Generiamo rendimenti gaussiani con media μ e vol σ giornaliere note.
    Sharpe annualizzato atteso = (μ/σ) * sqrt(252).
    """
    rng = np.random.default_rng(123)
    mu_daily, sigma_daily = 0.0008, 0.012  # ~20% annuo, ~19% vol
    n = 5_000  # campione grande per ridurre varianza statistica
    rets = pd.Series(rng.normal(mu_daily, sigma_daily, n))
    expected = (mu_daily / sigma_daily) * np.sqrt(252)
    actual = m.sharpe_ratio(rets, 252)
    # Tolleranza assoluta: SE annualizzato dello Sharpe stimato è ~0.22 con N=5000
    assert abs(actual - expected) < 0.2, (
        f"Sharpe fuori range: {actual:.3f} vs atteso {expected:.3f}"
    )
    print(f"  OK  sharpe: {actual:.3f} ~ atteso {expected:.3f}")


# -------------------- Sortino --------------------

def test_sortino_no_negative_returns() -> None:
    """Tutti rendimenti positivi → Sortino = inf."""
    rets = pd.Series([0.001, 0.002, 0.0015, 0.003])
    assert m.sortino_ratio(rets, 252) == float("inf")
    print("  OK  sortino: inf se nessuna perdita")


def test_sortino_greater_than_sharpe_when_skewed_positive() -> None:
    """Su rendimenti con asimmetria positiva, Sortino > Sharpe.
    Costruiamo serie con poche perdite piccole e tanti guadagni medi.
    """
    rng = np.random.default_rng(7)
    pos = rng.uniform(0.005, 0.01, 200)
    neg = rng.uniform(-0.002, 0.0, 50)
    rets = pd.Series(np.concatenate([pos, neg]))
    sh = m.sharpe_ratio(rets, 252)
    so = m.sortino_ratio(rets, 252)
    assert so > sh, f"Sortino ({so:.2f}) deve essere > Sharpe ({sh:.2f}) su skew positivo"
    print(f"  OK  sortino: {so:.2f} > sharpe {sh:.2f} (skew positivo)")


# -------------------- Drawdown --------------------

def test_max_drawdown_known() -> None:
    """Equity 100→120→60→90: max DD = 60/120 - 1 = -50%."""
    eq = pd.Series([100, 120, 60, 90])
    assert_close(m.max_drawdown(eq), -0.5, label="max DD -50%")
    print("  OK  max_drawdown: -50% verificato")


def test_max_drawdown_monotonic_growth() -> None:
    """Crescita pura → DD = 0."""
    eq = pd.Series([100, 110, 120, 130, 140])
    assert_close(m.max_drawdown(eq), 0.0, label="DD su crescita pura")
    print("  OK  max_drawdown: 0 su crescita pura")


def test_drawdown_series_shape() -> None:
    """La serie DD ha lo stesso indice dell'equity, parte sempre da 0."""
    eq = pd.Series([100, 110, 105, 120, 90])
    dd = m.drawdown_series(eq)
    assert len(dd) == len(eq)
    assert_close(dd.iloc[0], 0.0, label="DD iniziale = 0")
    assert (dd <= 1e-12).all(), "DD deve essere sempre <= 0"
    print("  OK  drawdown_series: shape e segno corretti")


# -------------------- Calmar --------------------

def test_calmar_known() -> None:
    """Equity sintetica con drop calibrato. Verifica coerenza CAGR/DD/Calmar."""
    daily = 1.10 ** (1 / 252)
    eq = pd.Series([daily ** i for i in range(252 + 1)])
    eq.iloc[126:] = eq.iloc[126:] * 0.8
    # DD esatto: peak a indice 125, low subito dopo a indice 126.
    # eq[126_post] / eq[125] = (daily^126 * 0.8) / daily^125 = 0.8 * daily.
    expected_dd = 0.8 * daily - 1
    cagr_val = m.cagr(eq, 252)
    mdd = m.max_drawdown(eq)
    expected_calmar = cagr_val / abs(mdd)
    assert_close(mdd, expected_dd, tol=1e-12, label="DD esatto da formula")
    assert_close(m.calmar_ratio(eq, 252), expected_calmar, tol=1e-9, label="Calmar coerente")
    print(f"  OK  calmar: {expected_calmar:.3f} (CAGR {cagr_val*100:+.1f}%, "
          f"DD {mdd*100:.2f}%)")
    
# -------------------- Win rate & profit factor --------------------

def test_win_rate() -> None:
    rets = pd.Series([0.01, -0.01, 0.02, 0.0, -0.005])
    # >0: 2 su 5 = 40%
    assert_close(m.win_rate(rets), 0.4, label="win rate 40%")
    print("  OK  win_rate: 40% verificato")


def test_profit_factor() -> None:
    """Gains: 0.01+0.02 = 0.03. Losses: -0.01-0.005 = -0.015. PF = 0.03/0.015 = 2."""
    rets = pd.Series([0.01, -0.01, 0.02, 0.0, -0.005])
    assert_close(m.profit_factor(rets), 2.0, tol=1e-12, label="PF = 2")
    print("  OK  profit_factor: 2.0 verificato")


# -------------------- Coerenza summarize --------------------

def test_summarize_sanity() -> None:
    """Su equity sintetica realistica, tutte le metriche devono essere finite e coerenti."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.012, 1000)  # ~12% annuo, 19% vol
    eq = m.returns_to_equity(pd.Series(rets))
    s = m.summarize(eq, 252)
    assert not pd.isna(s.cagr), "CAGR non deve essere NaN"
    assert not pd.isna(s.sharpe), "Sharpe non deve essere NaN"
    assert s.max_drawdown <= 0, "max DD deve essere <= 0"
    assert 0 <= s.win_rate <= 1, "win rate deve essere [0,1]"
    # n_periods = numero di rendimenti = len(equity) - 1 (pct_change droppa il primo)
    assert s.n_periods == 999, f"atteso 999 rendimenti, ottenuto {s.n_periods}"
    print(f"  OK  summarize: CAGR {s.cagr*100:+.1f}%, Sharpe {s.sharpe:.2f}, "
          f"DD {s.max_drawdown*100:.1f}%, n={s.n_periods}")


def test_equity_returns_roundtrip() -> None:
    """returns_to_equity ∘ equity_to_returns dovrebbe ricostruire l'equity (modulo punto iniziale)."""
    eq = pd.Series([1.0, 1.05, 1.03, 1.10, 1.08])
    rets = m.equity_to_returns(eq)
    rebuilt = m.returns_to_equity(rets, initial=eq.iloc[0])
    # rebuilt parte dal secondo punto
    for i in range(len(rebuilt)):
        assert_close(rebuilt.iloc[i], eq.iloc[i + 1], tol=1e-12, label=f"roundtrip [{i}]")
    print("  OK  equity↔returns: roundtrip esatto")


def main() -> None:
    print("Test metrics...\n")
    tests = [
        test_cagr_doubling_in_one_year_daily,
        test_cagr_quadrupling_in_two_years,
        test_cagr_negative,
        test_sharpe_constant_returns,
        test_sharpe_known_value,
        test_sortino_no_negative_returns,
        test_sortino_greater_than_sharpe_when_skewed_positive,
        test_max_drawdown_known,
        test_max_drawdown_monotonic_growth,
        test_drawdown_series_shape,
        test_calmar_known,
        test_win_rate,
        test_profit_factor,
        test_summarize_sanity,
        test_equity_returns_roundtrip,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print()
    if failed == 0:
        print("Tutti i test passati.")
        sys.exit(0)
    print(f"{failed} test falliti.")
    sys.exit(1)


if __name__ == "__main__":
    main()
