"""Test della pipeline di strategia. Lancia con: python -m src.tests_strategy"""
from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from . import strategy as st


def assert_close(actual: float, expected: float, tol: float = 1e-6, label: str = "") -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: atteso {expected}, ottenuto {actual}")


# -------------------- Sizing --------------------

def test_inverse_vol_no_cap_needed() -> None:
    """Pesi semplici inverse-vol senza che scatti il cap."""
    vols = pd.Series({"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20, "E": 0.20})
    weights = st.inverse_vol_weights_with_cap(vols, max_weight=0.35)
    for t in vols.index:
        assert_close(weights[t], 0.20, label=f"equal vol → equal weight ({t})")
    assert_close(weights.sum(), 1.0, label="somma pesi")
    print("  OK  inverse-vol: pesi uguali quando vol uguali")


def test_inverse_vol_cap_triggered() -> None:
    """Un asset super-stabile dovrebbe sforare il cap → cap + ridistribuzione."""
    vols = pd.Series({"STABLE": 0.05, "B": 0.30, "C": 0.30, "D": 0.30, "E": 0.30})
    weights = st.inverse_vol_weights_with_cap(vols, max_weight=0.35)
    # Senza cap: STABLE avrebbe (1/0.05)/(1/0.05 + 4*1/0.30) = 20/(20+13.33) ≈ 60%
    assert_close(weights["STABLE"], 0.35, label="STABLE deve essere al cap")
    # Gli altri sono uguali tra loro per simmetria, somma = 0.65, ognuno 0.1625
    for t in ["B", "C", "D", "E"]:
        assert_close(weights[t], 0.65 / 4, label=f"{t} ridistribuito")
    assert_close(weights.sum(), 1.0, tol=1e-9, label="somma pesi dopo cap")
    print("  OK  inverse-vol: cap + ridistribuzione")


def test_inverse_vol_degenere_pochi_titoli() -> None:
    """Con 2 asset e cap 0.35: max possibile è 0.70, NON si arriva a 1."""
    vols = pd.Series({"A": 0.10, "B": 0.20})
    weights = st.inverse_vol_weights_with_cap(vols, max_weight=0.35)
    assert_close(weights["A"], 0.35, label="A al cap")
    assert_close(weights["B"], 0.35, label="B al cap")
    assert_close(weights.sum(), 0.70, label="somma sotto 1 (resto cash)")
    print("  OK  inverse-vol: caso degenere (resta cash)")


# -------------------- Regime --------------------

def _make_latest_row(ticker: str, sector: str, close: float, ma200: float,
                     mom: float = 0.10, vol: float = 0.20, dv: float = 1e9) -> dict:
    return {
        "ticker": ticker, "sector": sector, "close": close, "ma200": ma200,
        "mom_12_1": mom, "vol_60": vol, "dollar_volume_20": dv,
    }


def test_regime_risk_on() -> None:
    latest = pd.DataFrame([_make_latest_row("SPY", "ETF Broad", 500, 450)])
    regime, _ = st.detect_regime(latest, st.StrategyConfig())
    assert regime == "RISK_ON", f"atteso RISK_ON, ottenuto {regime}"
    print("  OK  regime: SPY > MA200 → RISK_ON")


def test_regime_risk_off_defensive() -> None:
    latest = pd.DataFrame([
        _make_latest_row("SPY", "ETF Broad", 400, 450),
        _make_latest_row("GLD", "ETF Commodity", 200, 180),
        _make_latest_row("TLT", "ETF Bond", 80, 90),
    ])
    regime, _ = st.detect_regime(latest, st.StrategyConfig())
    assert regime == "RISK_OFF_DEFENSIVE", f"atteso DEFENSIVE, ottenuto {regime}"
    print("  OK  regime: SPY giù, GLD su → RISK_OFF_DEFENSIVE")


def test_regime_risk_off_cash() -> None:
    latest = pd.DataFrame([
        _make_latest_row("SPY", "ETF Broad", 400, 450),
        _make_latest_row("GLD", "ETF Commodity", 170, 180),
        _make_latest_row("TLT", "ETF Bond", 80, 90),
    ])
    regime, _ = st.detect_regime(latest, st.StrategyConfig())
    assert regime == "RISK_OFF_CASH", f"atteso CASH, ottenuto {regime}"
    print("  OK  regime: tutto giù → RISK_OFF_CASH")


def test_regime_buffer_keeps_state() -> None:
    """SPY in zona morta: previous_regime determina il macro stato."""
    cfg = st.StrategyConfig(regime_buffer_pct=0.02)
    # SPY 101, MA200 100 → upper 102, lower 98. 101 è dentro la zona morta.
    latest_dz = pd.DataFrame([
        _make_latest_row("SPY", "ETF Broad", 101, 100),
        _make_latest_row("GLD", "ETF Commodity", 200, 180),  # difensivo valido
        _make_latest_row("TLT", "ETF Bond", 80, 90),         # difensivo invalido
    ])

    # prev=RISK_ON in zona morta → resta RISK_ON
    regime, _ = st.detect_regime(latest_dz, cfg, previous_regime="RISK_ON")
    assert regime == "RISK_ON", f"atteso RISK_ON (sticky), ottenuto {regime}"

    # prev=RISK_OFF_CASH in zona morta → macro RISK_OFF, ma defensive valido
    # → DEFENSIVE (la sotto-decisione è sempre rivalutata sui dati correnti)
    regime, _ = st.detect_regime(latest_dz, cfg, previous_regime="RISK_OFF_CASH")
    assert regime == "RISK_OFF_DEFENSIVE", f"atteso DEFENSIVE, ottenuto {regime}"

    # prev=RISK_OFF_DEFENSIVE in zona morta → resta RISK_OFF, defensive valido → DEFENSIVE
    regime, _ = st.detect_regime(latest_dz, cfg, previous_regime="RISK_OFF_DEFENSIVE")
    assert regime == "RISK_OFF_DEFENSIVE", f"atteso DEFENSIVE, ottenuto {regime}"

    # Sanity: senza previous_regime, comportamento legacy (close=101 > MA200=100 → RISK_ON)
    regime, _ = st.detect_regime(latest_dz, cfg, previous_regime=None)
    assert regime == "RISK_ON", "senza prev_regime, fallback legacy"

    print("  OK  regime buffer: zona morta mantiene il macro stato")


def test_regime_buffer_switches_when_outside() -> None:
    """Fuori dalla zona morta il buffer non blocca lo switch, anche se prev contraddice."""
    cfg = st.StrategyConfig(regime_buffer_pct=0.02)

    # SPY 110 > 100*1.02=102 → forza RISK_ON anche se prev era RISK_OFF
    latest_up = pd.DataFrame([
        _make_latest_row("SPY", "ETF Broad", 110, 100),
    ])
    regime, _ = st.detect_regime(latest_up, cfg, previous_regime="RISK_OFF_CASH")
    assert regime == "RISK_ON", f"atteso RISK_ON, ottenuto {regime}"

    # SPY 90 < 100*0.98=98 → forza RISK_OFF anche se prev era RISK_ON
    latest_down = pd.DataFrame([
        _make_latest_row("SPY", "ETF Broad", 90, 100),
        _make_latest_row("GLD", "ETF Commodity", 170, 180),
        _make_latest_row("TLT", "ETF Bond", 80, 90),
    ])
    regime, _ = st.detect_regime(latest_down, cfg, previous_regime="RISK_ON")
    assert regime == "RISK_OFF_CASH", f"atteso CASH, ottenuto {regime}"

    print("  OK  regime buffer: switch normale fuori dalla zona morta")


# -------------------- Filtri --------------------

def test_filter_excludes_etf() -> None:
    latest = pd.DataFrame([
        _make_latest_row("AAPL", "Technology", 200, 180),
        _make_latest_row("SPY", "ETF Broad", 500, 450),
        _make_latest_row("XLK", "ETF Sector", 200, 180),
    ])
    out = st.filter_stocks(latest, st.StrategyConfig())
    tickers = set(out["ticker"])
    assert tickers == {"AAPL"}, f"atteso solo AAPL, ottenuto {tickers}"
    print("  OK  filtro: esclude ETF")


def test_filter_excludes_below_ma200() -> None:
    latest = pd.DataFrame([
        _make_latest_row("AAPL", "Technology", 200, 180),  # sopra
        _make_latest_row("BAD",  "Technology", 90,  100),  # sotto
    ])
    out = st.filter_stocks(latest, st.StrategyConfig())
    assert set(out["ticker"]) == {"AAPL"}
    print("  OK  filtro: esclude prezzo < MA200")


def test_filter_excludes_low_liquidity() -> None:
    latest = pd.DataFrame([
        _make_latest_row("AAPL", "Technology", 200, 180, dv=1e9),
        _make_latest_row("ILLIQ", "Technology", 200, 180, dv=100),
    ])
    cfg = st.StrategyConfig(min_dollar_volume=500_000)
    out = st.filter_stocks(latest, cfg)
    assert set(out["ticker"]) == {"AAPL"}
    print("  OK  filtro: esclude bassa liquidità")


# -------------------- End-to-end su dati reali --------------------

def test_end_to_end_today() -> None:
    """Smoke test: build_portfolio su data attuale deve produrre output coerente."""
    portfolio = st.build_portfolio(target_date=date.today())
    # Coerenza somma pesi
    total = sum(p.weight for p in portfolio.positions)
    assert total <= 1.0001, f"somma pesi > 1: {total}"
    if portfolio.regime == "RISK_ON":
        assert len(portfolio.positions) <= 5, "max 5 posizioni in RISK_ON"
        for p in portfolio.positions:
            assert p.weight <= 0.351, f"{p.ticker} sopra il cap: {p.weight}"
            assert p.weight > 0, f"{p.ticker} peso non positivo"
    elif portfolio.regime == "RISK_OFF_CASH":
        assert len(portfolio.positions) == 0
    print(f"  OK  end-to-end: regime={portfolio.regime}, "
          f"positions={len(portfolio.positions)}, cash={portfolio.cash_weight*100:.1f}%")


def main() -> None:
    print("Test strategy...\n")
    tests = [
        test_inverse_vol_no_cap_needed,
        test_inverse_vol_cap_triggered,
        test_inverse_vol_degenere_pochi_titoli,
        test_regime_risk_on,
        test_regime_risk_off_defensive,
        test_regime_risk_off_cash,
        test_regime_buffer_keeps_state,
        test_regime_buffer_switches_when_outside,
        test_filter_excludes_etf,
        test_filter_excludes_below_ma200,
        test_filter_excludes_low_liquidity,
        test_end_to_end_today,
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