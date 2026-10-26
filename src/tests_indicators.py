"""Test numerici degli indicatori. Si lanciano con: python -m src.tests_indicators

Filosofia: verifichiamo i calcoli su esempi controllati dove conosciamo la risposta giusta,
non su dati reali. I dati reali servono solo a smoke-test (no NaN inattesi, range sensati).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import indicators as ind
from .db import get_conn


def assert_close(actual: float, expected: float, tol: float = 1e-9, label: str = "") -> None:
    if pd.isna(actual) and pd.isna(expected):
        return
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: atteso {expected}, ottenuto {actual}")


# -------------------- Test su dati sintetici --------------------

def test_moving_average_constant() -> None:
    """MA di una serie costante = costante stessa, dopo i primi (window-1) NaN."""
    s = pd.Series([10.0] * 250)
    ma = ind.moving_average(s, 200)
    assert pd.isna(ma.iloc[198]), "indice 198: deve essere NaN (199 sample)"
    assert_close(ma.iloc[199], 10.0, label="MA200 su costante")
    assert_close(ma.iloc[249], 10.0, label="MA200 stabilizzata")
    print("  OK  moving_average: costante")


def test_moving_average_known_values() -> None:
    """MA su sequenza nota: 1..10, MA(5) all'indice 4 = (1+2+3+4+5)/5 = 3."""
    s = pd.Series(np.arange(1, 11, dtype=float))
    ma = ind.moving_average(s, 5)
    assert_close(ma.iloc[4], 3.0, label="MA5 su 1..5")
    assert_close(ma.iloc[9], 8.0, label="MA5 su 6..10")
    print("  OK  moving_average: valori noti")


def test_momentum_12_1_pure_growth() -> None:
    """Crescita +0.1% al giorno per 300 giorni.
    A indice 252+: ritorno tra close[t-252] e close[t-21] = (1.001^231) - 1.
    """
    daily = 0.001
    n = 300
    close = pd.Series((1 + daily) ** np.arange(n))
    mom = ind.momentum_12_1(close)
    expected = (1 + daily) ** (252 - 21) - 1
    assert_close(mom.iloc[260], expected, tol=1e-10, label="mom_12_1 crescita pura")
    # Prima di 252 giorni: NaN
    assert pd.isna(mom.iloc[251]), "mom_12_1 prima di 252gg deve essere NaN"
    print("  OK  momentum_12_1: crescita pura")


def test_momentum_12_1_excludes_last_month() -> None:
    """Verifica che gli ultimi 21 giorni NON contino nel calcolo.
    Setup: prezzo costante per 252 giorni, poi crash di -50% nell'ultimo mese.
    Il momentum 12-1 dovrebbe essere ~0% perché esclude proprio il crash.
    """
    n = 280
    close = pd.Series([100.0] * n)
    close.iloc[-21:] = 50.0  # crash nell'ultimo mese
    mom = ind.momentum_12_1(close)
    # All'ultimo indice: confronta close[-22]=100 con close[-253]=100 → mom=0
    assert_close(mom.iloc[-1], 0.0, tol=1e-9, label="mom_12_1 esclude ultimo mese")
    print("  OK  momentum_12_1: esclude correttamente l'ultimo mese")


def test_realized_volatility_range() -> None:
    """Volatilità di una random walk gaussiana con sigma giornaliera nota.
    sigma_daily = 0.01 → sigma_annua attesa ≈ 0.01 * sqrt(252) ≈ 0.1587.
    """
    rng = np.random.default_rng(42)
    daily_sigma = 0.01
    n = 500
    log_returns = rng.normal(0, daily_sigma, n)
    close = pd.Series(100.0 * np.exp(np.cumsum(log_returns)))
    vol = ind.realized_volatility(close, 60)
    expected = daily_sigma * np.sqrt(252)
    last = vol.iloc[-1]
    # Tolleranza 30% per noise statistico su 60 sample
    assert abs(last - expected) / expected < 0.3, (
        f"vol fuori range: {last:.4f} vs atteso ~{expected:.4f}"
    )
    print(f"  OK  realized_volatility: {last:.4f} ~ atteso {expected:.4f}")


def test_dollar_volume() -> None:
    close = pd.Series([10.0] * 25)
    volume = pd.Series([1_000_000] * 25)
    dv = ind.dollar_volume(close, volume, 20)
    assert pd.isna(dv.iloc[18]), "indice 18: NaN (19 sample)"
    assert_close(dv.iloc[19], 10_000_000.0, label="dollar_volume_20")
    print("  OK  dollar_volume")


def test_add_all_indicators_validation() -> None:
    """Verifica che la funzione rifiuti input malformati."""
    # Colonne mancanti
    bad = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "close": [100.0]})
    try:
        ind.add_all_indicators(bad)
        raise AssertionError("doveva sollevare ValueError per volume mancante")
    except ValueError as e:
        assert "volume" in str(e)

    # Date non ordinate
    bad2 = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")],
        "close": [100.0, 99.0],
        "volume": [1_000, 1_000],
    })
    try:
        ind.add_all_indicators(bad2)
        raise AssertionError("doveva sollevare ValueError per date non ordinate")
    except ValueError as e:
        assert "ordinato" in str(e)
    print("  OK  add_all_indicators: validazione input")


# -------------------- Smoke test su dati reali --------------------

def test_real_data_smoke() -> None:
    """Su dati reali AAPL: gli indicatori devono dare numeri sensati."""
    with get_conn() as con:
        df = con.execute(
            """
            SELECT date, close, volume
            FROM prices
            WHERE ticker = 'AAPL'
            ORDER BY date
            """
        ).df()

    if df.empty:
        print("  SKIP  dati AAPL non disponibili")
        return

    enriched = ind.add_all_indicators(df)
    last = enriched.iloc[-1]

    # MA200 deve essere positiva e nello stesso ordine di grandezza del close
    assert last["ma200"] > 0, "MA200 deve essere > 0"
    ratio = last["close"] / last["ma200"]
    assert 0.3 < ratio < 3.0, f"close/MA200 fuori range plausibile: {ratio:.2f}"

    # mom_12_1: tra -90% e +500% (range storico difensivo)
    assert -0.9 < last["mom_12_1"] < 5.0, f"momentum implausibile: {last['mom_12_1']:.2f}"

    # vol_60: tra 5% e 100% annuo per AAPL
    assert 0.05 < last["vol_60"] < 1.0, f"vol implausibile: {last['vol_60']:.2f}"

    # dollar_volume: AAPL ha sempre miliardi al giorno
    assert last["dollar_volume_20"] > 1e8, "dollar volume troppo basso per AAPL"

    print(f"  OK  smoke test AAPL @ {last['date']}:")
    print(f"        close={last['close']:.2f}  MA200={last['ma200']:.2f}  "
          f"ratio={ratio:.2f}")
    print(f"        mom_12_1={last['mom_12_1']*100:+.1f}%  "
          f"vol_60={last['vol_60']*100:.1f}%  "
          f"dv_20={last['dollar_volume_20']/1e9:.1f}B$")


def main() -> None:
    print("Test indicators...\n")
    tests = [
        test_moving_average_constant,
        test_moving_average_known_values,
        test_momentum_12_1_pure_growth,
        test_momentum_12_1_excludes_last_month,
        test_realized_volatility_range,
        test_dollar_volume,
        test_add_all_indicators_validation,
        test_real_data_smoke,
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