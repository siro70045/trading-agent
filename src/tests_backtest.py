"""Test del backtest. Lancia: python -m src.tests_backtest

Filosofia: test su dati reali (DB DuckDB), poiché il backtest è strettamente
accoppiato al DB. Periodi scelti per coprire condizioni diverse (regime CASH,
RISK_ON, transizioni).
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd

from . import backtest as bt
from .strategy import StrategyConfig


# -------------------- Helpers --------------------

def _has_db_data() -> bool:
    """Verifica che ci siano dati sufficienti nel DB per i test."""
    from .db import get_conn
    with get_conn() as con:
        n = con.execute("SELECT COUNT(*) FROM prices WHERE ticker='SPY'").fetchone()[0]
    return n > 200


# -------------------- Smoke test --------------------

def test_smoke_two_year_run() -> None:
    """Run su 2 anni: equity finita, lunghezza coerente, niente NaN."""
    res = bt.run_backtest(date(2020, 1, 1), date(2022, 1, 1))
    assert len(res.equity) >= 50, f"Atteso ≥50 settimane, ottenuto {len(res.equity)}"
    assert len(res.equity) == len(res.regime_history) == len(res.turnover_history), (
        "Lunghezze equity/regime/turnover devono coincidere"
    )
    assert len(res.returns_net) == len(res.equity) - 1, (
        f"returns_net dovrebbe avere len(equity)-1 = {len(res.equity)-1}, "
        f"ottenuto {len(res.returns_net)}"
    )
    assert not res.equity.isna().any(), "equity contiene NaN"
    assert not res.returns_net.isna().any(), "returns_net contiene NaN"
    assert not res.returns_gross.isna().any(), "returns_gross contiene NaN"
    assert np.isfinite(res.equity).all(), "equity non finita"
    assert (res.equity > 0).all(), "equity deve restare positiva"
    print(f"  OK  smoke 2y: {len(res.equity)} settimane, "
          f"equity finale {res.equity.iloc[-1]:.3f}, "
          f"{len(res.trades)} trades")


# -------------------- Costo zero vs costo standard --------------------

def test_zero_cost_dominates_positive_cost() -> None:
    """A parità di periodo e config, costo=0 → equity finale ≥ costo>0."""
    start, end = date(2018, 1, 1), date(2021, 1, 1)
    res_free = bt.run_backtest(start, end, cost_per_trade=0.0)
    res_paid = bt.run_backtest(start, end, cost_per_trade=0.001)  # 10bps, marcato
    eq_free = res_free.equity.iloc[-1]
    eq_paid = res_paid.equity.iloc[-1]
    assert eq_free >= eq_paid - 1e-12, (
        f"costo zero ({eq_free:.4f}) deve essere ≥ costo positivo ({eq_paid:.4f})"
    )
    # Verifica anche che le turnover_history siano identiche (la decisione non
    # dipende dai costi: dipendono solo dai prezzi).
    assert res_free.turnover_history.equals(res_paid.turnover_history), (
        "I turnover devono essere identici tra le due run"
    )
    # I rendimenti lordi devono coincidere; quelli netti no.
    assert np.allclose(res_free.returns_gross.values,
                       res_paid.returns_gross.values, atol=1e-12), (
        "returns_gross devono essere identici"
    )
    diff = (res_free.returns_net - res_paid.returns_net).sum()
    assert diff > 0, "Almeno un periodo deve avere costi → ret_net free > ret_net paid"
    print(f"  OK  costi: free {eq_free:.4f} ≥ paid {eq_paid:.4f}  "
          f"(Δret tot = {diff*100:+.2f}%)")


# -------------------- Cash regime → rendimento 0 --------------------

def test_cash_to_cash_zero_return() -> None:
    """Settimane CASH→CASH (no turnover) devono avere return_net esattamente 0.

    Usiamo il 2012 dove min_history_days=504 forza CASH (SPY ha ~165g di storia).
    """
    res = bt.run_backtest(date(2012, 1, 1), date(2013, 6, 1))
    regimes = res.regime_history
    turnovers = res.turnover_history
    returns_net = res.returns_net
    returns_gross = res.returns_gross

    # Trova settimane CASH→CASH: regime[i] == regime[i-1] == CASH e turnover[i] == 0.
    # Per quelle, returns_net[i] (indicizzato a exec_dates[i]) deve essere 0.
    exec_dates = list(regimes.index)
    n_checked = 0
    for i in range(1, len(exec_dates)):
        prev = regimes.iloc[i - 1]
        curr = regimes.iloc[i]
        to = turnovers.iloc[i]
        if prev == "RISK_OFF_CASH" and curr == "RISK_OFF_CASH":
            assert abs(to) < 1e-12, (
                f"CASH→CASH ma turnover non zero: {to} a {exec_dates[i].date()}"
            )
            ts = exec_dates[i]
            rn = returns_net.loc[ts]
            rg = returns_gross.loc[ts]
            assert abs(rg) < 1e-12, (
                f"CASH→CASH gross non zero ({rg}) a {ts.date()}"
            )
            assert abs(rn) < 1e-12, (
                f"CASH→CASH net non zero ({rn}) a {ts.date()}"
            )
            n_checked += 1

    assert n_checked > 10, (
        f"Atteso >10 settimane CASH→CASH nel 2012, trovate {n_checked}"
    )
    # Verifica anche che l'equity sia piatta in quei periodi
    cash_segment = res.equity[res.regime_history == "RISK_OFF_CASH"]
    if len(cash_segment) > 5:
        # tra il primo CASH e l'ultimo CASH consecutivo all'inizio,
        # l'equity non deve cambiare (al netto del costo di entry inizial).
        first = cash_segment.iloc[0]
        # Le prime N settimane sono tutte cash → equity costante
        first_n = (regimes == "RISK_OFF_CASH").cumprod().sum()
        # cumprod su booleani: 1 finché vero, 0 al primo falso e poi sempre 0
        if first_n > 1:
            initial_cash_segment = res.equity.iloc[: int(first_n)]
            assert (initial_cash_segment == first).all(), (
                "Equity deve restare piatta nelle prime settimane tutte cash"
            )

    print(f"  OK  CASH→CASH: verificate {n_checked} transizioni con ret=0")


# -------------------- Test sanità su trades --------------------

def test_trades_are_consistent() -> None:
    """I trade tracciati devono essere coerenti coi turnover registrati."""
    res = bt.run_backtest(date(2019, 6, 1), date(2020, 6, 1))
    # Per ogni exec_date, somma dei delta positivi == turnover one-way registrato
    by_exec: dict = {}
    for tr in res.trades:
        by_exec.setdefault(tr.exec_date, []).append(tr)

    mismatches = 0
    for exec_d, trade_list in by_exec.items():
        # Note: il turnover include anche la quota cash, mentre i trades NON
        # contengono righe esplicite per CASH. Quindi la somma dei delta>0
        # dei ticker può essere ≤ del turnover one-way (la differenza è
        # l'aumento della quota cash).
        positive_delta = sum(max(0.0, t.delta) for t in trade_list)
        ts = pd.Timestamp(exec_d)
        if ts in res.turnover_history.index:
            turnover = res.turnover_history.loc[ts]
            if positive_delta > turnover + 1e-9:
                mismatches += 1

    assert mismatches == 0, f"{mismatches} mismatch tra trade delta e turnover"
    print(f"  OK  trades coerenti su {len(by_exec)} exec_date")


# -------------------- Benchmark sanity --------------------

def test_benchmarks_smoke() -> None:
    """Benchmark indicizzati come il backtest e con valori finiti positivi."""
    from . import benchmarks as bm
    start, end = date(2020, 1, 1), date(2022, 1, 1)
    spy = bm.build_spy_buy_and_hold(start, end)
    sf = bm.build_60_40(start, end)
    assert len(spy) > 50, f"SPY B&H ha troppe poche settimane: {len(spy)}"
    assert len(sf) > 50, f"60/40 ha troppe poche settimane: {len(sf)}"
    assert (spy > 0).all() and (sf > 0).all(), "equity deve restare positiva"
    assert spy.iloc[0] == 1.0 and sf.iloc[0] == 1.0, "Equity normalizzata a 1"
    # Il 60/40 deve avere volatilità minore del 100% SPY
    spy_vol = spy.pct_change().std()
    sf_vol = sf.pct_change().std()
    assert sf_vol < spy_vol, (
        f"60/40 vol ({sf_vol:.4f}) deve essere < SPY vol ({spy_vol:.4f})"
    )
    print(f"  OK  benchmarks: SPY {spy.iloc[-1]:.3f}, 60/40 {sf.iloc[-1]:.3f}, "
          f"vol(60/40)={sf_vol*100:.2f}% < vol(SPY)={spy_vol*100:.2f}%")


# -------------------- Persistenza --------------------

def test_confirm_weeks_blocks_premature_switch() -> None:
    """Whipsaw RISK_ON → DEFENSIVE (1 wk) → RISK_ON nel marzo-aprile 2018:
    con confirm_weeks=2 il regime non deve mai diventare DEFENSIVE.

    La baseline (confirm=0) deve mostrare almeno una settimana DEFENSIVE
    (sanity: il whipsaw esiste nei dati).
    """
    start, end = date(2018, 3, 1), date(2018, 5, 1)
    cfg_base = StrategyConfig()
    cfg_conf = StrategyConfig(regime_confirm_weeks=2)

    res_base = bt.run_backtest(start, end, config=cfg_base)
    res_conf = bt.run_backtest(start, end, config=cfg_conf)

    has_def_base = (res_base.regime_history == "RISK_OFF_DEFENSIVE").any()
    has_def_conf = (res_conf.regime_history == "RISK_OFF_DEFENSIVE").any()

    assert has_def_base, (
        "Il test si basa sulla presenza di un whipsaw verso DEFENSIVE nella "
        "baseline 2018-03/04. Se non c'è, il test perde di significato."
    )
    assert not has_def_conf, (
        f"confirm_weeks=2 deve sopprimere il whipsaw, ma DEFENSIVE compare "
        f"nelle settimane {res_conf.regime_history[res_conf.regime_history == 'RISK_OFF_DEFENSIVE'].index.tolist()}"
    )
    # Lunghezze e finiteness preservate
    assert len(res_conf.equity) == len(res_base.equity)
    assert not res_conf.equity.isna().any()
    print(f"  OK  confirm_weeks=2: blocca il whipsaw 2018-03-26 (baseline ha "
          f"{int(has_def_base)} DEFENSIVE, confirm ha {int(has_def_conf)})")


def test_baseline_default_unchanged() -> None:
    """Smoke di non-regressione: con default StrategyConfig, equity ed equity
    finale devono essere stabili tra run consecutive (algoritmo deterministico)
    e identici a configurazioni che esplicitano i default a zero.
    """
    start, end = date(2019, 1, 1), date(2020, 1, 1)
    res_default = bt.run_backtest(start, end)
    res_zero = bt.run_backtest(start, end, config=StrategyConfig(
        regime_buffer_pct=0.0, regime_confirm_weeks=0,
    ))
    assert (res_default.equity.values == res_zero.equity.values).all(), (
        "Default e (buffer=0, confirm=0) devono dare equity identica"
    )
    print(f"  OK  baseline non-regressione: equity finale {res_default.equity.iloc[-1]:.4f}")


def test_save_and_reload() -> None:
    """Salva un backtest e verifica che sia rileggibile dal DB."""
    from .db import get_conn
    res = bt.run_backtest(date(2019, 1, 1), date(2019, 6, 1))
    run_id = bt.save_backtest(res)
    with get_conn() as con:
        header = con.execute(
            "SELECT n_periods, final_equity FROM backtest_runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
        n_eq = con.execute(
            "SELECT COUNT(*) FROM backtest_equity WHERE run_id = ?", [run_id]
        ).fetchone()[0]
        n_tr = con.execute(
            "SELECT COUNT(*) FROM backtest_trades WHERE run_id = ?", [run_id]
        ).fetchone()[0]
        # Cleanup per non sporcare il DB
        con.execute("DELETE FROM backtest_equity WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM backtest_trades WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM backtest_runs   WHERE run_id = ?", [run_id])
    assert header is not None, "Header backtest_runs non trovato"
    assert header[0] == len(res.returns_net), (
        f"n_periods nel DB ({header[0]}) ≠ in-memory ({len(res.returns_net)})"
    )
    assert abs(header[1] - res.equity.iloc[-1]) < 1e-9, "final_equity divergente"
    assert n_eq == len(res.equity), f"backtest_equity rows {n_eq} ≠ {len(res.equity)}"
    assert n_tr == len(res.trades), f"backtest_trades rows {n_tr} ≠ {len(res.trades)}"
    print(f"  OK  save/reload: header+{n_eq} equity rows + {n_tr} trades")


# -------------------- Main --------------------

def main() -> None:
    if not _has_db_data():
        print("DB senza dati prezzi: salto i test backtest")
        sys.exit(0)
    print("Test backtest...\n")
    tests = [
        test_smoke_two_year_run,
        test_zero_cost_dominates_positive_cost,
        test_cash_to_cash_zero_return,
        test_trades_are_consistent,
        test_benchmarks_smoke,
        test_confirm_weeks_blocks_premature_switch,
        test_baseline_default_unchanged,
        test_save_and_reload,
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
