"""Test diagnose. Lancia: python -m src.tests_diagnose

Filosofia: il modulo diagnose è essenzialmente formattazione + aggregazioni
note, quindi un singolo smoke test su run vuoto + un test sul run reale più
recente sono sufficienti a coprire i percorsi di errore.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from datetime import date, datetime

from . import diagnose as dg
from .db import get_conn, init_db


def test_diagnose_empty_run_does_not_crash() -> None:
    """Inserisce un backtest_run con solo header (no equity rows) e verifica
    che diagnose() stampi qualcosa senza sollevare eccezioni."""
    init_db()
    run_id = f"test_empty_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    try:
        with get_conn() as con:
            con.execute(
                """
                INSERT INTO backtest_runs
                    (run_id, run_at, start_date, end_date, cost_per_trade,
                     config_json, final_equity, n_periods)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [run_id, datetime.now(), date(2020, 1, 1), date(2020, 6, 30),
                 0.0005, "{}", None, 0],
            )

        run = dg.load_run(run_id)
        assert run is not None, "load_run su header esistente non deve dare None"
        assert len(run.equity) == 0, "equity deve essere vuota"
        assert len(run.regimes) == 0
        assert len(run.returns_net) == 0

        # diagnose() non deve crashare
        buf = io.StringIO()
        with redirect_stdout(buf):
            dg.diagnose(run)
        out = buf.getvalue()
        assert run_id in out, "L'output deve contenere il run_id"
        assert "niente da analizzare" in out.lower(), (
            "Deve segnalare assenza di dati"
        )

        # Test individuali sulle compute_*: devono ritornare strutture vuote
        # senza crashare
        dd = dg.compute_drawdown_anatomy(run.equity)
        assert dd["peak_date"] is None

        rs = dg.compute_regime_stats(run.regimes, run.returns_net)
        assert rs.empty, "regime_stats su run vuoto deve essere DataFrame vuoto"

        sw = dg.compute_regime_switches(run.regimes)
        assert sw["n_switches"] == 0
        assert sw["switches"] == []
        assert sw["whipsaws"] == []

        cs = dg.compute_turnover_and_costs(
            run.turnover, run.returns_gross, run.returns_net,
        )
        # Tutti NaN ma il dict non deve sollevare eccezioni
        assert "avg_turnover_weekly" in cs

        top, bottom = dg.compute_top_bottom_weeks(run.returns_net, run.regimes)
        assert top.empty and bottom.empty

        print("  OK  diagnose: run vuoto gestito senza crash")
    finally:
        # Cleanup
        with get_conn() as con:
            con.execute("DELETE FROM backtest_runs WHERE run_id = ?", [run_id])


def test_load_run_missing_returns_none() -> None:
    """run_id inesistente → load_run ritorna None senza errore."""
    res = dg.load_run("definitely_not_a_real_run_id_xyz")
    assert res is None
    print("  OK  diagnose: run_id sconosciuto → None")


def main() -> None:
    print("Test diagnose...\n")
    tests = [
        test_diagnose_empty_run_does_not_crash,
        test_load_run_missing_returns_none,
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
