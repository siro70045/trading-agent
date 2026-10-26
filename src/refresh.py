"""CLI: rinfresca universo e prezzi (idempotente)."""
from .db import init_db
from .prices import update_all
from .universe import list_active_tickers, sync_universe_to_db


def main() -> None:
    print("Inizializzo database...")
    init_db()

    n = sync_universe_to_db()
    print(f"Universo sincronizzato: {n} ticker.")

    tickers = list_active_tickers()
    print(f"Refresh prezzi per {len(tickers)} ticker...\n")

    results = update_all(tickers)

    ok   = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "up_to_date")
    err  = sum(1 for r in results if r["status"] == "error")
    rows = sum(r["rows"] for r in results)

    print("\n--- Riepilogo ---")
    print(f"  OK aggiornati:  {ok}")
    print(f"  Già aggiornati: {skip}")
    print(f"  Errori:         {err}")
    print(f"  Righe inserite: {rows:,}")


if __name__ == "__main__":
    main()