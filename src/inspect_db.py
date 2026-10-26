"""CLI: ispezione veloce dello stato del database."""
from .db import get_conn


def main() -> None:
    with get_conn() as con:
        n_tickers = con.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]
        n_prices  = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        print(f"Tickers in universo:  {n_tickers}")
        print(f"Righe prezzi totali:  {n_prices:,}")

        if n_prices == 0:
            return

        d_min, d_max = con.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()
        print(f"Range date:           {d_min} → {d_max}")

        print("\nDettaglio per ticker:")
        rows = con.execute("""
            SELECT t.ticker, t.sector, MAX(p.date) AS last_date, COUNT(p.date) AS bars
            FROM tickers t
            LEFT JOIN prices p ON p.ticker = t.ticker
            GROUP BY t.ticker, t.sector
            ORDER BY t.ticker
        """).fetchall()
        for ticker, sector, last_date, bars in rows:
            ld = str(last_date) if last_date else "—"
            print(f"  {ticker:<6}  {sector:<24}  {ld}  ({bars} bars)")


if __name__ == "__main__":
    main()