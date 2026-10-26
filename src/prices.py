"""Download e persistenza prezzi storici."""
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .config import DEFAULT_HISTORY_YEARS, DOWNLOAD_PAUSE_SECONDS
from .db import get_conn

PRICE_COLS = ["open", "high", "low", "close", "volume", "dividends", "splits"]


def get_last_date(ticker: str) -> date | None:
    with get_conn() as con:
        result = con.execute(
            "SELECT MAX(date) FROM prices WHERE ticker = ?", [ticker]
        ).fetchone()
    return result[0] if result and result[0] else None


def fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Scarica OHLCV+actions da Yahoo, ritorna DataFrame normalizzato."""
    t = yf.Ticker(ticker)
    df = t.history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # end è esclusivo in yfinance
        auto_adjust=False,
        actions=True,
    )
    if df.empty:
        return df

    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"stock_splits": "splits"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = ticker

    for col in PRICE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[["ticker", "date"] + PRICE_COLS]


def upsert_prices(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with get_conn() as con:
        con.register("new_prices", df)
        con.execute(
            """
            INSERT INTO prices
            SELECT * FROM new_prices
            ON CONFLICT (ticker, date) DO UPDATE SET
                open      = EXCLUDED.open,
                high      = EXCLUDED.high,
                low       = EXCLUDED.low,
                close     = EXCLUDED.close,
                volume    = EXCLUDED.volume,
                dividends = EXCLUDED.dividends,
                splits    = EXCLUDED.splits
            """
        )
        con.unregister("new_prices")
    return len(df)


def log_download(ticker, rows, last_date, status, error=None):
    with get_conn() as con:
        con.execute(
            """
            INSERT INTO download_log (ticker, rows_inserted, last_date, status, error_message)
            VALUES (?, ?, ?, ?, ?)
            """,
            [ticker, rows, last_date, status, error],
        )


def update_ticker(ticker: str, history_years: int = DEFAULT_HISTORY_YEARS) -> dict:
    """Update incrementale di un singolo ticker."""
    today = date.today()
    last = get_last_date(ticker)

    if last is None:
        start, mode = today - timedelta(days=365 * history_years), "full"
    elif last >= today - timedelta(days=1):
        return {"ticker": ticker, "status": "up_to_date", "rows": 0, "last_date": last}
    else:
        start, mode = last + timedelta(days=1), "incremental"

    try:
        df = fetch_history(ticker, start=start, end=today)
        rows = upsert_prices(df)
        new_last = df["date"].max() if not df.empty else last
        log_download(ticker, rows, new_last, "ok")
        return {"ticker": ticker, "status": "ok", "rows": rows,
                "mode": mode, "last_date": new_last}
    except Exception as e:
        log_download(ticker, 0, last, "error", str(e))
        return {"ticker": ticker, "status": "error", "rows": 0, "error": str(e)}


def update_all(tickers: list[str]) -> list[dict]:
    """Aggiorna tutti i ticker con piccola pausa per evitare rate-limit."""
    results = []
    icons = {"ok": "OK", "up_to_date": "==", "error": "FAIL"}
    for i, ticker in enumerate(tickers, 1):
        r = update_ticker(ticker)
        results.append(r)
        icon = icons.get(r["status"], "??")
        line = f"  [{i:>3}/{len(tickers)}] {icon:<4}  {ticker:<6}"
        if r["status"] == "ok":
            line += f"  {r['rows']:>5} rows  ({r['mode']})  -> {r['last_date']}"
        elif r["status"] == "up_to_date":
            line += f"  già aggiornato a {r['last_date']}"
        else:
            line += f"  {r.get('error', '')[:80]}"
        print(line)
        if i < len(tickers):
            time.sleep(DOWNLOAD_PAUSE_SECONDS)
    return results