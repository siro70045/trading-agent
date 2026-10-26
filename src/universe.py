"""Definizione e gestione dell'universo di trading."""
import csv

from .config import UNIVERSE_PATH
from .db import get_conn

# Universo starter: large cap diversificate per settore + ETF di riferimento.
# Abbastanza piccolo per scaricare in 1-2 minuti, abbastanza vario per sviluppo.
STARTER_UNIVERSE = [
    ("TLT",   "iShares 20+ Year Treasury",  "ETF Bond"),
    # Mega-cap tech / communication
    ("AAPL",  "Apple Inc.",                "Technology"),
    ("MSFT",  "Microsoft Corp.",           "Technology"),
    ("GOOGL", "Alphabet Inc.",             "Communication Services"),
    ("AMZN",  "Amazon.com Inc.",           "Consumer Discretionary"),
    ("META",  "Meta Platforms Inc.",       "Communication Services"),
    ("NVDA",  "NVIDIA Corp.",              "Technology"),
    # Healthcare
    ("JNJ",   "Johnson & Johnson",         "Healthcare"),
    ("UNH",   "UnitedHealth Group",        "Healthcare"),
    ("LLY",   "Eli Lilly and Co.",         "Healthcare"),
    # Financials
    ("JPM",   "JPMorgan Chase",            "Financials"),
    ("BRK-B", "Berkshire Hathaway",        "Financials"),
    ("V",     "Visa Inc.",                 "Financials"),
    # Industrials
    ("CAT",   "Caterpillar Inc.",          "Industrials"),
    ("HON",   "Honeywell International",   "Industrials"),
    # Consumer
    ("KO",    "Coca-Cola Co.",             "Consumer Staples"),
    ("PG",    "Procter & Gamble",          "Consumer Staples"),
    ("WMT",   "Walmart Inc.",              "Consumer Staples"),
    ("COST",  "Costco Wholesale",          "Consumer Staples"),
    # Energy
    ("XOM",   "Exxon Mobil",               "Energy"),
    ("CVX",   "Chevron Corp.",             "Energy"),
    # ETF benchmark
    ("SPY",   "SPDR S&P 500 ETF",          "ETF Broad"),
    ("QQQ",   "Invesco QQQ Trust",         "ETF Tech"),
    ("IWM",   "iShares Russell 2000",      "ETF Small-Cap"),
    ("EFA",   "iShares MSCI EAFE",         "ETF International"),
    ("EEM",   "iShares MSCI Emerging",     "ETF Emerging"),
    # ETF settoriali
    ("XLF",   "Financial Select SPDR",     "ETF Sector"),
    ("XLK",   "Technology Select SPDR",    "ETF Sector"),
    ("XLE",   "Energy Select SPDR",        "ETF Sector"),
    ("XLV",   "Health Care Select SPDR",   "ETF Sector"),
    ("GLD",   "SPDR Gold Trust",           "ETF Commodity"),
]


def write_starter_universe_csv() -> None:
    UNIVERSE_PATH.parent.mkdir(exist_ok=True)
    with UNIVERSE_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "name", "sector"])
        writer.writerows(STARTER_UNIVERSE)


def load_universe_from_csv() -> list[tuple[str, str, str]]:
    if not UNIVERSE_PATH.exists():
        write_starter_universe_csv()
    with UNIVERSE_PATH.open() as f:
        reader = csv.DictReader(f)
        return [(r["ticker"], r["name"], r["sector"]) for r in reader]


def sync_universe_to_db() -> int:
    """Idempotente: aggiorna la tabella `tickers` da CSV."""
    rows = load_universe_from_csv()
    with get_conn() as con:
        con.executemany(
            """
            INSERT INTO tickers (ticker, name, sector)
            VALUES (?, ?, ?)
            ON CONFLICT (ticker) DO UPDATE SET
                name   = EXCLUDED.name,
                sector = EXCLUDED.sector
            """,
            rows,
        )
        return len(rows)


def list_active_tickers() -> list[str]:
    with get_conn() as con:
        result = con.execute(
            "SELECT ticker FROM tickers WHERE active = TRUE ORDER BY ticker"
        ).fetchall()
    return [r[0] for r in result]
