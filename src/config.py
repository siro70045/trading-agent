"""Configurazione centralizzata: path e costanti."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "trading.duckdb"
UNIVERSE_PATH = DATA_DIR / "universe.csv"
DEFAULT_LOOKBACK_DAYS = 260 

DEFAULT_HISTORY_YEARS = 15
DOWNLOAD_PAUSE_SECONDS = 0.5
