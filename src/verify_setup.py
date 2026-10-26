"""Smoke test Fase 1: verifica che ogni dipendenza funzioni."""
import os
import sys
from pathlib import Path

import duckdb
import yfinance as yf
from anthropic import Anthropic
from dotenv import load_dotenv


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"  OK  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False


def test_yfinance() -> None:
    df = yf.download("SPY", period="5d", progress=False, auto_adjust=True)
    assert not df.empty, "nessun dato scaricato"
    assert "Close" in df.columns, "colonna Close mancante"


def test_duckdb() -> None:
    db_path = Path("data/trading.duckdb")
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE IF NOT EXISTS _smoke (x INTEGER)")
    con.execute("INSERT INTO _smoke VALUES (1)")
    (count,) = con.execute("SELECT COUNT(*) FROM _smoke").fetchone()
    con.execute("DROP TABLE _smoke")
    con.close()
    assert count >= 1


def test_anthropic() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY mancante nel file .env")
    client = Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": "Rispondi solo con: ok"}],
    )
    text = msg.content[0].text.strip().lower()
    assert "ok" in text, f"risposta inattesa: {text!r}"


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    print("Verifica ambiente...")
    results = [
        check("yfinance (download prezzi)", test_yfinance),
        check("duckdb (database locale)", test_duckdb),
        check("anthropic (API LLM)", test_anthropic),
    ]
    if all(results):
        print("\nFase 1 completa. Ambiente pronto.")
        sys.exit(0)
    print("\nQualcosa non va. Risolvi gli errori sopra prima di proseguire.")
    sys.exit(1)