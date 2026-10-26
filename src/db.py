"""Connessione DuckDB e schema."""
from contextlib import contextmanager

import duckdb

from .config import DATA_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickers (
    ticker     VARCHAR PRIMARY KEY,
    name       VARCHAR,
    sector     VARCHAR,
    added_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active     BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS prices (
    ticker     VARCHAR NOT NULL,
    date       DATE    NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    volume     BIGINT,
    dividends  DOUBLE,
    splits     DOUBLE,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);
CREATE INDEX IF NOT EXISTS idx_prices_date   ON prices(date);

CREATE TABLE IF NOT EXISTS download_log (
    ticker          VARCHAR   NOT NULL,
    run_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rows_inserted   INTEGER,
    last_date       DATE,
    status          VARCHAR,
    error_message   VARCHAR
);

CREATE TABLE IF NOT EXISTS signals (
    run_at        TIMESTAMP NOT NULL,
    target_date   DATE      NOT NULL,
    regime        VARCHAR   NOT NULL,
    ticker        VARCHAR   NOT NULL,
    weight        DOUBLE    NOT NULL,
    momentum      DOUBLE,
    volatility    DOUBLE,
    reason        VARCHAR,
    PRIMARY KEY (run_at, ticker)
);

CREATE INDEX IF NOT EXISTS idx_signals_target_date ON signals(target_date);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id          VARCHAR PRIMARY KEY,
    run_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    start_date      DATE      NOT NULL,
    end_date        DATE      NOT NULL,
    cost_per_trade  DOUBLE    NOT NULL,
    config_json     VARCHAR,
    final_equity    DOUBLE,
    n_periods       INTEGER
);

CREATE TABLE IF NOT EXISTS backtest_equity (
    run_id        VARCHAR NOT NULL,
    date          DATE    NOT NULL,
    equity        DOUBLE  NOT NULL,
    regime        VARCHAR,
    turnover      DOUBLE,
    return_gross  DOUBLE,
    return_net    DOUBLE,
    PRIMARY KEY (run_id, date)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    run_id        VARCHAR NOT NULL,
    decision_date DATE    NOT NULL,
    exec_date     DATE    NOT NULL,
    ticker        VARCHAR NOT NULL,
    weight_old    DOUBLE,
    weight_new    DOUBLE,
    delta         DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_bt_equity_run   ON backtest_equity(run_id);
CREATE INDEX IF NOT EXISTS idx_bt_trades_run   ON backtest_trades(run_id);
"""


@contextmanager
def get_conn():
    DATA_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        yield con
    finally:
        con.close()


def init_db() -> None:
    with get_conn() as con:
        con.execute(SCHEMA)
