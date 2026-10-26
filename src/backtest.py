"""Backtest settimanale: decisione venerdì close, esecuzione lunedì open.

Convenzioni:
- L'unità di tempo è la settimana. Per ogni venerdì t calcoliamo il portafoglio
  target via build_portfolio(t). L'esecuzione avviene al primo trading day > t
  (tipicamente lunedì), allo `open`. Il rendimento della settimana è dato dal
  movimento del prezzo open tra esecuzioni consecutive.
- Equity curve indicizzata per `exec_date`. Punto iniziale = 1.0.
- Costi: convenzione `cost = turnover_one_way * cost_per_trade`. Il turnover
  one-way è la somma degli aumenti di peso tra portafoglio precedente e nuovo
  (= somma decrementi). Si applica una sola volta per rebalance.
- CASH ha rendimento 0 (semplificazione: ignora interessi).
- Universo dinamico: gestito dentro build_portfolio via StrategyConfig.min_history_days.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from .db import get_conn, init_db
from .strategy import Portfolio, Regime, StrategyConfig, build_portfolio


# -------------------- Output --------------------

@dataclass
class Trade:
    decision_date: date
    exec_date: date
    ticker: str
    weight_old: float
    weight_new: float

    @property
    def delta(self) -> float:
        return self.weight_new - self.weight_old


@dataclass
class BacktestResult:
    equity: pd.Series                              # indicizzato per exec_date
    returns_gross: pd.Series                       # rendimento lordo settimanale (pre-costi)
    returns_net: pd.Series                         # rendimento netto settimanale
    regime_history: pd.Series                      # regime applicato in ciascuna settimana
    turnover_history: pd.Series                    # turnover one-way per settimana
    trades: list[Trade]                            # tutte le variazioni di peso
    config: StrategyConfig
    start_date: date
    end_date: date
    cost_per_trade: float

    def __post_init__(self) -> None:
        self.equity.name = "equity"
        self.returns_gross.name = "return_gross"
        self.returns_net.name = "return_net"
        self.regime_history.name = "regime"
        self.turnover_history.name = "turnover"


# -------------------- Helper: calendario e prezzi --------------------

def _trading_days(start: date, end: date, calendar_ticker: str = "SPY") -> list[date]:
    with get_conn() as con:
        rows = con.execute(
            """
            SELECT date FROM prices
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            [calendar_ticker, start, end],
        ).fetchall()
    return [r[0] for r in rows]


def _fridays(start: date, end: date) -> list[date]:
    fridays: list[date] = []
    cur = start + pd.Timedelta(days=(4 - start.weekday()) % 7)
    while cur <= end:
        fridays.append(cur)
        cur += pd.Timedelta(days=7)
    return fridays


def _load_opens_matrix(tickers: list[str], dates: list[date]) -> pd.DataFrame:
    """Matrice open price [date × ticker]. NaN dove manca il dato."""
    if not tickers or not dates:
        return pd.DataFrame()
    with get_conn() as con:
        df = con.execute(
            f"""
            SELECT ticker, date, open FROM prices
            WHERE ticker IN ({",".join(["?"] * len(tickers))})
              AND date BETWEEN ? AND ?
            """,
            [*tickers, dates[0], dates[-1]],
        ).df()
    if df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]))
    pivot = df.pivot(index="date", columns="ticker", values="open")
    pivot.index = pd.DatetimeIndex(pivot.index)
    return pivot.reindex(pd.DatetimeIndex([pd.Timestamp(d) for d in dates]))


# -------------------- Logica di rebalance e rendimento --------------------

def _portfolio_to_weights(portfolio: Portfolio) -> dict[str, float]:
    """Pesi {ticker: weight}. CASH è omesso ed è implicito (1 - somma)."""
    return {p.ticker: p.weight for p in portfolio.positions}


def _turnover_one_way(prev: dict[str, float], new: dict[str, float]) -> float:
    """Somma degli aumenti di peso. Equivale alla somma dei decrementi.

    Trattamento di CASH: lo includiamo esplicitamente come asset con peso
    `1 - somma(non-cash)`. Così, anche un passaggio da 100% azioni a 100% cash
    produce turnover = 1 (vendita totale).
    """
    keys = set(prev) | set(new)
    cash_prev = 1.0 - sum(prev.values())
    cash_new = 1.0 - sum(new.values())
    inc = sum(max(0.0, new.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
    inc += max(0.0, cash_new - cash_prev)
    return inc


def _gross_period_return(
    weights: dict[str, float],
    open_at_exec: pd.Series,    # serie indicizzata per ticker
    open_at_next: pd.Series,
) -> float:
    """Rendimento lordo: somma pesata dei rendimenti per ticker. CASH = 0."""
    if not weights:
        return 0.0
    total = 0.0
    for ticker, w in weights.items():
        if w == 0:
            continue
        p0 = open_at_exec.get(ticker)
        p1 = open_at_next.get(ticker)
        if p0 is None or p1 is None or pd.isna(p0) or pd.isna(p1) or p0 <= 0:
            # Prezzo mancante: assumiamo rendimento 0 sulla quota (conservativo)
            continue
        total += w * (p1 / p0 - 1.0)
    return total


# -------------------- Orchestrator --------------------

def run_backtest(
    start_date: date,
    end_date: date,
    config: StrategyConfig | None = None,
    cost_per_trade: float = 0.0005,
) -> BacktestResult:
    """Esegue il backtest settimanale. Ritorna BacktestResult.

    Algoritmo:
      1. Trova tutti i trading day in [start, end+7g] da SPY.
      2. Per ogni venerdì in [start, end]: chiama build_portfolio(venerdì, config).
         Se il primo trading day successivo (exec_date) supera end, fermati.
      3. Per ogni coppia (exec_i, exec_{i+1}): rendimento lordo = somma pesata
         dei movimenti open→open. Costo = turnover_one_way * cost_per_trade.
      4. Equity ricorsiva: equity[i+1] = equity[i] * (1 + return_net[i+1]).
    """
    config = config or StrategyConfig()

    # Ampliamo l'estremo destro di una settimana per trovare l'exec_date successivo
    # all'ultimo venerdì utile.
    cal_end = end_date + pd.Timedelta(days=10)
    days = _trading_days(start_date, cal_end.date() if hasattr(cal_end, "date") else cal_end)
    if len(days) < 2:
        raise ValueError(f"Calendario insufficiente in [{start_date}, {end_date}]")
    days_set = set(days)
    days_sorted = sorted(days)

    fridays = _fridays(start_date, end_date)
    if not fridays:
        raise ValueError(f"Nessun venerdì in [{start_date}, {end_date}]")

    # Mappa venerdì → primo trading day strettamente successivo
    exec_dates: list[date] = []
    decision_dates: list[date] = []
    for fri in fridays:
        nxt = next((d for d in days_sorted if d > fri), None)
        if nxt is None or nxt > end_date:
            continue
        if exec_dates and nxt == exec_dates[-1]:
            continue  # stesso lunedì già processato (mai dovrebbe accadere)
        exec_dates.append(nxt)
        decision_dates.append(fri)

    if len(exec_dates) < 2:
        raise ValueError(
            f"Servono almeno 2 exec_date per produrre un rendimento; "
            f"trovati {len(exec_dates)} in [{start_date}, {end_date}]"
        )

    # Costruisci portfolio target per ogni venerdì utile, applicando l'eventuale
    # filtro anti-whipsaw temporale (regime_confirm_weeks).
    #
    # Hysteresis: un regime "candidato" diverso da quello attivo deve essere
    # proposto per N settimane consecutive prima di diventare attivo. Se la
    # sequenza si interrompe, il candidato si resetta. Quando un cambio è
    # bloccato, il portafoglio della settimana viene ricostruito con
    # `force_regime` impostato sul regime ancora attivo.
    portfolios: list[Portfolio] = []
    active_regime: Regime | None = None
    candidate_regime: Regime | None = None
    candidate_count = 0
    confirm_n = max(0, int(config.regime_confirm_weeks))

    for fri in decision_dates:
        proposed_pf = build_portfolio(
            target_date=fri, config=config, previous_regime=active_regime,
        )
        proposed = proposed_pf.regime

        accept_proposed = (
            active_regime is None
            or confirm_n == 0
            or proposed == active_regime
        )
        if accept_proposed:
            actual_pf = proposed_pf
            candidate_regime = None
            candidate_count = 0
        else:
            # proposed != active e confirm > 0: serve hysteresis
            if proposed == candidate_regime:
                candidate_count += 1
            else:
                candidate_regime = proposed
                candidate_count = 1

            if candidate_count >= confirm_n:
                # Conferma sufficiente, switcha
                actual_pf = proposed_pf
                candidate_regime = None
                candidate_count = 0
            else:
                # Blocca: ricostruisci sotto il regime ancora attivo.
                # Se non realizzabile (es. force=DEFENSIVE ma difensivi giù),
                # la build può fallback a CASH; lo accettiamo.
                actual_pf = build_portfolio(
                    target_date=fri, config=config,
                    previous_regime=active_regime,
                    force_regime=active_regime,
                )

        active_regime = actual_pf.regime
        portfolios.append(actual_pf)

    # Tickers che potrebbero comparire nei portafogli (per matrice prezzi)
    all_tickers = sorted({p.ticker for pf in portfolios for p in pf.positions})

    # Carica matrice open per gli exec_date
    opens = _load_opens_matrix(all_tickers, exec_dates) if all_tickers else pd.DataFrame(
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in exec_dates])
    )

    # Iterazione: per ogni i in [1..N-1] calcoliamo il rendimento della settimana
    # tra exec_dates[i-1] e exec_dates[i], usando i pesi DECISI al venerdì precedente.
    n = len(exec_dates)
    ret_gross_vals: list[float] = []
    ret_net_vals: list[float] = []
    regime_vals: list[str] = [portfolios[0].regime]
    turnover_vals: list[float] = []
    trades: list[Trade] = []

    prev_weights: dict[str, float] = {}  # all'inizio: tutto cash
    # Costo iniziale: passaggio da 100% cash al portafoglio del primo venerdì.
    # Lo scarichiamo subito sull'equity di partenza, così la regola
    # "ogni rebalance sottrae turnover*cost_per_trade" è uniforme.
    initial_weights = _portfolio_to_weights(portfolios[0])
    initial_turnover = _turnover_one_way(prev_weights, initial_weights)
    turnover_vals.append(initial_turnover)
    equity_vals = [1.0 - initial_turnover * cost_per_trade]
    # Trades iniziali (open positions)
    for ticker, w in initial_weights.items():
        trades.append(Trade(
            decision_date=decision_dates[0],
            exec_date=exec_dates[0],
            ticker=ticker,
            weight_old=0.0,
            weight_new=w,
        ))
    prev_weights = initial_weights

    for i in range(1, n):
        exec_prev = exec_dates[i - 1]
        exec_curr = exec_dates[i]
        # Rendimento lordo: muoviamo i prezzi open[exec_prev] → open[exec_curr]
        # con i pesi DECISI al venerdì precedente (i-1).
        opens_prev = opens.loc[pd.Timestamp(exec_prev)] if not opens.empty else pd.Series(dtype=float)
        opens_curr = opens.loc[pd.Timestamp(exec_curr)] if not opens.empty else pd.Series(dtype=float)
        gross = _gross_period_return(prev_weights, opens_prev, opens_curr)

        # Pesi nuovi (decisi al venerdì i, eseguiti a exec_curr)
        new_weights = _portfolio_to_weights(portfolios[i])
        turnover = _turnover_one_way(prev_weights, new_weights)
        cost = turnover * cost_per_trade
        net = gross - cost

        # Trades (incl. uscite)
        all_keys = set(prev_weights) | set(new_weights)
        for ticker in all_keys:
            old_w = prev_weights.get(ticker, 0.0)
            new_w = new_weights.get(ticker, 0.0)
            if old_w == 0.0 and new_w == 0.0:
                continue
            if abs(new_w - old_w) > 1e-12:
                trades.append(Trade(
                    decision_date=decision_dates[i],
                    exec_date=exec_curr,
                    ticker=ticker,
                    weight_old=old_w,
                    weight_new=new_w,
                ))

        ret_gross_vals.append(gross)
        ret_net_vals.append(net)
        regime_vals.append(portfolios[i].regime)
        turnover_vals.append(turnover)
        equity_vals.append(equity_vals[-1] * (1.0 + net))
        prev_weights = new_weights

    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in exec_dates])
    equity = pd.Series(equity_vals, index=idx)
    # Returns hanno una riga in meno (il primo punto è "anchor")
    ret_idx = idx[1:]
    returns_gross = pd.Series(ret_gross_vals, index=ret_idx)
    returns_net = pd.Series(ret_net_vals, index=ret_idx)
    regime_history = pd.Series(regime_vals, index=idx)
    turnover_history = pd.Series(turnover_vals, index=idx)

    return BacktestResult(
        equity=equity,
        returns_gross=returns_gross,
        returns_net=returns_net,
        regime_history=regime_history,
        turnover_history=turnover_history,
        trades=trades,
        config=config,
        start_date=start_date,
        end_date=end_date,
        cost_per_trade=cost_per_trade,
    )


# -------------------- Persistenza --------------------

def _config_to_json(config: StrategyConfig) -> str:
    """Serializza StrategyConfig in JSON. Le tuple diventano liste."""
    d: dict[str, Any] = {}
    for k, v in config.__dict__.items():
        d[k] = list(v) if isinstance(v, tuple) else v
    return json.dumps(d, default=str)


def save_backtest(result: BacktestResult, run_id: str | None = None) -> str:
    """Persiste il risultato in DB. Ritorna il run_id usato."""
    init_db()
    run_id = run_id or f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    n_periods = len(result.returns_net)
    final_equity = float(result.equity.iloc[-1]) if len(result.equity) else float("nan")

    with get_conn() as con:
        con.execute(
            """
            INSERT INTO backtest_runs
                (run_id, run_at, start_date, end_date, cost_per_trade,
                 config_json, final_equity, n_periods)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id, datetime.now(), result.start_date, result.end_date,
                result.cost_per_trade, _config_to_json(result.config),
                final_equity, n_periods,
            ],
        )

        # backtest_equity: una riga per exec_date
        rows: list[tuple] = []
        for ts in result.equity.index:
            d = ts.date()
            ts_pd = pd.Timestamp(d)
            rg = (
                float(result.returns_gross.loc[ts_pd])
                if ts_pd in result.returns_gross.index else None
            )
            rn = (
                float(result.returns_net.loc[ts_pd])
                if ts_pd in result.returns_net.index else None
            )
            rows.append((
                run_id, d,
                float(result.equity.loc[ts_pd]),
                str(result.regime_history.loc[ts_pd]),
                float(result.turnover_history.loc[ts_pd]),
                rg, rn,
            ))
        con.executemany(
            """
            INSERT INTO backtest_equity
                (run_id, date, equity, regime, turnover, return_gross, return_net)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        if result.trades:
            trade_rows = [
                (run_id, t.decision_date, t.exec_date, t.ticker,
                 t.weight_old, t.weight_new, t.delta)
                for t in result.trades
            ]
            con.executemany(
                """
                INSERT INTO backtest_trades
                    (run_id, decision_date, exec_date, ticker,
                     weight_old, weight_new, delta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                trade_rows,
            )

    return run_id


# -------------------- CLI --------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Esegue un backtest settimanale.")
    p.add_argument("--start", type=str, default="2012-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--cost", type=float, default=0.0005)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = (
        datetime.strptime(args.end, "%Y-%m-%d").date() if args.end
        else date.today()
    )
    print(f"Backtest {start} → {end}, cost={args.cost*100:.3f}% per side")
    res = run_backtest(start, end, cost_per_trade=args.cost)
    print(f"  Periodi: {len(res.returns_net)}")
    print(f"  Equity finale: {res.equity.iloc[-1]:.4f}")
    print(f"  Turnover medio: {res.turnover_history.mean()*100:.1f}%")
    regime_counts = res.regime_history.value_counts()
    print("  Regimi:")
    for r, c in regime_counts.items():
        print(f"    {r:<22} {c:>4}")

    if not args.no_save:
        run_id = save_backtest(res)
        print(f"\nSalvato run_id = {run_id}")


if __name__ == "__main__":
    main()
