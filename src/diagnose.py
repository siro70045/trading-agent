"""Diagnostica testuale sul backtest_run più recente nel DB.

Carica equity, regimi, turnover e rendimenti dalle tabelle backtest_*, poi
calcola e stampa:
  1. Anatomia del max drawdown (peak, trough, recovery)
  2. Statistiche per regime (frequenza, rendimento medio, contributo)
  3. Switch di regime e whipsaw entro 4 settimane
  4. Turnover medio/annualizzato e costi totali in % del rendimento lordo
  5. Tabella per anno (Strategy / SPY / alpha / MDD / regime predominante)
  6. Top 5 e bottom 5 settimane per rendimento

Le funzioni compute_* sono pure (input pandas, output dict/DataFrame) e
testabili indipendentemente; main() carica dal DB e le orchestra.

Uso:
  python -m src.diagnose                    # ultimo run salvato
  python -m src.diagnose --run-id <id>      # run specifico
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .benchmarks import build_spy_buy_and_hold
from .db import get_conn


# -------------------- Caricamento dal DB --------------------

@dataclass
class RunData:
    run_id: str
    start_date: date
    end_date: date
    cost_per_trade: float
    final_equity: float
    n_periods: int
    equity: pd.Series          # indicizzato per data (DatetimeIndex)
    regimes: pd.Series
    turnover: pd.Series
    returns_gross: pd.Series   # uno in meno dell'equity (NaN sul primo punto)
    returns_net: pd.Series


def latest_run_id() -> str | None:
    with get_conn() as con:
        row = con.execute(
            "SELECT run_id FROM backtest_runs ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else None


def load_run(run_id: str) -> RunData | None:
    """Carica un run dal DB. Ritorna None se l'header non esiste."""
    with get_conn() as con:
        header = con.execute(
            """
            SELECT start_date, end_date, cost_per_trade, final_equity, n_periods
            FROM backtest_runs WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if header is None:
            return None
        eq_df = con.execute(
            """
            SELECT date, equity, regime, turnover, return_gross, return_net
            FROM backtest_equity WHERE run_id = ?
            ORDER BY date
            """,
            [run_id],
        ).df()

    if eq_df.empty:
        idx = pd.DatetimeIndex([])
        empty_f = pd.Series(dtype=float, index=idx)
        empty_o = pd.Series(dtype=object, index=idx)
        return RunData(
            run_id=run_id,
            start_date=header[0], end_date=header[1],
            cost_per_trade=header[2],
            final_equity=header[3] if header[3] is not None else float("nan"),
            n_periods=header[4] if header[4] is not None else 0,
            equity=empty_f, regimes=empty_o, turnover=empty_f,
            returns_gross=empty_f, returns_net=empty_f,
        )

    eq_df = eq_df.assign(date=pd.to_datetime(eq_df["date"])).set_index("date")
    return RunData(
        run_id=run_id,
        start_date=header[0], end_date=header[1],
        cost_per_trade=header[2],
        final_equity=header[3], n_periods=header[4],
        equity=eq_df["equity"],
        regimes=eq_df["regime"],
        turnover=eq_df["turnover"],
        returns_gross=eq_df["return_gross"].dropna(),
        returns_net=eq_df["return_net"].dropna(),
    )


# -------------------- 1. Drawdown --------------------

def compute_drawdown_anatomy(equity: pd.Series) -> dict:
    """Trova peak, trough e recovery del max drawdown.

    Recovery = primo punto >= peak DOPO il trough. None se non ancora avvenuto.
    Distanze espresse in 'settimane' = numero di periodi (l'equity è settimanale).
    """
    if len(equity) < 2:
        return {
            "max_dd": float("nan"), "peak_date": None, "peak_value": float("nan"),
            "trough_date": None, "trough_value": float("nan"),
            "weeks_to_trough": None, "recovery_date": None, "weeks_to_recovery": None,
        }
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    trough_idx = int(dd.values.argmin())
    trough_date = equity.index[trough_idx]
    peak_value = float(running_max.iloc[trough_idx])
    # Peak: ultimo punto <= trough_idx con equity == running_max[trough]
    eq_pre = equity.iloc[: trough_idx + 1]
    peak_idx = int(np.where(eq_pre.values >= peak_value - 1e-12)[0][-1])
    peak_date = equity.index[peak_idx]
    # Recovery: primo punto > trough_idx con equity >= peak_value
    eq_post = equity.iloc[trough_idx + 1:]
    rec_mask = eq_post.values >= peak_value - 1e-12
    if rec_mask.any():
        rec_offset = int(np.where(rec_mask)[0][0])
        rec_idx = trough_idx + 1 + rec_offset
        recovery_date = equity.index[rec_idx]
        weeks_to_recovery = rec_idx - peak_idx
    else:
        recovery_date = None
        weeks_to_recovery = None

    return {
        "max_dd": float(dd.iloc[trough_idx]),
        "peak_date": peak_date, "peak_value": peak_value,
        "trough_date": trough_date, "trough_value": float(equity.iloc[trough_idx]),
        "weeks_to_trough": trough_idx - peak_idx,
        "recovery_date": recovery_date,
        "weeks_to_recovery": weeks_to_recovery,
    }


# -------------------- 2. Regime stats --------------------

def compute_regime_stats(regimes: pd.Series, returns_net: pd.Series) -> pd.DataFrame:
    """Statistiche per regime. Il rendimento di una settimana viene attribuito
    al regime *attivo* in quella settimana (cioè decisione fatta al venerdì
    precedente, quel regime era quello vigente all'esecuzione successiva).

    Convenzione: returns_net[t] è il rendimento del periodo che termina a t,
    durante il quale era attivo il regime registrato al periodo precedente t-1.
    Quindi shift(1) dei regimi per allinearsi ai rendimenti.
    """
    if len(regimes) == 0 or len(returns_net) == 0:
        return pd.DataFrame(columns=[
            "n_weeks", "pct_time", "avg_return", "compound_return",
        ])
    aligned_regime = regimes.shift(1).reindex(returns_net.index)
    rows = []
    total_weeks = len(aligned_regime.dropna())
    for regime, group in returns_net.groupby(aligned_regime):
        if pd.isna(regime):
            continue
        compound = float((1.0 + group).prod() - 1.0)
        rows.append({
            "regime": regime,
            "n_weeks": int(len(group)),
            "pct_time": float(len(group) / total_weeks) if total_weeks else 0.0,
            "avg_return": float(group.mean()),
            "compound_return": compound,
        })
    return pd.DataFrame(rows).set_index("regime").sort_values("n_weeks", ascending=False)


# -------------------- 3. Switch & whipsaw --------------------

def compute_regime_switches(
    regimes: pd.Series, whipsaw_window_weeks: int = 4,
) -> dict:
    """Rileva switch di regime e whipsaw.

    Switch: regime[t] != regime[t-1].
    Whipsaw: switch A→B seguito entro `whipsaw_window_weeks` da B→A.
    """
    if len(regimes) < 2:
        return {"n_switches": 0, "switches": [], "whipsaws": []}
    prev = regimes.shift(1)
    switch_mask = (regimes != prev) & prev.notna()
    switch_idx = np.where(switch_mask.values)[0]
    switches = [
        (regimes.index[i], str(prev.iloc[i]), str(regimes.iloc[i]))
        for i in switch_idx
    ]
    whipsaws: list[tuple] = []
    for j, (date_j, from_j, to_j) in enumerate(switches):
        for k in range(j + 1, len(switches)):
            date_k, from_k, to_k = switches[k]
            # Distanza in n. di periodi dell'indice
            distance = switch_idx[k] - switch_idx[j]
            if distance > whipsaw_window_weeks:
                break
            if to_k == from_j:
                whipsaws.append((date_j, date_k, from_j, to_j, distance))
                break
    return {"n_switches": len(switches), "switches": switches, "whipsaws": whipsaws}


# -------------------- 4. Turnover & costi --------------------

def compute_turnover_and_costs(
    turnover: pd.Series, returns_gross: pd.Series, returns_net: pd.Series,
) -> dict:
    """Turnover medio/annualizzato e costo aggregato (geometrico).

    cost_drag = (1 + total_gross) - (1 + total_net) (espresso come differenza
    di rendimento composto). Il '% del rendimento lordo' = cost_drag / total_gross.
    """
    if len(turnover) == 0 or len(returns_gross) == 0:
        return {
            "avg_turnover_weekly": float("nan"),
            "annualized_turnover": float("nan"),
            "total_gross_return": float("nan"),
            "total_net_return": float("nan"),
            "cost_drag": float("nan"),
            "cost_pct_of_gross": float("nan"),
        }
    avg_w = float(turnover.mean())
    total_gross = float((1.0 + returns_gross).prod() - 1.0)
    total_net = float((1.0 + returns_net).prod() - 1.0)
    cost_drag = total_gross - total_net
    if total_gross != 0:
        ratio = cost_drag / total_gross
    else:
        ratio = float("nan")
    return {
        "avg_turnover_weekly": avg_w,
        "annualized_turnover": avg_w * 52.0,
        "total_gross_return": total_gross,
        "total_net_return": total_net,
        "cost_drag": cost_drag,
        "cost_pct_of_gross": ratio,
    }


# -------------------- 5. Tabella per anno --------------------

def compute_yearly_table(
    equity: pd.Series, regimes: pd.Series, returns_net: pd.Series,
    spy_equity: pd.Series,
) -> pd.DataFrame:
    """Per ogni anno: rendimento Strategy, rendimento SPY, alpha, MDD intra-anno,
    regime predominante.

    Il rendimento annuo è il fattore composto sui returns_net dell'anno.
    SPY: stesso schema, calcolato sull'equity SPY allineata.
    """
    if len(equity) == 0:
        return pd.DataFrame(columns=[
            "strategy_return", "spy_return", "alpha", "intra_year_mdd",
            "predominant_regime",
        ])
    spy_aligned = spy_equity.reindex(equity.index).ffill()
    spy_returns = spy_aligned.pct_change()

    years = equity.index.year.unique()
    rows = []
    for y in sorted(years):
        mask_year_eq = equity.index.year == y
        eq_y = equity[mask_year_eq]
        if len(eq_y) < 2:
            continue
        rn_y = returns_net.reindex(eq_y.index).dropna()
        spy_rn_y = spy_returns.reindex(eq_y.index).dropna()
        strat_ret = float((1.0 + rn_y).prod() - 1.0) if len(rn_y) else 0.0
        spy_ret = float((1.0 + spy_rn_y).prod() - 1.0) if len(spy_rn_y) else float("nan")
        # MDD intra-anno: ricostruisco un'equity intra-anno partendo da 1.0
        intra_eq = (1.0 + rn_y).cumprod() if len(rn_y) else pd.Series([1.0])
        if len(intra_eq):
            intra_mdd = float((intra_eq / intra_eq.cummax() - 1.0).min())
        else:
            intra_mdd = 0.0
        # Regime predominante (più settimane in quell'anno)
        reg_y = regimes[mask_year_eq]
        predominant = (
            reg_y.value_counts().idxmax() if len(reg_y) else "—"
        )
        rows.append({
            "year": y,
            "strategy_return": strat_ret,
            "spy_return": spy_ret,
            "alpha": strat_ret - spy_ret if not pd.isna(spy_ret) else float("nan"),
            "intra_year_mdd": intra_mdd,
            "predominant_regime": predominant,
        })
    return pd.DataFrame(rows).set_index("year")


# -------------------- 6. Top/bottom settimane --------------------

def compute_top_bottom_weeks(
    returns_net: pd.Series, regimes: pd.Series, n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(returns_net) == 0:
        empty = pd.DataFrame(columns=["return", "regime"])
        return empty, empty
    aligned_regime = regimes.shift(1).reindex(returns_net.index)
    df = pd.DataFrame({"return": returns_net, "regime": aligned_regime})
    top = df.nlargest(n, "return")
    bottom = df.nsmallest(n, "return")
    return top, bottom


# -------------------- Stampa --------------------

def _fmt_date(d) -> str:
    if d is None:
        return "—"
    if hasattr(d, "date"):
        d = d.date()
    return str(d)


def print_header(run: RunData) -> None:
    print("=" * 72)
    print(f"Backtest run: {run.run_id}")
    print(f"Periodo:      {run.start_date} → {run.end_date}")
    print(f"Costi:        {run.cost_per_trade*100:.3f}% per side")
    print(f"Equity finale: {run.final_equity:.4f}  ({run.n_periods} periodi)")
    print("=" * 72)


def print_drawdown(d: dict) -> None:
    print("\n--- 1. Drawdown massimo ---")
    if d["peak_date"] is None:
        print("  (dati insufficienti)")
        return
    print(f"  Peak:           {_fmt_date(d['peak_date'])}  (equity {d['peak_value']:.4f})")
    print(f"  Trough:         {_fmt_date(d['trough_date'])}  (equity {d['trough_value']:.4f})")
    print(f"  Drawdown:       {d['max_dd']*100:+.2f}%")
    print(f"  Durata caduta:  {d['weeks_to_trough']} settimane")
    if d["recovery_date"] is not None:
        print(f"  Recovery:       {_fmt_date(d['recovery_date'])}  "
              f"({d['weeks_to_recovery']} settimane peak→recovery)")
    else:
        print(f"  Recovery:       non ancora avvenuta")


def print_regime_stats(df: pd.DataFrame) -> None:
    print("\n--- 2. Statistiche per regime ---")
    if df.empty:
        print("  (dati insufficienti)")
        return
    print(f"  {'Regime':<22} {'N':>5} {'%time':>7} {'avg ret/wk':>11} {'contrib comp.':>14}")
    for regime, row in df.iterrows():
        print(
            f"  {regime:<22} {int(row['n_weeks']):>5} "
            f"{row['pct_time']*100:>6.1f}% "
            f"{row['avg_return']*100:>+10.3f}% "
            f"{row['compound_return']*100:>+13.1f}%"
        )


def print_switches(s: dict) -> None:
    print("\n--- 3. Switch di regime ---")
    print(f"  Totale switch:   {s['n_switches']}")
    print(f"  Whipsaw (≤4 wk): {len(s['whipsaws'])}")
    if s["switches"]:
        print("  Date di switch:")
        for d, frm, to in s["switches"]:
            print(f"    {_fmt_date(d)}  {frm} → {to}")
    if s["whipsaws"]:
        print("\n  Whipsaw entro 4 settimane:")
        for d1, d2, frm, to, dist in s["whipsaws"]:
            print(f"    {_fmt_date(d1)} → {_fmt_date(d2)}  "
                  f"{frm}→{to}→{frm}  ({dist} wk)")


def print_costs(c: dict) -> None:
    print("\n--- 4. Turnover e costi ---")
    if pd.isna(c["avg_turnover_weekly"]):
        print("  (dati insufficienti)")
        return
    print(f"  Turnover settimanale medio:   {c['avg_turnover_weekly']*100:>6.2f}%")
    print(f"  Turnover annualizzato:        {c['annualized_turnover']*100:>6.0f}%")
    print(f"  Rendimento composto LORDO:    {c['total_gross_return']*100:>+7.2f}%")
    print(f"  Rendimento composto NETTO:    {c['total_net_return']*100:>+7.2f}%")
    print(f"  Cost drag (geom):             {c['cost_drag']*100:>+7.2f}%")
    print(f"  Costo / rendimento lordo:     {c['cost_pct_of_gross']*100:>6.2f}%")


def print_yearly(df: pd.DataFrame) -> None:
    print("\n--- 5. Performance per anno ---")
    if df.empty:
        print("  (dati insufficienti)")
        return
    print(f"  {'Anno':<6} {'Strategy':>10} {'SPY':>10} {'Alpha':>9} "
          f"{'MDD intra':>11}  {'Regime predom.':<22}")
    for y, row in df.iterrows():
        print(
            f"  {y:<6} "
            f"{row['strategy_return']*100:>+9.2f}% "
            f"{row['spy_return']*100:>+9.2f}% "
            f"{row['alpha']*100:>+8.2f}% "
            f"{row['intra_year_mdd']*100:>+10.2f}%  "
            f"{row['predominant_regime']:<22}"
        )


def print_top_bottom(top: pd.DataFrame, bottom: pd.DataFrame) -> None:
    print("\n--- 6. Top 5 settimane ---")
    if top.empty:
        print("  (dati insufficienti)")
    else:
        print(f"  {'Data':<12} {'Ret':>8}    Regime")
        for ts, row in top.iterrows():
            print(f"  {_fmt_date(ts):<12} {row['return']*100:>+7.2f}%   "
                  f"{row['regime']}")

    print("\n--- Bottom 5 settimane ---")
    if bottom.empty:
        print("  (dati insufficienti)")
    else:
        print(f"  {'Data':<12} {'Ret':>8}    Regime")
        for ts, row in bottom.iterrows():
            print(f"  {_fmt_date(ts):<12} {row['return']*100:>+7.2f}%   "
                  f"{row['regime']}")


# -------------------- Orchestrator --------------------

def diagnose(run: RunData) -> None:
    """Stampa l'intera diagnostica per un RunData. Robusto a run vuoti."""
    print_header(run)
    if len(run.equity) == 0:
        print("\n(Run senza dati equity nel DB — niente da analizzare.)")
        return

    print_drawdown(compute_drawdown_anatomy(run.equity))
    print_regime_stats(compute_regime_stats(run.regimes, run.returns_net))
    print_switches(compute_regime_switches(run.regimes))
    print_costs(compute_turnover_and_costs(
        run.turnover, run.returns_gross, run.returns_net))

    spy = build_spy_buy_and_hold(run.start_date, run.end_date)
    print_yearly(compute_yearly_table(
        run.equity, run.regimes, run.returns_net, spy))

    top, bottom = compute_top_bottom_weeks(run.returns_net, run.regimes, n=5)
    print_top_bottom(top, bottom)


# -------------------- CLI --------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Diagnostica del backtest_run più recente.")
    p.add_argument("--run-id", type=str, default=None,
                   help="run_id specifico; default = ultimo salvato.")
    args = p.parse_args()

    run_id = args.run_id or latest_run_id()
    if run_id is None:
        print("Nessun backtest_run trovato nel DB.")
        return
    run = load_run(run_id)
    if run is None:
        print(f"run_id {run_id} non trovato.")
        return
    diagnose(run)


if __name__ == "__main__":
    main()
