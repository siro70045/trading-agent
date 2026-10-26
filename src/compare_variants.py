"""Confronto tra varianti di regime detector.

Lancia 3 backtest sullo stesso periodo (2012-01-01 → ultima data disponibile in DB)
con la stessa StrategyConfig di base, modificando solo la logica anti-whipsaw:

  - BASELINE:  buffer=0,    confirm=0    (legacy)
  - BUFFER:    buffer=0.02, confirm=0    (zona morta ±2% intorno a MA200)
  - CONFIRM:   buffer=0,    confirm=2    (richiede 2 settimane di conferma)

Salva i 3 run nel DB e stampa una tabella comparativa con metriche chiave.
Riusa le compute_* di src.diagnose, così la stessa definizione di metrica
viene usata in tutti i contesti.

Uso:
  python -m src.compare_variants
  python -m src.compare_variants --start 2012-01-01 --end 2026-04-29
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from .backtest import run_backtest, save_backtest
from .benchmarks import build_spy_buy_and_hold
from .db import get_conn
from .diagnose import (
    compute_regime_switches,
    compute_turnover_and_costs,
    compute_yearly_table,
)
from .metrics import summarize
from .strategy import StrategyConfig


VARIANTS: list[tuple[str, StrategyConfig]] = [
    ("BASELINE", StrategyConfig()),
    ("BUFFER",   StrategyConfig(regime_buffer_pct=0.02)),
    ("CONFIRM",  StrategyConfig(regime_confirm_weeks=2)),
]


@dataclass
class VariantSummary:
    name: str
    run_id: str
    cagr: float
    sharpe: float
    mdd: float
    n_switches: int
    n_whipsaws: int
    whipsaw_pct: float
    turnover_avg: float
    cost_drag_pct: float
    alpha_2015: float
    alpha_2025: float


def _last_data_date() -> date:
    with get_conn() as con:
        return con.execute(
            "SELECT MAX(date) FROM prices WHERE ticker='SPY'"
        ).fetchone()[0]


def collect(name: str, config: StrategyConfig, start: date, end: date,
            spy_eq: pd.Series) -> VariantSummary:
    print(f"  Running {name:<10} buffer={config.regime_buffer_pct:.2f} "
          f"confirm={config.regime_confirm_weeks} ...", flush=True)
    res = run_backtest(start, end, config=config)
    run_id = save_backtest(
        res, run_id=f"variant_{name.lower()}_"
                     f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
    )

    summ = summarize(res.equity, periods_per_year=52)
    sw = compute_regime_switches(res.regime_history)
    cs = compute_turnover_and_costs(
        res.turnover_history, res.returns_gross, res.returns_net,
    )
    yearly = compute_yearly_table(
        res.equity, res.regime_history, res.returns_net, spy_eq,
    )

    a15 = float(yearly.loc[2015, "alpha"]) if 2015 in yearly.index else float("nan")
    a25 = float(yearly.loc[2025, "alpha"]) if 2025 in yearly.index else float("nan")
    n_sw = sw["n_switches"]
    n_wh = len(sw["whipsaws"])
    return VariantSummary(
        name=name, run_id=run_id,
        cagr=summ.cagr, sharpe=summ.sharpe, mdd=summ.max_drawdown,
        n_switches=n_sw, n_whipsaws=n_wh,
        whipsaw_pct=(n_wh / n_sw) if n_sw else 0.0,
        turnover_avg=cs["avg_turnover_weekly"],
        cost_drag_pct=cs["cost_pct_of_gross"],
        alpha_2015=a15, alpha_2025=a25,
    )


def print_table(rows: list[VariantSummary]) -> None:
    metrics = [
        ("CAGR",                      "cagr",          "pct"),
        ("Sharpe",                    "sharpe",        "num"),
        ("Max drawdown",              "mdd",           "pct"),
        ("# switch totali",           "n_switches",    "int"),
        ("# whipsaw (≤4 wk)",         "n_whipsaws",    "int"),
        ("% whipsaw / switch",        "whipsaw_pct",   "pct"),
        ("Turnover settimanale med.", "turnover_avg",  "pct"),
        ("Cost drag % rend. lordo",   "cost_drag_pct", "pct"),
        ("Alpha 2015 vs SPY",         "alpha_2015",    "pct"),
        ("Alpha 2025 vs SPY",         "alpha_2025",    "pct"),
    ]
    col_w = 14
    label_w = 30

    print()
    print("=" * (label_w + col_w * len(rows) + 2))
    print("Confronto varianti")
    print("=" * (label_w + col_w * len(rows) + 2))
    header = " " * (label_w) + "".join(f"{r.name:>{col_w}}" for r in rows)
    print(header)
    print("-" * (label_w + col_w * len(rows)))
    for label, key, fmt in metrics:
        line = f"{label:<{label_w}}"
        for r in rows:
            v = getattr(r, key)
            if pd.isna(v):
                cell = "—"
            elif fmt == "pct":
                cell = f"{v * 100:+.2f}%"
            elif fmt == "int":
                cell = f"{int(v)}"
            else:
                cell = f"{v:.2f}"
            line += f"{cell:>{col_w}}"
        print(line)

    print()
    print("Run salvati:")
    for r in rows:
        print(f"  {r.name:<10} → {r.run_id}")


def main() -> None:
    p = argparse.ArgumentParser(description="Confronto varianti regime detector.")
    p.add_argument("--start", type=str, default="2012-01-01")
    p.add_argument("--end", type=str, default=None,
                   help="Default = ultima data SPY in DB.")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = (datetime.strptime(args.end, "%Y-%m-%d").date() if args.end
           else _last_data_date())

    print(f"Periodo: {start} → {end}")
    spy = build_spy_buy_and_hold(start, end)

    rows: list[VariantSummary] = []
    for name, cfg in VARIANTS:
        rows.append(collect(name, cfg, start, end, spy))

    print_table(rows)


if __name__ == "__main__":
    main()
