"""Analisi di SENSIBILITÀ di StrategyConfig.regime_buffer_pct.

Non è un'ottimizzazione: misura quanto le metriche dipendono dal buffer,
non sceglie un valore.

  - buffer da 0% a 4% con passo 0.25% (17 valori; unità = frazione, 0.02 = 2%)
  - regime_confirm_weeks = 0, tutto il resto = StrategyConfig() di default
  - costi 5 bps (default di run_backtest)
  - 3 periodi: intero, 2012-01-01→2018-12-31, 2019-01-01→end

Prima dello sweep verifica che buffer=0 e buffer=0.02 riproducano i run già
salvati nel DB (baseline e variante BUFFER); se no, si ferma.

I run NON vengono salvati nel DB. Output in outputs/buffer_sweep/.

Uso:
  python -m src.buffer_sweep
  python -m src.buffer_sweep --end 2026-04-29
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .db import get_conn
from .diagnose import compute_regime_switches, compute_turnover_and_costs
from .metrics import summarize
from .strategy import StrategyConfig


BASELINE_RUN_ID = "bt_20260430_155416_255140"
BUFFER_RUN_PREFIX = "variant_buffer_"
BUFFERS: list[float] = [round(i * 0.0025, 4) for i in range(17)]  # 0 → 0.04
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "buffer_sweep"
TOL = 1e-9


@dataclass
class SweepRow:
    buffer_pct: float          # in punti percentuali (es. 2.0)
    final_equity: float
    cagr: float
    sharpe: float
    max_dd: float
    n_switches: int
    n_whipsaws: int
    whipsaw_pct: float
    turnover_avg_weekly: float
    cost_drag_pct_gross: float


# -------------------- Sanity check vs DB --------------------

def _load_db_run(run_id: str) -> tuple[dict, pd.DataFrame]:
    with get_conn() as con:
        meta = con.execute(
            "SELECT run_id, start_date, end_date, cost_per_trade, final_equity, "
            "config_json FROM backtest_runs WHERE run_id = ?", [run_id],
        ).df()
        eq = con.execute(
            "SELECT date, equity, regime FROM backtest_equity "
            "WHERE run_id = ? ORDER BY date", [run_id],
        ).df()
    if meta.empty:
        raise SystemExit(f"Run {run_id} non trovato in backtest_runs")
    eq = eq.set_index(pd.DatetimeIndex(pd.to_datetime(eq.pop("date"))))
    eq.index.name = None
    m = meta.iloc[0].to_dict()
    m["start_date"] = pd.Timestamp(m["start_date"]).date()
    m["end_date"] = pd.Timestamp(m["end_date"]).date()
    return m, eq


def _latest_buffer_run_id() -> str:
    with get_conn() as con:
        row = con.execute(
            "SELECT run_id FROM backtest_runs WHERE run_id LIKE ? "
            "ORDER BY run_at DESC LIMIT 1", [BUFFER_RUN_PREFIX + "%"],
        ).fetchone()
    if row is None:
        raise SystemExit("Nessun run BUFFER salvato nel DB")
    return row[0]


def _check_reproduces(label: str, res: BacktestResult, run_id: str) -> None:
    meta, ref = _load_db_run(run_id)
    ours = pd.DataFrame({"equity": res.equity, "regime": res.regime_history})
    problems: list[str] = []
    if res.cost_per_trade != meta["cost_per_trade"]:
        problems.append(f"cost_per_trade {res.cost_per_trade} vs {meta['cost_per_trade']}")
    ours.index = pd.DatetimeIndex(ours.index.date)
    ref = ref.copy()
    ref.index = pd.DatetimeIndex(ref.index.date)
    if list(ours.index) != list(ref.index):
        problems.append(
            f"date diverse: {len(ours)} punti vs {len(ref)} nel DB "
            f"({ours.index.min().date()}→{ours.index.max().date()} vs "
            f"{ref.index.min().date()}→{ref.index.max().date()})"
        )
    else:
        diff = (ours["equity"] - ref["equity"]).abs() / ref["equity"]
        if diff.max() > TOL:
            first = diff[diff > TOL].index[0]
            problems.append(
                f"equity diverge dal {first.date()} "
                f"(max diff relativa {diff.max():.2e})"
            )
        reg_bad = ours["regime"] != ref["regime"]
        if reg_bad.any():
            problems.append(
                f"regime diverso in {int(reg_bad.sum())} settimane, "
                f"prima il {reg_bad[reg_bad].index[0].date()}"
            )
    final = float(res.equity.iloc[-1])
    status = "OK" if not problems else "FALLITO"
    print(f"  [{status}] {label}: equity finale {final:.4f} "
          f"vs DB {meta['final_equity']:.4f} ({run_id})")
    if problems:
        for p in problems:
            print(f"      - {p}")
        raise SystemExit(f"Sanity check '{label}' fallito: sweep interrotto.")


def sanity_checks(base_cfg: StrategyConfig) -> date:
    """Verifica baseline e BUFFER. Ritorna l'end_date dei run di riferimento."""
    print("Sanity check vs DB:")
    base_meta, _ = _load_db_run(BASELINE_RUN_ID)
    start, end = base_meta["start_date"], base_meta["end_date"]

    res0 = run_backtest(start, end, config=replace(base_cfg, regime_buffer_pct=0.0))
    _check_reproduces("buffer 0%", res0, BASELINE_RUN_ID)

    buf_id = _latest_buffer_run_id()
    buf_meta, _ = _load_db_run(buf_id)
    if (buf_meta["start_date"], buf_meta["end_date"]) != (start, end):
        raise SystemExit(f"Il run {buf_id} ha un periodo diverso dalla baseline")
    res2 = run_backtest(start, end, config=replace(base_cfg, regime_buffer_pct=0.02))
    _check_reproduces("buffer 2%", res2, buf_id)
    return end


# -------------------- Sweep --------------------

def evaluate(res: BacktestResult, buffer: float) -> SweepRow:
    summ = summarize(res.equity, periods_per_year=52)
    sw = compute_regime_switches(res.regime_history)
    cs = compute_turnover_and_costs(
        res.turnover_history, res.returns_gross, res.returns_net,
    )
    n_sw = sw["n_switches"]
    n_wh = len(sw["whipsaws"])
    return SweepRow(
        buffer_pct=buffer * 100,
        final_equity=float(res.equity.iloc[-1]),
        cagr=summ.cagr,
        sharpe=summ.sharpe,
        max_dd=summ.max_drawdown,
        n_switches=n_sw,
        n_whipsaws=n_wh,
        whipsaw_pct=(n_wh / n_sw) if n_sw else 0.0,
        turnover_avg_weekly=cs["avg_turnover_weekly"],
        cost_drag_pct_gross=cs["cost_pct_of_gross"],
    )


def sweep(base_cfg: StrategyConfig, start: date, end: date, label: str) -> pd.DataFrame:
    rows: list[SweepRow] = []
    for b in BUFFERS:
        print(f"  [{label}] buffer={b*100:.2f}% ...", flush=True)
        res = run_backtest(start, end, config=replace(base_cfg, regime_buffer_pct=b))
        rows.append(evaluate(res, b))
    return pd.DataFrame([asdict(r) for r in rows])


# -------------------- Output --------------------

def plot(results: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Slot categorici 1-3 della palette di riferimento + marker diversi
    # (codifica secondaria, l'identità non dipende solo dal colore).
    styles = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
    panels = [
        ("sharpe", "Sharpe ratio", 1.0),
        ("cagr", "Annual return, CAGR (%)", 100.0),
        ("max_dd", "Max drawdown (%)", 100.0),
        ("whipsaw_pct", "Whipsaws (% of regime switches reversed within 4 weeks)", 100.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (col, title, scale) in zip(axes.flat, panels):
        ax.set_facecolor("#fcfcfb")
        for (label, df), (color, marker) in zip(results.items(), styles):
            ax.plot(df["buffer_pct"], df[col] * scale, color=color, lw=2,
                    marker=marker, ms=5, label=label)
        ax.axvline(2.0, color="#52514e", lw=1, ls="--")
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.grid(True, color="#e5e4df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e", labelsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Dead band around the 200-day average (%)", color="#52514e")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Sensitivity to the width of the dead band (weekly backtest, 5 bps costs; "
                 "dashed line = 2%, chosen before the sweep)", fontsize=12, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _monotone_desc(s: pd.Series) -> str:
    d = s.diff().dropna()
    ups = s.index[1:][d.values > 1e-12]
    if len(ups) == 0:
        return "sì (non crescente)"
    return f"no ({len(ups)} risalite)"


def facts(label: str, df: pd.DataFrame) -> list[str]:
    i_max = df["sharpe"].idxmax()
    band = df[(df["buffer_pct"] >= 1.0) & (df["buffer_pct"] <= 3.0)]["sharpe"]
    return [
        f"{label}: Sharpe max {df.loc[i_max, 'sharpe']:.3f} a buffer "
        f"{df.loc[i_max, 'buffer_pct']:.2f}% (min {df['sharpe'].min():.3f} a "
        f"{df.loc[df['sharpe'].idxmin(), 'buffer_pct']:.2f}%); "
        f"fascia 1-3%: {band.min():.3f}–{band.max():.3f} (ampiezza {band.max()-band.min():.3f}); "
        f"monotonia ↓ switch {_monotone_desc(df['n_switches'])}, "
        f"whipsaw% {_monotone_desc(df['whipsaw_pct'])}, "
        f"turnover {_monotone_desc(df['turnover_avg_weekly'])}"
    ]


def print_table(label: str, df: pd.DataFrame) -> None:
    out = pd.DataFrame({
        "buffer%": df["buffer_pct"].map(lambda v: f"{v:.2f}"),
        "eq_fin": df["final_equity"].map(lambda v: f"{v:.4f}"),
        "CAGR": df["cagr"].map(lambda v: f"{v*100:.2f}%"),
        "Sharpe": df["sharpe"].map(lambda v: f"{v:.3f}"),
        "MaxDD": df["max_dd"].map(lambda v: f"{v*100:.2f}%"),
        "switch": df["n_switches"],
        "whips": df["n_whipsaws"],
        "whips%": df["whipsaw_pct"].map(lambda v: f"{v*100:.1f}%"),
        "turn/wk": df["turnover_avg_weekly"].map(lambda v: f"{v*100:.2f}%"),
        "cost%gross": df["cost_drag_pct_gross"].map(lambda v: f"{v*100:.2f}%"),
    })
    print(f"\n=== {label} ===")
    print(out.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="Sensitivity sweep di regime_buffer_pct.")
    p.add_argument("--end", type=str, default=None,
                   help="Fine dei periodi. Default = end_date della baseline in DB.")
    p.add_argument("--plot-only", action="store_true",
                   help="Redraw the chart from the saved CSVs, without re-running any backtest.")
    args = p.parse_args()

    if args.plot_only:
        files = [("Full period (2012-2026)", "sweep_full.csv"),
                 ("2012-2018", "sweep_2012_2018.csv"),
                 ("2019-2026", "sweep_2019_2026.csv")]
        results = {label: pd.read_csv(OUT_DIR / f) for label, f in files}
        plot(results, OUT_DIR / "sweep_buffer.png")
        print(f"Chart redrawn: {OUT_DIR / 'sweep_buffer.png'}")
        return

    base_cfg = StrategyConfig(regime_confirm_weeks=0)
    ref_end = sanity_checks(base_cfg)
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else ref_end

    periods = [
        ("full",      date(2012, 1, 1), end,                "sweep_full.csv"),
        ("2012-2018", date(2012, 1, 1), date(2018, 12, 31), "sweep_2012_2018.csv"),
        (f"2019-{end.year}", date(2019, 1, 1), end,          "sweep_2019_2026.csv"),
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    for label, s, e, fname in periods:
        print(f"\nPeriodo {label}: {s} → {e}")
        df = sweep(base_cfg, s, e, label)
        df.to_csv(OUT_DIR / fname, index=False)
        results[f"{label} ({s}→{e})"] = df

    plot(results, OUT_DIR / "sweep_buffer.png")

    for label, df in results.items():
        print_table(label, df)
    print("\nSintesi (solo fatti):")
    for label, df in results.items():
        for line in facts(label.split(" ")[0], df):
            print(f"  - {line}")
    print(f"\nOutput in {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
