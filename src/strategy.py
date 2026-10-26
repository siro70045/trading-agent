"""Pipeline di selezione del portafoglio target.

Architettura: ogni step della pipeline è una funzione pura testabile.
`build_portfolio(target_date, config)` orchestra il tutto.

Pipeline:
  1. Regime di mercato       → SPY > MA200?
  2a. RISK_ON: filtra azioni → liquidità + trend individuale
      ranking momentum 12-1  → top N
      sizing inverse-vol     → con cap
  2b. RISK_OFF: difensivi    → GLD/TLT solo se sopra MA200, altrimenti CASH
  3. Output                  → Portfolio con date, posizioni, regime, motivi
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import pandas as pd

from .config import DEFAULT_LOOKBACK_DAYS
from .db import get_conn
from .indicators import add_all_indicators

Regime = Literal["RISK_ON", "RISK_OFF_DEFENSIVE", "RISK_OFF_CASH"]


# -------------------- Configurazione --------------------

@dataclass(frozen=True)
class StrategyConfig:
    # Universe
    market_proxy: str = "SPY"
    defensive_assets: tuple[str, ...] = ("GLD", "TLT")
    excluded_etf_sectors: tuple[str, ...] = (
        "ETF Broad", "ETF Tech", "ETF Small-Cap",
        "ETF International", "ETF Emerging", "ETF Sector",
        "ETF Commodity", "ETF Bond",
    )
    # Filtri
    min_dollar_volume: float = 1_000_000.0  # 1M$/giorno (passa-tutti per starter universe)
    # Ranking & sizing
    n_positions: int = 5
    max_weight: float = 0.35
    # Trend
    ma_window: int = 200
    # Universo dinamico: minimo di osservazioni storiche richieste a un ticker
    # perché possa entrare nel pool a una certa target_date. 504 ≈ 2 anni di trading.
    min_history_days: int = 504
    # Anti-whipsaw: zona morta percentuale intorno a MA200 dentro la quale
    # il regime macro (ON ↔ OFF) viene preservato. 0.0 → comportamento legacy.
    regime_buffer_pct: float = 0.0
    # Anti-whipsaw temporale: numero di settimane consecutive in cui un regime
    # diverso da quello attivo deve essere proposto prima di switchare.
    # 0 → switch immediato (legacy). Logica gestita in backtest.py.
    regime_confirm_weeks: int = 0


# -------------------- Output --------------------

@dataclass
class Position:
    ticker: str
    weight: float
    momentum: float | None = None  # per debug/report
    volatility: float | None = None
    reason: str = ""


@dataclass
class Portfolio:
    target_date: date
    regime: Regime
    positions: list[Position] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def cash_weight(self) -> float:
        return max(0.0, 1.0 - sum(p.weight for p in self.positions))

    def summary(self) -> str:
        lines = [
            f"Target date: {self.target_date}",
            f"Regime:      {self.regime}",
            f"Positions:   {len(self.positions)}",
        ]
        if self.positions:
            lines.append("")
            lines.append(f"  {'Ticker':<6} {'Weight':>8} {'Mom12-1':>9} {'Vol60':>8}  Reason")
            for p in self.positions:
                mom = f"{p.momentum*100:+.1f}%" if p.momentum is not None else "—"
                vol = f"{p.volatility*100:.1f}%" if p.volatility is not None else "—"
                lines.append(
                    f"  {p.ticker:<6} {p.weight*100:>7.1f}% {mom:>9} {vol:>8}  {p.reason}"
                )
        if self.cash_weight > 0.001:
            lines.append(f"  {'CASH':<6} {self.cash_weight*100:>7.1f}%")
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


# -------------------- Data loading --------------------

def load_enriched_prices(
    target_date: date,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_history_days: int = 0,
) -> pd.DataFrame:
    """Carica prezzi + sector e calcola indicatori, fino a target_date inclusa.

    Ritorna un DataFrame con: ticker, sector, date, close, volume,
    ma200, mom_12_1, vol_60, dollar_volume_20.

    Se `min_history_days > 0`, esclude i ticker che a `target_date` hanno
    meno osservazioni totali nel DB. Il conteggio è separato dalla finestra
    di lookback usata per gli indicatori.
    """
    start = target_date - timedelta(days=int(lookback_days * 1.6))  # buffer per weekend/festivi
    with get_conn() as con:
        eligible_clause = ""
        params: list = [start, target_date]
        if min_history_days > 0:
            eligible_clause = (
                "AND t.ticker IN ("
                "  SELECT ticker FROM prices WHERE date <= ?"
                "  GROUP BY ticker HAVING COUNT(*) >= ?"
                ")"
            )
            params.extend([target_date, min_history_days])
        df = con.execute(
            f"""
            SELECT t.ticker, t.sector, p.date, p.close, p.volume
            FROM prices p
            JOIN tickers t ON t.ticker = p.ticker
            WHERE p.date BETWEEN ? AND ?
              AND t.active = TRUE
              {eligible_clause}
            ORDER BY p.ticker, p.date
            """,
            params,
        ).df()

    if df.empty:
        return df

    # Calcola indicatori per-ticker iterando esplicitamente. Più leggibile e
    # immune a quirks di pandas groupby+apply tra versioni.
    enriched_pieces: list[pd.DataFrame] = []
    for ticker, group in df.groupby("ticker", sort=False):
        # add_all_indicators richiede solo (date, close, volume)
        prices_only = group[["date", "close", "volume"]].reset_index(drop=True)
        enriched = add_all_indicators(prices_only)
        # Riattacca ticker e sector
        enriched["ticker"] = ticker
        enriched["sector"] = group["sector"].iloc[0]
        enriched_pieces.append(enriched)

    return pd.concat(enriched_pieces, ignore_index=True)

def latest_row_per_ticker(enriched: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Per ogni ticker, l'ultima riga con date <= target_date."""
    on_or_before = enriched[enriched["date"] <= pd.Timestamp(target_date)]
    return on_or_before.sort_values("date").groupby("ticker", as_index=False).tail(1)


# -------------------- Pipeline steps --------------------

def detect_regime(
    latest: pd.DataFrame,
    config: StrategyConfig,
    previous_regime: Regime | None = None,
) -> tuple[Regime, list[str]]:
    """Determina il regime in base a SPY vs MA200 e ai difensivi.

    Se `config.regime_buffer_pct > 0` e `previous_regime` è noto, applica una
    zona morta [MA200*(1-b), MA200*(1+b)] entro la quale il macro-stato
    (RISK_ON vs RISK_OFF) viene preservato; la sotto-decisione defensive/cash
    viene comunque rivalutata sui dati correnti.

    Con buffer=0 o previous_regime=None il comportamento è identico al legacy
    (`close > ma200` → RISK_ON, altrimenti RISK_OFF).
    """
    notes: list[str] = []

    spy = latest[latest["ticker"] == config.market_proxy]
    if spy.empty or pd.isna(spy.iloc[0]["ma200"]):
        notes.append(f"Dati {config.market_proxy} insufficienti, default RISK_OFF_CASH")
        return "RISK_OFF_CASH", notes

    spy_row = spy.iloc[0]
    close = spy_row["close"]
    ma200 = spy_row["ma200"]
    buffer = config.regime_buffer_pct

    if buffer > 0 and previous_regime is not None:
        upper = ma200 * (1.0 + buffer)
        lower = ma200 * (1.0 - buffer)
        if close > upper:
            risk_on = True
            notes.append(
                f"{config.market_proxy} {close:.2f} > soglia upper {upper:.2f} → RISK_ON"
            )
        elif close < lower:
            risk_on = False
            notes.append(
                f"{config.market_proxy} {close:.2f} < soglia lower {lower:.2f} → RISK_OFF"
            )
        else:
            risk_on = previous_regime == "RISK_ON"
            notes.append(
                f"{config.market_proxy} {close:.2f} in zona morta "
                f"[{lower:.2f}, {upper:.2f}], mantiene macro = "
                f"{'RISK_ON' if risk_on else 'RISK_OFF'}"
            )
    else:
        risk_on = close > ma200
        if risk_on:
            notes.append(
                f"{config.market_proxy} {close:.2f} > MA200 {ma200:.2f} → RISK_ON"
            )
        else:
            notes.append(
                f"{config.market_proxy} {close:.2f} < MA200 {ma200:.2f} → modalità difensiva"
            )

    if risk_on:
        return "RISK_ON", notes

    # RISK_OFF macro: defensive vs cash sempre valutati su dati correnti
    defs = latest[latest["ticker"].isin(config.defensive_assets)]
    valid_defs = defs[defs["close"] > defs["ma200"]]
    if valid_defs.empty:
        notes.append("Anche difensivi sotto MA200 → 100% CASH")
        return "RISK_OFF_CASH", notes
    notes.append(f"Difensivi sopra MA200: {', '.join(valid_defs['ticker'].tolist())}")
    return "RISK_OFF_DEFENSIVE", notes


def filter_stocks(latest: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Applica i filtri di base sull'universo azionario per modalità RISK_ON."""
    df = latest.copy()
    # Escludi ETF (vogliamo solo singole azioni in RISK_ON)
    df = df[~df["sector"].isin(config.excluded_etf_sectors)]
    # Indicatori disponibili (non NaN)
    df = df.dropna(subset=["ma200", "mom_12_1", "vol_60", "dollar_volume_20"])
    # Liquidità
    df = df[df["dollar_volume_20"] >= config.min_dollar_volume]
    # Trend individuale
    df = df[df["close"] > df["ma200"]]
    return df


def rank_and_select(filtered: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Ordina per momentum 12-1 decrescente, prende top N."""
    return filtered.sort_values("mom_12_1", ascending=False).head(config.n_positions)


def inverse_vol_weights_with_cap(
    vols: pd.Series, max_weight: float, max_iter: int = 50
) -> pd.Series:
    """Pesi proporzionali a 1/vol, normalizzati a 1, con cap per posizione.

    Algoritmo iterativo: applica il cap, ridistribuisci l'eccesso tra gli scoperti
    in proporzione ai loro pesi raw. Itera fino a stabilità (o max_iter).
    Caso limite: se N * max_weight < 1.0 → impossibile, alziamo i pesi al cap
    e accettiamo che la somma sia < 1 (resta cash).
    """
    if len(vols) == 0:
        return pd.Series(dtype=float)

    raw = 1.0 / vols
    weights = raw / raw.sum()

    # Caso degenere: se anche tutti al cap non arriviamo a 1, mettili al cap
    if len(weights) * max_weight <= 1.0:
        return pd.Series(max_weight, index=weights.index)

    for _ in range(max_iter):
        over = weights > max_weight
        if not over.any():
            break
        excess = (weights[over] - max_weight).sum()
        weights[over] = max_weight
        under = ~over
        # Ridistribuisci proporzionalmente ai pesi raw degli scoperti
        if under.any():
            under_raw = raw[under]
            weights[under] = weights[under] + excess * (under_raw / under_raw.sum())
        else:
            break  # tutti al cap, esci
    return weights


def build_defensive_portfolio(
    latest: pd.DataFrame, config: StrategyConfig, target_date: date, notes: list[str]
) -> Portfolio:
    """Costruisce un portafoglio difensivo: GLD/TLT che sono sopra MA200, equipesati."""
    defs = latest[latest["ticker"].isin(config.defensive_assets)]
    valid = defs[defs["close"] > defs["ma200"]]

    portfolio = Portfolio(target_date=target_date, regime="RISK_OFF_DEFENSIVE", notes=notes)
    if valid.empty:
        portfolio.regime = "RISK_OFF_CASH"
        return portfolio

    weight_each = 1.0 / len(valid)
    for _, row in valid.iterrows():
        portfolio.positions.append(Position(
            ticker=row["ticker"],
            weight=weight_each,
            momentum=row["mom_12_1"] if not pd.isna(row["mom_12_1"]) else None,
            volatility=row["vol_60"] if not pd.isna(row["vol_60"]) else None,
            reason=f"defensive (close {row['close']:.2f} > MA200 {row['ma200']:.2f})",
        ))
    return portfolio


# -------------------- Orchestrator --------------------

def build_portfolio(
    target_date: date | None = None,
    config: StrategyConfig | None = None,
    previous_regime: Regime | None = None,
    force_regime: Regime | None = None,
) -> Portfolio:
    """Entry point: costruisce il portafoglio target per una data.

    `previous_regime`: regime attivo al passo precedente; usato dal buffer in
    detect_regime per applicare la zona morta.

    `force_regime`: se non None, salta detect_regime e costruisce il portafoglio
    sotto questo regime. Usato dal backtest quando confirm_weeks blocca uno
    switch e occorre riprodurre il portafoglio sotto il regime ancora attivo.
    Se `force_regime` è strutturalmente impossibile (es. DEFENSIVE ma nessun
    difensivo sopra MA200), il portafoglio risultante può comunque finire
    in CASH per fail-safe.
    """
    config = config or StrategyConfig()
    target_date = target_date or date.today()

    enriched = load_enriched_prices(
        target_date, min_history_days=config.min_history_days
    )
    if enriched.empty:
        return Portfolio(
            target_date=target_date,
            regime="RISK_OFF_CASH",
            notes=[f"Nessun dato disponibile fino al {target_date}"],
        )

    latest = latest_row_per_ticker(enriched, target_date)
    if force_regime is not None:
        regime = force_regime
        notes = [f"regime forzato esternamente (anti-whipsaw): {regime}"]
    else:
        regime, notes = detect_regime(latest, config, previous_regime)

    if regime == "RISK_OFF_CASH":
        return Portfolio(target_date=target_date, regime=regime, notes=notes)

    if regime == "RISK_OFF_DEFENSIVE":
        return build_defensive_portfolio(latest, config, target_date, notes)

    # RISK_ON
    candidates = filter_stocks(latest, config)
    notes.append(f"Candidati dopo filtri: {len(candidates)}")
    if len(candidates) < config.n_positions:
        notes.append(
            f"Avviso: solo {len(candidates)} candidati passano i filtri "
            f"(target {config.n_positions})"
        )
    if candidates.empty:
        return Portfolio(
            target_date=target_date, regime="RISK_OFF_CASH",
            notes=notes + ["Nessun titolo passa i filtri → CASH"],
        )

    selected = rank_and_select(candidates, config).set_index("ticker")
    weights = inverse_vol_weights_with_cap(selected["vol_60"], config.max_weight)

    portfolio = Portfolio(target_date=target_date, regime="RISK_ON", notes=notes)
    for ticker, weight in weights.items():
        row = selected.loc[ticker]
        portfolio.positions.append(Position(
            ticker=ticker,
            weight=float(weight),
            momentum=float(row["mom_12_1"]),
            volatility=float(row["vol_60"]),
            reason=f"momentum rank, sector={row['sector']}",
        ))
    return portfolio
