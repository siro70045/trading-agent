# State Report — `trading-agent` (HMM-ready)

_Report di stato generato il 2026-06-10. Solo lettura del codice in `src/` + interrogazione del DB `data/trading.duckdb`. Nessun file è stato modificato._

Obiettivo dell'analisi: preparare la sostituzione del regime detector binario (SPY vs MA200 → `RISK_ON` / `RISK_OFF_DEFENSIVE` / `RISK_OFF_CASH`) con un HMM che produce probabilità di regime continue, da usare per modulare l'esposizione invece di switchare nettamente. La diagnostica esistente mostra **46% di whipsaw** sulla variante baseline — è il numero che l'HMM dovrebbe abbattere.

---

## 1. Dove nascono i segnali di regime

### Il punto binario attuale (SPY vs MA200)

Funzione: **`detect_regime()` in `src/strategy.py:173-241`**.

Il confronto core è in `src/strategy.py:221`:

```python
risk_on = close > ma200
```

dove `close` e `ma200` sono presi dalla riga SPY (`config.market_proxy = "SPY"`, `src/strategy.py:34`) della tabella `latest` (ultima osservazione ≤ target_date). La MA200 è calcolata in `src/indicators.py:moving_average` (finestra 200, `min_periods=200`) e attaccata come colonna `ma200` in `add_all_indicators` (`src/indicators.py:59-82`).

Struttura della decisione:
- **SPY non disponibile / `ma200` NaN** → `RISK_OFF_CASH` (fail-safe, `src/strategy.py:191-193`).
- **`close > ma200`** → `RISK_ON` (`src/strategy.py:231-232`).
- **`close ≤ ma200`** → macro difensivo, poi sotto-decisione:
  - difensivi (`GLD`, `TLT`, `src/strategy.py:35`) sopra la **loro** MA200 → `RISK_OFF_DEFENSIVE` (`src/strategy.py:240-241`);
  - nessun difensivo sopra MA200 → `RISK_OFF_CASH` (`src/strategy.py:237-239`).

La funzione ritorna `tuple[Regime, list[str]]` — il regime discreto più una lista di note testuali. **Questo è il punto di innesto.**

### Estensioni anti-whipsaw già presenti dentro `detect_regime`

`detect_regime` accetta già `previous_regime` e implementa una **zona morta** (buffer percentuale) attorno a MA200 quando `config.regime_buffer_pct > 0` (`src/strategy.py:200-219`): dentro `[MA200·(1-b), MA200·(1+b)]` il macro-stato precedente viene preservato. Con `buffer=0` o `previous_regime=None` il comportamento è identico al legacy.

### Come il regime determina il resto della pipeline

Orchestratore: **`build_portfolio()` in `src/strategy.py:325-396`**. Branch sul regime:

| Regime | Cosa costruisce | Codice |
|---|---|---|
| `RISK_OFF_CASH` | Portafoglio vuoto (100% cash implicito) | `src/strategy.py:363-364` |
| `RISK_OFF_DEFENSIVE` | `build_defensive_portfolio` → GLD/TLT sopra MA200, **equipesati** | `src/strategy.py:366-367`, `299-320` |
| `RISK_ON` | `filter_stocks` → `rank_and_select` (momentum 12-1, top N) → `inverse_vol_weights_with_cap` | `src/strategy.py:369-396` |

In `RISK_ON`:
- **filtro** (`filter_stocks`, `src/strategy.py:244-255`): esclude ETF, richiede indicatori non-NaN, liquidità ≥ `min_dollar_volume`, trend individuale `close > ma200`;
- **ranking** (`rank_and_select`, `src/strategy.py:258-260`): ordina per `mom_12_1` decrescente, prende `n_positions` (default 5);
- **sizing** (`inverse_vol_weights_with_cap`, `src/strategy.py:263-296`): pesi ∝ 1/vol_60, normalizzati a 1, con cap per posizione (`max_weight=0.35`) ridistribuito iterativamente.

Quindi il regime è un **gate netto a tre stati**: decide quale dei tre rami costruire. Non c'è oggi alcuna nozione di esposizione parziale/continua — è esattamente ciò che l'HMM deve introdurre.

### Switch difensivi a livello backtest

Sopra `detect_regime` c'è un secondo strato anti-whipsaw **temporale** nel backtest: l'hysteresis `regime_confirm_weeks` in `src/backtest.py:210-255`. Un regime "candidato" diverso da quello attivo deve essere proposto per N settimane consecutive prima di switchare; finché è bloccato, il portafoglio viene ricostruito con `build_portfolio(..., force_regime=active_regime)` (`src/backtest.py:248-252`, parametro `force_regime` definito in `src/strategy.py:329` e gestito a `357-359`).

### Punto di innesto preciso per il detector probabilistico

> **In `detect_regime()` di `src/strategy.py` (righe 173-241): oggi ritorna `tuple[Regime, list[str]]`. Andrebbe estesa a calcolare/ritornare anche `P(regime)` — es. `tuple[Regime, list[str], dict[str, float]]` o un dataclass `RegimeSignal` con `probs`.** Il consumo di quelle probabilità (traduzione P→peso) va poi agganciato in **`build_portfolio()` (`src/strategy.py:325-396`)**, dove oggi il regime discreto sceglie il ramo: lì si introduce uno scaling dell'esposizione complessiva. `previous_regime` e `force_regime` esistono già come canali di stato tra backtest e strategia e sono riusabili dall'HMM (es. per lo smoothing/stato latente precedente).

---

## 2. Data layer

DB: **DuckDB** in `data/trading.duckdb`. Schema in `src/db.py:8-90`.

### Tabelle e colonne

| Tabella | Colonne (tipo) | Ruolo |
|---|---|---|
| `tickers` | `ticker VARCHAR PK`, `name VARCHAR`, `sector VARCHAR`, `added_at TIMESTAMP`, `active BOOLEAN` | universo |
| `prices` | `ticker VARCHAR`, `date DATE`, `open/high/low/close DOUBLE`, `volume BIGINT`, `dividends DOUBLE`, `splits DOUBLE`, PK `(ticker,date)` | OHLCV giornaliero |
| `download_log` | `ticker`, `run_at`, `rows_inserted`, `last_date`, `status`, `error_message` | audit download |
| `signals` | `run_at TS`, `target_date DATE`, `regime VARCHAR`, `ticker`, `weight DOUBLE`, `momentum`, `volatility`, `reason`, PK `(run_at,ticker)` | segnali live |
| `backtest_runs` | `run_id VARCHAR PK`, `run_at`, `start_date`, `end_date`, `cost_per_trade DOUBLE`, `config_json VARCHAR`, `final_equity DOUBLE`, `n_periods INT` | header run |
| `backtest_equity` | `run_id`, `date`, `equity`, `regime`, `turnover`, `return_gross`, `return_net`, PK `(run_id,date)` | curva equity per run |
| `backtest_trades` | `run_id`, `decision_date`, `exec_date`, `ticker`, `weight_old`, `weight_new`, `delta` | trade per run |

Conteggi attuali nel DB: `prices` 116.576 righe, `signals` 38, `backtest_runs` 4, `backtest_equity` 2.988, `backtest_trades` 12.760, `download_log` 31.

### Frequenza dati prezzi

**Giornaliera** (trading days). Verificato sulle prime date SPY: `2011-05-04, 05-05, 05-06, 05-09, 05-10, …` (salta i weekend). Gli indicatori sono tutti giornalieri (`src/indicators.py`: MA200, mom 12-1 = `shift(21)/shift(252)`, vol_60, dollar_volume_20). Il **backtest** opera invece a **cadenza settimanale** (decisione venerdì close → esecuzione lunedì open, vedi §3), campionando la serie giornaliera.

### Universo: 31 ticker, ruoli speciali

Definito in `src/universe.py:9-49` (`STARTER_UNIVERSE`), sincronizzato via `sync_universe_to_db`. Conteggio nel DB: **31 ticker attivi**.

- **Market proxy**: `SPY` (`src/strategy.py:34`) — unico driver del regime macro e calendario di trading del backtest (`src/backtest.py:66`, `_trading_days(calendar_ticker="SPY")`).
- **Difensivi**: `GLD`, `TLT` (`src/strategy.py:35`) — usati in `RISK_OFF_DEFENSIVE` e nel benchmark 60/40.
- **Azioni singole** (21): AAPL, MSFT, GOOGL, AMZN, META, NVDA, JNJ, UNH, LLY, JPM, BRK-B, V, CAT, HON, KO, PG, WMT, COST, XOM, CVX — il pool selezionabile in `RISK_ON`.
- **ETF esclusi dal pool azionario** (`excluded_etf_sectors`, `src/strategy.py:36-40`): SPY, QQQ, IWM, EFA, EEM, XLF, XLK, XLE, XLV, GLD, TLT — figurano come dati ma non come candidati azionari.

⚠️ Nessun **VIX** né dati di volatilità implicita nel DB: l'unica volatilità disponibile è **realizzata** (`vol_60`). Se l'HMM dovesse usare il VIX andrebbe aggiunto all'universo e scaricato.

### Storia disponibile

- Range prezzi globale: **2011-05-04 → 2026-04-29**.
- Quasi tutti i ticker: 3.769 bar fino al 2026-04-29. Eccezione: **META 3.506 bar** (IPO 2012) — più corta.
- I backtest salvati coprono **2012-01-01 → 2026-04-29** (l'inizio 2012 lascia il buffer per MA200/mom 12-1 che richiedono ~252 giorni pieni).

**Per l'HMM**: ~15 anni di daily su SPY (≈3.770 osservazioni giornaliere, ≈746 settimanali) come orizzonte di training. Feature direttamente derivabili dai daily SPY senza nuovi download: rendimenti, vol realizzata, distanza da MA200, momentum.

---

## 3. Motore di backtest

File: **`src/backtest.py`**. Entry point: **`run_backtest(start_date, end_date, config=None, cost_per_trade=0.0005)` (`src/backtest.py:153-351`)**.

### Frequenza ed esecuzione

- **Settimanale**: decisione ogni **venerdì close**, esecuzione al **primo trading day strettamente successivo (≈lunedì) all'`open`** (`src/backtest.py:1-14`, mapping venerdì→exec a `186-194` via `_fridays` `79-85` e `_trading_days` `66-76`).
- Rendimento settimanale = movimento `open→open` tra exec consecutivi, pesato sui pesi **decisi al venerdì precedente** (`_gross_period_return`, `src/backtest.py:130-148`; loop principale `293-329`).
- Equity ricorsiva `equity[i+1] = equity[i]·(1+net)`, indicizzata per `exec_date`, punto iniziale `1.0` meno il costo iniziale di apertura (`src/backtest.py:278-281`).
- CASH ha rendimento 0 (ignora interessi, `src/backtest.py:12`).

### Modello di costi

- `cost = turnover_one_way · cost_per_trade`, applicato una sola volta per rebalance (`src/backtest.py:305`).
- **Turnover one-way** = somma degli aumenti di peso (≡ somma decrementi), **CASH incluso** come asset esplicito di peso `1−Σ` (`_turnover_one_way`, `src/backtest.py:115-127`). Così anche 100%→cash genera turnover 1.
- Costo iniziale: passaggio da 100% cash al primo portafoglio, scaricato subito sull'equity di partenza (`src/backtest.py:275-281`).
- Default `cost_per_trade = 0.0005` (5 bps per side).

### Universo dinamico (`min_history_days`)

Gestito **dentro `build_portfolio` → `load_enriched_prices`** (`src/strategy.py:109-163`), non nel backtest. `config.min_history_days` (default **504 ≈ 2 anni**, `src/strategy.py:50`) esclude a ogni `target_date` i ticker con meno di N osservazioni totali ≤ data (subquery `src/strategy.py:127-134`). Il conteggio è separato dalla finestra di lookback degli indicatori. Effetto: META e ticker giovani entrano nel pool solo quando hanno storia sufficiente → niente look-ahead survivorship.

### Cosa viene salvato in DB per run

`save_backtest` (`src/backtest.py:364-432`): un header in `backtest_runs` (incl. `config_json` serializzato da `_config_to_json`, `356-361`), una riga per `exec_date` in `backtest_equity` (equity, regime, turnover, return_gross/net), e tutte le variazioni di peso in `backtest_trades`. `run_id` autogenerato `bt_<timestamp>` se non passato.

### Metriche prodotte e dove

- **`src/metrics.py`** — funzioni pure su equity curve: `cagr` (`34-45`), `total_return`, `sharpe_ratio` (`57-70`), `sortino_ratio`, `max_drawdown` (`101-105`), `calmar_ratio`, `win_rate`, `profit_factor`, riepilogo in `summarize()` (`177-195`, ritorna `PerformanceSummary`). Annualizzazione via `periods_per_year` (52 per settimanale).
- **`src/diagnose.py`** — metriche di regime/whipsaw/turnover/yearly (vedi §4).
- **`src/benchmarks.py`** — `build_spy_buy_and_hold` (`92-101`) e `build_60_40` (`106-153`), indicizzati sugli stessi `exec_date` del backtest per confronto riga-per-riga.

La CLI `main()` (`src/backtest.py:437-467`) stampa periodi, equity finale, turnover medio, conteggio regimi, e salva (default ON, `--no-save` per disattivare).

---

## 4. Diagnostica whipsaw — la metrica del 46%

### Dove e come è calcolata

File **`src/diagnose.py`**, funzione **`compute_regime_switches(regimes, whipsaw_window_weeks=4)` (`src/diagnose.py:185-213`)**.

### Definizione operativa

- **Switch**: `regime[t] != regime[t-1]` (con `prev` non-NaN), `src/diagnose.py:196-197`.
- **Whipsaw**: uno switch A→B **seguito entro `whipsaw_window_weeks=4` periodi** da un B→A — cioè il ritorno al regime di partenza entro 4 settimane (`src/diagnose.py:202-212`). La distanza è in numero di periodi indice; per ogni switch si guarda al primo switch successivo che inverte (`break` al primo match). Nota: la window di default è **4 settimane**, non parametrizzata da CLI.

L'output è `{"n_switches", "switches", "whipsaws"}`, stampato da `print_switches` (`src/diagnose.py:371-383`). Metriche correlate: `compute_turnover_and_costs` (`218-250`), `compute_yearly_table` (`255-303`, incl. alpha vs SPY).

### Output reale — run più recente (`latest_run_id()` → variante CONFIRM)

`python -m src.diagnose` (run di default = `variant_confirm_20260501_002847_392674`):

```
========================================================================
Backtest run: variant_confirm_20260501_002847_392674
Periodo:      2012-01-01 → 2026-04-29
Costi:        0.050% per side
Equity finale: 6.7792  (746 periodi)
========================================================================

--- 1. Drawdown massimo ---
  Peak:           2020-02-18  (equity 3.0519)
  Trough:         2020-03-16  (equity 2.2158)
  Drawdown:       -27.40%
  Durata caduta:  4 settimane
  Recovery:       2020-08-31  (28 settimane peak→recovery)

--- 2. Statistiche per regime ---
  Regime                     N   %time  avg ret/wk  contrib comp.
  RISK_ON                  557   74.7%     +0.388%        +630.6%
  RISK_OFF_CASH            121   16.2%     -0.002%          -0.3%
  RISK_OFF_DEFENSIVE        68    9.1%     -0.075%          -6.9%

--- 3. Switch di regime ---
  Totale switch:   30
  Whipsaw (≤4 wk): 8
  ...
  Whipsaw entro 4 settimane:
    2014-10-20 → 2014-11-03  RISK_ON→RISK_OFF_DEFENSIVE→RISK_ON  (2 wk)
    2015-12-28 → 2016-01-19  RISK_OFF_DEFENSIVE→RISK_OFF_CASH→RISK_OFF_DEFENSIVE  (3 wk)
    2022-04-04 → 2022-04-18  RISK_OFF_DEFENSIVE→RISK_ON→RISK_OFF_DEFENSIVE  (2 wk)
    2022-05-16 → 2022-05-31  RISK_OFF_DEFENSIVE→RISK_OFF_CASH→RISK_OFF_DEFENSIVE  (2 wk)
    2022-05-31 → 2022-06-21  RISK_OFF_CASH→RISK_OFF_DEFENSIVE→RISK_OFF_CASH  (3 wk)
    2023-03-20 → 2023-04-03  RISK_ON→RISK_OFF_DEFENSIVE→RISK_ON  (2 wk)
    2023-10-30 → 2023-11-13  RISK_ON→RISK_OFF_DEFENSIVE→RISK_ON  (2 wk)
    2026-03-30 → 2026-04-20  RISK_ON→RISK_OFF_DEFENSIVE→RISK_ON  (3 wk)

--- 4. Turnover e costi ---
  Turnover settimanale medio:    15.11%
  Turnover annualizzato:           786%
  Rendimento composto LORDO:    +617.28%
  Rendimento composto NETTO:    +577.92%
  Cost drag (geom):              +39.36%
  Costo / rendimento lordo:       6.38%
```

### Output reale — il 46% (run BASELINE, `bt_20260430_155416_255140`)

Il **46%** citato non è del run più recente ma della **BASELINE legacy** (buffer=0, confirm=0). `python -m src.diagnose --run-id bt_20260430_155416_255140`:

```
--- 3. Switch di regime ---
  Totale switch:   50
  Whipsaw (≤4 wk): 23           →  23/50 = 46.0%

--- 4. Turnover e costi ---
  Turnover settimanale medio:    17.66%
  Turnover annualizzato:           918%
  Rendimento composto LORDO:    +743.41%
  Rendimento composto NETTO:    +689.42%
  Cost drag (geom):              +53.99%
  Costo / rendimento lordo:       7.26%
```

(La drawdown anatomy del baseline è diversa: peak 2021-11-08 → trough 2023-03-13, **−23.53%**, 70 settimane di caduta, recovery 2023-07-31.)

### Sintesi metriche whipsaw

| Metrica | BASELINE (il "46%") | CONFIRM (run più recente) |
|---|---|---|
| Totale switch | 50 | 30 |
| # whipsaw (≤4 wk) | 23 | 8 |
| **% whipsaw / switch** | **46.0%** | 26.7% |
| Turnover settimanale medio | 17.66% | 15.11% |
| Turnover annualizzato | 918% | 786% |
| Cost drag (% rend. lordo) | 7.26% | 6.38% |

### Metriche correlate (BASELINE)

- **Turnover settimanale 17.66%** → annualizzato **918%** (portafoglio quasi interamente rivoltato ~9×/anno).
- **Cost drag** geometrico **+53.99%** di rendimento composto, **7.26% del lordo**.
- **Anni con alpha negativo vs SPY: 6 su 15** → 2012 (−9.11%), 2013 (−22.25%), 2014 (−5.98%), 2015 (−14.91%), 2019 (−7.35%), 2025 (−16.25%). I peggiori (2013, 2025) sono anni di forte bull SPY in cui la strategia è stata sotto-esposta/defensive — esattamente il costo dello switching netto che l'esposizione continua dell'HMM mira a mitigare.

---

## 5. Stato esperimenti anti-whipsaw già fatti

**Esistono, e sono già stati eseguiti.** Non si parte da zero.

### Parametri in `StrategyConfig`

Presenti entrambi (`src/strategy.py:51-57`):
- `regime_buffer_pct: float = 0.0` — zona morta % attorno a MA200 (logica in `detect_regime`, `src/strategy.py:200-219`).
- `regime_confirm_weeks: int = 0` — settimane di conferma prima di switchare (logica hysteresis in `src/backtest.py:210-255`).

Default = comportamento legacy (nessun anti-whipsaw).

### `compare_variants.py`

Esiste: **`src/compare_variants.py`**. Definisce 3 varianti (`src/compare_variants.py:38-42`):
- `BASELINE` — `StrategyConfig()` (buffer=0, confirm=0)
- `BUFFER` — `StrategyConfig(regime_buffer_pct=0.02)` (zona morta ±2%)
- `CONFIRM` — `StrategyConfig(regime_confirm_weeks=2)` (2 settimane di conferma)

Lancia i 3 backtest, li salva, e stampa una tabella comparativa riusando le `compute_*` di `diagnose` + `summarize` di `metrics`.

### Risultati salvati nel DB (run del 2026-05-01)

I 3 run `variant_*` sono persistiti. Numeri reali ricalcolati dai run salvati:

| Metrica | BASELINE | BUFFER (±2%) | CONFIRM (2 wk) |
|---|---|---|---|
| Final equity | 7.894 | **8.340** | 6.779 |
| CAGR | +15.49% | **+15.93%** | +14.27% |
| Sharpe | 0.98 | **1.00** | 0.90 |
| Max drawdown | −23.53% | **−21.37%** | −27.40% |
| # switch | 50 | **24** | 30 |
| # whipsaw (≤4 wk) | 23 | **6** | 8 |
| **% whipsaw / switch** | 46.0% | **25.0%** | 26.7% |
| Turnover sett. medio | 17.66% | **14.38%** | 15.11% |
| Cost drag % gross | 7.26% | **5.91%** | 6.38% |

**Lettura**: il **BUFFER ±2% domina** — taglia i whipsaw da 46% a 25%, riduce turnover e drawdown, e **migliora anche CAGR/Sharpe** (non è un trade-off, è un free lunch su questo dataset). Il CONFIRM riduce i whipsaw ma **peggiora** CAGR, Sharpe e drawdown (la conferma ritardata fa entrare tardi nei rimbalzi, vedi il MDD −27.40% del 2020).

**Implicazione per l'HMM**: c'è già un baseline anti-whipsaw forte da battere (BUFFER). L'HMM non va valutato contro la sola BASELINE legacy ma **contro la variante BUFFER** — altrimenti il confronto è truccato a suo favore. Il buffer è essenzialmente un'hysteresis deterministica a 1 feature; l'HMM deve giustificare la complessità extra superando un 25% di whipsaw / Sharpe 1.00.

---

## 6. Pianificazione integrazione HMM (prospettica)

_Questa sezione è una proposta, non un dato di fatto._

### File/funzione da modificare

1. **`src/strategy.py:detect_regime` (173-241)** — punto di innesto primario. Estendere il valore di ritorno per includere `P(regime)`. Opzione pulita: introdurre un dataclass

   ```python
   @dataclass
   class RegimeSignal:
       regime: Regime          # stato MAP, per retrocompatibilità/log
       probs: dict[str, float] # P(bull/neutral/bear) o P(RISK_ON) continua
       notes: list[str]
   ```

   e far ritornare `detect_regime` un `RegimeSignal`. Branch su `config.regime_mode` (vedi sotto): `"binary"` → percorso attuale impacchettato in `probs` degeneri {0/1}; `"hmm"` → inferenza HMM.

2. **`src/strategy.py:build_portfolio` (325-396)** — dove `P(regime)` viene tradotta in **scaling dell'esposizione complessiva**: dopo aver costruito i pesi `RISK_ON`, moltiplicarli per `exposure = f(P(bull))`, lasciando il resto in cash. Questo mantiene intatta la pipeline di selezione/sizing e inserisce l'HMM come **overlay** sopra di essa.

3. **Nuovo modulo `src/regime_hmm.py`** — isola fit/inferenza HMM (training su rolling/expanding window, niente look-ahead: a ogni `target_date` il modello vede solo dati ≤ data). Mantiene `strategy.py` sottile.

### Feature di input suggerite

Tutte derivabili dai daily SPY già in DB (nessun nuovo download), in ordine di priorità:

1. **Rendimenti giornalieri SPY** (log-returns) — l'osservabile classico per un HMM a 2-3 stati (bull/bear ± neutral); cattura il livello di drift.
2. **Volatilità realizzata** (es. `vol_60`, o una 20g più reattiva) — gli stati bear sono ad alta vol; è il discriminante più forte. Già calcolata in `indicators.py:realized_volatility`.
3. **Distanza da MA200** (`close/ma200 − 1`) — continuità col segnale legacy, dà all'HMM lo stesso input del detector binario in forma continua.
4. **Cross-sectional momentum / dispersione** dei 21 titoli del pool (breadth) — opzionale, cattura la salute interna del mercato oltre l'indice.
5. **Volume / dollar-volume** — debole come segnale di regime, lo terrei fuori dal PoC.
6. **VIX** — teoricamente ottimo (vol implicita forward-looking) ma **non in DB**: richiede di aggiungerlo all'universo e scaricarlo. Da rimandare a dopo il PoC.

Raccomandazione PoC: **2 feature — log-return SPY + vol realizzata**, HMM gaussiano a 2-3 stati. È il minimo difendibile e già potenzialmente migliore del buffer.

### Traduzione P(regime) → peso di portafoglio

- **Opzione A — esposizione lineare** `exposure = P(bull)`. Massima continuità, zero soglie, ma esposizione raramente a 0/100% → turnover potenzialmente *alto* (peso che oscilla a ogni settimana). Va abbinata a smoothing.
- **Opzione B — step function** (es. `<0.3 → cash`, `0.3–0.7 → 50%`, `>0.7 → pieno`). Reintroduce discretizzazione → rischio di whipsaw sulle soglie, ma turnover più controllato. È di fatto un buffer a 3 livelli.
- **Opzione C — smoothing temporale** su `P(bull)` (EMA) prima della traduzione lineare. **Raccomandata**: combina la continuità di A con stabilità; l'HMM è già intrinsecamente persistente (matrice di transizione), ma una EMA leggera sulla posterior taglia il jitter residuo che genera turnover.

Proposta concreta: **C poi A** — `exposure = clip(EMA_k(P(bull)), 0, 1)`, con dead-band opzionale (non muovere il peso se |Δexposure| < ε) per contenere il turnover. Misurare turnover/whipsaw esattamente con le `compute_*` esistenti.

### Integrazione senza rompere il backtest

- Aggiungere a `StrategyConfig` un campo **opzionale**:
  ```python
  regime_mode: Literal["binary", "hmm"] = "binary"
  ```
  più i parametri HMM (`hmm_n_states`, `hmm_features`, `hmm_smoothing_k`, `hmm_train_window`). Default `"binary"` → **comportamento bit-identico all'attuale** (i 4 run salvati restano riproducibili). Il campo va aggiunto in coda al dataclass (`src/strategy.py:31-57`) e viene serializzato automaticamente da `_config_to_json` (`src/backtest.py:356-361`) → tracciabilità nel DB senza modifiche.
- `compare_variants.py` guadagna una 4ª variante `HMM` accanto a BASELINE/BUFFER/CONFIRM → confronto diretto sulle stesse metriche, stesso periodo.
- L'overlay di esposizione si innesta solo nel ramo `RISK_ON`; `RISK_OFF_*` possono restare come fallback quando `P(bull)` è bassa (oppure essere assorbiti: `P(bull)→0` ≡ cash). Da decidere nel PoC.

### Test di non-regressione necessari

- **Identità binary-mode**: con `regime_mode="binary"`, `run_backtest` su 2012-01-01→2026-04-29 deve riprodurre **esattamente** l'equity di `bt_20260430_155416_255140` (final_equity 7.8942, 50 switch). Test golden sull'equity curve.
- **No look-ahead nell'HMM**: a ogni `target_date` l'inferenza usa solo dati ≤ data. Test che il `P(regime)` a una data non cambia se si aggiungono dati futuri.
- **Determinismo**: stesso seed → stesso fit (gli HMM EM sono sensibili all'init; fissare `random_state`).
- **Coerenza pesi**: `Σ pesi + cash == 1`, esposizione ∈ [0,1], nessun NaN — riusare gli assert impliciti di `inverse_vol_weights_with_cap`.
- Estendere i test esistenti (`src/tests_strategy.py`, `src/tests_backtest.py`, `src/tests_diagnose.py`).

### Librerie candidate

- **`hmmlearn`** — raccomandata per il PoC. `GaussianHMM` copre esattamente il caso (osservazioni continue, 2-3 stati), API matura, scikit-learn-like, dipendenza leggera. Unica accortezza: l'ordinamento degli stati non è semantico (lo stato 0 non è "per forza" bear) → mappare gli stati a bull/bear via la media dei rendimenti per stato dopo il fit.
- **`pomegranate`** — più flessibile (distribuzioni custom, GPU) ma API più volatile tra versioni e overkill per un PoC a 2 feature.
- **Implementazione custom** (Baum-Welch + Viterbi) — solo se serve controllo totale sul training online/incrementale; non per il primo PoC.

Da aggiungere a `requirements.txt`: `hmmlearn` (porta con sé `scikit-learn`/`scipy`, compatibili con numpy/pandas già presenti).

### Stima di lavoro per un PoC funzionante

| Blocco | Ore-uomo |
|---|---|
| `src/regime_hmm.py`: fit gaussiano 2-3 stati, mapping stati→regime, inferenza walk-forward no-look-ahead | 4–6 |
| Estensione `detect_regime`/`RegimeSignal` + `regime_mode` in `StrategyConfig` + serializzazione | 2–3 |
| Overlay esposizione in `build_portfolio` (opzione C+A con dead-band) | 2–3 |
| Test di non-regressione (golden binary-mode, no-look-ahead, determinismo) | 3–4 |
| 4ª variante in `compare_variants` + giro di tuning iniziale (n_states, smoothing) | 2–3 |
| **Totale PoC** | **≈ 13–19 ore-uomo (~2–3 giorni)** |

Esclusi: aggiunta del VIX (download + universo, ~1h), tuning approfondito, validazione out-of-sample formale.

---

### TL;DR per la decisione

- Innesto pulito e già predisposto: `detect_regime` (`src/strategy.py:173`) per produrre `P(regime)`, `build_portfolio` (`src/strategy.py:325`) per tradurla in esposizione; canali di stato (`previous_regime`, `force_regime`, `config_json`) già esistenti.
- Dati pronti: 15 anni di daily SPY, nessun VIX.
- **Il vero benchmark da battere non è il 46% baseline ma il 25% del BUFFER ±2%**, che già migliora anche CAGR/Sharpe. L'HMM deve giustificare la complessità superando quello. Se l'obiettivo è solo ridurre il whipsaw col minimo sforzo, il BUFFER è già lì; l'HMM ha senso se si vuole esposizione **continua** (sizing graduale del rischio), non solo meno switch.
