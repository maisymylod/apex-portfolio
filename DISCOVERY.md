# DISCOVERY — current daily-run pipeline (as of 2026-06-05)

Phase 0 map of the actual code paths, written before any analytics changes.
Repo state read at commit `8ab9553` (merge apex-quant). Inception 2026-05-20,
2 history rows on disk, no trades since seed (shares are unchanged since day 1
— this matters for backfill, see §4).

---

## 1. Data flow

```
GitHub Actions (.github/workflows/daily.yml, 20:30 UTC Mon–Fri, also manual dispatch)
  └─ python -m agent.daily  →  agent/daily.py:main()
       │
       ├─ portfolio.load()                          state/portfolio.json
       │    └─ (first run only) seed_portfolio() — buys planner.TARGET_WEIGHTS at close
       │
       ├─ prices.fetch_close_prices(position tickers)        [yfinance call #1, ~8d window]
       │    └─ abort (exit 1) if dict is EMPTY; silently drops individual missing tickers
       │
       ├─ portfolio.mark_to_market(state, px)  →  snapshot dict
       │    └─ positions sorted by MV desc; weight_pct vs total; pnl vs cost basis
       │    └─ NOTE: a position whose price is missing is SKIPPED ENTIRELY —
       │       its MV vanishes from holdings_value/total_value (silent fake loss)
       │
       ├─ risk.risk_report(snapshot)  →  static checks
       │    └─ hardcoded sector_map (power/hyperscaler/semi/miners), 20% name cap,
       │       50% sector cap → flags + GREEN/RED status
       │
       ├─ prior_total = float(history.csv rows[-1].split(",")[1])      ← POSITIONAL READ
       │
       ├─ risk.update_biases(snapshot, prior_total)  →  learning loop
       │    └─ reads+writes state/learning.json: per-ticker bias = decay·prior + nudge,
       │       nudge = clip(pnl_pct_since_inception/100·0.1, ±0.05), cap ±0.25
       │    └─ appends {as_of, delta_pct} to calibration[] (keeps 30)
       │    └─ ⚠ docstring says "5-day rolling return vs SPY" — code actually uses
       │       since-inception pnl_pct, no SPY involved (doc/code mismatch)
       │    └─ ⚠ learning.json is OVERWRITTEN each run — no per-day bias history persists
       │       (yesterday's biases survive only inside journal markdown)
       │
       ├─ compute_quant_block(snapshot, learning)              [yfinance call #2]
       │    ├─ prices.fetch_history(universe = 13 portfolio tickers + SPY, XLU, SMH,
       │    │                        days=BL_HISTORY_DAYS=90)  → DataFrame (ffilled closes)
       │    │   └─ ⚠ SPY IS ALREADY FETCHED EVERY RUN (planner.BENCHMARKS) — a benchmark
       │    │      series exists in-process today; it just isn't persisted or used
       │    │      outside BL views
       │    ├─ planner.compute_bl_weights(history, biases)
       │    │   └─ rolling_cov(60d) → equilibrium Π → posterior μ_BL (4 THESIS_VIEWS)
       │    │      → optimal_weights (long-only, 20% cap) → apply_bias(learning)
       │    ├─ rebalance candidates where |Δ| ≥ 3pp
       │    └─ returns {bl_weights, mu_bl, n_views, rebalance, bl_run_ts,
       │               history ← the DataFrame, kept in-memory for MC/BSM}
       │    └─ whole block: try/except → None + traceback print (run continues)
       │
       ├─ risk.mc_risk_report(snapshot, quant["history"], mu_bl)   [only if quant]
       │    └─ quant/mc.py: 10,000 correlated-GBM paths × 252d, seed=20260520 (fixed),
       │       weights = MV/gross_holdings (CASH EXCLUDED — sim is invested sleeve only,
       │       but ruin_threshold = 0.70 × total_value INCLUDING cash)
       │    └─ VaR/CVaR(95%,1d), median MaxDD, sim Sharpe/return/vol, P(ruin)
       │    └─ veto thresholds in risk.py ($80 VaR / $120 CVaR / −25% MaxDD / 15% ruin);
       │       veto ⇒ risk status forced RED + flags appended
       │    └─ try/except → None
       │
       ├─ compute_options_overlay(snapshot, quant, mc_report)     [only if quant]
       │    └─ quant/bsm.py: NVDA protective put, K=0.9·spot, T=60d, σ=30d realized
       │       → value, greeks, 4 trigger conditions → ACTIVE/MONITORING (paper signal only)
       │    └─ try/except → None
       │
       ├─ portfolio.append_history(snapshot, mc_report, bl_run)   → state/history.csv
       ├─ portfolio.save(state)  (adds bl_weights, bl_run_ts, mc_report, options_overlay,
       │                          last_run to portfolio.json — additive keys)
       ├─ write_journal(...)     → journal/YYYY-MM-DD.md   ⚠ same-day rerun OVERWRITES
       ├─ update_readme(snapshot, history_count = len(rows)−1)  → README live P&L block
       │
       └─ workflow: cp state/{portfolio.json,history.csv,learning.json} docs/data/
                    git add state/ journal/ docs/data/ README.md; commit; push
                         │
                         └─ GitHub Pages → docs/index.html + docs/app.js
                              ├─ fetch data/portfolio.json  → positions, bl_weights,
                              │    mc_report, options_overlay (key-based access — additive-safe)
                              ├─ fetch data/history.csv     → parseHistory() POSITIONAL
                              └─ fetch data/learning.json   (optional, try/catch)
```

Failure containment today: quant, MC, and options each fail soft (None + log).
`fetch_close_prices` returning a *partial* dict does **not** fail soft — it
silently shrinks the book (see §5 gap G).

## 2. history.csv schema and positional readers

`portfolio.HISTORY_HEADER` (11 columns, in code):

```
idx  0 date_utc      4 pnl_usd       8 sim_sharpe
     1 total_value   5 pnl_pct       9 p_ruin
     2 cash          6 var_95_usd   10 bl_run
     3 holdings      7 cvar_95_usd
```

⚠ The file on disk still has the **old 6-column header**
(`date_utc,total_value,cash,holdings_value,pnl_usd,pnl_pct`) and two 6-field
rows. `append_history` migrates the header line in place on next run (exact
list-equality check, header line rewritten, data rows untouched) — so old rows
stay 6 fields and new rows have 11. Every reader must keep tolerating short rows.

Positional readers (the reason columns must only ever be APPENDED at the end):

| Reader | Location | What breaks on insert/reorder |
|---|---|---|
| `rows[-1].split(",")[1]` | agent/daily.py:370 (prior_total for learning loop) | index 1 must stay `total_value` forever |
| `const [date, total, cash, holdings, pnl, pnlPct, var95, cvar95, sharpe, pRuin, blRun] = r.split(',')` | docs/app.js:27 parseHistory | indices 0–10 fixed; extra trailing fields are silently ignored by destructuring → **appending is safe**, inserting shifts everything |
| `len(rows) − 1` as "Day N" | daily.py:397→update_readme; app.js renderStats (`history.length`) | row-count only — column-safe, but double-runs per day inflate the day count (already happened: two 2026-05-20 rows) |
| header-equality migration | portfolio.append_history:103 | extending `HISTORY_HEADER` at the END reuses this path automatically; reordering would silently mislabel old data |

portfolio.json consumers are all key-based (`p.positions`, `p.bl_weights`,
`p.mc_report`, `p.options_overlay` in app.js; `state[...]` in Python) —
**adding keys is safe; renaming/removing is not.**

journal/*.md and README are render-only (no parser reads them back).

## 3. Gap list — value vs effort

Scale: value ●●● high / ●● med / ● low; effort in rough implementation size.

| # | Gap | Value | Effort | Notes |
|---|---|---|---|---|
| A | No benchmark anywhere: P&L is absolute-only; SPY is fetched every run for BL and then discarded | ●●● | S–M | Highest leverage. Series already in `quant["history"]`; needs persistence + alignment + stats |
| B | No realized metrics: every Sharpe/vol/return shown is MC-simulated; equity curve in history.csv is unused analytically | ●●● | S | Pure pandas/numpy on existing data; must handle 2-row history |
| G | Silent data-quality failures: missing ticker ⇒ position silently dropped from totals ⇒ fake P&L swing recorded permanently in history.csv | ●●● | S | Cheap to detect (compare px keys vs positions; staleness vs last trading day); should land first or with A/B since corrupt history poisons all realized metrics |
| C | No attribution: can't say which name/sector drove a day's P&L; ~16% cash drag never quantified | ●● | M | Needs per-position prior-day prices (in `quant["history"]`) |
| D | Risk is portfolio-total only: no per-position component VaR, no HHI/effective-N, no diversification ratio | ●● | S–M | Analytic (Σ-based) version is closed-form from the existing cov matrix; no extra MC needed |
| E | No stress tests tied to the thesis (AI-capex drawdown, power/rates, miners/BTC) | ●● | S | Deterministic shock vectors × current weights; trivially seedless/reproducible |
| F | Learning loop is unmeasured: biases applied to BL weights but never validated against realized outcomes; also doc/code mismatch (no SPY-relative logic) and bias history not persisted | ●● | M | Requires persisting per-day bias snapshots FIRST; metrics only become meaningful after ~weeks of data |
| — | Same-day rerun artifacts: duplicate history rows, journal overwrite, inflated Day-N | ● | S | Hygiene; bundle with G |
| — | learning nudge uses since-inception pnl_pct, not daily delta (code comment admits this) | ● | S | Fix naturally falls out of having a real prior-close column |

## 4. Backfillability

"Backfillable" = computable today for all dates since inception (2026-05-20),
not just from now on. Key enabling fact: **no trades have occurred since seed**
— shares per ticker are constants, so any per-position daily series can be
reconstructed as `shares × close(t)` from yfinance history.

| Analytic | Backfillable? | From |
|---|---|---|
| A: benchmark curve, active return, beta/alpha, TE, IR, capture | **Yes, fully** | yfinance SPY/^GSPC closes since inception_date (16 days). The existing `fetch_history` call already covers the window; just extend `days` to `max(90, days_since_inception + buffer)` as the book ages |
| B: realized return/vol/Sharpe/Sortino/Calmar/drawdown/underwater | **Yes** | history.csv `total_value` column (dedup same-date rows, keep last). Today: 2 points → most stats undefined; must degrade gracefully |
| C: per-position / per-sector attribution, cash drag | **Yes, while untraded** | shares (constant) × yfinance daily closes; once a rebalance ever executes, prior days freeze and it becomes forward-tracked from persisted rows |
| D: component VaR, HHI, diversification ratio | n/a (point-in-time) | current weights + 60d rolling cov already computed for BL/MC |
| E: scenario shocks | n/a (point-in-time) | current weights + static shock definitions |
| F: bias hit-rate / rank-IC vs realized next-day returns | **No — forward only** | per-day bias snapshots are not persisted (learning.json overwritten; journals are markdown). Must start persisting a bias history file before any validation is possible. (Parsing old journal tables could recover ~2 weeks but is fragile; not worth it) |
| G: data-quality status | n/a (per-run) | px dict vs position keys; last-bar date vs expected trading day |

## 5. Current per-run network budget (constraint baseline)

Two yfinance batch downloads per run: (1) close prices, 13 tickers × ~8 calendar
days; (2) history, 16 tickers × ~144 calendar days. All new analytics can run off
download (2) — widening its window covers benchmark backfill with **zero new
network calls**. Reproducibility precedent: MC seed fixed at 20260520; any new
stochastic code must follow suit (stress tests as specced are deterministic anyway).

## 6. Phase 1 addendum (implemented 2026-06-05)

Data-flow changes shipped in Phase 1 (G + A + B):

- **history.csv 11 → 13 columns**: appended `bench_value` (SPY curve scaled to
  $1k at inception, as-of run date) and `data_status` (`ok`/`degraded`) at the
  END. Header migrates in place via the existing equality check; legacy short
  rows untouched; all positional readers (§2) unaffected.
- **portfolio.json new keys** (additive): `realized` (cum/ann return, vol,
  Sharpe, Sortino, Calmar, drawdowns, underwater series, trailing window),
  `benchmark` (source, backfilled curve since inception, active return, beta,
  alpha, TE, IR, capture), `data_quality` (status, issues, missing_prices,
  last_bar).
- **agent/analytics.py**: new pure module (no network, no RNG, no writes) —
  all of the above metrics. New code reads history.csv by column NAME.
- **daily.py**: BL price window widened to `max(90, days since inception + 10)`
  so the single existing yfinance fetch always covers inception (benchmark
  backfill with zero new network calls; ^GSPC fallback fetch only if SPY is
  absent). Three new fail-soft blocks follow the compute_quant_block pattern.
  Journal gains Data quality / Realized performance / Benchmark sections; the
  MC section is now explicitly labelled SIMULATED.
- **Annualization guard**: annualized return and Calmar are None until 20 runs
  (annualizing a days-old curve produces absurd numbers); beta/alpha/TE/IR
  need 3 paired returns. n_obs is reported everywhere.
- **agent/tests/**: known-value, edge-case, header-migration, and fully mocked
  end-to-end tests; tests.yml now runs both quant/tests/ and agent/tests/.

## 7. Phase 2 addendum (implemented 2026-06-05)

Attribution + cash drag (gap C):

- **portfolio.json new key** (additive): `attribution` — per-position and
  per-sector contribution-to-return (1d + cumulative) plus a `cash_drag`
  block (sleeve vs book vs benchmark, 1d + cumulative). No new history.csv
  columns; everything derives from existing data.
- Cumulative contribution = position P&L / starting capital, exact while no
  trades have executed; it sums to total_pnl_pct (cash contributes zero) —
  asserted in the integration test. Daily contribution uses prior closes
  from the existing history frame + the prior recorded row; a sector's 1d
  figure is None if any member name is missing (no misleading partial sums).
- The sector map moved to `risk.SECTOR_MAP` (module constant) so risk_report
  and attribution share one source.
- Journal gains an Attribution section (positions, sectors, cash drag).
  Zero new network calls; fail-soft block like the others.

## 8. Phase 3 addendum (implemented 2026-06-05)

Risk decomposition + thesis stress tests (gaps D, E):

- **portfolio.json new keys** (additive): `risk_decomposition` — parametric
  Gaussian 1d 95% VaR on the invested sleeve with Euler component VaR per
  position (sums exactly to total), marginal VaR per +1pp weight, risk
  contribution %, diversification ratio, HHI, effective names; `stress_tests`
  — six deterministic scenarios tied to the planner priors (AI capex
  drawdown, rate shock, semis export controls, BTC crash, oil spike, AGI
  melt-up) with book vs assumed-SPY impact and active spread.
- Same sleeve weighting as mc_risk_report (MV / gross over names present in
  the history frame); names missing from the frame are listed in `excluded`.
- Both closed-form/deterministic (no RNG), zero new network calls, no new
  history.csv columns. Journal gains Risk decomposition + Stress tests
  sections, placed next to the MC block with the parametric/simulated
  distinction stated.

## 9. Test/CI baseline

`tests.yml` runs `pytest quant/tests/` only (12 tests, no network, no agent/
coverage). There are currently **zero tests for agent/** (portfolio I/O,
history schema, learning loop, journal). New analytics tests should establish
an `agent/`-level test dir and get added to CI's pytest path.
