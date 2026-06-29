# apex-portfolio

A paper portfolio that mirrors the known thesis and 13F positions of
Situational Awareness Capital (Leopold Aschenbrenner). Starting capital:
**$1,000**. No real broker. No real money.

A GitHub Actions cron runs `python -m agent.daily` each weekday at 4:30pm
ET. Each run pulls fresh prices via yfinance, marks the portfolio to
market, computes realized and SPY-relative analytics plus a data-quality
check, updates a self-supervised conviction-bias file, writes a journal
entry to `journal/`, and refreshes the live P&L table below.

## Live P&L

<!-- LIVE_PNL_START -->
**As of 2026-06-29 (UTC)** — Day 32

| Total value | P&L | P&L % | Cash | Holdings |
|-------------|-----|-------|------|----------|
| $951.22 | $-48.78 (DOWN) | -4.88% | $159.98 | $791.24 |

### Top holdings

| Ticker | Weight | P&L % |
|--------|--------|-------|
| BE | 12.1% | -4.14% |
| CEG | 9.7% | -7.43% |
| VST | 9.5% | +12.98% |
| MSFT | 9.2% | -12.34% |
| GEV | 9.0% | +6.76% |
| GOOG | 7.7% | -8.46% |

<!-- LIVE_PNL_END -->

## How "self-training" works

`agent/risk.py` keeps a per-ticker bias in `state/learning.json` that
nudges up when a position outperforms and decays back toward zero
otherwise. It is intentionally small (capped at +/- 25%) so it can
influence future rebalances without overruling the static thesis priors
in `agent/planner.py`.

## Layout

```
agent/
  planner.py     static thesis priors (target weights, conviction, thesis)
  prices.py     yfinance wrapper
  portfolio.py   state I/O, mark-to-market
  risk.py        sector/concentration flags + learning loop
  analytics.py   realized + benchmark analytics, data-quality guard (no network)
  daily.py       entrypoint for the cron
  tests/         no-network tests for the agent layer
state/
  portfolio.json current positions, cash, cost basis, realized/benchmark blocks
  history.csv    daily snapshot (equity curve, MC risk, benchmark value, data status)
  learning.json  rolling conviction biases
journal/
  YYYY-MM-DD.md  one file per run
prompts/
  system.md      APEX master prompt (reference, not executed)
.github/workflows/daily.yml  cron action
```

## Running locally

```bash
pip install -r requirements.txt
python -m agent.daily
```

## Not investment advice

This is a personal experiment. Positions are inferred from a stale 13F
(filed 2026-05-18, as of 2026-03-31) and from public essays. The fund's
options direction (long puts vs short puts) cannot be determined from
the 13F itself, so the put overlay on semis is not mirrored here.
