# apex-portfolio

A paper portfolio that mirrors the known thesis and 13F positions of
Situational Awareness Capital (Leopold Aschenbrenner). Starting capital:
**$1,000**. No real broker. No real money.

A GitHub Actions cron runs `python -m agent.daily` each weekday at 4:30pm
ET. Each run pulls fresh prices via yfinance, marks the portfolio to
market, updates a self-supervised conviction-bias file, writes a journal
entry to `journal/`, and refreshes the live P&L table below.

## Live P&L

<!-- LIVE_PNL_START -->
**As of 2026-06-01 (UTC)** — Day 12

| Total value | P&L | P&L % | Cash | Holdings |
|-------------|-----|-------|------|----------|
| $1027.16 | $+27.16 (UP) | +2.72% | $159.98 | $867.18 |

### Top holdings

| Ticker | Weight | P&L % |
|--------|--------|-------|
| BE | 11.1% | -4.66% |
| MSFT | 10.7% | +9.52% |
| CEG | 9.2% | -5.15% |
| VST | 8.4% | +7.67% |
| GOOG | 7.6% | -2.91% |
| GEV | 7.2% | -7.95% |

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
  daily.py       entrypoint for the cron
state/
  portfolio.json current positions, cash, cost basis
  history.csv    daily total-value snapshot
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
