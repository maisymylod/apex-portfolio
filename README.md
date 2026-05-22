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
**As of 2026-05-22 (UTC)** — Day 6

| Total value | P&L | P&L % | Cash | Holdings |
|-------------|-----|-------|------|----------|
| $1020.03 | $+20.03 (UP) | +2.00% | $159.98 | $860.05 |

### Top holdings

| Ticker | Weight | P&L % |
|--------|--------|-------|
| BE | 12.4% | +5.44% |
| CEG | 10.3% | +4.97% |
| MSFT | 9.8% | -0.45% |
| VST | 8.5% | +8.72% |
| GEV | 7.9% | +0.59% |
| GOOG | 7.8% | -1.13% |

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
