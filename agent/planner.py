"""Static thesis priors mirroring Situational Awareness LP Q1 2026 13F + essays.

The static TARGET_WEIGHTS dict is the prior (used to seed the portfolio on
day 1 and as the equilibrium anchor for Black-Litterman). The THESIS_VIEWS
list is what the quant layer consumes to derive a posterior weight vector.

Edit TARGET_WEIGHTS or THESIS_VIEWS to rebalance. The daily agent reads
both on every run.
"""
from __future__ import annotations

import pandas as pd

from quant import bl

# ticker -> (target_weight_pct, conviction, one_line_thesis)
TARGET_WEIGHTS = {
    "BE":   (12, "HIGH", "Bloom Energy. Fund's largest equity long ($879M)."),
    "CEG":  (10, "HIGH", "Constellation. Nuclear baseload for AI datacenters."),
    "MSFT": (10, "HIGH", "OpenAI exposure plus Azure hyperscaler."),
    "VST":  (8,  "HIGH", "Vistra. Gas plus nuclear, AI datacenter PPAs."),
    "GEV":  (8,  "HIGH", "GE Vernova. Turbines for AI datacenter buildout."),
    "GOOG": (8,  "MED",  "DeepMind plus internal TPU compute."),
    "NVDA": (5,  "MED",  "Compute core, but fund hedged with $1.57B puts."),
    "META": (5,  "MED",  "Llama stack, ad business funds capex."),
    "CLSK": (4,  "MED",  "CleanSpark. HPC-pivot miner (fund ramped 7x in Q1)."),
    "RIOT": (4,  "MED",  "Riot. HPC-pivot miner (fund 2x'd in Q1)."),
    "TSM":  (4,  "MED",  "Foundry chokepoint."),
    "AVGO": (3,  "MED",  "Custom AI silicon for hyperscalers."),
    "BITF": (3,  "MED",  "Bitfarms. HPC-pivot miner (fund 3x'd in Q1)."),
}

CASH_TARGET_PCT = 16

# Conviction multiplier applied to position sizing during rebalance.
CONVICTION_MULT = {"HIGH": 1.0, "MED": 1.0, "LOW": 0.7}

STARTING_CAPITAL_USD = 1000.0

# Benchmarks added to the BL universe so views can reference them. Their
# weights from BL are discarded — we only deploy into the 13 portfolio names.
BENCHMARKS = ["SPY", "XLU", "SMH"]

# Annualized expected outperformance for each view, expressed as a linear
# combination of tickers. Confidence maps to Ω diagonal via CONFIDENCE_OMEGA.
THESIS_VIEWS: list[bl.View] = [
    # "AI infrastructure outperforms SPY by 15% annualized" — HIGH
    bl.View(pick={"NVDA": 1.0, "MSFT": 0.5, "SPY": -1.0}, q=0.15, confidence="HIGH"),
    # "Nuclear/power outperforms utilities index by 20% ann" — MED
    bl.View(pick={"CEG": 0.5, "VST": 0.5, "XLU": -1.0}, q=0.20, confidence="MED"),
    # "Semi longs hedged — small long, let puts carry" — LOW
    bl.View(pick={"NVDA": 1.0, "SMH": -1.0}, q=0.05, confidence="LOW"),
    # "Hyperscalers compound — MSFT + GOOG outperform SPY by 10% ann" — MED
    bl.View(pick={"MSFT": 0.5, "GOOG": 0.5, "SPY": -1.0}, q=0.10, confidence="MED"),
]


def portfolio_tickers() -> list[str]:
    return list(TARGET_WEIGHTS.keys())


def universe_tickers() -> list[str]:
    return portfolio_tickers() + BENCHMARKS


def prior_weights() -> pd.Series:
    """Static TARGET_WEIGHTS expressed as a normalized Series over the universe.

    Benchmarks get zero prior weight — they exist only to anchor views.
    """
    pw = pd.Series({t: w[0] for t, w in TARGET_WEIGHTS.items()}, dtype=float)
    pw = pw / pw.sum()
    full = pd.Series(0.0, index=universe_tickers())
    full.update(pw)
    return full


def compute_bl_weights(
    prices_history: pd.DataFrame,
    biases: dict[str, float] | None = None,
) -> dict:
    """Run the BL pipeline and return weights + diagnostics.

    Returns:
        {
          "weights":      dict[ticker -> float],  # over portfolio tickers, sums to 1
          "mu_bl":        dict[ticker -> float],  # posterior μ, annualized, all universe
          "pi":           dict[ticker -> float],  # equilibrium prior μ
          "tickers_used": list[str],
          "n_views":      int,
        }
    """
    from quant import stats as qstats

    biases = biases or {}
    universe = [t for t in universe_tickers() if t in prices_history.columns]
    prices_u = prices_history[universe]
    cov = qstats.rolling_cov(prices_u, window=60)
    prior = prior_weights().reindex(universe).fillna(0.0)
    pi = bl.equilibrium_returns(prior, cov)
    # only keep views whose tickers are all in the universe
    active_views = [v for v in THESIS_VIEWS if all(t in universe for t in v.pick)]
    mu = bl.posterior_returns(pi, cov, active_views)

    # solve weights over the portfolio sub-universe only
    port_tickers = [t for t in portfolio_tickers() if t in universe]
    sub_cov = cov.loc[port_tickers, port_tickers]
    sub_mu = mu.loc[port_tickers]
    w = bl.optimal_weights(sub_mu, sub_cov, weight_cap=0.20)
    w_biased = bl.apply_bias(w, biases)

    return {
        "weights": {t: float(v) for t, v in w_biased.items()},
        "mu_bl": {t: float(v) for t, v in mu.items()},
        "pi": {t: float(v) for t, v in pi.items()},
        "tickers_used": port_tickers,
        "n_views": len(active_views),
    }
