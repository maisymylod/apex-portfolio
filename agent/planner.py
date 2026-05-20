"""Static thesis priors mirroring Situational Awareness LP Q1 2026 13F + essays.

Edit TARGET_WEIGHTS to rebalance. The daily agent reads this on every run.
"""

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
