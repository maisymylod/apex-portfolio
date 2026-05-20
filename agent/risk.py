"""Risk computation and conviction adjustment.

The 'self-training' loop lives here: each day we update a rolling
conviction multiplier per ticker based on its 5-day rolling return
vs SPY (relative outperformance ⇒ +bias, underperformance ⇒ -bias).
The bias is capped and decays so a single bad day doesn't blow up
the prior.

A Monte-Carlo risk pass (quant.mc) is layered on top of the static
concentration checks. It can issue a VETO when 1-day VaR, max-drawdown
distribution, or probability of ruin breach thresholds.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from quant import mc, stats as qstats

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
LEARN_PATH = STATE_DIR / "learning.json"

BIAS_CAP = 0.25  # +/- 25% max conviction adjustment
DECAY = 0.85     # daily decay toward 0

# MC veto thresholds (sized for a $1k book per the master prompt)
VAR_LIMIT_USD = 80.0          # 1-day 95% VaR cap = 8% of $1k
CVAR_LIMIT_USD = 120.0        # 1-day 95% CVaR cap
MEDIAN_MAXDD_LIMIT = -0.25    # median simulated max drawdown floor (negative)
P_RUIN_LIMIT = 0.15           # 15% probability of breaching ruin threshold
RUIN_THRESHOLD_FRAC = 0.70    # ruin = drop below 70% of current value

MC_N_PATHS = 10_000
MC_HORIZON = 252


def load_learning() -> dict:
    if not LEARN_PATH.exists():
        return {"biases": {}, "calibration": []}
    with LEARN_PATH.open() as f:
        return json.load(f)


def save_learning(d: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LEARN_PATH.open("w") as f:
        json.dump(d, f, indent=2)


def update_biases(snapshot: dict, prior_total: float | None) -> dict:
    """Adjust per-ticker conviction bias based on daily P&L direction."""
    d = load_learning()
    biases = d.get("biases", {})
    for p in snapshot["positions"]:
        t = p["ticker"]
        prior = biases.get(t, 0.0) * DECAY
        # Reward positions with positive day P&L, penalize negatives.
        # Use pnl_pct day-over-day if we had it; for v1 use total pnl_pct
        # scaled small. Replace with daily-delta once we have ≥2 history rows.
        nudge = max(-0.05, min(0.05, p["pnl_pct"] / 100 * 0.1))
        biases[t] = max(-BIAS_CAP, min(BIAS_CAP, prior + nudge))
    d["biases"] = biases
    # Track portfolio-level calibration: did total go up or down?
    if prior_total is not None:
        d.setdefault("calibration", []).append({
            "as_of": snapshot["as_of"][:10],
            "delta_pct": (snapshot["total_value"] - prior_total) / prior_total * 100,
        })
        d["calibration"] = d["calibration"][-30:]  # keep 30 days
    save_learning(d)
    return d


def risk_report(snapshot: dict) -> dict:
    positions = snapshot["positions"]
    largest = max(positions, key=lambda p: p["weight_pct"]) if positions else None
    # crude sector buckets matching the planner thesis
    sector_map = {
        "BE": "power", "CEG": "power", "VST": "power", "GEV": "power",
        "MSFT": "hyperscaler", "GOOG": "hyperscaler", "META": "hyperscaler",
        "NVDA": "semi", "TSM": "semi", "AVGO": "semi",
        "CLSK": "miners", "RIOT": "miners", "BITF": "miners",
    }
    sector_weight: dict[str, float] = {}
    for p in positions:
        s = sector_map.get(p["ticker"], "other")
        sector_weight[s] = sector_weight.get(s, 0.0) + p["weight_pct"]
    flags = []
    if largest and largest["weight_pct"] > 20:
        flags.append(f"concentration: {largest['ticker']} {largest['weight_pct']:.1f}% > 20%")
    for s, w in sector_weight.items():
        if w > 50:
            flags.append(f"sector concentration: {s} {w:.1f}% > 50%")
    return {
        "largest_position": largest,
        "sector_weights": sector_weight,
        "flags": flags,
        "status": "RED" if flags else "GREEN",
    }


def mc_risk_report(
    snapshot: dict,
    history: pd.DataFrame,
    mu_bl: dict[str, float] | None = None,
    seed: int | None = 20260520,
) -> dict | None:
    """Run a MC simulation on the current portfolio.

    Drift μ comes from BL posterior if supplied, otherwise from rolling
    historical means. Σ is the 60-day rolling annualized covariance.
    Returns a dict of metrics + flag list + veto bool, or None on failure.
    """
    try:
        positions = snapshot["positions"]
        if not positions or history is None or history.empty:
            return None
        tickers = [p["ticker"] for p in positions if p["ticker"] in history.columns]
        if len(tickers) < 2:
            return None
        prices_p = history[tickers]
        cov = qstats.rolling_cov(prices_p, window=60)
        if mu_bl:
            mu = pd.Series({t: mu_bl.get(t, 0.0) for t in tickers})
        else:
            mu = qstats.historical_drift(prices_p, window=60)
        total_value = snapshot["total_value"]
        mv_by = {p["ticker"]: p["market_value"] for p in positions}
        gross_holdings = sum(mv_by.get(t, 0.0) for t in tickers)
        if gross_holdings <= 0:
            return None
        weights = pd.Series({t: mv_by.get(t, 0.0) / gross_holdings for t in tickers})
        metrics = mc.evaluate(
            weights, mu, cov, starting_value=gross_holdings,
            n_paths=MC_N_PATHS, horizon_days=MC_HORIZON, seed=seed,
            ruin_threshold=RUIN_THRESHOLD_FRAC * total_value,
        )
        flags = []
        if metrics["var_1d_usd"] > VAR_LIMIT_USD:
            flags.append(f"MC VaR 95%: ${metrics['var_1d_usd']:.2f} > ${VAR_LIMIT_USD:.0f}")
        if metrics["cvar_1d_usd"] > CVAR_LIMIT_USD:
            flags.append(f"MC CVaR 95%: ${metrics['cvar_1d_usd']:.2f} > ${CVAR_LIMIT_USD:.0f}")
        if metrics["median_max_drawdown"] < MEDIAN_MAXDD_LIMIT:
            flags.append(
                f"Median MaxDD: {metrics['median_max_drawdown']*100:.1f}% "
                f"< {MEDIAN_MAXDD_LIMIT*100:.0f}%"
            )
        if metrics["p_ruin"] > P_RUIN_LIMIT:
            flags.append(
                f"P(ruin < ${metrics['ruin_threshold_usd']:.0f}): "
                f"{metrics['p_ruin']*100:.1f}% > {P_RUIN_LIMIT*100:.0f}%"
            )
        return {
            **metrics,
            "veto": bool(flags),
            "veto_flags": flags,
            "drift_source": "bl" if mu_bl else "historical",
        }
    except Exception:
        traceback.print_exc()
        print("[risk] MC pass failed, continuing without MC report")
        return None
