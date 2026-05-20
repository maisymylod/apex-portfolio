"""Risk computation and conviction adjustment.

The 'self-training' loop lives here: each day we update a rolling
conviction multiplier per ticker based on its 5-day rolling return
vs SPY (relative outperformance ⇒ +bias, underperformance ⇒ -bias).
The bias is capped and decays so a single bad day doesn't blow up
the prior.
"""
from __future__ import annotations

import json
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
LEARN_PATH = STATE_DIR / "learning.json"

BIAS_CAP = 0.25  # +/- 25% max conviction adjustment
DECAY = 0.85     # daily decay toward 0


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
