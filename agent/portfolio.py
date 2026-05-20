"""Portfolio state I/O and metrics."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
PORTFOLIO_PATH = STATE_DIR / "portfolio.json"
HISTORY_PATH = STATE_DIR / "history.csv"


@dataclass
class Position:
    ticker: str
    shares: float
    cost_basis: float  # per-share avg
    thesis: str
    conviction: str


def load() -> dict:
    if not PORTFOLIO_PATH.exists():
        return {
            "starting_capital": None,
            "cash": None,
            "positions": {},
            "inception_date": None,
            "last_run": None,
        }
    with PORTFOLIO_PATH.open() as f:
        return json.load(f)


def save(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    with PORTFOLIO_PATH.open("w") as f:
        json.dump(state, f, indent=2)


def mark_to_market(state: dict, prices: dict[str, float]) -> dict:
    """Return a snapshot dict with current values, P&L, etc."""
    positions = []
    holdings_value = 0.0
    for ticker, pos in state["positions"].items():
        px = prices.get(ticker)
        if px is None:
            continue
        mv = pos["shares"] * px
        cost = pos["shares"] * pos["cost_basis"]
        pnl = mv - cost
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0
        positions.append({
            "ticker": ticker,
            "shares": pos["shares"],
            "cost_basis": pos["cost_basis"],
            "price": px,
            "market_value": mv,
            "pnl_usd": pnl,
            "pnl_pct": pnl_pct,
            "weight_pct": 0.0,  # filled below
            "conviction": pos.get("conviction", ""),
            "thesis": pos.get("thesis", ""),
        })
        holdings_value += mv
    total_value = holdings_value + state["cash"]
    for p in positions:
        p["weight_pct"] = (p["market_value"] / total_value * 100) if total_value else 0.0
    starting = state["starting_capital"]
    total_pnl = total_value - starting if starting else 0.0
    total_pnl_pct = (total_pnl / starting * 100) if starting else 0.0
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "positions": sorted(positions, key=lambda x: -x["market_value"]),
        "cash": state["cash"],
        "holdings_value": holdings_value,
        "total_value": total_value,
        "starting_capital": starting,
        "total_pnl_usd": total_pnl,
        "total_pnl_pct": total_pnl_pct,
    }


def append_history(snapshot: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date_utc", "total_value", "cash", "holdings_value", "pnl_usd", "pnl_pct"])
        w.writerow([
            snapshot["as_of"][:10],
            f"{snapshot['total_value']:.2f}",
            f"{snapshot['cash']:.2f}",
            f"{snapshot['holdings_value']:.2f}",
            f"{snapshot['total_pnl_usd']:.2f}",
            f"{snapshot['total_pnl_pct']:.2f}",
        ])
