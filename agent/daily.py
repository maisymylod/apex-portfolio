"""Daily entrypoint. On first run, seeds the portfolio from planner targets.
On subsequent runs, marks to market, updates the learning loop, writes a
journal entry, appends to history, and refreshes the README P&L table.

No real broker. No real money. Paper only.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from . import planner, portfolio, prices, risk

ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DIR = ROOT / "journal"
README = ROOT / "README.md"


def seed_portfolio() -> dict:
    """First-run seed: buy target weights at today's close."""
    tickers = list(planner.TARGET_WEIGHTS.keys())
    px = prices.fetch_close_prices(tickers)
    starting = planner.STARTING_CAPITAL_USD
    positions = {}
    spent = 0.0
    for t, (weight_pct, conv, thesis) in planner.TARGET_WEIGHTS.items():
        if t not in px:
            print(f"[seed] missing price for {t}, skipping")
            continue
        budget = starting * weight_pct / 100
        shares = round(budget / px[t], 4)
        positions[t] = {
            "shares": shares,
            "cost_basis": px[t],
            "thesis": thesis,
            "conviction": conv,
            "bought_at": datetime.now(timezone.utc).isoformat(),
        }
        spent += shares * px[t]
    cash = starting - spent
    state = {
        "starting_capital": starting,
        "cash": cash,
        "positions": positions,
        "inception_date": datetime.now(timezone.utc).date().isoformat(),
        "last_run": None,
    }
    portfolio.save(state)
    return state


def write_journal(snapshot: dict, risk_rep: dict, learning: dict) -> Path:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    date = snapshot["as_of"][:10]
    path = JOURNAL_DIR / f"{date}.md"
    lines = [
        f"# Journal: {date}",
        "",
        f"**Total value:** ${snapshot['total_value']:.2f}  ",
        f"**P&L since inception:** ${snapshot['total_pnl_usd']:+.2f} ({snapshot['total_pnl_pct']:+.2f}%)  ",
        f"**Cash:** ${snapshot['cash']:.2f}  ",
        f"**Risk status:** {risk_rep['status']}",
        "",
    ]
    if risk_rep["flags"]:
        lines.append("## Risk flags")
        for f in risk_rep["flags"]:
            lines.append(f"- {f}")
        lines.append("")
    lines.append("## Positions")
    lines.append("")
    lines.append("| Ticker | Shares | Cost | Price | MV | P&L | Weight | Conviction |")
    lines.append("|--------|--------|------|-------|-----|-----|--------|------------|")
    for p in snapshot["positions"]:
        lines.append(
            f"| {p['ticker']} | {p['shares']:.4f} | ${p['cost_basis']:.2f} | ${p['price']:.2f} | "
            f"${p['market_value']:.2f} | ${p['pnl_usd']:+.2f} ({p['pnl_pct']:+.2f}%) | "
            f"{p['weight_pct']:.1f}% | {p['conviction']} |"
        )
    lines.append("")
    lines.append("## Sector exposure")
    for s, w in sorted(risk_rep["sector_weights"].items(), key=lambda x: -x[1]):
        lines.append(f"- {s}: {w:.1f}%")
    lines.append("")
    lines.append("## Learning loop (conviction biases)")
    biases = learning.get("biases", {})
    if biases:
        for t, b in sorted(biases.items(), key=lambda x: -x[1]):
            lines.append(f"- {t}: {b:+.3f}")
    else:
        lines.append("- (no biases yet, first run)")
    path.write_text("\n".join(lines) + "\n")
    return path


def update_readme(snapshot: dict, history_count: int) -> None:
    if not README.exists():
        return
    text = README.read_text()
    marker_start = "<!-- LIVE_PNL_START -->"
    marker_end = "<!-- LIVE_PNL_END -->"
    if marker_start not in text or marker_end not in text:
        return
    pnl = snapshot["total_pnl_usd"]
    pnl_pct = snapshot["total_pnl_pct"]
    arrow = "UP" if pnl >= 0 else "DOWN"
    block_lines = [
        marker_start,
        f"**As of {snapshot['as_of'][:10]} (UTC)** — Day {history_count}",
        "",
        f"| Total value | P&L | P&L % | Cash | Holdings |",
        f"|-------------|-----|-------|------|----------|",
        f"| ${snapshot['total_value']:.2f} | ${pnl:+.2f} ({arrow}) | {pnl_pct:+.2f}% | ${snapshot['cash']:.2f} | ${snapshot['holdings_value']:.2f} |",
        "",
        "### Top holdings",
        "",
        "| Ticker | Weight | P&L % |",
        "|--------|--------|-------|",
    ]
    for p in snapshot["positions"][:6]:
        block_lines.append(f"| {p['ticker']} | {p['weight_pct']:.1f}% | {p['pnl_pct']:+.2f}% |")
    block_lines.append("")
    block_lines.append(marker_end)
    block = "\n".join(block_lines)
    pre = text.split(marker_start)[0]
    post = text.split(marker_end)[1]
    README.write_text(pre + block + post)


def main() -> int:
    state = portfolio.load()
    if not state.get("starting_capital"):
        print("[daily] seeding new portfolio")
        state = seed_portfolio()

    tickers = list(state["positions"].keys())
    px = prices.fetch_close_prices(tickers)
    if not px:
        print("[daily] no prices fetched, aborting")
        return 1

    snapshot = portfolio.mark_to_market(state, px)
    risk_rep = risk.risk_report(snapshot)

    # learning loop needs the prior total to compute day delta
    history = portfolio.HISTORY_PATH
    prior_total = None
    if history.exists():
        rows = history.read_text().strip().splitlines()
        if len(rows) > 1:
            prior_total = float(rows[-1].split(",")[1])

    learning = risk.update_biases(snapshot, prior_total)

    portfolio.append_history(snapshot)
    portfolio.save(state)

    write_journal(snapshot, risk_rep, learning)

    history_rows = history.read_text().strip().splitlines() if history.exists() else []
    update_readme(snapshot, max(0, len(history_rows) - 1))

    print(f"[daily] total ${snapshot['total_value']:.2f}, "
          f"P&L ${snapshot['total_pnl_usd']:+.2f} ({snapshot['total_pnl_pct']:+.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
