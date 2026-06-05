"""Daily entrypoint. On first run, seeds the portfolio from planner targets.
On subsequent runs, marks to market, updates the learning loop, writes a
journal entry, appends to history, and refreshes the README P&L table.

No real broker. No real money. Paper only.
"""
from __future__ import annotations

import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

from . import analytics, planner, portfolio, prices, risk
from quant import bsm

BL_HISTORY_DAYS = 90
REBALANCE_THRESHOLD_PCT = 3.0

# BSM put overlay parameters
PUT_TICKER = "NVDA"
PUT_NVDA_WEIGHT_TRIGGER = 0.15
PUT_REALIZED_VOL_TRIGGER = 0.35
PUT_P_RUIN_TRIGGER = 0.08
PUT_MIN_CASH_USD = 50.0
PUT_STRIKE_FRAC = 0.90
PUT_TENOR_DAYS = 60
PUT_MAX_SPEND_USD = 30.0
RISK_FREE_RATE = 0.045

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


def compute_quant_block(snapshot: dict, learning: dict, days: int = BL_HISTORY_DAYS) -> dict | None:
    """Pull price history, run BL, return a quant snapshot dict.

    Returns None and logs the traceback on any failure — the daily cron
    must stay green even if yfinance, scipy, or BL math hiccup.
    """
    try:
        biases = learning.get("biases", {}) if learning else {}
        history = prices.fetch_history(planner.universe_tickers(), days=days)
        if history.empty or history.shape[0] < 20:
            print("[quant] insufficient price history, skipping BL")
            return None
        bl_out = planner.compute_bl_weights(history, biases=biases)
        current = {p["ticker"]: p["weight_pct"] / 100.0 for p in snapshot["positions"]}
        rebalance = []
        for t, target in bl_out["weights"].items():
            cur = current.get(t, 0.0)
            delta = target - cur
            if abs(delta) * 100 >= REBALANCE_THRESHOLD_PCT:
                rebalance.append({
                    "ticker": t,
                    "current_pct": cur * 100,
                    "target_pct": target * 100,
                    "delta_pct": delta * 100,
                })
        rebalance.sort(key=lambda x: -abs(x["delta_pct"]))
        return {
            "bl_weights": bl_out["weights"],
            "mu_bl": bl_out["mu_bl"],
            "n_views": bl_out["n_views"],
            "rebalance": rebalance,
            "bl_run_ts": datetime.now(timezone.utc).isoformat(),
            "history": history,  # kept for downstream MC/BSM, not serialized
        }
    except Exception:
        traceback.print_exc()
        print("[quant] BL pipeline failed, continuing without quant block")
        return None


def compute_options_overlay(snapshot: dict, quant: dict | None, mc_report: dict | None) -> dict | None:
    """Paper-only BSM put overlay on NVDA. Returns None if data is missing.

    Always returns the BSM values and the trigger evaluation. The journal
    will show STATUS=ACTIVE when all four trigger conditions are met,
    MONITORING otherwise.
    """
    if not quant or quant.get("history") is None:
        return None
    history = quant["history"]
    if PUT_TICKER not in history.columns:
        return None
    try:
        nvda_prices = history[PUT_TICKER].dropna()
        if len(nvda_prices) < 5:
            return None
        spot = float(nvda_prices.iloc[-1])
        strike = round(spot * PUT_STRIKE_FRAC, 2)
        tail = nvda_prices.tail(30)
        import numpy as np
        daily_log_r = np.log(tail / tail.shift(1)).dropna().tolist()
        sigma = bsm.realized_vol_annual(daily_log_r)
        if sigma <= 0:
            return None
        T = PUT_TENOR_DAYS / 252.0
        put_v = bsm.put_price(spot, strike, RISK_FREE_RATE, sigma, T)
        g = bsm.greeks(spot, strike, RISK_FREE_RATE, sigma, T, option="put")

        nvda_position = next((p for p in snapshot["positions"] if p["ticker"] == PUT_TICKER), None)
        nvda_weight = (nvda_position["weight_pct"] / 100.0) if nvda_position else 0.0
        p_ruin = mc_report["p_ruin"] if mc_report else 0.0
        cash = snapshot["cash"]
        triggers = {
            "nvda_weight_gt_15pct": nvda_weight > PUT_NVDA_WEIGHT_TRIGGER,
            "realized_vol_gt_35pct": sigma > PUT_REALIZED_VOL_TRIGGER,
            "p_ruin_gt_8pct": p_ruin > PUT_P_RUIN_TRIGGER,
            "cash_gt_50": cash > PUT_MIN_CASH_USD,
        }
        active = all(triggers.values())
        nvda_shares = nvda_position["shares"] if nvda_position else 0.0
        notional_hedge = put_v * nvda_shares
        return {
            "ticker": PUT_TICKER,
            "spot": spot,
            "strike": strike,
            "sigma": sigma,
            "tenor_days": PUT_TENOR_DAYS,
            "put_value_per_share": put_v,
            "delta": g.delta,
            "gamma": g.gamma,
            "theta_per_day": g.theta,
            "vega": g.vega,
            "notional_hedge_usd": notional_hedge,
            "max_spend_usd": PUT_MAX_SPEND_USD,
            "triggers": triggers,
            "status": "ACTIVE" if active else "MONITORING",
            "nvda_weight_pct": nvda_weight * 100,
        }
    except Exception:
        traceback.print_exc()
        print("[options] BSM overlay failed, continuing")
        return None


def _history_window_days(state: dict) -> int:
    """Price-history window: wide enough to always cover inception so the
    benchmark curve backfills from day 1 off the same single yfinance call."""
    try:
        inception = state.get("inception_date")
        if not inception:
            return BL_HISTORY_DAYS
        age = (datetime.now(timezone.utc).date() - date.fromisoformat(inception)).days
        return max(BL_HISTORY_DAYS, age + 10)
    except Exception:
        return BL_HISTORY_DAYS


def _port_curve_with_today(snapshot: dict) -> list[tuple[str, float]]:
    """Recorded equity curve plus today's (not yet appended) snapshot point.

    A same-date rerun replaces the recorded point rather than duplicating it.
    """
    today = (snapshot["as_of"][:10], snapshot["total_value"])
    curve = [c for c in analytics.load_equity_curve(portfolio.HISTORY_PATH) if c[0] != today[0]]
    return curve + [today]


def compute_realized_block(state: dict, snapshot: dict) -> dict | None:
    """REALIZED performance from the recorded equity curve (not simulated)."""
    try:
        return analytics.realized_metrics(
            _port_curve_with_today(snapshot),
            state["starting_capital"],
            rf_annual=RISK_FREE_RATE,
        )
    except Exception:
        traceback.print_exc()
        print("[realized] realized-metrics block failed, continuing")
        return None


def compute_benchmark_block(state: dict, snapshot: dict, history, days: int) -> dict | None:
    """Benchmark curve (backfilled from inception) + benchmark-relative stats.

    SPY already rides along in the BL history fetch; ^GSPC is fetched only
    if SPY is missing, so the happy path adds zero network calls.
    """
    try:
        inception = state.get("inception_date")
        if not inception:
            return None
        series = None
        source = None
        if history is not None and not history.empty and "SPY" in history.columns:
            s = history["SPY"].dropna()
            if not s.empty:
                series, source = s, "SPY (adjusted close)"
        if series is None:
            fb = prices.fetch_history(["^GSPC"], days=days)
            if not fb.empty and "^GSPC" in fb.columns:
                series, source = fb["^GSPC"].dropna(), "^GSPC (price)"
        if series is None or series.empty:
            print("[benchmark] no benchmark series available, skipping")
            return None
        curve = analytics.benchmark_curve(series, inception, state["starting_capital"])
        if not curve:
            print("[benchmark] benchmark curve empty after inception filter, skipping")
            return None
        metrics = analytics.benchmark_metrics(
            _port_curve_with_today(snapshot), curve,
            state["starting_capital"], rf_annual=RISK_FREE_RATE,
        )
        return {"source": source, "asof": snapshot["as_of"], "curve": curve, **metrics}
    except Exception:
        traceback.print_exc()
        print("[benchmark] benchmark block failed, continuing")
        return None


def compute_attribution_block(state: dict, snapshot: dict, history, benchmark: dict | None) -> dict | None:
    """Per-position/sector contribution-to-return + explicit cash drag.

    Daily figures use prior closes from the already-fetched history frame
    and the prior recorded history row; cumulative figures need only cost
    basis. Fail-soft like every other analytics block.
    """
    try:
        import pandas as pd

        today_s = snapshot["as_of"][:10]
        rows = analytics.load_history_rows(portfolio.HISTORY_PATH)
        prior_rows = [r for r in rows if r.get("date_utc") and r["date_utc"] < today_s]
        prior_row = prior_rows[-1] if prior_rows else None
        prior_total = None
        if prior_row:
            try:
                prior_total = float(prior_row["total_value"])
            except (KeyError, TypeError, ValueError):
                prior_total = None

        prior_closes = None
        if history is not None and not history.empty:
            past = history[history.index < pd.Timestamp(today_s)]
            if not past.empty:
                last = past.iloc[-1]
                prior_closes = {
                    t: float(last[t]) for t in history.columns if pd.notna(last[t])
                }

        result = analytics.attribution(
            snapshot["positions"], state["starting_capital"],
            prior_closes=prior_closes, prior_total=prior_total,
            sector_map=risk.SECTOR_MAP,
        )
        result["cash_drag"] = analytics.cash_drag_metrics(snapshot, prior_row)

        # Names excluded from the mark (no price) hit the book total but can't
        # be attributed — name them and quantify the unattributed residual.
        marked = {p["ticker"] for p in snapshot["positions"]}
        result["excluded_from_mark"] = sorted(t for t in state["positions"] if t not in marked)
        day_vals = [r["day_contribution_pct"] for r in result["positions"]]
        attributed = (
            sum(v for v in day_vals if v is not None)
            if any(v is not None for v in day_vals) else None
        )
        result["day_attributed_pct"] = attributed
        book_day = result["cash_drag"]["book_day_return_pct"]
        result["day_residual_pct"] = (
            book_day - attributed
            if book_day is not None and attributed is not None else None
        )

        if benchmark:
            result["cash_drag"]["bench_cum_return_pct"] = benchmark["bench_cum_return_pct"]
            curve = benchmark.get("curve") or []
            if len(curve) >= 2 and curve[-2]["value"] > 0:
                result["cash_drag"]["bench_day_return_pct"] = (
                    (curve[-1]["value"] / curve[-2]["value"] - 1.0) * 100
                )
        result["asof"] = snapshot["as_of"]
        return result
    except Exception:
        traceback.print_exc()
        print("[attribution] attribution block failed, continuing")
        return None


def compute_data_quality_block(state: dict, px: dict, history, snapshot: dict) -> dict | None:
    """Flag missing/stale price data instead of letting it pass silently."""
    try:
        as_of = date.fromisoformat(snapshot["as_of"][:10])
        return analytics.data_quality_report(
            list(state["positions"].keys()), px, history, as_of,
        )
    except Exception:
        traceback.print_exc()
        print("[data-quality] check failed, continuing")
        return None


def _fmt(v: float | None, pattern: str = "{:+.2f}%", na: str = "n/a") -> str:
    return pattern.format(v) if v is not None else na


def write_journal(
    snapshot: dict,
    risk_rep: dict,
    learning: dict,
    quant: dict | None = None,
    mc_report: dict | None = None,
    options: dict | None = None,
    realized: dict | None = None,
    benchmark: dict | None = None,
    attribution: dict | None = None,
    data_quality: dict | None = None,
) -> Path:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    date = snapshot["as_of"][:10]
    path = JOURNAL_DIR / f"{date}.md"
    lines = [
        f"# Journal: {date}",
        "",
        f"**Total value:** ${snapshot['total_value']:.2f}  ",
        f"**P&L since inception:** ${snapshot['total_pnl_usd']:+.2f} ({snapshot['total_pnl_pct']:+.2f}%)  ",
        f"**Cash:** ${snapshot['cash']:.2f}  ",
        f"**Risk status:** {risk_rep['status']}  ",
        f"**Data quality:** {data_quality['status'] if data_quality else 'unknown'}",
        "",
    ]
    if data_quality and data_quality["issues"]:
        lines.append("## Data quality flags")
        for issue in data_quality["issues"]:
            lines.append(f"- {issue}")
        lines.append("")
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

    if realized:
        lines.append("## Realized performance")
        lines.append(
            f"_From the recorded equity curve ({realized['n_obs']} runs) — REALIZED, not "
            f"simulated. Annualized return and Calmar appear after "
            f"{analytics.MIN_OBS_ANNUALIZE} runs._"
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Cumulative return | {_fmt(realized['cum_return_pct'])} |")
        lines.append(f"| Annualized return | {_fmt(realized['ann_return_pct'])} |")
        lines.append(f"| Realized vol (ann) | {_fmt(realized['vol_ann_pct'], '{:.2f}%')} |")
        lines.append(f"| Realized Sharpe | {_fmt(realized['sharpe'], '{:.2f}')} |")
        lines.append(f"| Sortino | {_fmt(realized['sortino'], '{:.2f}')} |")
        lines.append(f"| Calmar | {_fmt(realized['calmar'], '{:.2f}')} |")
        lines.append(f"| Current drawdown | {_fmt(realized['current_drawdown_pct'])} |")
        lines.append(f"| Max drawdown | {_fmt(realized['max_drawdown_pct'])} |")
        if realized.get("trailing"):
            t = realized["trailing"]
            lines.append(f"| Trailing {t['window_runs']}-run return | {_fmt(t['return_pct'])} |")
        lines.append("")

    if benchmark:
        lines.append(f"## Benchmark comparison — {benchmark['source']}")
        lines.append(
            f"_Backfilled from inception. {benchmark['n_obs']} aligned daily returns; "
            f"beta/alpha/TE/IR/capture need {analytics.MIN_OBS_BETA}+._"
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Portfolio cum return | {_fmt(benchmark['port_cum_return_pct'])} |")
        lines.append(f"| Benchmark cum return | {_fmt(benchmark['bench_cum_return_pct'])} |")
        lines.append(f"| Active return (cum) | {_fmt(benchmark['active_return_cum_pct'])} |")
        lines.append(f"| Active return (1d) | {_fmt(benchmark['active_return_1d_pct'])} |")
        lines.append(f"| Beta | {_fmt(benchmark['beta'], '{:.2f}')} |")
        lines.append(f"| Alpha (CAPM, ann) | {_fmt(benchmark['alpha_ann_pct'])} |")
        lines.append(f"| Tracking error (ann) | {_fmt(benchmark['tracking_error_ann_pct'], '{:.2f}%')} |")
        lines.append(f"| Information ratio | {_fmt(benchmark['info_ratio'], '{:.2f}')} |")
        lines.append(f"| Up capture | {_fmt(benchmark['up_capture'], '{:.2f}')} |")
        lines.append(f"| Down capture | {_fmt(benchmark['down_capture'], '{:.2f}')} |")
        lines.append("")

    if attribution:
        lines.append("## Attribution")
        lines.append(
            "_Contribution to book return (pp = percentage points). Cumulative sums to "
            "total P&L %; cash contributes zero. 1d figures need prior closes + a prior "
            "recorded row._"
        )
        lines.append("")
        lines.append("| Ticker | Sector | 1d P&L | 1d contrib | Cum P&L | Cum contrib |")
        lines.append("|--------|--------|--------|------------|---------|-------------|")
        for r in attribution["positions"]:
            lines.append(
                f"| {r['ticker']} | {r['sector']} | {_fmt(r['day_pnl_usd'], '${:+.2f}')} | "
                f"{_fmt(r['day_contribution_pct'], '{:+.2f}pp')} | "
                f"{_fmt(r['cum_pnl_usd'], '${:+.2f}')} | "
                f"{_fmt(r['cum_contribution_pct'], '{:+.2f}pp')} |"
            )
        lines.append("")
        lines.append("### Sectors")
        lines.append("| Sector | Weight | 1d contrib | Cum contrib |")
        lines.append("|--------|--------|------------|-------------|")
        for s in attribution["sectors"]:
            lines.append(
                f"| {s['sector']} | {s['weight_pct']:.1f}% | "
                f"{_fmt(s['day_contribution_pct'], '{:+.2f}pp')} | "
                f"{_fmt(s['cum_contribution_pct'], '{:+.2f}pp')} |"
            )
        if attribution.get("day_attributed_pct") is not None:
            lines.append("")
            lines.append(
                f"_1d attributed: {attribution['day_attributed_pct']:+.2f}pp of a "
                f"{_fmt(attribution['cash_drag'].get('book_day_return_pct'))} book move; "
                f"residual {_fmt(attribution['day_residual_pct'], '{:+.2f}pp')} "
                f"(excluded marks, adjustment/timing)._"
            )
        if attribution.get("excluded_from_mark"):
            lines.append("")
            lines.append(
                f"_EXCLUDED from mark (no price — P&L hits the book total but cannot be "
                f"attributed): {', '.join(attribution['excluded_from_mark'])}._"
            )
        cd = attribution.get("cash_drag")
        if cd:
            lines.append("")
            lines.append("### Cash drag")
            lines.append(
                f"_Book holds {_fmt(cd['cash_weight_pct'], '{:.1f}%')} idle cash "
                f"(${cd['cash_usd']:.2f}). Sleeve = invested positions only; "
                f"drag = book minus sleeve._"
            )
            lines.append("")
            lines.append("| Return | 1d | Cumulative |")
            lines.append("|--------|----|------------|")
            lines.append(
                f"| Invested sleeve | {_fmt(cd['sleeve_day_return_pct'])} | "
                f"{_fmt(cd['sleeve_cum_return_pct'])} |"
            )
            lines.append(
                f"| Total book | {_fmt(cd['book_day_return_pct'])} | "
                f"{_fmt(cd['book_cum_return_pct'])} |"
            )
            lines.append(
                f"| Benchmark | {_fmt(cd.get('bench_day_return_pct'))} | "
                f"{_fmt(cd.get('bench_cum_return_pct'))} |"
            )
            lines.append(
                f"| Cash drag | {_fmt(cd['day_drag_pct'], '{:+.2f}pp')} | "
                f"{_fmt(cd['cum_drag_pct'], '{:+.2f}pp')} |"
            )
        lines.append("")

    lines.append("## Learning loop (conviction biases)")
    biases = learning.get("biases", {})
    if biases:
        for t, b in sorted(biases.items(), key=lambda x: -x[1]):
            lines.append(f"- {t}: {b:+.3f}")
    else:
        lines.append("- (no biases yet, first run)")

    if quant:
        lines.append("")
        lines.append("## Quant analytics")
        lines.append(f"- BL views applied: {quant['n_views']}")
        lines.append(f"- BL run: {quant['bl_run_ts']}")
        lines.append("")
        lines.append("### BL target weights vs current")
        lines.append("| Ticker | Current % | BL target % | Δ |")
        lines.append("|--------|-----------|-------------|---|")
        cur_by = {p["ticker"]: p["weight_pct"] for p in snapshot["positions"]}
        for t in sorted(quant["bl_weights"], key=lambda x: -quant["bl_weights"][x]):
            cur = cur_by.get(t, 0.0)
            tgt = quant["bl_weights"][t] * 100
            lines.append(f"| {t} | {cur:.1f}% | {tgt:.1f}% | {tgt - cur:+.1f}pp |")
        if quant["rebalance"]:
            lines.append("")
            lines.append("### Rebalance candidates (|Δ| ≥ 3pp)")
            for r in quant["rebalance"]:
                lines.append(
                    f"- {r['ticker']}: {r['current_pct']:.1f}% → {r['target_pct']:.1f}% "
                    f"({r['delta_pct']:+.1f}pp)"
                )
        else:
            lines.append("")
            lines.append("_No rebalance candidates above threshold._")

    if mc_report:
        lines.append("")
        lines.append("## Quant risk snapshot (Monte Carlo — SIMULATED)")
        lines.append(
            f"_{mc_report['n_paths']:,} paths × {mc_report['horizon_days']}d, "
            f"drift={mc_report['drift_source']}. Forward-looking simulation — "
            f"distinct from the realized metrics above._"
        )
        lines.append("")
        lines.append("| Metric | Value | Flag |")
        lines.append("|--------|-------|------|")

        def _flag(cond: bool) -> str:
            return "**VETO**" if cond else "OK"

        from .risk import (
            VAR_LIMIT_USD,
            CVAR_LIMIT_USD,
            MEDIAN_MAXDD_LIMIT,
            P_RUIN_LIMIT,
        )
        lines.append(f"| VaR 95% (1d) | ${mc_report['var_1d_usd']:.2f} | "
                     f"{_flag(mc_report['var_1d_usd'] > VAR_LIMIT_USD)} |")
        lines.append(f"| CVaR 95% (1d) | ${mc_report['cvar_1d_usd']:.2f} | "
                     f"{_flag(mc_report['cvar_1d_usd'] > CVAR_LIMIT_USD)} |")
        lines.append(f"| Median MaxDD | {mc_report['median_max_drawdown']*100:.1f}% | "
                     f"{_flag(mc_report['median_max_drawdown'] < MEDIAN_MAXDD_LIMIT)} |")
        lines.append(f"| Sim Sharpe (ann) | {mc_report['sim_sharpe']:.2f} | --- |")
        lines.append(f"| Sim return (ann) | {mc_report['sim_return_ann']*100:+.1f}% | --- |")
        lines.append(f"| Sim vol (ann) | {mc_report['sim_vol_ann']*100:.1f}% | --- |")
        lines.append(
            f"| P(ruin < ${mc_report['ruin_threshold_usd']:.0f}) | "
            f"{mc_report['p_ruin']*100:.2f}% | "
            f"{_flag(mc_report['p_ruin'] > P_RUIN_LIMIT)} |"
        )
        if mc_report["veto"]:
            lines.append("")
            lines.append("**MC VETO active** — executor would not trade:")
            for fl in mc_report["veto_flags"]:
                lines.append(f"- {fl}")

    if options:
        lines.append("")
        lines.append("## Options overlay (paper only)")
        lines.append(
            f"_{options['ticker']} protective put — "
            f"K=${options['strike']:.2f}, T={options['tenor_days']}d, "
            f"σ={options['sigma']*100:.1f}%, STATUS: **{options['status']}**_"
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Spot | ${options['spot']:.2f} |")
        lines.append(f"| BSM put value | ${options['put_value_per_share']:.2f}/share |")
        lines.append(f"| Delta | {options['delta']:+.3f} |")
        lines.append(f"| Gamma | {options['gamma']:.4f} |")
        lines.append(f"| Theta (daily) | ${options['theta_per_day']:+.3f}/share |")
        lines.append(f"| Vega (per 1 vol pt) | ${options['vega']:.3f} |")
        lines.append(f"| Notional hedge (BSM × shares) | ${options['notional_hedge_usd']:.2f} |")
        lines.append(f"| Budget cap | ${options['max_spend_usd']:.2f} |")
        lines.append("")
        lines.append("### Trigger conditions")
        for k, v in options["triggers"].items():
            lines.append(f"- {'PASS' if v else 'FAIL'} — {k}")

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

    hist_days = _history_window_days(state)
    quant = compute_quant_block(snapshot, learning, days=hist_days)
    mc_report = None
    if quant:
        state["bl_weights"] = quant["bl_weights"]
        state["bl_run_ts"] = quant["bl_run_ts"]
        mc_report = risk.mc_risk_report(snapshot, quant["history"], mu_bl=quant["mu_bl"])
        if mc_report:
            state["mc_report"] = {k: v for k, v in mc_report.items() if k != "veto_flags"}
            state["mc_report"]["veto_flags"] = mc_report["veto_flags"]
            if mc_report["veto"]:
                risk_rep["status"] = "RED"
                risk_rep["flags"].extend(mc_report["veto_flags"])

    options = compute_options_overlay(snapshot, quant, mc_report)
    if options:
        state["options_overlay"] = {k: v for k, v in options.items() if k != "triggers"}
        state["options_overlay"]["triggers"] = options["triggers"]

    quant_history = quant["history"] if quant else None
    realized = compute_realized_block(state, snapshot)
    benchmark = compute_benchmark_block(state, snapshot, quant_history, hist_days)
    attribution = compute_attribution_block(state, snapshot, quant_history, benchmark)
    data_quality = compute_data_quality_block(state, px, quant_history, snapshot)
    if realized:
        state["realized"] = realized
    if benchmark:
        state["benchmark"] = benchmark
    if attribution:
        state["attribution"] = attribution
    if data_quality:
        state["data_quality"] = data_quality

    bench_today = None
    if benchmark:
        today_s = snapshot["as_of"][:10]
        past = [p for p in benchmark["curve"] if p["date"] <= today_s]
        if past:
            bench_today = past[-1]["value"]

    portfolio.append_history(
        snapshot, mc_report=mc_report, bl_run=bool(quant),
        bench_value=bench_today,
        data_status=data_quality["status"] if data_quality else "",
    )
    portfolio.save(state)

    write_journal(
        snapshot, risk_rep, learning, quant=quant, mc_report=mc_report, options=options,
        realized=realized, benchmark=benchmark, attribution=attribution,
        data_quality=data_quality,
    )

    history_rows = history.read_text().strip().splitlines() if history.exists() else []
    update_readme(snapshot, max(0, len(history_rows) - 1))

    print(f"[daily] total ${snapshot['total_value']:.2f}, "
          f"P&L ${snapshot['total_pnl_usd']:+.2f} ({snapshot['total_pnl_pct']:+.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
