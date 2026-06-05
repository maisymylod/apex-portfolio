"""Realized + benchmark analytics from the recorded equity curve.

Everything in this module is REALIZED — computed from what the paper book
actually did (state/history.csv) plus price series the run already fetched
for BL (which include SPY). The simulated metrics live in quant/mc.py; the
two must stay clearly labelled wherever they are surfaced.

Pure functions: no network, no RNG, no file writes (only load_equity_curve
reads). Callers (agent/daily.py) pass series in and wrap every call in the
standard try/except-and-continue pattern so the daily run never dies here.

Unit conventions: keys suffixed _pct are percent (matching pnl_pct
elsewhere); Sharpe/Sortino/Calmar/beta/IR/capture are unitless ratios.
"""
from __future__ import annotations

import csv
import math
from bisect import bisect_right
from datetime import date
from pathlib import Path

import pandas as pd

TRADING_DAYS = 252
MIN_OBS_ANNUALIZE = 20     # runs needed before an annualized return is honest
MIN_OBS_BETA = 3           # paired daily returns needed for beta/alpha/TE/IR
TRAILING_WINDOW_RUNS = 21  # ~1 trading month
STALE_BAR_MAX_DAYS = 4     # calendar days; covers a weekend plus one holiday


def load_equity_curve(history_path: Path) -> list[tuple[str, float]]:
    """(date, total_value) points from history.csv, read by column NAME.

    Positional readers elsewhere must never break, but new code reads by
    header so it survives future appended columns. Same-date rows (manual
    re-runs) dedup keeping the LAST row. Legacy short rows parse fine via
    DictReader. Returns [] if the file is missing.
    """
    if not history_path.exists():
        return []
    out: dict[str, float] = {}
    with history_path.open() as f:
        for row in csv.DictReader(f):
            try:
                out[row["date_utc"]] = float(row["total_value"])
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out.items())


def _returns(values: list[float]) -> list[float]:
    return [cur / prev - 1.0 for prev, cur in zip(values, values[1:]) if prev > 0]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _stdev(xs: list[float]) -> float | None:
    """Sample stdev (ddof=1); None below 2 observations."""
    if len(xs) < 2:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def realized_metrics(
    curve: list[tuple[str, float]],
    starting_capital: float,
    rf_annual: float = 0.045,
) -> dict:
    """Realized performance stats from the recorded (date, total_value) curve.

    The inception point (starting_capital) is prepended so the seed-day move
    counts as the first return. Annualized return and Calmar stay None until
    MIN_OBS_ANNUALIZE runs — annualizing a days-old curve produces absurd
    numbers. Sortino uses MAR = daily risk-free; downside deviation is the
    full-sample (n-denominator) root mean square of below-MAR excess.
    """
    out = {
        "n_obs": len(curve),
        "cum_return_pct": None,
        "ann_return_pct": None,
        "vol_ann_pct": None,
        "sharpe": None,
        "sortino": None,
        "calmar": None,
        "current_drawdown_pct": None,
        "max_drawdown_pct": None,
        "underwater": [],
        "trailing": None,
    }
    if not curve or not starting_capital:
        return out
    values = [v for _, v in curve]
    rets = _returns([float(starting_capital)] + values)
    rf_d = rf_annual / TRADING_DAYS

    out["cum_return_pct"] = (values[-1] / starting_capital - 1.0) * 100

    # drawdown / underwater — peak tracking includes starting capital
    peak = float(starting_capital)
    max_dd = 0.0
    underwater = []
    for d, v in curve:
        peak = max(peak, v)
        dd = v / peak - 1.0
        max_dd = min(max_dd, dd)
        underwater.append({"date": d, "drawdown_pct": dd * 100})
    out["underwater"] = underwater
    out["max_drawdown_pct"] = max_dd * 100
    out["current_drawdown_pct"] = underwater[-1]["drawdown_pct"]

    if len(rets) >= MIN_OBS_ANNUALIZE:
        ann = (values[-1] / starting_capital) ** (TRADING_DAYS / len(rets)) - 1.0
        out["ann_return_pct"] = ann * 100
        if max_dd < 0:
            out["calmar"] = ann / abs(max_dd)

    sd = _stdev(rets)
    if sd and sd > 0:
        out["vol_ann_pct"] = sd * math.sqrt(TRADING_DAYS) * 100
        out["sharpe"] = (_mean(rets) - rf_d) / sd * math.sqrt(TRADING_DAYS)

    downside = [min(r - rf_d, 0.0) for r in rets]
    if rets and any(d < 0 for d in downside):
        dd_dev = math.sqrt(sum(d * d for d in downside) / len(rets))
        out["sortino"] = (_mean(rets) - rf_d) / dd_dev * math.sqrt(TRADING_DAYS)

    if len(rets) > TRAILING_WINDOW_RUNS:
        win = rets[-TRAILING_WINDOW_RUNS:]
        wsd = _stdev(win)
        out["trailing"] = {
            "window_runs": TRAILING_WINDOW_RUNS,
            "return_pct": (math.prod(1.0 + r for r in win) - 1.0) * 100,
            "vol_ann_pct": wsd * math.sqrt(TRADING_DAYS) * 100 if wsd else None,
            "sharpe": (
                (_mean(win) - rf_d) / wsd * math.sqrt(TRADING_DAYS)
                if wsd and wsd > 0 else None
            ),
        }
    return out


def benchmark_curve(
    closes: pd.Series,
    inception_date: str,
    starting_capital: float,
) -> list[dict]:
    """Benchmark closes rescaled to a starting_capital book at inception.

    First bar on/after inception_date becomes starting_capital; everything
    earlier is dropped. Returns [{date, value}, ...] (JSON-ready)."""
    s = closes.dropna()
    if s.empty:
        return []
    if getattr(s.index, "tz", None) is not None:
        s = s.tz_localize(None)
    s = s[s.index >= pd.Timestamp(inception_date)]
    if s.empty:
        return []
    base = float(s.iloc[0])
    if base <= 0:
        return []
    return [
        {"date": ts.date().isoformat(), "value": float(v) / base * float(starting_capital)}
        for ts, v in s.items()
    ]


def benchmark_metrics(
    port_curve: list[tuple[str, float]],
    bench_curve: list[dict],
    starting_capital: float,
    rf_annual: float = 0.045,
) -> dict:
    """Benchmark-relative stats on date-aligned daily returns.

    Each portfolio point pairs with the benchmark value as-of (<=) its date.
    Beta/alpha/TE/IR/capture need MIN_OBS_BETA paired returns; cumulative
    comparisons work from the first paired point. Alpha is CAPM, annualized
    arithmetically; IR = annualized mean active return / tracking error.
    """
    out = {
        "n_obs": 0,
        "port_cum_return_pct": None,
        "bench_cum_return_pct": None,
        "active_return_cum_pct": None,
        "active_return_1d_pct": None,
        "beta": None,
        "alpha_ann_pct": None,
        "tracking_error_ann_pct": None,
        "info_ratio": None,
        "up_capture": None,
        "down_capture": None,
    }
    if not port_curve or not bench_curve or not starting_capital:
        return out
    bdates = [p["date"] for p in bench_curve]
    bvals = [p["value"] for p in bench_curve]

    def asof(d: str) -> float | None:
        i = bisect_right(bdates, d) - 1
        return bvals[i] if i >= 0 else None

    pairs = [(v, asof(d)) for d, v in port_curve]
    pairs = [(p, b) for p, b in pairs if b is not None]
    if not pairs:
        return out
    pv = [p for p, _ in pairs]
    bv = [b for _, b in pairs]
    out["port_cum_return_pct"] = (pv[-1] / starting_capital - 1.0) * 100
    out["bench_cum_return_pct"] = (bv[-1] / starting_capital - 1.0) * 100
    out["active_return_cum_pct"] = out["port_cum_return_pct"] - out["bench_cum_return_pct"]

    pr = _returns(pv)
    br = _returns(bv)
    out["n_obs"] = len(pr)
    if pr and br:
        out["active_return_1d_pct"] = (pr[-1] - br[-1]) * 100
    if len(pr) < MIN_OBS_BETA or len(pr) != len(br):
        return out

    mp, mb = _mean(pr), _mean(br)
    var_b = sum((b - mb) ** 2 for b in br) / (len(br) - 1)
    cov = sum((p - mp) * (b - mb) for p, b in zip(pr, br)) / (len(pr) - 1)
    rf_d = rf_annual / TRADING_DAYS
    if var_b > 0:
        beta = cov / var_b
        out["beta"] = beta
        out["alpha_ann_pct"] = ((mp - rf_d) - beta * (mb - rf_d)) * TRADING_DAYS * 100

    active = [p - b for p, b in zip(pr, br)]
    te = _stdev(active)
    if te and te > 0:
        out["tracking_error_ann_pct"] = te * math.sqrt(TRADING_DAYS) * 100
        out["info_ratio"] = _mean(active) * TRADING_DAYS / (te * math.sqrt(TRADING_DAYS))

    up = [(p, b) for p, b in zip(pr, br) if b > 0]
    down = [(p, b) for p, b in zip(pr, br) if b < 0]
    if up:
        mb_up = _mean([b for _, b in up])
        if mb_up != 0:
            out["up_capture"] = _mean([p for p, _ in up]) / mb_up
    if down:
        mb_dn = _mean([b for _, b in down])
        if mb_dn != 0:
            out["down_capture"] = _mean([p for p, _ in down]) / mb_dn
    return out


def data_quality_report(
    position_tickers: list[str],
    px: dict[str, float],
    history,
    as_of: date,
    bench_ticker: str = "SPY",
) -> dict:
    """Detect missing or stale price data instead of letting it pass silently.

    A position missing from px has been EXCLUDED from the mark (its market
    value silently vanished from totals) — that is the worst failure mode
    because it records a fake P&L swing into history.csv permanently.
    """
    issues = []
    missing = sorted(t for t in position_tickers if t not in px)
    if missing:
        issues.append(
            "missing close price (position EXCLUDED from mark, totals understated): "
            + ", ".join(missing)
        )
    last_bar = None
    if history is None or getattr(history, "empty", True):
        issues.append("no price history frame: BL/MC/benchmark skipped this run")
    else:
        ts = history.index[-1]
        last_bar_d = ts.date() if hasattr(ts, "date") else date.fromisoformat(str(ts)[:10])
        age = (as_of - last_bar_d).days
        if age > STALE_BAR_MAX_DAYS:
            issues.append(f"price history stale: last bar {last_bar_d.isoformat()} is {age}d old")
        if bench_ticker not in history.columns:
            issues.append(f"benchmark {bench_ticker} missing from history frame")
        last_bar = last_bar_d.isoformat()
    return {
        "status": "ok" if not issues else "degraded",
        "issues": issues,
        "missing_prices": missing,
        "last_bar": last_bar,
    }
