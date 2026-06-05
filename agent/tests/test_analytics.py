"""Known-value + edge-case tests for agent/analytics.py. Fast, no network."""
from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from agent import analytics

SQ252 = math.sqrt(252)


# ---------------------------------------------------------------- equity curve

def test_load_equity_curve_dedup_and_legacy_rows(tmp_path):
    f = tmp_path / "history.csv"
    f.write_text(
        "date_utc,total_value,cash,holdings_value,pnl_usd,pnl_pct,"
        "var_95_usd,cvar_95_usd,sim_sharpe,p_ruin,bl_run,bench_value,data_status\n"
        # legacy 6-field row (pre-migration data line)
        "2026-05-20,999.98,159.98,840.00,-0.02,-0.00\n"
        # same-date rerun: keep LAST
        "2026-05-20,999.86,159.98,839.88,-0.14,-0.01\n"
        # full current-width row
        "2026-05-21,1005.00,159.98,845.02,5.00,0.50,12.00,18.00,0.800,0.0100,1,1002.50,ok\n"
    )
    curve = analytics.load_equity_curve(f)
    assert curve == [("2026-05-20", 999.86), ("2026-05-21", 1005.00)]


def test_load_equity_curve_missing_file(tmp_path):
    assert analytics.load_equity_curve(tmp_path / "nope.csv") == []


# ------------------------------------------------------------ realized metrics

def test_realized_metrics_known_values():
    # start 100 -> 110 -> 99 -> 110.88 : returns +10%, -10%, +12%
    curve = [("2026-01-01", 110.0), ("2026-01-02", 99.0), ("2026-01-03", 110.88)]
    m = analytics.realized_metrics(curve, 100.0, rf_annual=0.0)

    assert m["n_obs"] == 3
    assert m["cum_return_pct"] == pytest.approx(10.88)

    # returns [0.10, -0.10, 0.12]: mean 0.04, sample sd = sqrt(0.0296/2)
    sd = math.sqrt(0.0296 / 2)
    assert m["vol_ann_pct"] == pytest.approx(sd * SQ252 * 100)
    assert m["sharpe"] == pytest.approx(0.04 / sd * SQ252)

    # sortino (MAR=0): downside dev = sqrt(0.10^2 / 3)
    dd = math.sqrt(0.01 / 3)
    assert m["sortino"] == pytest.approx(0.04 / dd * SQ252)

    # drawdown: peak 110 -> trough 99 = -10%; last point is a new peak
    assert m["max_drawdown_pct"] == pytest.approx(-10.0)
    assert m["current_drawdown_pct"] == pytest.approx(0.0)
    assert [u["drawdown_pct"] for u in m["underwater"]] == pytest.approx([0.0, -10.0, 0.0])

    # too few runs to annualize honestly
    assert m["ann_return_pct"] is None
    assert m["calmar"] is None
    assert m["trailing"] is None


def test_realized_metrics_annualized_after_min_obs():
    # 25 runs of +1% except run 11 at -5%
    rets = [0.01] * 25
    rets[10] = -0.05
    v = 1000.0
    curve = []
    for i, r in enumerate(rets):
        v *= 1.0 + r
        curve.append((f"2026-02-{i + 1:02d}", v))
    m = analytics.realized_metrics(curve, 1000.0, rf_annual=0.0)

    total = math.prod(1.0 + r for r in rets)
    assert m["cum_return_pct"] == pytest.approx((total - 1.0) * 100)
    expected_ann = (total ** (252 / 25) - 1.0) * 100
    assert m["ann_return_pct"] == pytest.approx(expected_ann)
    # the -5% day is exactly the max drawdown (prior day was the peak)
    assert m["max_drawdown_pct"] == pytest.approx(-5.0)
    assert m["calmar"] == pytest.approx((expected_ann / 100) / 0.05)
    # trailing window: last 21 returns (indices 4..24) include the -5% dip
    assert m["trailing"]["window_runs"] == 21
    expected_trailing = (math.prod(1.0 + r for r in rets[-21:]) - 1.0) * 100
    assert m["trailing"]["return_pct"] == pytest.approx(expected_trailing)


def test_realized_metrics_empty_and_single_point():
    empty = analytics.realized_metrics([], 1000.0)
    assert empty["n_obs"] == 0
    assert empty["cum_return_pct"] is None
    assert empty["sharpe"] is None
    assert empty["underwater"] == []

    one = analytics.realized_metrics([("2026-01-01", 1010.0)], 1000.0, rf_annual=0.0)
    assert one["cum_return_pct"] == pytest.approx(1.0)
    assert one["vol_ann_pct"] is None       # single return, sample sd undefined
    assert one["sortino"] is None           # no downside observation
    assert one["max_drawdown_pct"] == pytest.approx(0.0)


def test_realized_metrics_zero_starting_capital():
    m = analytics.realized_metrics([("2026-01-01", 1000.0)], 0.0)
    assert m["cum_return_pct"] is None


# ------------------------------------------------------------ benchmark curve

def test_benchmark_curve_scaling_and_inception_filter():
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    s = pd.Series([490.0, 500.0, 505.0, 495.0, 510.0], index=idx)
    # inception on the 2nd bar: the 490 bar must be dropped, 500 becomes $1000
    curve = analytics.benchmark_curve(s, "2026-01-02", 1000.0)
    assert curve[0] == {"date": "2026-01-02", "value": 1000.0}
    assert curve[-1]["value"] == pytest.approx(1020.0)
    assert len(curve) == 4


def test_benchmark_curve_empty_cases():
    assert analytics.benchmark_curve(pd.Series(dtype=float), "2026-01-01", 1000.0) == []
    idx = pd.date_range("2026-01-01", periods=2, freq="D")
    s = pd.Series([100.0, 101.0], index=idx)
    # inception after all bars
    assert analytics.benchmark_curve(s, "2026-02-01", 1000.0) == []


# ---------------------------------------------------------- benchmark metrics

def test_benchmark_metrics_known_values():
    # bench daily returns [0.01, -0.02, 0.03]; portfolio exactly 2x
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    bvals = [1000.0, 1010.0, 989.8, 1019.494]
    pvals = [1000.0, 1020.0, 979.2, 1037.952]
    bench_curve = [{"date": d, "value": v} for d, v in zip(dates, bvals)]
    port_curve = list(zip(dates, pvals))

    m = analytics.benchmark_metrics(port_curve, bench_curve, 1000.0, rf_annual=0.0)
    assert m["n_obs"] == 3
    assert m["beta"] == pytest.approx(2.0)
    assert m["alpha_ann_pct"] == pytest.approx(0.0, abs=1e-9)
    assert m["up_capture"] == pytest.approx(2.0)
    assert m["down_capture"] == pytest.approx(2.0)

    # active returns = bench returns here: [0.01, -0.02, 0.03]
    active = [0.01, -0.02, 0.03]
    ma = sum(active) / 3
    te = math.sqrt(sum((a - ma) ** 2 for a in active) / 2)
    assert m["tracking_error_ann_pct"] == pytest.approx(te * SQ252 * 100, rel=1e-3)
    assert m["info_ratio"] == pytest.approx(ma * 252 / (te * SQ252), rel=1e-3)

    assert m["port_cum_return_pct"] == pytest.approx(3.7952)
    assert m["bench_cum_return_pct"] == pytest.approx(1.9494)
    assert m["active_return_cum_pct"] == pytest.approx(3.7952 - 1.9494)
    assert m["active_return_1d_pct"] == pytest.approx((0.06 - 0.03) * 100, rel=1e-3)


def test_benchmark_metrics_asof_alignment():
    # portfolio ran on a date with no bench bar: pairs with the prior bench close
    bench_curve = [
        {"date": "2026-01-02", "value": 1000.0},
        {"date": "2026-01-05", "value": 1010.0},
    ]
    port_curve = [("2026-01-02", 1000.0), ("2026-01-03", 1004.0), ("2026-01-05", 1012.0)]
    m = analytics.benchmark_metrics(port_curve, bench_curve, 1000.0)
    assert m["n_obs"] == 2  # 3 paired points -> 2 returns
    assert m["port_cum_return_pct"] == pytest.approx(1.2)
    assert m["bench_cum_return_pct"] == pytest.approx(1.0)


def test_benchmark_metrics_insufficient_data():
    m = analytics.benchmark_metrics(
        [("2026-01-01", 1000.0)],
        [{"date": "2026-01-01", "value": 1000.0}],
        1000.0,
    )
    assert m["n_obs"] == 0
    assert m["beta"] is None
    assert m["tracking_error_ann_pct"] is None
    assert m["port_cum_return_pct"] == pytest.approx(0.0)

    empty = analytics.benchmark_metrics([], [], 1000.0)
    assert empty["n_obs"] == 0 and empty["active_return_cum_pct"] is None


# ---------------------------------------------------------------- attribution

def _positions():
    # AAA: 10 sh, cost 10 -> price 12 (cum +20); BBB: 5 sh, cost 20 -> price 18 (cum -10)
    return [
        {"ticker": "AAA", "shares": 10.0, "cost_basis": 10.0, "price": 12.0,
         "market_value": 120.0, "pnl_usd": 20.0, "weight_pct": 60.0},
        {"ticker": "BBB", "shares": 5.0, "cost_basis": 20.0, "price": 18.0,
         "market_value": 90.0, "pnl_usd": -10.0, "weight_pct": 45.0},
    ]


def test_attribution_known_values():
    out = analytics.attribution(
        _positions(), 200.0,
        prior_closes={"AAA": 11.0, "BBB": 19.0}, prior_total=205.0,
        sector_map={"AAA": "x", "BBB": "y"},
    )
    by = {r["ticker"]: r for r in out["positions"]}
    # cumulative: pnl / starting capital
    assert by["AAA"]["cum_contribution_pct"] == pytest.approx(10.0)
    assert by["BBB"]["cum_contribution_pct"] == pytest.approx(-5.0)
    # cumulative contributions sum to the book's total pnl pct (cash adds zero)
    assert sum(r["cum_contribution_pct"] for r in out["positions"]) == pytest.approx(5.0)
    # daily: shares x (price - prior close) / prior total
    assert by["AAA"]["day_pnl_usd"] == pytest.approx(10.0)
    assert by["AAA"]["day_contribution_pct"] == pytest.approx(10.0 / 205.0 * 100)
    assert by["BBB"]["day_pnl_usd"] == pytest.approx(-5.0)
    assert by["BBB"]["day_contribution_pct"] == pytest.approx(-5.0 / 205.0 * 100)
    # sorted by cumulative contribution desc
    assert [r["ticker"] for r in out["positions"]] == ["AAA", "BBB"]
    sec = {s["sector"]: s for s in out["sectors"]}
    assert sec["x"]["cum_contribution_pct"] == pytest.approx(10.0)
    assert sec["y"]["day_contribution_pct"] == pytest.approx(-5.0 / 205.0 * 100)


def test_attribution_missing_prior_close_degrades_day_fields():
    out = analytics.attribution(
        _positions(), 200.0,
        prior_closes={"AAA": 11.0}, prior_total=205.0,  # BBB prior missing
        sector_map={"AAA": "x", "BBB": "x"},
    )
    by = {r["ticker"]: r for r in out["positions"]}
    assert by["AAA"]["day_contribution_pct"] is not None
    assert by["BBB"]["day_pnl_usd"] is None
    assert by["BBB"]["cum_contribution_pct"] == pytest.approx(-5.0)  # cum still exact
    # shared sector: day figure must be None, not a misleading partial sum
    assert out["sectors"][0]["day_contribution_pct"] is None
    assert out["sectors"][0]["cum_contribution_pct"] == pytest.approx(5.0)


def test_attribution_no_priors_cum_only():
    out = analytics.attribution(_positions(), 200.0)
    assert all(r["day_pnl_usd"] is None for r in out["positions"])
    assert all(r["cum_contribution_pct"] is not None for r in out["positions"])


# ------------------------------------------------------------------ cash drag

def test_cash_drag_known_values():
    # invested cost 200, holdings 220 -> sleeve +10%; cash 100, total 320,
    # starting 300 -> book +6.6667%; drag = book - sleeve
    snapshot = {
        "cash": 100.0, "holdings_value": 220.0, "total_value": 320.0,
        "total_pnl_pct": 20.0 / 300.0 * 100,
        "positions": _positions(),  # cost = 10x10 + 5x20 = 200
    }
    prior_row = {"total_value": "310", "holdings_value": "210"}
    cd = analytics.cash_drag_metrics(snapshot, prior_row)
    assert cd["cash_weight_pct"] == pytest.approx(100.0 / 320.0 * 100)
    assert cd["sleeve_cum_return_pct"] == pytest.approx(10.0)
    assert cd["cum_drag_pct"] == pytest.approx(20.0 / 3 - 10.0)
    assert cd["book_day_return_pct"] == pytest.approx((320.0 / 310.0 - 1) * 100)
    assert cd["sleeve_day_return_pct"] == pytest.approx((220.0 / 210.0 - 1) * 100)
    assert cd["day_drag_pct"] == pytest.approx(
        cd["book_day_return_pct"] - cd["sleeve_day_return_pct"]
    )


def test_cash_drag_no_prior_row():
    snapshot = {
        "cash": 100.0, "holdings_value": 220.0, "total_value": 320.0,
        "total_pnl_pct": 6.6667, "positions": _positions(),
    }
    cd = analytics.cash_drag_metrics(snapshot, None)
    assert cd["sleeve_cum_return_pct"] == pytest.approx(10.0)
    assert cd["book_day_return_pct"] is None
    assert cd["day_drag_pct"] is None


def test_load_history_rows_by_name(tmp_path):
    f = tmp_path / "history.csv"
    f.write_text(
        "date_utc,total_value,cash,holdings_value,pnl_usd,pnl_pct\n"
        "2026-05-20,999.86,159.98,839.88,-0.14,-0.01\n"
    )
    rows = analytics.load_history_rows(f)
    assert rows[0]["holdings_value"] == "839.88"
    assert analytics.load_history_rows(tmp_path / "nope.csv") == []


# -------------------------------------------------------------- data quality

def _hist(last_date: str, cols=("SPY", "NVDA")):
    idx = pd.date_range(end=last_date, periods=5, freq="D")
    return pd.DataFrame({c: range(1, 6) for c in cols}, index=idx)


def test_data_quality_ok():
    dq = analytics.data_quality_report(
        ["NVDA"], {"NVDA": 100.0}, _hist("2026-06-05"), date(2026, 6, 5),
    )
    assert dq["status"] == "ok"
    assert dq["issues"] == []
    assert dq["missing_prices"] == []
    assert dq["last_bar"] == "2026-06-05"


def test_data_quality_flags_missing_and_stale():
    dq = analytics.data_quality_report(
        ["NVDA", "MSFT"], {"NVDA": 100.0}, _hist("2026-05-26"), date(2026, 6, 5),
    )
    assert dq["status"] == "degraded"
    assert dq["missing_prices"] == ["MSFT"]
    assert any("EXCLUDED from mark" in i for i in dq["issues"])
    assert any("stale" in i for i in dq["issues"])


def test_data_quality_no_history_and_missing_benchmark():
    dq = analytics.data_quality_report(["NVDA"], {"NVDA": 1.0}, None, date(2026, 6, 5))
    assert dq["status"] == "degraded"
    assert any("no price history" in i for i in dq["issues"])
    assert dq["last_bar"] is None

    dq2 = analytics.data_quality_report(
        ["NVDA"], {"NVDA": 1.0}, _hist("2026-06-05", cols=("NVDA",)), date(2026, 6, 5),
    )
    assert dq2["status"] == "degraded"
    assert any("benchmark SPY missing" in i for i in dq2["issues"])
