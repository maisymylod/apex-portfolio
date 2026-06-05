"""End-to-end agent.daily runs with mocked yfinance + tmp state. No network.

Verifies the whole orchestration: seed -> mark -> risk -> learning -> BL ->
MC -> options -> realized/benchmark/data-quality -> history append -> journal,
including the history.csv header migration and the append-only column contract.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent import daily, planner, portfolio, prices, risk


@pytest.fixture
def fake_market():
    """Deterministic synthetic closes for the whole universe, ending today."""
    rng = np.random.default_rng(7)
    tickers = planner.universe_tickers()
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    idx = pd.date_range(end=end, periods=120, freq="D")
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, size=(120, len(tickers))), axis=0))
    return pd.DataFrame(base, index=idx, columns=tickers)


@pytest.fixture
def sandbox(monkeypatch, tmp_path, fake_market):
    """Redirect every state/journal path into tmp and stub both yfinance calls."""
    state_dir = tmp_path / "state"
    monkeypatch.setattr(portfolio, "STATE_DIR", state_dir)
    monkeypatch.setattr(portfolio, "PORTFOLIO_PATH", state_dir / "portfolio.json")
    monkeypatch.setattr(portfolio, "HISTORY_PATH", state_dir / "history.csv")
    monkeypatch.setattr(risk, "STATE_DIR", state_dir)
    monkeypatch.setattr(risk, "LEARN_PATH", state_dir / "learning.json")
    monkeypatch.setattr(daily, "JOURNAL_DIR", tmp_path / "journal")
    monkeypatch.setattr(daily, "README", tmp_path / "README.md")  # absent: update skipped

    def fake_close(tickers):
        return {t: float(fake_market[t].iloc[-1]) for t in tickers if t in fake_market.columns}

    def fake_history(tickers, days=90):
        cols = [t for t in tickers if t in fake_market.columns]
        return fake_market[cols].tail(days)

    monkeypatch.setattr(prices, "fetch_close_prices", fake_close)
    monkeypatch.setattr(prices, "fetch_history", fake_history)
    return tmp_path


def test_daily_end_to_end_two_runs(sandbox):
    assert daily.main() == 0  # run 1: seeds, then marks
    assert daily.main() == 0  # run 2: same-day rerun

    state = json.loads((sandbox / "state" / "portfolio.json").read_text())
    for key in ("realized", "benchmark", "attribution", "data_quality", "bl_weights", "mc_report"):
        assert key in state, f"portfolio.json missing {key}"
    assert state["data_quality"]["status"] == "ok"
    assert state["benchmark"]["source"].startswith("SPY")
    assert state["benchmark"]["curve"], "benchmark curve should not be empty"
    assert state["realized"]["n_obs"] >= 1
    assert state["realized"]["cum_return_pct"] is not None
    # additive-only contract: original keys all still present
    for key in ("starting_capital", "cash", "positions", "inception_date", "last_run"):
        assert key in state

    rows = (sandbox / "state" / "history.csv").read_text().strip().splitlines()
    header = rows[0].split(",")
    assert header == portfolio.HISTORY_HEADER
    assert header[1] == "total_value"               # positional contract (daily.py)
    assert header[-2:] == ["bench_value", "data_status"]
    assert len(rows) == 3                            # header + one row per run
    last = rows[-1].split(",")
    assert len(last) == len(portfolio.HISTORY_HEADER)
    float(last[1])                                   # parses as the prior-total read does
    assert last[-1] == "ok"
    assert float(last[-2]) > 0                       # bench_value populated

    # attribution invariant: position cum contributions sum to the book's
    # recorded pnl_pct (cash contributes zero); csv rounds to 2dp
    contribs = [p["cum_contribution_pct"] for p in state["attribution"]["positions"]]
    assert sum(contribs) == pytest.approx(float(last[5]), abs=0.01)
    cd = state["attribution"]["cash_drag"]
    assert cd["cash_weight_pct"] is not None
    assert cd["sleeve_cum_return_pct"] is not None
    assert "bench_cum_return_pct" in cd
    assert state["attribution"]["excluded_from_mark"] == []

    journals = list((sandbox / "journal").glob("*.md"))
    assert len(journals) == 1
    text = journals[0].read_text()
    for section in (
        "**Data quality:** ok",
        "## Realized performance",
        "REALIZED, not simulated",
        "## Benchmark comparison",
        "## Attribution",
        "### Cash drag",
        "Monte Carlo — SIMULATED",
    ):
        assert section in text, f"journal missing: {section!r}"


def test_daily_survives_missing_quant_history(sandbox, monkeypatch):
    """If the history fetch dies, the run must still mark, append, and journal."""
    assert daily.main() == 0  # seed first with healthy data

    def broken_history(tickers, days=90):
        raise RuntimeError("yfinance fell over")

    monkeypatch.setattr(prices, "fetch_history", broken_history)
    assert daily.main() == 0

    rows = (sandbox / "state" / "history.csv").read_text().strip().splitlines()
    assert len(rows) == 3
    last = rows[-1].split(",")
    assert last[-1] == "degraded"                    # data-quality guard fired
    assert last[-2] == ""                            # no benchmark value
    state = json.loads((sandbox / "state" / "portfolio.json").read_text())
    assert any("no price history" in i for i in state["data_quality"]["issues"])
    # realized metrics still computed: they come from history.csv, not yfinance
    assert state["realized"]["cum_return_pct"] is not None
    # attribution degrades to cumulative-only: cum needs no fresh prices
    attr = state["attribution"]
    assert all(p["cum_contribution_pct"] is not None for p in attr["positions"])
    assert all(p["day_contribution_pct"] is None for p in attr["positions"])
    assert attr["cash_drag"]["sleeve_cum_return_pct"] is not None


def test_compute_attribution_block_day_and_residual(tmp_path, monkeypatch):
    """Day attribution against a prior recorded row, excluded names surfaced."""
    monkeypatch.setattr(portfolio, "HISTORY_PATH", tmp_path / "history.csv")
    (tmp_path / "history.csv").write_text(
        "date_utc,total_value,cash,holdings_value,pnl_usd,pnl_pct\n"
        "2026-06-04,1000.00,100.00,900.00,0.00,0.00\n"
    )
    snapshot = {
        "as_of": "2026-06-05T20:30:00+00:00", "total_value": 1010.0, "cash": 100.0,
        "holdings_value": 910.0, "total_pnl_pct": 1.0,
        "positions": [{
            "ticker": "AAA", "shares": 10.0, "cost_basis": 89.0, "price": 91.0,
            "market_value": 910.0, "pnl_usd": 20.0, "weight_pct": 90.1,
        }],
    }
    state = {"starting_capital": 1000.0, "positions": {"AAA": {}, "ZZZ": {}}}
    hist = pd.DataFrame(
        {"AAA": [90.0, 91.0]}, index=pd.to_datetime(["2026-06-04", "2026-06-05"]),
    )
    out = daily.compute_attribution_block(state, snapshot, hist, None)
    assert out["excluded_from_mark"] == ["ZZZ"]
    # AAA day pnl = 10 x (91 - 90) = $10 on a $1000 prior book -> +1.00pp
    assert out["positions"][0]["day_contribution_pct"] == pytest.approx(1.0)
    assert out["day_attributed_pct"] == pytest.approx(1.0)
    # book day = 1010/1000 - 1 = +1.00% -> fully attributed, residual 0
    assert out["day_residual_pct"] == pytest.approx(0.0)
    assert out["cash_drag"]["sleeve_day_return_pct"] == pytest.approx((910 / 900 - 1) * 100)


def test_append_history_migrates_legacy_header(tmp_path, monkeypatch):
    monkeypatch.setattr(portfolio, "STATE_DIR", tmp_path)
    monkeypatch.setattr(portfolio, "HISTORY_PATH", tmp_path / "history.csv")
    (tmp_path / "history.csv").write_text(
        "date_utc,total_value,cash,holdings_value,pnl_usd,pnl_pct\n"
        "2026-05-20,999.98,159.98,840.00,-0.02,-0.00\n"
    )
    snapshot = {
        "as_of": "2026-06-05T20:30:00+00:00", "total_value": 1010.0, "cash": 160.0,
        "holdings_value": 850.0, "total_pnl_usd": 10.0, "total_pnl_pct": 1.0,
    }
    portfolio.append_history(
        snapshot, mc_report=None, bl_run=False, bench_value=1005.5, data_status="ok",
    )
    rows = (tmp_path / "history.csv").read_text().strip().splitlines()
    assert rows[0].split(",") == portfolio.HISTORY_HEADER
    assert rows[1].split(",")[1] == "999.98"  # legacy data row untouched
    new = rows[2].split(",")
    assert new[portfolio.HISTORY_HEADER.index("bench_value")] == "1005.50"
    assert new[portfolio.HISTORY_HEADER.index("data_status")] == "ok"
