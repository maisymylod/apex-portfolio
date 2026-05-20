"""Sanity tests for quant/ — fast, deterministic, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import bl, bsm, kelly, mc, stats as qstats


@pytest.fixture
def prices():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    tickers = ["A", "B", "C"]
    # mild trend + noise so log returns are well-defined
    base = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=(120, 3)), axis=0))
    return pd.DataFrame(base, index=dates, columns=tickers)


def test_log_returns_shape(prices):
    r = qstats.log_returns(prices)
    assert r.shape == (119, 3)
    assert not r.isna().any().any()


def test_rolling_cov_psd(prices):
    cov = qstats.rolling_cov(prices, window=60)
    assert cov.shape == (3, 3)
    eigvals = np.linalg.eigvalsh(cov.values)
    assert (eigvals > -1e-8).all()  # positive semi-definite


def test_cholesky_roundtrip(prices):
    cov = qstats.rolling_cov(prices, window=60).values
    L = qstats.cholesky(cov)
    assert np.allclose(L @ L.T, cov, atol=1e-6)


def test_bl_no_views_returns_pi(prices):
    cov = qstats.rolling_cov(prices, window=60)
    pi = pd.Series([0.08, 0.10, 0.05], index=cov.columns)
    mu = bl.posterior_returns(pi, cov, views=[])
    assert np.allclose(mu.values, pi.values)


def test_bl_views_shift_mu(prices):
    cov = qstats.rolling_cov(prices, window=60)
    pi = pd.Series([0.05, 0.05, 0.05], index=cov.columns)
    views = [bl.View(pick={"A": 1.0, "B": -1.0}, q=0.20, confidence="HIGH")]
    mu = bl.posterior_returns(pi, cov, views=views)
    assert mu["A"] > pi["A"]
    assert mu["B"] < pi["B"]


def test_optimal_weights_sum_to_one(prices):
    cov = qstats.rolling_cov(prices, window=60)
    mu = pd.Series([0.08, 0.10, 0.05], index=cov.columns)
    w = bl.optimal_weights(mu, cov, weight_cap=0.6)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()
    assert (w <= 0.6 + 1e-9).all()


def test_apply_bias_normalizes():
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    out = bl.apply_bias(w, {"A": 0.10, "B": -0.05, "C": 0.0})
    assert abs(out.sum() - 1.0) < 1e-9


def test_mc_metrics_reasonable(prices):
    cov = qstats.rolling_cov(prices, window=60)
    mu = pd.Series([0.10, 0.08, 0.06], index=cov.columns)
    w = pd.Series([0.4, 0.4, 0.2], index=cov.columns)
    out = mc.evaluate(w, mu, cov, starting_value=1000.0, n_paths=2000, horizon_days=60, seed=42)
    assert out["var_1d_usd"] > 0
    assert out["cvar_1d_usd"] >= out["var_1d_usd"]
    assert -1.0 < out["median_max_drawdown"] <= 0.0
    assert 0.0 <= out["p_ruin"] <= 1.0
    assert out["n_paths"] == 2000


def test_mc_deterministic_with_seed(prices):
    cov = qstats.rolling_cov(prices, window=60)
    mu = pd.Series([0.10, 0.08, 0.06], index=cov.columns)
    w = pd.Series([0.4, 0.4, 0.2], index=cov.columns)
    a = mc.evaluate(w, mu, cov, 1000.0, n_paths=500, horizon_days=30, seed=7)
    b = mc.evaluate(w, mu, cov, 1000.0, n_paths=500, horizon_days=30, seed=7)
    assert a["var_1d_usd"] == b["var_1d_usd"]
    assert a["sim_sharpe"] == b["sim_sharpe"]


def test_bsm_put_call_parity():
    S, K, r, sigma, T = 100.0, 100.0, 0.04, 0.25, 0.5
    c = bsm.call_price(S, K, r, sigma, T)
    p = bsm.put_price(S, K, r, sigma, T)
    # C - P = S - K e^(-rT)
    import math
    lhs = c - p
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-6


def test_bsm_greeks_put_delta_negative():
    g = bsm.greeks(100, 95, 0.04, 0.30, 60 / 252, option="put")
    assert -1.0 < g.delta < 0.0
    assert g.gamma > 0
    assert g.theta < 0
    assert g.vega > 0


def test_kelly_weights_capped(prices):
    cov = qstats.rolling_cov(prices, window=60)
    mu = pd.Series([0.40, 0.30, 0.20], index=cov.columns)  # juicy μ to trigger cap
    w = kelly.kelly_weights(mu, cov, risk_free=0.045, max_per_name=0.20)
    assert (w <= 0.20 + 1e-9).all()
    assert (w >= 0).all()
