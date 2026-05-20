"""Black-Scholes-Merton call/put pricer + greeks.

Used by the optional protective-put overlay in agent/daily.py. The overlay
is a paper-only signal — there is no broker hooked up — but we price the
put properly so the journal shows BSM value, delta, gamma, theta, vega.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float  # per-day (we convert from per-year inside)
    vega: float   # per 1.00 vol change
    rho: float    # per 1.00 rate change


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("BSM requires positive T and sigma")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def greeks(S: float, K: float, r: float, sigma: float, T: float, option: str = "put") -> Greeks:
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    pdf = norm.pdf(d1)
    sign = 1.0 if option.lower() == "call" else -1.0
    nd1 = norm.cdf(sign * d1)
    nd2 = norm.cdf(sign * d2)
    delta = sign * nd1
    gamma = pdf / (S * sigma * math.sqrt(T))
    theta_annual = (
        -S * pdf * sigma / (2 * math.sqrt(T))
        - sign * r * K * math.exp(-r * T) * nd2
    )
    theta_daily = theta_annual / 365.0
    vega = S * pdf * math.sqrt(T) / 100.0  # per 1 vol pt
    rho = sign * K * T * math.exp(-r * T) * nd2 / 100.0
    return Greeks(delta=delta, gamma=gamma, theta=theta_daily, vega=vega, rho=rho)


def realized_vol_annual(returns_daily: list[float] | None) -> float:
    """Stdev of daily log returns, annualized. Returns 0 if no data."""
    if not returns_daily or len(returns_daily) < 2:
        return 0.0
    import statistics
    sd = statistics.stdev(returns_daily)
    return sd * math.sqrt(252)
