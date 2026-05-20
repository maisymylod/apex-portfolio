"""Return-history utilities: log returns, rolling covariance, Cholesky.

All inputs are pandas DataFrames indexed by date, columns = tickers, values
= adjusted close. All outputs are numpy arrays (or DataFrames where
labelled). Keeps the heavy lifting in numpy/scipy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns. Drops the first row (NaN)."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def rolling_cov(prices: pd.DataFrame, window: int = 60, annualize: bool = True) -> pd.DataFrame:
    """Sample covariance of daily log returns over the last `window` days.

    Returned matrix is annualized by default (×252). Diagonal is variance,
    off-diagonal is covariance. Column/index labels are tickers.
    """
    r = log_returns(prices).tail(window)
    if r.shape[0] < 2:
        # degenerate: too little history, return identity scaled by tiny var
        n = prices.shape[1]
        return pd.DataFrame(np.eye(n) * 1e-4, index=prices.columns, columns=prices.columns)
    cov = r.cov()
    if annualize:
        cov = cov * TRADING_DAYS
    return cov


def cholesky(cov: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Lower-triangular Cholesky factor with a tiny diagonal jitter for PSD safety."""
    m = np.asarray(cov, dtype=float)
    jitter = 1e-10 * np.trace(m) / max(m.shape[0], 1)
    return np.linalg.cholesky(m + np.eye(m.shape[0]) * jitter)


def historical_drift(prices: pd.DataFrame, window: int = 60, annualize: bool = True) -> pd.Series:
    """Mean of daily log returns over the last `window` days, optionally annualized."""
    r = log_returns(prices).tail(window)
    mu = r.mean()
    if annualize:
        mu = mu * TRADING_DAYS
    return mu
