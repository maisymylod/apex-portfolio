"""Monte Carlo risk engine.

Simulates correlated GBM paths for the whole portfolio:

    dS_i = S_i (μ_i dt + σ_i dW_i),    dW = L dZ,   L = chol(Σ_daily)

Returns a metrics dict consumed by risk.py: VaR/CVaR at 95%, max-drawdown
distribution, simulated Sharpe, and P(ruin) at a configurable threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as qstats

TRADING_DAYS = 252


def simulate_portfolio(
    weights: pd.Series,
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    starting_value: float,
    n_paths: int = 10_000,
    horizon_days: int = TRADING_DAYS,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate portfolio value paths under correlated GBM.

    Returns array of shape (n_paths, horizon_days + 1) with starting_value
    in column 0. Annualized μ and Σ are converted to per-day inside.
    """
    tickers = list(cov_annual.columns)
    w = weights.reindex(tickers).fillna(0.0).values.astype(float)
    mu = mu_annual.reindex(tickers).fillna(0.0).values.astype(float) / TRADING_DAYS
    cov_d = cov_annual.values.astype(float) / TRADING_DAYS

    rng = np.random.default_rng(seed)
    L = qstats.cholesky(cov_d)
    n = len(tickers)

    # Per-asset paths: simulate log returns, exponentiate, then aggregate to
    # portfolio level using fixed weights (assume daily rebalance to target).
    z = rng.standard_normal(size=(horizon_days, n_paths, n))
    correlated = z @ L.T  # (horizon, paths, n)
    drift = (mu - 0.5 * np.diag(cov_d))  # Ito correction
    daily_log_r = drift + correlated  # (horizon, paths, n)
    daily_simple_r = np.expm1(daily_log_r)  # (horizon, paths, n)
    # daily-rebalanced portfolio simple return = w · r_assets
    port_r = daily_simple_r @ w  # (horizon, paths)
    growth = np.cumprod(1.0 + port_r, axis=0)  # (horizon, paths)
    paths = np.empty((n_paths, horizon_days + 1))
    paths[:, 0] = starting_value
    paths[:, 1:] = starting_value * growth.T
    return paths


def risk_metrics(
    paths: np.ndarray,
    starting_value: float,
    ruin_threshold: float | None = None,
    var_alpha: float = 0.05,
) -> dict:
    """Reduce a paths array into the metrics risk.py reports.

    paths: (n_paths, horizon_days + 1) — output of simulate_portfolio.
    """
    n_paths, horizon_plus = paths.shape
    horizon = horizon_plus - 1
    if ruin_threshold is None:
        ruin_threshold = 0.7 * starting_value

    # 1-day P&L distribution: paths[:, 1] vs starting
    day1 = paths[:, 1] - starting_value
    var_1d = -np.quantile(day1, var_alpha)  # loss exceeded alpha% of the time, positive number
    tail = day1[day1 <= -var_1d]
    cvar_1d = -tail.mean() if tail.size else float(var_1d)

    # max drawdown per path (peak-to-trough on each path)
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max  # negative or zero
    max_dd_per_path = drawdowns.min(axis=1)  # most negative per path
    median_max_dd = float(np.median(max_dd_per_path))  # negative number

    # Sharpe (sim) from terminal log returns
    terminal_log_r = np.log(paths[:, -1] / starting_value)
    ann_factor = TRADING_DAYS / horizon
    mu_ann = float(terminal_log_r.mean() * ann_factor)
    sd_ann = float(terminal_log_r.std(ddof=1) * np.sqrt(ann_factor))
    sharpe = mu_ann / sd_ann if sd_ann > 0 else 0.0

    # Probability of ruin: any path that breaches threshold at any point
    breached = (paths.min(axis=1) <= ruin_threshold).mean()

    return {
        "var_1d_usd": float(var_1d),
        "cvar_1d_usd": float(cvar_1d),
        "median_max_drawdown": median_max_dd,  # negative fraction, e.g. -0.142
        "sim_sharpe": float(sharpe),
        "sim_return_ann": mu_ann,
        "sim_vol_ann": sd_ann,
        "p_ruin": float(breached),
        "ruin_threshold_usd": float(ruin_threshold),
        "n_paths": int(n_paths),
        "horizon_days": int(horizon),
    }


def evaluate(
    weights: pd.Series,
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    starting_value: float,
    n_paths: int = 10_000,
    horizon_days: int = TRADING_DAYS,
    seed: int | None = None,
    ruin_threshold: float | None = None,
) -> dict:
    """Single-call helper used by risk.py."""
    paths = simulate_portfolio(
        weights, mu_annual, cov_annual, starting_value,
        n_paths=n_paths, horizon_days=horizon_days, seed=seed,
    )
    return risk_metrics(paths, starting_value, ruin_threshold=ruin_threshold)
