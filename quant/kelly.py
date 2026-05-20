"""Kelly criterion position sizer.

Mean-variance Kelly for a single asset:
    f* = (μ - r) / σ²

For a portfolio, the full-Kelly vector is f* = Σ^-1 (μ - r·1). We use
HALF-Kelly throughout (standard practice for parameter uncertainty), and
cap any single weight at `max_per_name`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALF = 0.5


def kelly_weights(
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    risk_free: float = 0.045,
    fraction: float = HALF,
    max_per_name: float = 0.20,
) -> pd.Series:
    """Half-Kelly weight vector. Normalizes to sum to 1 if total > 1."""
    tickers = list(cov_annual.columns)
    mu = mu_annual.reindex(tickers).fillna(0.0).values.astype(float)
    sigma = cov_annual.values.astype(float)
    excess = mu - risk_free
    try:
        raw = np.linalg.solve(sigma, excess)
    except np.linalg.LinAlgError:
        raw = excess / np.diag(sigma).clip(min=1e-6)
    w = pd.Series(raw * fraction, index=tickers)
    w = w.clip(lower=0.0)
    w = w.clip(upper=max_per_name)
    s = w.sum()
    if s > 1.0:
        w = w / s
    return w


def blend(bl_weights: pd.Series, kelly: pd.Series, alpha: float = 0.5) -> pd.Series:
    """Convex combination of BL weights and Kelly weights, then renormalize."""
    idx = bl_weights.index.union(kelly.index)
    a = bl_weights.reindex(idx).fillna(0.0)
    b = kelly.reindex(idx).fillna(0.0)
    blended = alpha * a + (1 - alpha) * b
    s = blended.sum()
    return blended / s if s > 0 else a
