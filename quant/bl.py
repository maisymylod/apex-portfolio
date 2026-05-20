"""Black-Litterman optimizer.

    μ_BL = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 Π + P'Ω^-1 q]
    w*   = (δΣ)^-1 μ_BL      (mean-variance weights given posterior μ)

Π is the equilibrium return vector — reverse-optimized from the prior
weights via Π = δ Σ w_prior. δ is the implied risk-aversion (default 2.5).

Views are described in plain English upstream and converted into (P, q, Ω)
by the agent layer. This module just does the math.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TAU = 0.05  # scalar uncertainty in the prior
DEFAULT_DELTA = 2.5  # risk aversion (Litterman: 2.5 for equities)
CONFIDENCE_OMEGA = {"HIGH": 0.02**2, "MED": 0.05**2, "LOW": 0.10**2}


@dataclass
class View:
    """One Black-Litterman view.

    pick:        dict ticker -> coefficient (signed). e.g. {"NVDA": 1, "SPY": -1}
    q:           expected return of the linear combination, annualized.
    confidence:  "HIGH" | "MED" | "LOW" — maps to Ω diagonal entry.
    """
    pick: dict[str, float]
    q: float
    confidence: str = "MED"


def equilibrium_returns(w_prior: pd.Series, cov: pd.DataFrame, delta: float = DEFAULT_DELTA) -> pd.Series:
    """Reverse-optimize: given prior weights and Σ, recover Π = δ Σ w_prior."""
    w = w_prior.reindex(cov.columns).fillna(0.0).values.astype(float)
    pi = delta * cov.values.astype(float) @ w
    return pd.Series(pi, index=cov.columns)


def _pick_matrix(views: list[View], tickers: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build P (k×n), q (k,), Ω diagonal (k,) from a list of View objects."""
    k = len(views)
    n = len(tickers)
    P = np.zeros((k, n))
    q = np.zeros(k)
    omega_diag = np.zeros(k)
    idx = {t: i for i, t in enumerate(tickers)}
    for r, v in enumerate(views):
        for tkr, coef in v.pick.items():
            if tkr in idx:
                P[r, idx[tkr]] = coef
        q[r] = v.q
        omega_diag[r] = CONFIDENCE_OMEGA.get(v.confidence.upper(), CONFIDENCE_OMEGA["MED"])
    return P, q, omega_diag


def posterior_returns(
    pi: pd.Series,
    cov: pd.DataFrame,
    views: list[View],
    tau: float = DEFAULT_TAU,
) -> pd.Series:
    """Compute μ_BL given prior Π, covariance Σ, and a list of views.

    Returns a Series indexed like Π. If `views` is empty, returns Π.
    """
    tickers = list(cov.columns)
    pi_arr = pi.reindex(tickers).fillna(0.0).values.astype(float)
    sigma = cov.values.astype(float)
    if not views:
        return pd.Series(pi_arr, index=tickers)
    P, q, omega_diag = _pick_matrix(views, tickers)
    omega = np.diag(omega_diag)
    tau_sigma_inv = np.linalg.inv(tau * sigma)
    omega_inv = np.linalg.inv(omega)
    a = tau_sigma_inv + P.T @ omega_inv @ P
    b = tau_sigma_inv @ pi_arr + P.T @ omega_inv @ q
    mu_bl = np.linalg.solve(a, b)
    return pd.Series(mu_bl, index=tickers)


def optimal_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    delta: float = DEFAULT_DELTA,
    long_only: bool = True,
    weight_cap: float = 0.20,
) -> pd.Series:
    """Mean-variance weights w* = (δΣ)^-1 μ, normalized to sum to 1.

    Optionally enforce long-only (clip negatives) and a per-name cap. After
    clipping/capping we renormalize. This is intentionally simple; we are
    not running a QP here.
    """
    tickers = list(cov.columns)
    sigma = cov.values.astype(float)
    mu_arr = mu.reindex(tickers).fillna(0.0).values.astype(float)
    raw = np.linalg.solve(delta * sigma, mu_arr)
    w = pd.Series(raw, index=tickers)
    if long_only:
        w = w.clip(lower=0.0)
    if w.sum() <= 0:
        # fallback: equal weight if the math degenerates (e.g. all-negative μ)
        w = pd.Series(1.0 / len(tickers), index=tickers)
    w = w / w.sum()
    if weight_cap is not None:
        for _ in range(50):  # iterative cap-and-renormalize
            over = w > weight_cap
            if not over.any():
                break
            excess = (w[over] - weight_cap).sum()
            w[over] = weight_cap
            under = ~over
            if w[under].sum() > 0:
                w[under] = w[under] + excess * w[under] / w[under].sum()
            else:
                break
    return w / w.sum()


def apply_bias(weights: pd.Series, biases: dict[str, float]) -> pd.Series:
    """Multiply by (1 + bias) and renormalize. Used by the learning loop."""
    factors = pd.Series({t: 1.0 + biases.get(t, 0.0) for t in weights.index})
    scaled = weights * factors
    s = scaled.sum()
    return scaled / s if s > 0 else weights
