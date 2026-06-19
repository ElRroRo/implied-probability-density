"""Black-Scholes call pricing and implied-volatility inversion.

Used in Step 2 to (a) invert market mid-prices into implied vols and
(b) convert the smooth, fitted vol smile back into a continuous call
price curve c(X).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_call(S: float, K, r: float, sigma, tau: float) -> np.ndarray:
    """Black-Scholes price of a European call.

    Vectorised over K and/or sigma. Handles tau<=0 and sigma<=0 by
    returning the intrinsic value.
    """
    S = float(S)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    intrinsic = np.maximum(S - K * np.exp(-r * tau), 0.0)
    if tau <= 0:
        return intrinsic

    with np.errstate(divide="ignore", invalid="ignore"):
        sig_sqrt = sigma * np.sqrt(tau)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / sig_sqrt
        d2 = d1 - sig_sqrt
        price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    # Fall back to intrinsic where sigma is non-positive / numerically bad.
    price = np.where(sig_sqrt > 0, price, intrinsic)
    return price


def bs_vega(S: float, K, r: float, sigma, tau: float) -> np.ndarray:
    """Black-Scholes vega (dPrice/dSigma)."""
    S = float(S)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if tau <= 0:
        return np.zeros_like(K)
    sig_sqrt = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / sig_sqrt
    return S * norm.pdf(d1) * np.sqrt(tau)


def implied_vol(price: float, S: float, K: float, r: float, tau: float,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Invert a single call mid-price into an implied vol via Brent.

    Returns ``np.nan`` if the price is outside no-arbitrage bounds or the
    solver fails to bracket a root.
    """
    if tau <= 0 or price <= 0:
        return np.nan

    intrinsic = max(S - K * np.exp(-r * tau), 0.0)
    upper = S  # call price can never exceed spot
    if price <= intrinsic + 1e-8 or price >= upper:
        return np.nan

    def objective(sigma):
        return float(bs_call(S, K, r, sigma, tau)) - price

    try:
        return brentq(objective, lo, hi, maxiter=100, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan
