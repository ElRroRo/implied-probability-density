"""Risk-free term structure from US Treasury yields (via yfinance).

Option expirations vary, so we fetch several constant-maturity Treasury
yields and interpolate the continuously-relevant rate to each option's
time-to-expiry tau.
"""
from __future__ import annotations

import numpy as np
import yfinance as yf

# (yfinance symbol, tenor in years). ^IRX=13wk, ^FVX=5y, ^TNX=10y, ^TYX=30y.
_TENORS = [("^IRX", 0.25), ("^FVX", 5.0), ("^TNX", 10.0), ("^TYX", 30.0)]
_DEFAULT_RATE = 0.04


def fetch_curve() -> tuple[np.ndarray, np.ndarray]:
    """Return (tenors, rates) sorted by tenor. Rates are decimals.

    Symbols quote yields in percent, so we divide by 100. Any symbol that
    fails to fetch is skipped; if all fail we return a flat default curve.
    """
    tenors, rates = [], []
    for sym, tenor in _TENORS:
        try:
            hist = yf.Ticker(sym).history(period="5d")
            val = float(hist["Close"].dropna().iloc[-1]) / 100.0
            if np.isfinite(val) and val > 0:
                tenors.append(tenor)
                rates.append(val)
        except Exception:
            continue

    if not tenors:
        return np.array([0.25, 30.0]), np.array([_DEFAULT_RATE, _DEFAULT_RATE])

    order = np.argsort(tenors)
    return np.asarray(tenors)[order], np.asarray(rates)[order]


def rate_for_tenor(tau: float, curve: tuple[np.ndarray, np.ndarray] | None = None) -> float:
    """Linearly interpolate the risk-free rate at maturity ``tau`` (years).

    Flat extrapolation beyond the shortest/longest available tenor.
    """
    tenors, rates = curve if curve is not None else fetch_curve()
    if len(tenors) == 1:
        return float(rates[0])
    return float(np.interp(tau, tenors, rates))
