"""Step 2 - Continuous call-price function c(X).

The Breeden-Litzenberger identity needs a second derivative of call price
w.r.t. strike, so the discrete market quotes must be turned into a smooth
curve. We fit a smoothing cubic spline across strike-implied-vol space,
then convert that smooth smile back into call prices via Black-Scholes on
a dense strike grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline

from .black_scholes import bs_call, implied_vol


@dataclass
class PriceCurve:
    """Dense, differentiable call-price curve and the smile behind it."""
    grid: np.ndarray   # dense strikes
    call: np.ndarray   # c(X) on the grid
    iv: np.ndarray     # fitted implied vol on the grid
    dX: float          # grid step
    knots_K: np.ndarray  # observed strikes used for the fit
    knots_iv: np.ndarray  # implied vols at those strikes


def invert_smile(strikes, mid, spot, r, tau, iv_market=None):
    """Invert market mid-prices into implied vols.

    Uses our own Black-Scholes inversion for consistency, falling back to
    Yahoo's reported IV where the solver fails. Returns the strikes/ivs
    that ended up usable.
    """
    strikes = np.asarray(strikes, float)
    mid = np.asarray(mid, float)
    iv = np.array([implied_vol(p, spot, k, r, tau) for p, k in zip(mid, strikes)])

    if iv_market is not None:
        iv_market = np.asarray(iv_market, float)
        fallback = ~np.isfinite(iv) & np.isfinite(iv_market) & (iv_market > 0)
        iv[fallback] = iv_market[fallback]

    good = np.isfinite(iv) & (iv > 0)
    return strikes[good], iv[good]


def fit_smile(strikes, mid, spot, r, tau, smoothing=None, iv_market=None,
              dX=None, pad=0.0):
    """Fit the vol smile and rebuild a dense call-price curve.

    Parameters
    ----------
    smoothing : float or None
        Multiplier on a data-scaled base ``s`` factor. ``None`` -> 1.0
        (the gentle auto default). 0 -> interpolating spline; larger ->
        smoother (more bias, less noise).
    dX : float or None
        Dense grid step. ``None`` -> ~400 points across the strike range.
    pad : float
        Fractional extension beyond observed strikes (0 = no extrapolation).
    """
    K, iv = invert_smile(strikes, mid, spot, r, tau, iv_market)
    if len(K) < 4:
        raise ValueError("Not enough valid implied-vol points to fit a smile.")

    k = min(3, len(K) - 1)  # spline degree (cubic when possible)
    base_s = len(K) * np.nanvar(iv) * 0.5  # gentle data-scaled default
    mult = 1.0 if smoothing is None else float(smoothing)
    spline = UnivariateSpline(K, iv, k=k, s=base_s * mult)

    lo, hi = K.min(), K.max()
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    if dX is None:
        dX = span / 400.0
    grid = np.arange(lo, hi + dX, dX)

    iv_grid = np.clip(spline(grid), 1e-4, None)
    call = bs_call(spot, grid, r, iv_grid, tau)

    return PriceCurve(grid=grid, call=np.asarray(call, float), iv=iv_grid,
                      dX=float(dX), knots_K=K, knots_iv=iv)
