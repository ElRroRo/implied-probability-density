"""End-to-end orchestration of the 4-step pipeline.

    Step 1  data.get_snapshot       option chain + filtering
    Step 2  smoothing.fit_smile     continuous c(X) curve
    Step 3  density.risk_neutral_density   q(S_T)
    Step 4  density.crra_transform        p(S_T)

This module is the only glue between the otherwise-independent math steps,
keeping the foundation easy to extend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import GAMMA
from . import data, density, rates, smoothing


@dataclass
class DensityResult:
    ticker: str
    expiry: str
    spot: float
    r: float
    tau: float          # years to expiry
    gamma: float
    grid: np.ndarray    # terminal price grid S_T
    q: np.ndarray       # risk-neutral density
    p: np.ndarray       # real-world (CRRA) density
    curve: smoothing.PriceCurve = field(repr=False, default=None)
    summary: dict = field(default_factory=dict)


def _year_fraction(expiry: str) -> float:
    exp = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    days = max((exp - now).total_seconds() / 86400.0, 1.0 / 24)  # >= ~1h
    return days / 365.0


def compute_densities(ticker: str, expiry: str | None = None,
                      spline_s: float | None = None,
                      gamma: float = GAMMA,
                      rate_curve=None,
                      spot: float | None = None) -> DensityResult:
    """Run all four steps and return densities + summary stats."""
    # Step 1 -------------------------------------------------------------
    snap = data.get_snapshot(ticker, expiry, spot=spot)
    tau = _year_fraction(snap.expiry)
    r = rates.rate_for_tenor(tau, rate_curve)

    # Step 2 -------------------------------------------------------------
    curve = smoothing.fit_smile(snap.strikes, snap.mid, snap.spot, r, tau,
                                smoothing=spline_s, iv_market=snap.iv_market)

    # Step 3 -------------------------------------------------------------
    q = density.risk_neutral_density(curve.call, curve.dX, r, tau)
    q = density.normalize(q, curve.grid)

    # Step 4 -------------------------------------------------------------
    p = density.crra_transform(curve.grid, q, gamma=gamma)

    q_mean, q_std = density.moments(curve.grid, q)
    p_mean, p_std = density.moments(curve.grid, p)
    forward = snap.spot * np.exp(r * tau)

    summary = {
        "spot": snap.spot,
        "forward": forward,
        "r": r,
        "tau_days": tau * 365.0,
        "gamma": gamma,
        "q_mean": q_mean, "q_std": q_std,
        "p_mean": p_mean, "p_std": p_std,
        "n_strikes": len(snap.strikes),
        "q_area": float(np.trapezoid(q, curve.grid)),
        "p_area": float(np.trapezoid(p, curve.grid)),
    }

    return DensityResult(
        ticker=snap.ticker, expiry=snap.expiry, spot=snap.spot, r=r, tau=tau,
        gamma=gamma, grid=curve.grid, q=q, p=p, curve=curve, summary=summary,
    )


@dataclass
class SurfaceResult:
    """A term structure of densities = the forward probability cone."""
    ticker: str
    spot: float
    price: np.ndarray            # common price axis (shared by all expiries)
    prob: np.ndarray             # (len(price), n_expiry) real-world density p
    taus: np.ndarray             # years-to-expiry per column
    expiries: list[str]          # expiry date string per column
    results: list = field(default_factory=list, repr=False)  # per-expiry DensityResult


def _support(grid: np.ndarray, p: np.ndarray, lo_q=0.01, hi_q=0.99) -> tuple[float, float]:
    """Central [lo_q, hi_q] price range of a density (trims extreme tails)."""
    cdf = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(grid))])
    if cdf[-1] <= 0:
        return float(grid[0]), float(grid[-1])
    cdf /= cdf[-1]
    return float(np.interp(lo_q, cdf, grid)), float(np.interp(hi_q, cdf, grid))


def compute_surface(ticker: str, expiries: list[str],
                    spline_s: float | None = None, gamma: float = GAMMA,
                    rate_curve=None, n_price: int = 240) -> SurfaceResult:
    """Compute one density per expiry and resample them onto a shared price
    grid, producing the forward probability cone across time.

    Expiries that fail (thin/illiquid chains) are skipped.
    """
    curve = rate_curve if rate_curve is not None else rates.fetch_curve()
    spot = data.get_spot(ticker)

    results: list[DensityResult] = []
    for exp in expiries:
        try:
            results.append(compute_densities(ticker, exp, spline_s, gamma,
                                             curve, spot=spot))
        except Exception:
            continue
    if not results:
        raise ValueError(f"No usable expiries for {ticker}.")

    results.sort(key=lambda r: r.tau)

    # Shared price axis from the union of per-expiry central supports, so the
    # cone widens with horizon but isn't dominated by extreme far-dated tails.
    lows, highs = zip(*(_support(r.grid, r.p) for r in results))
    price = np.linspace(min(lows), max(highs), n_price)

    prob = np.zeros((n_price, len(results)))
    for j, r in enumerate(results):
        prob[:, j] = np.interp(price, r.grid, r.p, left=0.0, right=0.0)

    return SurfaceResult(
        ticker=results[0].ticker, spot=spot, price=price, prob=prob,
        taus=np.array([r.tau for r in results]),
        expiries=[r.expiry for r in results], results=results,
    )
