"""Steps 3 & 4 - Risk-neutral and real-world densities.

Step 3 applies the Breeden-Litzenberger identity to the smooth call curve
to extract the risk-neutral density q(S_T). Step 4 reweights it by the
CRRA marginal utility (gamma = 2.5) to recover the real-world density
p(S_T).
"""
from __future__ import annotations

import numpy as np


def risk_neutral_density(call: np.ndarray, dX: float, r: float, tau: float) -> np.ndarray:
    """Breeden-Litzenberger: q(S_T) = e^{r*tau} * d^2c/dX^2.

    Central second finite difference. Endpoints are padded by copying the
    nearest interior value so the output matches the grid length. Tiny
    negative values (numerical noise) are clipped to zero.
    """
    call = np.asarray(call, float)
    q = np.empty_like(call)
    q[1:-1] = (call[2:] - 2.0 * call[1:-1] + call[:-2]) / (dX ** 2)
    q[0], q[-1] = q[1], q[-2]
    q *= np.exp(r * tau)
    return np.clip(q, 0.0, None)


def normalize(density: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Scale a density so it integrates to 1 over the grid (trapezoid)."""
    area = np.trapezoid(density, grid)
    if area <= 0:
        return density
    return density / area


def crra_transform(grid: np.ndarray, q: np.ndarray, gamma: float = 2.5) -> np.ndarray:
    """Step 4 - CRRA reweighting: p(S_T) ∝ S_T^gamma * q(S_T).

    A risk-averse investor (gamma>0) fears down-moves, so this lifts mass on
    low terminal prices and trims aggressive up-moves. Renormalised to
    integrate to exactly 1.
    """
    grid = np.asarray(grid, float)
    weighted = (grid ** gamma) * np.asarray(q, float)
    return normalize(weighted, grid)


def interval_probability(grid: np.ndarray, density: np.ndarray,
                         lo: float | None = None, hi: float | None = None) -> float:
    """Cumulative probability that the price lands in [lo, hi].

    ``lo``/``hi`` of None mean unbounded on that side. Partial edge bins are
    handled by interpolation so the result is accurate for arbitrary bounds.
    Returns a probability in [0, 1].
    """
    g = np.asarray(grid, float)
    d = normalize(np.asarray(density, float), g)
    a = g[0] if lo is None else max(float(lo), g[0])
    b = g[-1] if hi is None else min(float(hi), g[-1])
    if b <= a:
        return 0.0
    inside = g[(g > a) & (g < b)]
    xs = np.concatenate([[a], inside, [b]])
    ys = np.interp(xs, g, d)
    return float(np.clip(np.trapezoid(ys, xs), 0.0, 1.0))


def moments(grid: np.ndarray, density: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) of a density defined on ``grid``."""
    grid = np.asarray(grid, float)
    p = normalize(np.asarray(density, float), grid)
    mean = np.trapezoid(grid * p, grid)
    var = np.trapezoid((grid - mean) ** 2 * p, grid)
    return float(mean), float(np.sqrt(max(var, 0.0)))
