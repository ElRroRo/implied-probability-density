# Implied Probability Density

Extract the **probability distribution of a stock's future price** directly from
its live options market, and explore it in an interactive, professional UI.

Given a ticker, the app pulls the option chain from `yfinance`, fits the implied
volatility smile, applies the **Breeden–Litzenberger** identity to recover the
risk-neutral density `q(S_T)`, and reweights it through a **CRRA** pricing kernel
(γ = 2.5) to obtain the real-world density `p(S_T)`.

> 📐 For the complete derivation of every formula — risk-neutral valuation,
> Black–Scholes inversion, the Breeden–Litzenberger theorem, the pricing-kernel
> change of measure, the cone, and the IV surface — see **[`MATH.md`](MATH.md)**.

---

## The 4-step pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1. Option data | `src/data.py` | Fetch spot + call chain, filter illiquid / wide-spread strikes, compute mids |
| 2. Smooth c(K) | `src/black_scholes.py`, `src/smoothing.py` | Invert mids → IV, fit a cubic smoothing spline over the smile, rebuild a dense Black–Scholes call curve |
| 3. Breeden–Litzenberger | `src/density.py` | $q(S_T) = e^{r\tau}\,\partial^2 c/\partial K^2$ (central 2nd difference), normalized |
| 4. CRRA transform | `src/density.py` | $p(S_T) \propto S_T^{\gamma}\,q(S_T)$, renormalized so $\int p = 1$ |

The risk-free rate `r` is interpolated from a fetched US Treasury term structure
(`src/rates.py`) to each expiry's tenor τ. `src/pipeline.py` is the only glue
between the steps; `compute_surface` extends Step 1–4 across many expiries to
build the term structure used by the cone and surface.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter a liquid ticker (SPY, AAPL, QQQ…), pick an expiration, choose a timeframe,
and adjust the smoothing slider. Toggle Light/Dark at the top-right.

### Use the model without the UI

```python
from src.pipeline import compute_densities, compute_surface

res = compute_densities("SPY", expiry=None)   # None = first listed expiry
print(res.summary)        # spot, forward, r, τ, means/stds, integration checks
res.grid, res.q, res.p    # numpy arrays: terminal price, q-density, p-density

surf = compute_surface("SPY", ["2026-07-17", "2026-08-21"])
surf.price, surf.prob     # shared price grid, (price × expiry) density matrix
```

---

## Interface

- **Light / Dark theme toggle** (top-right). Clean, professional palette —
  green/red candles, blue risk-neutral curve, violet real-world curve and cone,
  cool→warm volatility surface.
- **Price chart + forward probability cone.** Candlesticks (1H / 1D / 1W) with a
  term-structure cone: each listed expiry contributes its own density at its real
  date, interpolated over time so the cloud fans out. TradingView-style mechanics
  — scroll over the body to zoom time (price auto-fits), scroll over an axis to
  scale just that axis, drag to pan, crosshair, weekend gaps collapsed. The cone
  is computed **once per ticker** and cached for the session — switching
  timeframes only redraws candles. Use **Refresh** to re-pull live data.
- **Probability calculator.** Pick one or more expiry dates and query the
  cumulative probability *between*, *below*, or *above* chosen prices, under the
  real-world (p) or risk-neutral (q) measure. The selection is shaded on the
  density chart.
- **3D implied-volatility surface.** Fitted IV across moneyness (K/S) and
  maturity, from the cone's expiries. Drag to rotate, scroll to zoom; gaps mark
  strikes that don't trade at a given maturity.

---

## Project layout

```
app.py                 Streamlit UI, theming, chart builders
src/
├── data.py            Step 1: option chain, OHLC history, filtering
├── black_scholes.py   Black–Scholes price/vega + implied-vol inversion
├── smoothing.py       Step 2: smoothing spline → dense call curve c(K)
├── density.py         Step 3 (Breeden–Litzenberger) + Step 4 (CRRA) + probabilities
├── rates.py           Risk-free term structure
└── pipeline.py        Orchestration: compute_densities, compute_surface
MATH.md                Full mathematical derivation
```

---

## Notes & caveats

- **γ = 2.5 is hardcoded** (`src/__init__.py: GAMMA`).
- $p \propto S^{\gamma} q$ is the standard risk-neutral→physical transform; it
  lifts the right tail, so the real-world mean sits **above** the forward — the
  equity risk premium. (See [`MATH.md` §10](MATH.md#10-crra-utility-and-the-change-of-measure).)
- Data is **delayed** (Yahoo Finance) and quality depends on liquidity; the
  Step-1 filter drops zero-volume / wide-spread strikes before differentiating.
- Densities are supported only across **traded** strikes (no tail extrapolation),
  and the cone's between-expiry interpolation is a display convenience, not a
  no-arbitrage term-structure model. Full list in
  [`MATH.md` §15](MATH.md#15-assumptions-error-sources-and-limitations).

---

## Disclaimer

For research and educational use only. Nothing here is investment advice. Option
data is delayed and the extracted densities are model estimates, not guarantees.
