# Mathematical Breakdown

A complete derivation of every step in the pipeline, from raw option quotes to
the real-world probability density, the forward probability cone, and the
implied-volatility surface.

---

## Table of contents

1. [Notation and setup](#1-notation-and-setup)
2. [Risk-neutral valuation](#2-risk-neutral-valuation)
3. [The Black–Scholes call price](#3-the-blackscholes-call-price)
4. [Implied volatility inversion](#4-implied-volatility-inversion)
5. [The volatility smile and the smoothing spline](#5-the-volatility-smile-and-the-smoothing-spline)
6. [The Breeden–Litzenberger theorem](#6-the-breedenlitzenberger-theorem)
7. [Numerical second derivative](#7-numerical-second-derivative)
8. [The risk-free term structure](#8-the-risk-free-term-structure)
9. [From risk-neutral to real-world: the pricing kernel](#9-from-risk-neutral-to-real-world-the-pricing-kernel)
10. [CRRA utility and the change of measure](#10-crra-utility-and-the-change-of-measure)
11. [Normalization](#11-normalization)
12. [Moments and cumulative probabilities](#12-moments-and-cumulative-probabilities)
13. [The forward probability cone](#13-the-forward-probability-cone)
14. [The implied-volatility surface](#14-the-implied-volatility-surface)
15. [Assumptions, error sources, and limitations](#15-assumptions-error-sources-and-limitations)
16. [Symbol glossary](#16-symbol-glossary)

---

## 1. Notation and setup

We fix a single underlying asset and a single valuation instant $t=0$.

| Symbol | Meaning |
|--------|---------|
| $S_0$ | current spot price of the underlying |
| $S_T$ | (random) price of the underlying at maturity $T$ |
| $\tau = T - t$ | time to expiry, in years |
| $K$ (or $X$) | option strike |
| $r$ | continuously-compounded risk-free rate for horizon $\tau$ |
| $\sigma$ | volatility (Black–Scholes parameter) |
| $c(K)$ | price of a European **call** with strike $K$ |
| $q(S_T)$ | **risk-neutral** density of the terminal price |
| $p(S_T)$ | **real-world** (physical) density of the terminal price |
| $\gamma$ | coefficient of relative risk aversion (fixed at $2.5$) |

We work with **European** options (exercise only at $T$), which is what the
Breeden–Litzenberger identity requires. Listed US equity options are American,
which is a known approximation discussed in §15.

---

## 2. Risk-neutral valuation

Under the **risk-neutral measure** $\mathbb{Q}$, the price of any European
claim equals the discounted expectation of its payoff. For a call with strike
$K$ and payoff $(S_T - K)^+$:

$$
c(K) \;=\; e^{-r\tau}\, \mathbb{E}^{\mathbb{Q}}\!\left[(S_T - K)^+\right]
\;=\; e^{-r\tau} \int_{0}^{\infty} (S - K)^+ \, q(S)\, dS .
$$

Because $(S-K)^+ = 0$ for $S < K$, the lower limit collapses to $K$:

$$
\boxed{\;c(K) \;=\; e^{-r\tau} \int_{K}^{\infty} (S - K)\, q(S)\, dS\;}
\quad (2.1)
$$

This single equation is the foundation: it links the **observable** call-price
function $c(K)$ to the **unobservable** density $q$. Breeden–Litzenberger (§6)
inverts it.

The risk-neutral density is a genuine probability density:

$$
q(S) \ge 0, \qquad \int_0^\infty q(S)\,dS = 1,
$$

and the discounted underlying is a $\mathbb{Q}$-martingale, so the mean of $q$
is the **forward price**:

$$
\mathbb{E}^{\mathbb{Q}}[S_T] = \int_0^\infty S\,q(S)\,dS = S_0 e^{r\tau} =: F.
\quad (2.2)
$$

Equation (2.2) is the model's primary internal sanity check (the code reports
`q_mean` vs `forward`).

---

## 3. The Black–Scholes call price

Black–Scholes assumes the underlying follows geometric Brownian motion under
$\mathbb{Q}$:

$$
dS_t = r\,S_t\,dt + \sigma\,S_t\,dW_t^{\mathbb{Q}},
$$

which gives a **lognormal** terminal distribution:

$$
\ln S_T \sim \mathcal{N}\!\left(\ln S_0 + \left(r - \tfrac{1}{2}\sigma^2\right)\tau,\; \sigma^2\tau\right).
$$

Substituting this lognormal $q$ into (2.1) and integrating yields the closed
form:

$$
\boxed{\;c_{\mathrm{BS}}(S_0,K,r,\sigma,\tau) = S_0\,\Phi(d_1) - K e^{-r\tau}\,\Phi(d_2)\;}
\quad (3.1)
$$

$$
d_1 = \frac{\ln(S_0/K) + \left(r + \tfrac{1}{2}\sigma^2\right)\tau}{\sigma\sqrt{\tau}},
\qquad
d_2 = d_1 - \sigma\sqrt{\tau},
$$

where $\Phi$ is the standard normal CDF. The **vega** (sensitivity to
volatility), needed for the inversion below and always positive, is

$$
\frac{\partial c_{\mathrm{BS}}}{\partial \sigma} = S_0\,\phi(d_1)\sqrt{\tau} \;>\; 0,
\quad (3.2)
$$

with $\phi$ the standard normal PDF.

> In the code, (3.1)–(3.2) are `black_scholes.bs_call` and `bs_vega`. We do
> **not** assume the market is Black–Scholes — we use BS only as an invertible,
> monotonic "unit converter" between price and implied volatility (§4–§5).

---

## 4. Implied volatility inversion

The market quotes prices, not volatilities. For each strike $K_i$ with market
mid-price $c_i^{\mathrm{mkt}} = \tfrac12(\text{bid}+\text{ask})$, the **implied
volatility** $\sigma_i$ is the unique value solving

$$
c_{\mathrm{BS}}(S_0, K_i, r, \sigma_i, \tau) = c_i^{\mathrm{mkt}}.
\quad (4.1)
$$

Uniqueness is guaranteed because $c_{\mathrm{BS}}$ is **strictly increasing**
in $\sigma$ (vega $>0$ in (3.2)), with no-arbitrage bounds

$$
(S_0 - K e^{-r\tau})^+ \;\le\; c^{\mathrm{mkt}} \;\le\; S_0 .
$$

We solve (4.1) numerically with Brent's method (`black_scholes.implied_vol`,
bracketing $\sigma \in [10^{-4}, 5]$). Quotes outside the no-arbitrage band, or
where the solver fails, fall back to Yahoo's reported IV and are otherwise
dropped.

**Why convert to IV at all?** Prices vary over orders of magnitude across
strikes (deep ITM calls cost ~$S_0$, deep OTM ~0), so smoothing them directly
is ill-conditioned. Implied vols live in a narrow band (≈ 0.1–1.0) and form a
smooth, gently-curved "smile," which is far better behaved for interpolation.

---

## 5. The volatility smile and the smoothing spline

We have discrete pairs $(K_i, \sigma_i)$. Breeden–Litzenberger needs a
**twice-differentiable** call-price function, so we first build a smooth
volatility function $\sigma(K)$.

We fit a **cubic smoothing spline** $\hat\sigma(K)$ (degree $k=3$) minimizing

$$
\sum_{i} \big(\hat\sigma(K_i) - \sigma_i\big)^2 \;\le\; s,
\quad (5.1)
$$

where $s$ is a smoothing budget. With $s=0$ the spline interpolates every point
(noisy second derivative); larger $s$ trades fidelity for smoothness. We scale a
data-driven default,

$$
s = m \cdot n \cdot \widehat{\text{Var}}(\sigma_i) \cdot \tfrac12,
$$

where $n$ is the number of strikes and $m\in[0,2]$ is the UI "smoothing"
multiplier (`smoothing.fit_smile`). Cubic order matters: the call price's
**second** derivative inherits two derivatives' worth of roughness from the
smile, so a $C^2$ smile keeps $q$ continuous.

We then evaluate the smile on a **dense strike grid** $\{K_j\}$ (≈ 400 points,
step $\Delta K$) spanning the observed strike range — **no extrapolation beyond
traded strikes** — and convert back to prices with Black–Scholes:

$$
\boxed{\;c(K_j) \;=\; c_{\mathrm{BS}}\!\big(S_0,\,K_j,\,r,\,\hat\sigma(K_j),\,\tau\big)\;}
\quad (5.2)
$$

The result is a smooth, dense, arbitrage-respecting call curve $c(K)$ ready for
differentiation. (Using BS per-strike with its *own* implied vol reproduces the
market price exactly at the knots; it is **not** an assumption that a single
$\sigma$ governs all strikes.)

---

## 6. The Breeden–Litzenberger theorem

**Claim.**

$$
\boxed{\;q(S_T) \;=\; e^{r\tau}\,\frac{\partial^2 c(K)}{\partial K^2}\bigg|_{K=S_T}\;}
\quad (6.1)
$$

**Derivation.** Start from (2.1) and differentiate with respect to $K$. The
integrand and the lower limit both depend on $K$, so we use the Leibniz rule:

$$
\frac{\partial}{\partial K}\int_{K}^{\infty} g(S,K)\,dS
= -\,g(K,K) + \int_{K}^{\infty} \frac{\partial g}{\partial K}\,dS .
$$

With $g(S,K) = e^{-r\tau}(S-K)q(S)$:

- the boundary term vanishes because $g(K,K) = e^{-r\tau}(K-K)q(K) = 0$;
- inside, $\dfrac{\partial}{\partial K}(S-K) = -1$.

Hence the **first derivative** is

$$
\frac{\partial c}{\partial K}
= -\,e^{-r\tau}\int_{K}^{\infty} q(S)\,dS
= -\,e^{-r\tau}\,\big[1 - Q(K)\big]
= e^{-r\tau}\big[Q(K) - 1\big],
\quad (6.2)
$$

where $Q(K)=\int_0^K q(S)\,dS$ is the risk-neutral CDF. Equation (6.2) is itself
useful: it says the (undiscounted) risk-neutral probability of finishing in the
money is

$$
\mathbb{Q}(S_T > K) = 1 - Q(K) = -\,e^{r\tau}\,\frac{\partial c}{\partial K}.
$$

Differentiating (6.2) once more, using $Q'(K) = q(K)$:

$$
\frac{\partial^2 c}{\partial K^2} = e^{-r\tau}\,q(K).
$$

Solving for $q$ and renaming the evaluation point $K = S_T$ gives (6.1). $\;\blacksquare$

Intuitively, a *tight* butterfly spread — long calls at $K-\Delta K$ and
$K+\Delta K$, short two at $K$ — pays off only if $S_T \approx K$. Its
(discounted) price per unit width² is exactly the probability mass near $K$,
i.e. the density. Equation (6.1) is the continuum limit of that butterfly.

---

## 7. Numerical second derivative

On the dense uniform grid (step $\Delta K$) we approximate (6.1) with the
**central second difference**:

$$
\boxed{\;q(K_j) \;\approx\; e^{r\tau}\,\frac{c(K_{j+1}) - 2\,c(K_j) + c(K_{j-1})}{(\Delta K)^2}\;}
\quad (7.1)
$$

A Taylor expansion shows this is second-order accurate:

$$
\frac{c_{j+1} - 2c_j + c_{j-1}}{(\Delta K)^2}
= c''(K_j) + \frac{(\Delta K)^2}{12}\,c^{(4)}(K_j) + O\!\big((\Delta K)^4\big).
$$

There is a genuine tension here: shrinking $\Delta K$ reduces the
**truncation** error $\propto (\Delta K)^2$ but amplifies **noise** in $c$
(differentiation is ill-posed; errors scale like $1/(\Delta K)^2$). The smile
smoothing of §5 is what makes the small-$\Delta K$ regime usable.

Post-processing:

- **Clip negatives.** Tiny $q<0$ from numerical noise (or residual butterfly
  arbitrage in the quotes) are set to $0$.
- **Renormalize** (see §11) so $\int q\,dK = 1$ on the grid.

Endpoints (where the central stencil is unavailable) copy the nearest interior
value; their contribution is negligible after normalization.

---

## 8. The risk-free term structure

Each expiry has its own horizon $\tau$, so a single rate is wrong. We build a
discount curve from US Treasury constant-maturity yields:

| Instrument | Tenor |
|------------|-------|
| `^IRX` | 0.25 y (13-week T-bill) |
| `^FVX` | 5 y |
| `^TNX` | 10 y |
| `^TYX` | 30 y |

Yields (quoted in percent) are converted to decimals, $\rho_k = y_k/100$, and
the rate at horizon $\tau$ is obtained by **linear interpolation** on the
$(\text{tenor}, \rho)$ points, with flat extrapolation past the ends:

$$
r(\tau) = \text{interp}\big(\tau;\ \{(\text{tenor}_k, \rho_k)\}\big).
\quad (8.1)
$$

This $r(\tau)$ feeds the discounting in (3.1), (5.2), (6.1), (7.1) and the
forward in (2.2). If the fetch fails entirely, a flat $r=0.04$ fallback is used.

---

## 9. From risk-neutral to real-world: the pricing kernel

The risk-neutral density $q$ already prices everything, so why transform it? Because $q$ is **not** the distribution the world actually follows. $\mathbb{Q}$
embeds risk compensation: it over-weights bad states (low $S_T$) relative to
their true likelihood, which is why insurance-like OTM puts look "expensive."
The **real-world** (physical) density $p$ is what we want for statements like
"the probability the stock finishes above \$X."

The two measures are linked by the **stochastic discount factor** (pricing
kernel) $M$. For any payoff $h(S_T)$, price equals the *physical* expectation of
discounted, kernel-weighted payoff:

$$
e^{-r\tau}\,\mathbb{E}^{\mathbb{Q}}[h(S_T)]
= \mathbb{E}^{\mathbb{P}}\!\big[M(S_T)\,h(S_T)\big].
$$

Writing both expectations as integrals against their densities,

$$
e^{-r\tau}\!\int h(S)\,q(S)\,dS = \int M(S)\,h(S)\,p(S)\,dS
\qquad \forall\, h,
$$

forces the integrands to match pointwise:

$$
e^{-r\tau}\,q(S) = M(S)\,p(S)
\quad\Longrightarrow\quad
\boxed{\;\frac{q(S)}{p(S)} \;\propto\; M(S)\;}
\quad (9.1)
$$

So the **Radon–Nikodym derivative** $\frac{d\mathbb{Q}}{d\mathbb{P}}$ is
proportional to the pricing kernel. To go from $q$ to $p$ we need a model for
$M$ — which is where utility theory enters.

---

## 10. CRRA utility and the change of measure

Model the market with a representative investor who has **constant relative
risk aversion (CRRA)** utility over terminal wealth (taken proportional to
$S_T$):

$$
U(S) = \frac{S^{\,1-\gamma}}{1-\gamma}\quad(\gamma \ne 1),
\qquad
U'(S) = S^{-\gamma}.
$$

In equilibrium the pricing kernel is the investor's **marginal rate of
substitution** — the discounted ratio of marginal utility in the future state
to marginal utility today:

$$
M(S) = e^{-\delta\tau}\,\frac{U'(S)}{U'(S_0)}
= e^{-\delta\tau}\left(\frac{S}{S_0}\right)^{-\gamma}
\;\propto\; S^{-\gamma},
\quad (10.1)
$$

where $\delta$ is the subjective discount rate (an irrelevant constant here, as
it cancels in normalization). $M$ is **decreasing** in $S$: a dollar in a crash
state ($S$ small) is worth more than a dollar in a boom — the formal statement
of risk aversion.

Combining (9.1) and (10.1):

$$
\frac{q(S)}{p(S)} \propto S^{-\gamma}
\quad\Longrightarrow\quad
p(S) \propto S^{\gamma}\,q(S).
$$

After normalizing to a density (§11), with $\gamma = 2.5$:

$$
\boxed{\;p(S_T) \;=\; \frac{S_T^{\,\gamma}\,q(S_T)}{\displaystyle\int_0^\infty S^{\,\gamma}\,q(S)\,dS}\;}
\quad (10.2)
$$

**Direction of the shift.** The weight $S^{\gamma}$ is increasing, so it moves
probability mass to *higher* prices: $p$ has a **higher mean than $q$**. This is
exactly the **equity risk premium** — under the physical measure the asset is
expected to grow faster than the risk-free forward, compensating the
risk-averse investor. Concretely,

$$
\mathbb{E}^{\mathbb{P}}[S_T] \;>\; \mathbb{E}^{\mathbb{Q}}[S_T] = S_0 e^{r\tau}.
$$

> **A note on intuition vs. the formula.** It is tempting to say risk aversion
> "raises the probability of down-moves." That describes the *kernel*
> $M\propto S^{-\gamma}$ (which up-weights bad states in *pricing*). The
> transformation from $q$ to the *real-world* $p$ is the **inverse**: we divide
> out that pricing distortion, $p\propto q/M \propto S^{\gamma} q$, which shifts
> the real-world distribution **up**. The implemented formula (10.2) is the
> standard, internally consistent one; the code's diagnostics report
> $\mathbb{E}^{\mathbb{P}}[S_T] - \mathbb{E}^{\mathbb{Q}}[S_T] > 0$ as the
> premium.

---

## 11. Normalization

A density extracted numerically will not integrate to exactly $1$. For any
non-negative array $f$ on grid $\{S_j\}$ we rescale by the **trapezoidal**
integral:

$$
\hat f(S_j) = \frac{f(S_j)}{\displaystyle\int f\,dS},
\qquad
\int f\,dS \approx \sum_{j} \frac{f(S_{j+1}) + f(S_j)}{2}\,(S_{j+1}-S_j).
\quad (11.1)
$$

This is applied to $q$ after (7.1) and again to the CRRA numerator in (10.2),
guaranteeing $\int q\,dS = \int p\,dS = 1$ (the code asserts both are $\approx
1.0000$). The discrete analogue of (10.2) actually used is

$$
p(S_j) = \frac{S_j^{\gamma}\,q(S_j)}{\sum_k S_k^{\gamma}\,q(S_k)\,\Delta S_k}.
$$

---

## 12. Moments and cumulative probabilities

**Mean and standard deviation** of any density $f$ on the grid:

$$
\mu = \int S\,f(S)\,dS, \qquad
\sigma_{\mathrm{dist}} = \sqrt{\int (S-\mu)^2 f(S)\,dS},
$$

both via trapezoidal quadrature (`density.moments`). Reported for $q$ and $p$.

**Interval / tail probabilities** (the Probability Calculator). For bounds
$[a,b]$ (either may be open):

$$
\mathbb{P}(a \le S_T \le b) = \int_a^b f(S)\,dS,
\qquad
\mathbb{P}(S_T \le a) = \int_{S_{\min}}^{a} f,
\qquad
\mathbb{P}(S_T \ge b) = \int_{b}^{S_{\max}} f.
$$

Partial edge bins are handled by linearly interpolating $f$ at the exact bounds
before integrating (`density.interval_probability`), so the answer is accurate
for arbitrary $a,b$ — not snapped to grid points. Choosing $f=p$ gives the
real-world probability; $f=q$ gives the risk-neutral one.

The **hover** readout converts density to an interpretable number: the chance
of landing within a \$1 window around price $S$ is $f(S)\cdot \$1$, displayed as
$f(S)\times 100\%$ "per \$1."

---

## 13. The forward probability cone

The cone visualizes how the distribution evolves across **maturities**. For
each cone expiry $T_k$ (with $\tau_k$, density $p_k$ on its own grid) we:

1. **Resample** every $p_k$ onto a shared price axis $\{S_j\}$ (union of the
   per-expiry central 1–99% supports), zero outside each expiry's traded range.
   This yields a matrix $P_{jk} = p_k(S_j)$.
2. **Anchor at "now"** with a narrow Gaussian $p_0$ centered at $S_0$
   (the price is known today), at $\tau_0 = 0$.
3. **Interpolate in time.** For a dense set of future business days with
   horizons $\{\tau^{*}_\ell\}$, linearly interpolate each price row across the
   knot maturities $\{0, \tau_1, \tau_2, \dots\}$:

$$
\tilde p(S_j, \tau^{*}_\ell) = \text{interp}\big(\tau^{*}_\ell;\ \{(\tau_k, P_{jk})\}\big).
$$

4. **Render.** Cell color uses the **per-column-normalized** density
   $\tilde p / \max_j \tilde p$ (so the widening cloud stays visible at every
   horizon), while the **hover reports the true** $\tilde p(S_j)\times100\%$ per
   \$1. The cloud therefore fans out as uncertainty accumulates, exactly as a
   variance that grows roughly like $\sigma^2\tau$ should.

> The cone is a *visualization* layered on the per-expiry densities; the
> time-interpolation between listed expiries is linear in density (a display
> choice), whereas each expiry slice itself is fully model-derived.

---

## 14. The implied-volatility surface

The 3-D surface plots the **fitted** smile of each cone expiry against
moneyness and maturity:

$$
\Sigma(m, \tau_k) = \hat\sigma_k\!\big(m \cdot S_0\big)\times 100\%,
\qquad m = \frac{K}{S_0},
$$

where $\hat\sigma_k$ is the §5 smoothing spline for expiry $k$. Each smile is
resampled onto a shared moneyness grid $m\in[0.80, 1.20]$; points outside an
expiry's traded strike range are left as **gaps** (no extrapolation), which is
why near-dated maturities show a narrow ATM ribbon that widens with $\tau$.

Two cross-sections of this surface have names:

- **Smile / skew** — a slice at fixed $\tau$. Equity index smiles slope down
  (OTM puts richer than OTM calls), reflecting crash fear and the negative
  skew of $q$.
- **Term structure** — a slice at fixed $m$ (e.g. ATM), showing how IV varies
  with horizon.

---

## 15. Assumptions, error sources, and limitations

**Modeling assumptions**

- **European exercise.** Breeden–Litzenberger (§6) assumes European options;
  listed US equity options are American. For non-dividend-paying names and
  short horizons the early-exercise premium is small, but it biases the
  extracted $q$ for deep ITM strikes and dividend-payers.
- **Representative CRRA investor with constant $\gamma=2.5$.** The $q\to p$ map
  (§10) is only as good as this preference model. $\gamma$ is a fixed input,
  not estimated; the real-world density scales with it. A different utility
  (e.g. habit, prospect theory) would give a different kernel and different $p$.
- **No-arbitrage in quotes.** The smile fit and clipping mitigate, but cannot
  fully cure, butterfly/calendar arbitrage present in noisy market quotes.

**Numerical error sources**

- **Ill-posed differentiation** (§7): the second derivative amplifies quote
  noise; the smoothing budget $s$ (§5) is the main control. Too little → spiky,
  possibly negative $q$; too much → biased, over-smoothed $q$.
- **No tail extrapolation.** Densities are only supported across *traded*
  strikes. True tail mass beyond the listed strikes is missing, so tail
  probabilities (§12) are slightly understated and the densities are
  renormalized over the observed range.
- **Discretization.** Trapezoidal integration and central differences are
  $O(\Delta K^2)$; with ~400 grid points this is well below quote noise.
- **Data quality.** Everything inherits Yahoo Finance's delayed, sometimes
  stale quotes; illiquid names with wide spreads produce unreliable smiles
  (hence the liquidity filter in Step 1).

**Time interpolation.** The cone's between-expiry interpolation (§13) is a
display convenience (linear in density), not a no-arbitrage-consistent
term-structure model.

---

## 16. Symbol glossary

| Symbol | Meaning | First used |
|--------|---------|-----------|
| $S_0,\,S_T$ | spot now / terminal price | §1 |
| $\tau,\,T$ | time to expiry / maturity | §1 |
| $K$ | strike | §1 |
| $r(\tau)$ | risk-free rate at horizon $\tau$ | §8 |
| $\sigma,\,\hat\sigma(K)$ | volatility / fitted smile | §3, §5 |
| $\Phi,\,\phi$ | normal CDF / PDF | §3 |
| $c(K)$ | call price function | §2 |
| $Q(K),\,q(S)$ | risk-neutral CDF / density | §6, §2 |
| $p(S)$ | real-world density | §10 |
| $F$ | forward price $S_0 e^{r\tau}$ | §2 |
| $M(S)$ | stochastic discount factor (kernel) | §9 |
| $U,\,U'$ | utility / marginal utility | §10 |
| $\gamma$ | relative risk aversion ($=2.5$) | §10 |
| $\delta$ | subjective discount rate | §10 |
| $\mathbb{Q},\,\mathbb{P}$ | risk-neutral / physical measure | §2, §9 |
| $\Delta K,\,\Delta S$ | grid step | §7, §11 |
| $m$ | moneyness $K/S_0$ | §14 |

---

*This document describes the model implemented in `src/`. See
[`README.md`](README.md) for how the four pipeline steps map onto these
modules.*
