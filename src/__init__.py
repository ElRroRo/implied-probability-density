"""Implied probability density model package.

Foundation modules implementing the 4-step pipeline:
    Step 1  data.py          option chain + filtering
    Step 2  black_scholes.py + smoothing.py   continuous c(X) curve
    Step 3  density.py        Breeden-Litzenberger risk-neutral q(S_T)
    Step 4  density.py        CRRA transformation -> real-world p(S_T)
"""

GAMMA = 2.5  # hardcoded risk-aversion coefficient (CRRA)
