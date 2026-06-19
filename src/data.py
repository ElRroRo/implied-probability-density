"""Step 1 - Option data collection and filtering (via yfinance).

Provides spot price, available expirations, and a cleaned call-option
chain (strikes + mid-prices) ready to feed into the smoothing step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class OptionSnapshot:
    """A filtered single-expiry call chain."""
    ticker: str
    expiry: str          # 'YYYY-MM-DD'
    spot: float
    strikes: np.ndarray  # filtered strikes (sorted)
    mid: np.ndarray      # corresponding mid-prices
    iv_market: np.ndarray  # Yahoo's reported implied vols (fallback)
    raw: pd.DataFrame    # the filtered DataFrame, for inspection


def _yf(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker.strip().upper())


def get_spot(ticker: str) -> float:
    """Latest traded price, with a history fallback."""
    t = _yf(ticker)
    try:
        px = float(t.fast_info["last_price"])
        if px > 0:
            return px
    except Exception:
        pass
    hist = t.history(period="5d")
    if hist.empty:
        raise ValueError(f"No price data for ticker '{ticker}'.")
    return float(hist["Close"].dropna().iloc[-1])


# Timeframe -> yfinance fetch config. Only native, official intervals.
TIMEFRAMES: dict[str, dict] = {
    "1H": dict(interval="60m", period="1mo"),
    "1D": dict(interval="1d", period="1y"),
    "1W": dict(interval="1wk", period="5y"),
}


def get_history(ticker: str, timeframe: str = "1D") -> pd.DataFrame:
    """OHLCV candles for a timeframe ('1H', '1D', '1W')."""
    cfg = TIMEFRAMES[timeframe]
    df = _yf(ticker).history(period=cfg["period"], interval=cfg["interval"])
    if df.empty:
        raise ValueError(f"No price history for '{ticker}' at {timeframe}.")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def get_expirations(ticker: str) -> list[str]:
    """All available option expiration dates (strings)."""
    exps = _yf(ticker).options
    if not exps:
        raise ValueError(f"No listed options for ticker '{ticker}'.")
    return list(exps)


def get_call_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """Raw calls DataFrame for one expiry."""
    chain = _yf(ticker).option_chain(expiry)
    return chain.calls.copy()


def filter_chain(df: pd.DataFrame, spot: float,
                 max_rel_spread: float = 0.5,
                 lo_mult: float = 0.4, hi_mult: float = 1.8,
                 min_quotes: int = 6) -> pd.DataFrame:
    """Drop noisy / illiquid quotes before differentiating later.

    Keeps rows with two-sided quotes, some activity (volume or open
    interest), a tolerable relative bid-ask spread, and strikes within a
    sane band around spot. Falls back to looser rules if too few rows
    survive (so thin chains still produce a curve).
    """
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    two_sided = (df["bid"] > 0) & (df["ask"] > 0) & (df["mid"] > 0)
    active = (df.get("volume", 0).fillna(0) > 0) | (df.get("openInterest", 0).fillna(0) > 0)
    rel_spread = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    tight = rel_spread <= max_rel_spread
    band = (df["strike"] >= lo_mult * spot) & (df["strike"] <= hi_mult * spot)

    out = df[two_sided & active & tight & band]
    if len(out) < min_quotes:  # relax: keep any two-sided quote in band
        out = df[two_sided & band]
    if len(out) < min_quotes:  # last resort: any two-sided quote
        out = df[two_sided]

    return out.sort_values("strike").drop_duplicates("strike").reset_index(drop=True)


def get_snapshot(ticker: str, expiry: str | None = None,
                 spot: float | None = None) -> OptionSnapshot:
    """Convenience: fetch spot + (optionally first) expiry chain, filtered.

    ``spot`` may be supplied to skip a redundant price fetch (e.g. when
    building a multi-expiry surface).
    """
    if expiry is None:
        expiry = get_expirations(ticker)[0]
    if spot is None:
        spot = get_spot(ticker)
    filtered = filter_chain(get_call_chain(ticker, expiry), spot)
    if len(filtered) < 4:
        raise ValueError(
            f"Only {len(filtered)} usable strikes for {ticker} {expiry}; "
            "need >= 4 for a stable density. Try another expiry/ticker."
        )
    return OptionSnapshot(
        ticker=ticker.upper(),
        expiry=expiry,
        spot=spot,
        strikes=filtered["strike"].to_numpy(float),
        mid=filtered["mid"].to_numpy(float),
        iv_market=filtered.get("impliedVolatility", pd.Series(np.nan, index=filtered.index)).to_numpy(float),
        raw=filtered,
    )
