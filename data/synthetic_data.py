"""
data/synthetic_data.py
Synthetic stock price generation for training augmentation.

FIXES
──────
• trend_synthetic() previously used pd.date_range(end=today, freq="B") which
  returns n-1 dates when today is a business day (pandas off-by-one bug).
  Fixed: uses _biz_dates() like random_walk() does.

• merge_real_synthetic() shifts synthetic dates to strictly BEFORE the real
  data window, guaranteeing zero date overlap and zero length inflation.

• clean_df() from fetch_data is applied to the merged result so the caller
  always receives a dedup'd, NaN-free DataFrame.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SYNTHETIC_RATIO


# ─────────────────────────────────────────────────────────────────────────────
# Date helper
# ─────────────────────────────────────────────────────────────────────────────

def _biz_dates(n: int) -> pd.DatetimeIndex:
    """
    Return exactly *n* past business dates ending yesterday.

    Avoids the pandas edge case where bdate_range(end=today, periods=n)
    returns n-1 dates when today itself is a business day.
    """
    end   = datetime.today() - timedelta(days=1)
    start = end - timedelta(days=n * 3)       # generous buffer for weekends/holidays
    dates = pd.bdate_range(start=start, end=end)
    return dates[-n:]


# ─────────────────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────────────────

def random_walk(
    start_price: float = 100.0,
    n_days: int        = 365,
    drift: float       = 0.0005,
    volatility: float  = 0.015,
    seed: int | None   = None,
) -> pd.DataFrame:
    """Geometric Brownian Motion (log-normal random walk)."""
    rng         = np.random.default_rng(seed)
    log_returns = rng.normal(drift, volatility, n_days)
    prices      = start_price * np.exp(np.cumsum(log_returns))
    prices      = np.insert(prices, 0, start_price)[:-1]   # shift so first == start_price

    dates  = _biz_dates(n_days)
    volume = rng.lognormal(mean=14, sigma=1, size=n_days).astype(int)

    return pd.DataFrame(
        {
            "Open":   prices * (1 + rng.normal(0, 0.003, n_days)),
            "High":   prices * (1 + np.abs(rng.normal(0, 0.008, n_days))),
            "Low":    prices * (1 - np.abs(rng.normal(0, 0.008, n_days))),
            "Close":  prices,
            "Volume": volume,
        },
        index=dates,
    )


def trend_synthetic(
    start_price: float = 100.0,
    n_days: int        = 365,
    trend: str         = "bullish",   # "bullish" | "bearish" | "sideways"
    noise_std: float   = 0.01,
    seed: int | None   = None,
) -> pd.DataFrame:
    """
    Deterministic linear trend + Gaussian noise.

    FIX: uses _biz_dates() instead of pd.date_range(end=today, freq='B')
    which was returning n-1 dates on business days.
    """
    rng   = np.random.default_rng(seed)
    daily = {"bullish": 0.001, "bearish": -0.001, "sideways": 0.0}[trend]

    t      = np.arange(n_days)
    signal = start_price * (1 + daily) ** t
    noise  = rng.normal(0, noise_std * start_price, n_days)
    prices = np.maximum(signal + noise, 1.0)

    # FIX: was pd.date_range(end=datetime.today(), periods=n_days, freq="B")
    # which returns n_days - 1 entries when today is a business day.
    dates  = _biz_dates(n_days)
    volume = rng.lognormal(14, 0.8, n_days).astype(int)

    df = pd.DataFrame(
        {
            "Open":   prices * (1 + rng.normal(0, 0.003, n_days)),
            "High":   prices + np.abs(rng.normal(0, noise_std * start_price, n_days)),
            "Low":    prices - np.abs(rng.normal(0, noise_std * start_price, n_days)),
            "Close":  prices,
            "Volume": volume,
        },
        index=dates,
    )
    df["Low"]  = df[["Low",  "Open", "Close"]].min(axis=1) * 0.99
    df["High"] = df[["High", "Open", "Close"]].max(axis=1) * 1.01
    return df


def augment_with_noise(
    df: pd.DataFrame,
    noise_std: float = 0.005,
    seed: int | None = None,
) -> pd.DataFrame:
    """Add small Gaussian noise to OHLC columns for augmentation."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n   = len(df)
    for col in ["Open", "High", "Low", "Close"]:
        if col in out.columns:
            out[col] = out[col] * (1 + rng.normal(0, noise_std, n))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_real_synthetic(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    ratio: float = SYNTHETIC_RATIO,
) -> pd.DataFrame:
    """
    Concatenate a fraction of synthetic rows with real data FOR TRAINING ONLY.

    Synthetic dates are shifted to strictly BEFORE real_df.index[0] so they
    never overlap the real data window.  The merged result is cleaned via
    clean_df() so it is always dedup'd and NaN-free.

    Returns
    -------
    pd.DataFrame with len >= len(real_df) (extra synthetic rows prepended)
    """
    from data.fetch_data import clean_df

    n_synthetic  = int(len(real_df) * ratio)
    synth_sample = synthetic_df.tail(n_synthetic).copy()

    if len(synth_sample) > 0:
        # Shift synthetic dates to well before real data starts
        earliest_real      = real_df.index[0]
        gap                = earliest_real - synth_sample.index[-1] - pd.Timedelta(days=2)
        synth_sample.index = synth_sample.index + gap

    combined = pd.concat([real_df, synth_sample]).sort_index()
    # Apply canonical cleaning so the merged df has no duplicates / NaN
    return clean_df(combined)
