"""
data/fetch_data.py
Fetches historical OHLCV data from Yahoo Finance.

ROOT CAUSE FIX — Length mismatch
──────────────────────────────────
Every yfinance download previously used today as the implicit end-date.
During market hours, a second call within the same session could receive an
extra partial bar for "today", making the returned DataFrame 1 row larger than
the first call.  That 1-row difference propagated all the way to the LSTM
prediction alignment and caused:
    "Length of values (1848) does not match length of index (1847)"

Fix: ALL download calls now explicitly pass end=YESTERDAY so every call
within the same calendar day returns EXACTLY the same rows, regardless of
whether the market is open, closed, or mid-session.

A single canonical clean_df() function is used everywhere so deduplication,
today-bar removal, and NaN-dropping happen in one place only.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MAX_LOOKBACK_YEARS


# ─────────────────────────────────────────────────────────────────────────────
# Canonical data-cleaning function  (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical OHLCV cleaner used by EVERY data path in this project.

    Steps (order matters):
    1. Flatten multi-level columns (yfinance >= 0.2 returns them).
    2. Parse and tz-strip the DatetimeIndex.
    3. Sort chronologically.
    4. Drop duplicate index entries (keep the last — most recent adjustment).
    5. Drop today's bar if present (partial / not yet closed).
    6. Keep only the five OHLCV columns that exist.
    7. Drop any row with NaN in any kept column.

    Returns a clean DataFrame or an empty DataFrame if input was empty.
    The returned index is a tz-naive DatetimeIndex with NO duplicates and
    NO today-or-future dates.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # ── 1. Flatten multi-level columns ──────────────────────────────────────
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # ── 2. Parse index ───────────────────────────────────────────────────────
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # ── 3. Sort ──────────────────────────────────────────────────────────────
    df = df.sort_index()

    # ── 4. Deduplicate index ─────────────────────────────────────────────────
    df = df[~df.index.duplicated(keep="last")]

    # ── 5. Drop today's partial bar ──────────────────────────────────────────
    # We lock downloads to end=yesterday, but apply this guard defensively.
    today = pd.Timestamp.today().normalize()
    df    = df[df.index < today]

    # ── 6. Keep OHLCV columns ────────────────────────────────────────────────
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df   = df[keep]

    # ── 7. Drop NaN rows ─────────────────────────────────────────────────────
    df = df.dropna()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Internal download helper
# ─────────────────────────────────────────────────────────────────────────────

def _yesterday() -> str:
    """Return yesterday's date as 'YYYY-MM-DD' string (used as end= in all downloads)."""
    return (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")


def _try_download(ticker: str, **kwargs) -> pd.DataFrame:
    """
    Single yf.download attempt.  Always passes end=yesterday.
    Returns empty DataFrame on any error.
    """
    # Always lock end-date to yesterday — prevents partial-bar race condition
    kwargs.setdefault("end", _yesterday())
    try:
        df = yf.download(ticker, progress=False, auto_adjust=True, **kwargs)
        return clean_df(df)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(
    ticker: str,
    years: int = MAX_LOOKBACK_YEARS,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for *ticker* with a progressive fallback:
        15y → 10y → 5y → 2y → max (period)

    All downloads end at yesterday — identical row count across every call
    within the same calendar day, eliminating the length-mismatch race condition.

    Returns
    -------
    pd.DataFrame  columns: Open, High, Low, Close, Volume
                  index: tz-naive DatetimeIndex, no duplicates, no NaN
    Raises
    ------
    ValueError  if no data (>= 30 rows) can be fetched after all fallbacks.
    """
    fallback_years = sorted(set([years, 10, 5, 2]), reverse=True)

    for yrs in fallback_years:
        start = (datetime.today() - timedelta(days=yrs * 365)).strftime("%Y-%m-%d")
        df    = _try_download(ticker, start=start)
        if len(df) >= 30:
            return df

    # Final fallback: "max" period (new / limited-history stocks)
    df = _try_download(ticker, period="max")
    if len(df) >= 30:
        return df

    raise ValueError(
        f"No usable data for '{ticker}'. "
        "Check the symbol or your internet connection."
    )


def get_current_price(ticker: str) -> float:
    """
    Return the latest closing price for *ticker*.
    Triple-layer fallback: fast_info → recent history → 0.0.
    Never raises; returns 0.0 if all methods fail.
    """
    try:
        fi    = yf.Ticker(ticker).fast_info
        price = getattr(fi, "last_price", None)
        if price is not None and float(price) > 0:
            return float(price)
    except Exception:
        pass

    try:
        df = _try_download(ticker, period="5d")
        if not df.empty and "Close" in df.columns:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass

    return 0.0


def get_stock_info(ticker: str) -> dict:
    """
    Return a metadata dict for display.
    Guards against t.info returning None (some tickers / yfinance versions).
    """
    _default = {
        "name":     ticker,
        "sector":   "N/A",
        "currency": "INR" if (".NS" in ticker or ".BO" in ticker) else "USD",
        "exchange": "N/A",
    }
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "name":     info.get("longName")  or info.get("shortName") or ticker,
            "sector":   info.get("sector")    or "N/A",
            "currency": info.get("currency")  or _default["currency"],
            "exchange": info.get("exchange")  or "N/A",
        }
    except Exception:
        return _default


def get_data_years(df: pd.DataFrame) -> float:
    """Number of years spanned by df's index (for UI display)."""
    if len(df) < 2:
        return 0.0
    return round((df.index[-1] - df.index[0]).days / 365.25, 1)
