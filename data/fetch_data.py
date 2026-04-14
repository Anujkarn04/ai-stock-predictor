"""
data/fetch_data.py
FINAL ROBUST VERSION — production safe

Fixes:
- Length mismatch (kept your yesterday logic)
- Empty dataframe issues
- Missing OHLC columns ('Open' error)
- Yahoo API instability
"""

import pandas as pd
import yfinance as yf
import time
from datetime import datetime, timedelta
import sys, os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MAX_LOOKBACK_YEARS


# ─────────────────────────────────────────────────────────────────────────────
# Canonical cleaner (UNCHANGED — your logic is correct)
# ─────────────────────────────────────────────────────────────────────────────

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Parse index
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Sort
    df = df.sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep="last")]

    # Remove today's partial data
    today = pd.Timestamp.today().normalize()
    df = df[df.index < today]

    # Keep required columns
    required_cols = ["Open", "High", "Low", "Close"]
    available_cols = [c for c in required_cols if c in df.columns]

    if len(available_cols) < 4:
        return pd.DataFrame()

    df = df[available_cols + ([ "Volume"] if "Volume" in df.columns else [])]

    # Drop NaN
    df = df.dropna()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _yesterday() -> str:
    return (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")


def _try_download(ticker: str, period="15y") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            end=_yesterday(),   # 🔥 critical fix
            progress=False,
            auto_adjust=True
        )
        return clean_df(df)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 🔥 FINAL FIXED MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """
    🔥 PRODUCTION VERSION
    - Retries API calls
    - Handles Yahoo failures
    - Fallbacks for .NS tickers
    """

    def attempt(t, period="15y"):
        df = _try_download(t, period=period)
        return df if df is not None and not df.empty else pd.DataFrame()

    # ── Retry system ────────────────────────────────────────────────────────
    for i in range(3):  # try 3 times
        df = attempt(ticker)

        if not df.empty:
            return df

        time.sleep(1)  # wait before retry

    # ── Fallback: remove .NS ───────────────────────────────────────────────
    if ticker.endswith(".NS"):
        alt = ticker.replace(".NS", "")

        for i in range(2):
            df = attempt(alt, period="5y")

            if not df.empty:
                return df

            time.sleep(1)

    # ── FINAL fallback ─────────────────────────────────────────────────────
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Price + info (UNCHANGED — already good)
# ─────────────────────────────────────────────────────────────────────────────

def get_current_price(ticker: str) -> float:
    try:
        fi = yf.Ticker(ticker).fast_info
        price = getattr(fi, "last_price", None)
        if price and float(price) > 0:
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
    _default = {
        "name": ticker,
        "sector": "N/A",
        "currency": "INR" if (".NS" in ticker or ".BO" in ticker) else "USD",
        "exchange": "N/A",
    }
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "N/A",
            "currency": info.get("currency") or _default["currency"],
            "exchange": info.get("exchange") or "N/A",
        }
    except Exception:
        return _default


def get_data_years(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    return round((df.index[-1] - df.index[0]).days / 365.25, 1)