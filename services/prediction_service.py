"""
services/prediction_service.py
High-level prediction API.

🔥 DEPLOYMENT FIX ADDED
───────────────────────
TensorFlow is NOT available on Streamlit Cloud.
So LSTM is safely disabled using try/except.

→ App will NOT crash
→ Falls back to Linear Regression
→ Shows clean error message
"""

import numpy as np
import pandas as pd
from typing import Dict
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    USE_SYNTHETIC, PREDICTION_DAYS,
    RISK_LOW_THRESHOLD, RISK_MEDIUM_THRESHOLD,
)
from data.fetch_data import fetch_stock_data, get_data_years, clean_df
from data.synthetic_data import random_walk, merge_real_synthetic
from models.regression_model import StockLinearRegressor

# 🚨 IMPORTANT: Safe import for LSTM
try:
    from models.lstm_model import StockLSTM
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_training_df(real_df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic data only used for LR"""
    if USE_SYNTHETIC:
        start = float(real_df["Close"].iloc[0])
        synth = random_walk(start_price=start, n_days=len(real_df) // 2)
        return merge_real_synthetic(real_df, synth)
    return real_df


def _ensure_trained(
    ticker: str,
) -> tuple:
    """
    Load or train models.

    LR → always available  
    LSTM → only if TensorFlow exists
    """
    raw_df   = fetch_stock_data(ticker)
    real_df  = clean_df(raw_df)
    train_df = _get_training_df(real_df)

    lr = StockLinearRegressor(ticker)

    if not lr.load():
        lr.train(train_df)
        lr.save()

    # ✅ LSTM safe handling
    lstm = None
    if LSTM_AVAILABLE:
        try:
            lstm = StockLSTM(ticker)

            if not lstm.load():
                lstm.train(real_df=real_df)
                lstm.save()

        except Exception:
            lstm = None  # fallback safely

    return lr, lstm, real_df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_next_days(
    ticker: str,
    days: int       = PREDICTION_DAYS,
    model_type: str = "LSTM",
) -> Dict:

    lr, lstm, real_df = _ensure_trained(ticker)

    future_dates = pd.bdate_range(
        start=real_df.index[-1] + pd.Timedelta(days=1),
        periods=days,
    )

    result: Dict = {
        "ticker":         ticker,
        "days":           days,
        "dates":          [d.strftime("%Y-%m-%d") for d in future_dates],
        "predictions":    [],
        "lr_predictions": None,
        "historical":     real_df,
    }

    # 🚨 SAFE LSTM
    if model_type in ("LSTM", "Both"):
        if lstm is None:
            raise RuntimeError(
                "LSTM is not available in deployed version. "
                "Please use Linear Regression."
            )
        result["predictions"] = lstm.predict_next_days(real_df, days).tolist()

    # ✅ LR always works
    if model_type in ("LR", "Both"):
        lr_preds = lr.predict_next_days(real_df, days).tolist()
        if model_type == "LR":
            result["predictions"] = lr_preds
        else:
            result["lr_predictions"] = lr_preds

    return result


def get_history_with_predictions(ticker: str) -> Dict:

    lr, lstm, real_df = _ensure_trained(ticker)

    out = real_df.copy()

    lr_pred = lr.predict_on_history(out)
    out["LR_Pred"] = lr_pred.reindex(out.index)

    # 🚨 SAFE LSTM
    if lstm is not None:
        lstm_pred = lstm.predict_on_history(out)
        out["LSTM_Pred"] = lstm_pred.reindex(out.index)
    else:
        out["LSTM_Pred"] = np.nan

    return {"df": out, "ticker": ticker}


def get_metrics(ticker: str) -> Dict[str, Dict]:

    raw_df   = fetch_stock_data(ticker)
    real_df  = clean_df(raw_df)
    train_df = _get_training_df(real_df)

    data_rows  = len(real_df)
    data_years = get_data_years(real_df)

    lr = StockLinearRegressor(ticker)
    lr_m = lr.train(train_df)
    lr.save()

    result = {"LR": lr_m}

    # 🚨 SAFE LSTM
    if LSTM_AVAILABLE:
        try:
            lstm = StockLSTM(ticker)
            lstm_m = lstm.train(real_df=real_df)
            lstm.save()

            lstm_m["data_rows"]  = data_rows
            lstm_m["data_years"] = data_years

            result["LSTM"] = lstm_m

        except Exception:
            result["LSTM"] = {
                "error": "LSTM not available in deployed version"
            }
    else:
        result["LSTM"] = {
            "error": "LSTM not available in deployed version"
        }

    lr_m["data_rows"]  = data_rows
    lr_m["data_years"] = data_years

    return result


def get_risk_score(df: pd.DataFrame) -> Dict:

    daily_returns = df["Close"].pct_change().dropna()
    std_daily     = float(daily_returns.std())
    annual_vol    = std_daily * (252 ** 0.5)

    if std_daily < RISK_LOW_THRESHOLD:
        level, badge = "Low",    "🟢"
    elif std_daily < RISK_MEDIUM_THRESHOLD:
        level, badge = "Medium", "🟡"
    else:
        level, badge = "High",   "🔴"

    sharpe = (
        round((daily_returns.mean() / std_daily) * (252 ** 0.5), 2)
        if std_daily > 0 else 0.0
    )

    return {
        "daily_std":     round(std_daily * 100, 3),
        "annual_vol":    round(annual_vol * 100, 2),
        "risk_level":    level,
        "risk_badge":    badge,
        "sharpe_approx": sharpe,
    }