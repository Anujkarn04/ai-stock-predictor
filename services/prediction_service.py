"""
services/prediction_service.py
High-level prediction API.

ROOT CAUSE FIX
───────────────
LSTM training previously received train_df = merge_real_synthetic(real_df, synth).
Synthetic data is PREPENDED (earlier dates), so after chronological split:
  X_train ≈ synthetic sequences with NEGATIVE scaled values [-0.15, 0.9]
  X_test  ≈ real sequences with values in [0.75, 1.0]

The model learned the synthetic distribution and failed completely on real data
→ RMSE=9000+, MAPE=200%+, Accuracy=0%.

THE FIX
────────
lstm.train(real_df=real_df) — LSTM trains on real_df sequences only.
train_df (with synthetic) is passed only to LR, which is insensitive to this
distribution-shift problem.

STALE MODEL DETECTION
──────────────────────
_ensure_trained() calls _stale_lstm() which now also checks whether the
loaded scaler's data_min_ is consistent with the current real_df prices.
If a previously saved scaler was fitted on merged data (lower min), it is
detected as stale and the model is retrained with the correct pipeline.
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
from models.lstm_model import StockLSTM


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_training_df(real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return augmented training DataFrame for LINEAR REGRESSION only.
    LSTM always uses real_df directly (no synthetic).
    """
    if USE_SYNTHETIC:
        start = float(real_df["Close"].iloc[0])
        synth = random_walk(start_price=start, n_days=len(real_df) // 2)
        return merge_real_synthetic(real_df, synth)
    return real_df


def _stale_lstm(lstm: StockLSTM, real_df: pd.DataFrame) -> bool:
    """
    Return True if the loaded LSTM should be retrained.

    Checks:
    1. Old architecture (no BatchNormalization) → retrain.
    2. Scaler data_min_ significantly lower than real_df.Close.min() →
       scaler was fitted on merged data (synthetic contamination) → retrain.
    """
    if lstm.model is None:
        return True

    # Check 1: architecture
    has_batchnorm = "BatchNormalization" in {type(l).__name__ for l in lstm.model.layers}
    if not has_batchnorm:
        return True

    # Check 2: stale scaler (fitted on merged data with lower price floor)
    if lstm._is_scaler_stale(real_df):
        return True

    return False


def _ensure_trained(
    ticker: str,
) -> tuple[StockLinearRegressor, StockLSTM, pd.DataFrame]:
    """
    Load or train LR + LSTM models for *ticker*.

    LR:   trained on train_df (may include synthetic — LR is robust to this).
    LSTM: trained on real_df ONLY (synthetic causes distribution shift → RMSE=9000+).

    Returns (lr, lstm, real_df).
    """
    raw_df   = fetch_stock_data(ticker)
    real_df  = clean_df(raw_df)
    train_df = _get_training_df(real_df)   # used for LR only

    lr   = StockLinearRegressor(ticker)
    lstm = StockLSTM(ticker)

    # LR: train on merged data (robust to distribution differences)
    if not lr.load():
        lr.train(train_df)
        lr.save()

    # LSTM: retrain if missing, stale architecture, or stale scaler
    if not lstm.load() or _stale_lstm(lstm, real_df):
        # Pass ONLY real_df — train_df argument is ignored inside StockLSTM.train()
        lstm.train(real_df=real_df)
        lstm.save()

    return lr, lstm, real_df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_next_days(
    ticker: str,
    days: int       = PREDICTION_DAYS,
    model_type: str = "LSTM",
) -> Dict:
    """
    Forecast next *days* business days.
    result["historical"] is the canonical clean real_df.
    All prediction lists have length == days.
    """
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

    if model_type in ("LSTM", "Both"):
        result["predictions"] = lstm.predict_next_days(real_df, days).tolist()

    if model_type in ("LR", "Both"):
        lr_preds = lr.predict_next_days(real_df, days).tolist()
        if model_type == "LR":
            result["predictions"] = lr_preds
        else:
            result["lr_predictions"] = lr_preds

    return result


def get_history_with_predictions(ticker: str) -> Dict:
    """
    Return historical df with aligned LR_Pred + LSTM_Pred columns.
    All predictions are in original price space (₹).
    """
    lr, lstm, real_df = _ensure_trained(ticker)

    out = real_df.copy()

    lstm_pred = lstm.predict_on_history(out)
    lr_pred   = lr.predict_on_history(out)

    out["LR_Pred"]   = lr_pred.reindex(out.index)
    out["LSTM_Pred"] = lstm_pred.reindex(out.index)

    return {"df": out, "ticker": ticker}


def get_metrics(ticker: str) -> Dict[str, Dict]:
    """
    Force-retrain both models and return evaluation metrics.
    LSTM trains on real_df only; LR uses merged (real + synthetic) data.
    """
    raw_df   = fetch_stock_data(ticker)
    real_df  = clean_df(raw_df)
    train_df = _get_training_df(real_df)

    data_rows  = len(real_df)
    data_years = get_data_years(real_df)

    lr   = StockLinearRegressor(ticker)
    lstm = StockLSTM(ticker)

    # LR: train on augmented data
    lr_m = lr.train(train_df)
    lr.save()

    # LSTM: train on REAL DATA ONLY — no synthetic contamination
    lstm_m = lstm.train(real_df=real_df)
    lstm.save()

    for m in (lr_m, lstm_m):
        m["data_rows"]  = data_rows
        m["data_years"] = data_years

    return {"LR": lr_m, "LSTM": lstm_m}


def get_risk_score(df: pd.DataFrame) -> Dict:
    """Volatility-based risk classification."""
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
