"""
data/preprocess.py
Normalisation, feature engineering, and train/test splitting.

LENGTH-MISMATCH FIX
────────────────────
All functions that previously did their own  df[~df.index.duplicated(...)].copy()
now call fetch_data.clean_df() instead.  There is exactly ONE deduplication
function in the entire codebase.  This guarantees that every stage of the
pipeline sees the same row count for the same input DataFrame.

scale_with_fitted_scaler() no longer deduplicates internally — the caller
is responsible for passing a clean df (produced by clean_df()).  This removes
a hidden second deduplication that could theoretically produce a different
row count than the caller expected.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SEQUENCE_LENGTH, TEST_SPLIT


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append MA, EMA, MACD, Volume-MA to df.  Drops leading NaN rows."""
    df = df.copy()
    df["MA7"]        = df["Close"].rolling(7).mean()
    df["MA21"]       = df["Close"].rolling(21).mean()
    df["EMA12"]      = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"]      = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]       = df["EMA12"] - df["EMA26"]
    df["Volume_MA7"] = df["Volume"].rolling(7).mean() if "Volume" in df.columns else 0
    df.dropna(inplace=True)
    return df


# ── Scaling ───────────────────────────────────────────────────────────────────

def scale_close(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Fit a MinMaxScaler on df['Close'] and return (scaled_array, scaler).

    The df is cleaned via the canonical clean_df() before scaling so that
    len(scaled) == len(clean_df(df)) with certainty.

    Use ONLY during training.  For prediction use scale_with_fitted_scaler().
    """
    from data.fetch_data import clean_df

    clean      = clean_df(df)
    close_vals = clean["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_vals)

    return scaled, scaler


def scale_with_fitted_scaler(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Transform df['Close'] using an ALREADY-FITTED scaler (no refit).

    The caller MUST pass a DataFrame that has already been cleaned by
    clean_df().  This function does NOT re-clean internally — doing so
    would risk producing a different row count than the caller expects,
    which is exactly the source of the 1848 vs 1847 bug.

    Returns
    -------
    np.ndarray  shape (len(df), 1)
    """
    close_vals = df["Close"].values.reshape(-1, 1)
    return scaler.transform(close_vals)


# ── Sequence Builder ──────────────────────────────────────────────────────────

def build_sequences(
    scaled: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) look-back sequences.

    X[i] = scaled[i : i+seq_len]
    y[i] = scaled[i+seq_len]

    Invariant: len(X) == len(y) == len(scaled) - seq_len
    """
    n = len(scaled)

    X = np.array([scaled[i: i + seq_len, 0] for i in range(n - seq_len)])
    y = np.array([scaled[i + seq_len, 0]    for i in range(n - seq_len)])

    return X.reshape(-1, seq_len, 1), y


# ── Train / Test Split ────────────────────────────────────────────────────────

def train_test_split_ts(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SPLIT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split (NO shuffle to avoid leakage)."""
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]


# ── Regression Features ───────────────────────────────────────────────────────

def build_regression_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Lag + technical indicator features for Linear Regression.
    """
    from data.fetch_data import clean_df

    clean = clean_df(df)
    feat  = add_technical_indicators(clean)

    for lag in [1, 2, 3, 5, 7]:
        feat[f"lag_{lag}"] = feat["Close"].shift(lag)

    feat.dropna(inplace=True)

    feature_cols = ["MA7", "MA21", "MACD",
                    "lag_1", "lag_2", "lag_3", "lag_5", "lag_7"]

    return feat[feature_cols], feat["Close"]


# ── Direction Accuracy ────────────────────────────────────────────────────────

def compute_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Direction Accuracy:
    % of times model correctly predicts UP/DOWN movement.
    """

    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    if len(y_true) < 2:
        return 0.0

    actual_dir = np.sign(np.diff(y_true))
    pred_dir   = np.sign(np.diff(y_pred))

    correct = actual_dir == pred_dir

    return float(np.mean(correct) * 100)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    RMSE, MAE, MAPE, Accuracy and Direction Accuracy.

    IMPORTANT:
    Direction Accuracy = REAL metric for trading usefulness
    """

    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"compute_metrics: length mismatch y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))

    mask = np.abs(y_true) > 1e-8

    mape = float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    ) if mask.sum() > 0 else 100.0

    accuracy = max(0.0, 100.0 - mape)

    # ✅ NEW REAL METRIC
    direction_acc = compute_direction_accuracy(y_true, y_pred)

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 4),
        "Accuracy": round(accuracy, 2),
        "Direction Accuracy": round(direction_acc, 2),  # 🔥 NEW
    }