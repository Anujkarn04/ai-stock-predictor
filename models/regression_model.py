"""
models/regression_model.py
Linear Regression baseline with lag + technical indicator features.

LENGTH-MISMATCH FIX
────────────────────
predict_on_history() now calls clean_df() once (canonical, from fetch_data)
instead of doing its own deduplication.  The returned Series uses clean.index.
The caller (prediction_service) aligns via reindex(out.index).
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict

from config import MODEL_DIR, TEST_SPLIT
from data.preprocess import (
    build_regression_features,
    add_technical_indicators,
    compute_metrics,
)


class StockLinearRegressor:
    """LinearRegression wrapper with full metric suite and stable alignment."""

    def __init__(self, ticker: str):
        self.ticker    = ticker
        self.model     = LinearRegression()
        self.scaler    = StandardScaler()
        self._trained  = False
        self._features: list[str] = []

    # ── paths ─────────────────────────────────────────────────────────────────

    def _model_path(self) -> str:
        safe = self.ticker.replace(".", "_").replace("/", "_")
        return os.path.join(MODEL_DIR, f"lr_{safe}.pkl")

    def _scaler_path(self) -> str:
        safe = self.ticker.replace(".", "_").replace("/", "_")
        return os.path.join(MODEL_DIR, f"lr_scaler_{safe}.pkl")

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Chronological split; returns RMSE, MAE, MAPE, Accuracy."""
        X, y   = build_regression_features(df)
        split  = int(len(X) * (1 - TEST_SPLIT))

        X_train, X_test = X.iloc[:split],  X.iloc[split:]
        y_train, y_test = y.iloc[:split],  y.iloc[split:]

        X_tr_sc = self.scaler.fit_transform(X_train)
        X_te_sc = self.scaler.transform(X_test)

        self.model.fit(X_tr_sc, y_train)
        preds = self.model.predict(X_te_sc)

        self._trained  = True
        self._features = list(X.columns)

        return compute_metrics(y_test.values, preds)

    # ── predict FUTURE N days ─────────────────────────────────────────────────

    def predict_next_days(self, df: pd.DataFrame, days: int = 7) -> np.ndarray:
        """Iterative multi-step forecast. Returns ndarray shape (days,)."""
        if not self._trained:
            raise RuntimeError("Model not trained.")

        history = add_technical_indicators(df.copy())
        preds   = []

        for _ in range(days):
            row = {
                "MA7":  history["Close"].rolling(7).mean().iloc[-1],
                "MA21": history["Close"].rolling(21).mean().iloc[-1],
                "MACD": (
                    history["Close"].ewm(span=12, adjust=False).mean().iloc[-1]
                    - history["Close"].ewm(span=26, adjust=False).mean().iloc[-1]
                ),
            }
            for lag in [1, 2, 3, 5, 7]:
                row[f"lag_{lag}"] = history["Close"].iloc[-lag]

            feat_arr = np.array([[row[c] for c in self._features]])
            pred     = float(self.model.predict(self.scaler.transform(feat_arr))[0])
            preds.append(pred)

            new_row          = history.iloc[[-1]].copy()
            new_row.index    = [history.index[-1] + pd.Timedelta(days=1)]
            new_row["Close"] = pred
            history          = pd.concat([history, new_row])

        return np.array(preds)

    # ── predict ON history ────────────────────────────────────────────────────

    def predict_on_history(self, df: pd.DataFrame) -> pd.Series:
        """
        Run model over all rows with sufficient feature data.
        Returns pd.Series with index == clean_df(df).index.
        Rows without enough lag data are NaN.
        Caller aligns via reindex(out.index).
        """
        if not self._trained:
            raise RuntimeError("Model not trained.")

        from data.fetch_data import clean_df
        clean = clean_df(df)

        full, _ = build_regression_features(clean)
        X_sc    = self.scaler.transform(full)
        preds   = self.model.predict(X_sc)

        pred_series = pd.Series(preds, index=full.index, name="LR_Pred")
        # Align to clean.index (fills NaN for rows dropped by lag/dropna).
        # Caller's reindex(out.index) is the final alignment step.
        return pred_series.reindex(clean.index)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self):
        joblib.dump(self.model,  self._model_path())
        joblib.dump(self.scaler, self._scaler_path())

    def load(self) -> bool:
        mp, sp = self._model_path(), self._scaler_path()
        if os.path.exists(mp) and os.path.exists(sp):
            self.model    = joblib.load(mp)
            self.scaler   = joblib.load(sp)
            self._trained = True
            self._features = list(
                getattr(self.scaler, "feature_names_in_",
                        ["MA7","MA21","MACD",
                         "lag_1","lag_2","lag_3","lag_5","lag_7"])
            )
            return True
        return False
