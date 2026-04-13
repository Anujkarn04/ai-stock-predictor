"""
models/train_model.py
CLI entry-point for training both models on a given ticker.

Usage:
    python models/train_model.py --ticker TCS.NS --days 7
"""

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetch_data import fetch_stock_data
from data.synthetic_data import random_walk, merge_real_synthetic
from models.regression_model import StockLinearRegressor
from models.lstm_model import StockLSTM
from config import USE_SYNTHETIC, PREDICTION_DAYS


def train_all(ticker: str, days: int = PREDICTION_DAYS):
    print(f"\n{'='*55}")
    print(f"  Training models for: {ticker}")
    print(f"{'='*55}")

    # 1. Fetch real data
    print("\n[1/4] Fetching historical data …")
    df = fetch_stock_data(ticker)
    print(f"      {len(df)} trading days loaded.")

    # 2. Optionally augment with synthetic data
    if USE_SYNTHETIC:
        print("[2/4] Generating synthetic augmentation data …")
        start = float(df["Close"].iloc[0])
        synth = random_walk(start_price=start, n_days=len(df)//2)
        df = merge_real_synthetic(df, synth)
        print(f"      Combined dataset: {len(df)} rows.")
    else:
        print("[2/4] Synthetic data disabled (USE_SYNTHETIC=False). Skipping.")

    # 3. Train Linear Regression
    print("\n[3/4] Training Linear Regression …")
    lr = StockLinearRegressor(ticker)
    lr_metrics = lr.train(df)
    lr.save()
    print(f"      LR  → RMSE: {lr_metrics['RMSE']}  |  MAE: {lr_metrics['MAE']}")

    # 4. Train LSTM
    print("\n[4/4] Training LSTM (this may take a few minutes) …")
    lstm = StockLSTM(ticker)
    lstm_metrics = lstm.train(df)
    lstm.save()
    print(f"      LSTM → RMSE: {lstm_metrics['RMSE']}  |  MAE: {lstm_metrics['MAE']}")

    print(f"\n✅  Models saved to saved_models/")
    return {"LR": lr_metrics, "LSTM": lstm_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    parser.add_argument("--ticker", default="TCS.NS", help="Stock ticker symbol")
    parser.add_argument("--days",   type=int, default=PREDICTION_DAYS,
                        help="Prediction horizon (for display only)")
    args = parser.parse_args()
    train_all(args.ticker, args.days)
