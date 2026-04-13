"""
config.py — Central configuration.
"""
import os

# ── Stocks ────────────────────────────────────────────────────────────────────
DEFAULT_STOCKS = [
    "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "WIPRO.NS",
    "SBIN.NS", "IRFC.NS", "RVNL.NS", "KALYANKJIL.NS",
    "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
]

STOCK_LABELS: dict[str, str] = {
    "TCS.NS":        "TCS (Tata Consultancy)",
    "INFY.NS":       "Infosys",
    "RELIANCE.NS":   "Reliance Industries",
    "HDFCBANK.NS":   "HDFC Bank",
    "WIPRO.NS":      "Wipro",
    "SBIN.NS":       "SBI (State Bank of India)",
    "IRFC.NS":       "IRFC (Indian Railway Finance)",
    "RVNL.NS":       "RVNL (Rail Vikas Nigam)",
    "KALYANKJIL.NS": "Kalyan Jewellers",
    "AAPL":          "Apple (AAPL)",
    "GOOGL":         "Alphabet / Google (GOOGL)",
    "MSFT":          "Microsoft (MSFT)",
    "TSLA":          "Tesla (TSLA)",
    "AMZN":          "Amazon (AMZN)",
}

DEFAULT_STOCK = "TCS.NS"

# ── Data history ──────────────────────────────────────────────────────────────
# Fetch up to 15 years with progressive fallback: 15y → 10y → 5y → 2y → max
MAX_LOOKBACK_YEARS = 15
LOOKBACK_YEARS     = MAX_LOOKBACK_YEARS   # backward compatibility alias

PREDICTION_DAYS = 7

# ── Synthetic data ────────────────────────────────────────────────────────────
USE_SYNTHETIC   = True
SYNTHETIC_RATIO = 0.3

# ── LSTM hyper-parameters ─────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
LSTM_EPOCHS     = 5
LSTM_BATCH_SIZE = 32
LSTM_UNITS      = 64
TEST_SPLIT      = 0.2

# ── Reproducibility seed ──────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "saved_models")
DATABASE_PATH = os.path.join(BASE_DIR, "database", "trading.db")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

# ── Trading ───────────────────────────────────────────────────────────────────
INITIAL_BALANCE = 10_000.0

# ── Risk ──────────────────────────────────────────────────────────────────────
RISK_LOW_THRESHOLD    = 0.01
RISK_MEDIUM_THRESHOLD = 0.025

# ── Auth ──────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-32chars!!")
