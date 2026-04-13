# 📈 AI Stock Market Predictor & Trading Simulator

> A full-stack web application that predicts stock prices with Machine Learning and lets you practise trading with virtual money — no real money at risk.

---

## 🚀 Live Demo

Deploy instantly on **Streamlit Cloud** (see Deployment section below).

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 📊 Dashboard | Candlestick + volume charts, 52-week stats |
| 🔮 Prediction | LSTM & Linear Regression, 1–30 day forecast |
| 💹 Trading Simulator | Buy / Sell with virtual ₹10,000 wallet |
| 💼 Portfolio | Holdings, P&L, pie-chart allocation |
| 🔍 Insights | Volatility, risk score, moving averages, model comparison |
| 🧬 Synthetic Data | GBM random walk + trend-based augmentation |

---

## 🧠 Tech Stack

- **Frontend / UI**: Streamlit
- **ML**: TensorFlow / Keras (LSTM), scikit-learn (Linear Regression)
- **Data**: yfinance (real-time), custom synthetic generator
- **Visualisation**: Plotly
- **Database**: SQLite (via Python stdlib)
- **Language**: Python 3.10+

---

## 📁 Project Structure

```
stock_predictor/
├── app.py                   ← Streamlit entry point
├── config.py                ← All settings & constants
├── requirements.txt
│
├── data/
│   ├── fetch_data.py        ← yfinance wrapper
│   ├── preprocess.py        ← scaling, sequences, features
│   └── synthetic_data.py    ← GBM + trend + noise augmentation
│
├── models/
│   ├── regression_model.py  ← Linear Regression wrapper
│   ├── lstm_model.py        ← LSTM wrapper
│   └── train_model.py       ← CLI training script
│
├── services/
│   ├── prediction_service.py ← predict_next_days() API
│   └── trading_service.py    ← buy / sell / portfolio logic
│
├── utils/
│   └── helpers.py           ← Plotly charts + formatters
│
├── database/
│   └── db.py                ← SQLite CRUD helpers
│
├── saved_models/            ← Auto-created; stores .h5 + .pkl
└── .streamlit/
    └── config.toml          ← Dark theme
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10 or 3.11
- pip

### Step-by-step

```bash
# 1. Clone / download the project
git clone https://github.com/your-username/stock-predictor.git
cd stock-predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-train models for a ticker
python models/train_model.py --ticker TCS.NS --days 7

# 5. Launch the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

> 💡 Models are automatically trained on first use if `saved_models/` is empty.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this folder to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repo, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Deploy** — done!

### Memory optimisation for cloud
- Set `LSTM_EPOCHS = 10` in `config.py` to reduce training time.
- Set `USE_SYNTHETIC = False` if RAM is tight.
- The app uses `@st.cache_data` to avoid redundant API calls.

---

## 🔧 Configuration

Edit `config.py` to customise:

```python
INITIAL_BALANCE  = 10_000.0   # virtual wallet in ₹
PREDICTION_DAYS  = 7           # default forecast horizon
USE_SYNTHETIC    = True        # toggle synthetic data augmentation
LSTM_EPOCHS      = 30          # reduce for faster training on cloud
SEQUENCE_LENGTH  = 60          # LSTM look-back window
```

---

## 📸 Screenshots

| Dashboard | Prediction | Portfolio |
|-----------|-----------|-----------|
| *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## 🔮 Future Improvements

- [ ] News sentiment analysis (NewsAPI integration)
- [ ] Transformer / Attention model
- [ ] Options pricing simulator
- [ ] Email / Telegram trade alerts
- [ ] Multi-user authentication
- [ ] Real broker API integration (Zerodha Kite)

---

## 📄 License

MIT — free to use, fork, and build upon.
