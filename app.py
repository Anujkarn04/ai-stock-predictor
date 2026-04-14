"""
app.py — AI Stock Predictor & Trading Simulator
Entry point for Streamlit.  Run: streamlit run app.py

Architecture
------------
* All business logic lives in services/, models/, database/
* app.py only handles UI routing and calls service functions
* Auth state: st.session_state["logged_in"], ["user_id"], ["username"]
"""

import streamlit as st
import pandas as pd
import sys, os
IS_CLOUD = os.getenv("STREAMLIT_SERVER_RUN_ON_SAVE") is not None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Bootstrap DB ─────────────────────────────────────────────────────────────
from database.db import init_db
init_db()

# ── Services ──────────────────────────────────────────────────────────────────
from services.auth_service     import is_logged_in, current_user, current_user_id, do_logout, render_auth_page
from services.prediction_service import predict_next_days, get_risk_score, get_metrics, get_history_with_predictions
from services.trading_service  import buy_stock, sell_stock, get_portfolio_summary, get_transaction_history
from database.db               import reset_account

# ── Data / Config ─────────────────────────────────────────────────────────────
from data.fetch_data  import fetch_stock_data, get_current_price, get_stock_info, get_data_years
from config           import DEFAULT_STOCKS, DEFAULT_STOCK, PREDICTION_DAYS, INITIAL_BALANCE, STOCK_LABELS

# ── Helpers ───────────────────────────────────────────────────────────────────
from utils.helpers import (
    fmt_inr, fmt_pct, pnl_colour,
    plot_price_history, plot_prediction, plot_actual_vs_predicted,
    plot_portfolio_pie, plot_portfolio_pnl,
    plot_model_comparison, plot_moving_averages, plot_returns_dist,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --accent: #FFD700; --green: #00C49F;
    --red: #FF6B6B;    --blue: #7EB8F7;
    --bg:  #0E0E1A;    --card: #1E1E2E;
}
div[data-testid="metric-container"] {
    background: var(--card); border-radius: 12px;
    padding: 14px 18px; border: 1px solid #2a2a3e;
}
.stMetric label { font-size: .82rem !important; color: #aaa !important; }
h1 { color: var(--accent) !important; }
h2 { color: #e8e8f0 !important; }
/* Badge pills */
.badge-green { background:#00C49F22; color:#00C49F; padding:3px 10px;
               border-radius:20px; font-size:.82rem; font-weight:600; }
.badge-yellow{ background:#FFD70022; color:#FFD700; padding:3px 10px;
               border-radius:20px; font-size:.82rem; font-weight:600; }
.badge-red   { background:#FF6B6B22; color:#FF6B6B; padding:3px 10px;
               border-radius:20px; font-size:.82rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUTH GATE  — show login page if not authenticated
# ─────────────────────────────────────────────────────────────────────────────
if not is_logged_in():
    render_auth_page()
    st.stop()          # nothing below renders until logged in


# ─────────────────────────────────────────────────────────────────────────────
# From here on the user IS authenticated
# ─────────────────────────────────────────────────────────────────────────────
uid  = current_user_id()
user = current_user()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=56)
    st.title("AI Stock Predictor")
    st.caption("LSTM · Linear Regression · Live data")
    st.divider()

    # User info
    st.markdown(f"👤 **{user['username']}**")
    st.caption(f"Member since {user['created_at'][:10]}")
    if st.button("🚪 Logout", use_container_width=True):
        do_logout()

    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 Predict", "💹 Trade", "💼 Portfolio", "🔍 Insights"],
        label_visibility="collapsed",
    )
    st.divider()

    # Stock picker
    # Build human-readable labels for the dropdown
    _labels = [STOCK_LABELS.get(s, s) for s in DEFAULT_STOCKS]
    _sel_idx = DEFAULT_STOCKS.index(DEFAULT_STOCK)
    _selected_label = st.selectbox("Select Stock", _labels, index=_sel_idx)
    ticker = DEFAULT_STOCKS[_labels.index(_selected_label)]
    custom = st.text_input("Custom ticker", placeholder="e.g. BAJFINANCE.NS")
    if custom.strip():
        ticker = custom.strip().upper()

    st.divider()

    # Live wallet KPIs (user-specific)
    summary = get_portfolio_summary(uid)
    st.metric("💰 Cash",      fmt_inr(summary["cash_balance"]))
    st.metric("📦 Net Worth", fmt_inr(summary["net_worth"]))

    pnl = summary["total_pnl"]
    st.metric(
        "📈 Unrealised P&L",
        fmt_inr(pnl),
        delta=f"{pnl / max(summary['total_invested'], 1) * 100:+.2f}%",
    )
    st.divider()

    if st.button("🔄 Reset Account", use_container_width=True):
        reset_account(uid)
        st.success(f"Reset to {fmt_inr(INITIAL_BALANCE)}")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Cached helpers
# ─────────────────────────────────────────────────────────────────────────────

# @st.cache_data  — serialisable return values (DataFrames, dicts, lists).
#                   TTL expires after N seconds; refreshed on next call.
# @st.cache_resource — non-serialisable objects kept in memory for the
#                   lifetime of the Streamlit server process (model objects,
#                   DB connections, etc.).  NEVER reloads from disk on rerun.

@st.cache_data(ttl=900, show_spinner=False)
def _fetch(t: str):
    """Fetch OHLCV data once; cached 5 min per ticker."""
    return fetch_stock_data(t)

@st.cache_data(ttl=600, show_spinner=False)
def _predict(t: str, days: int, model: str):
    """Cached forecast — only recomputes if ticker/days/model changes."""
    return predict_next_days(t, days, model)

@st.cache_data(ttl=600, show_spinner=False)
def _metrics(t: str):
    """Cached evaluation metrics — button-triggered, never auto-rerun."""
    return get_metrics(t)

@st.cache_data(ttl=600, show_spinner=False)
def _history_preds(t: str):
    """Cached in-sample predictions aligned to historical index."""
    return get_history_with_predictions(t)

@st.cache_resource(show_spinner=False)
def _load_models(t: str):
    """
    Load (or train) LR + LSTM models for ticker *t* and keep them in memory.

    @st.cache_resource means this runs ONCE per (ticker, server process) and
    the model objects are reused on every subsequent Streamlit rerun — no
    repeated disk I/O, no retraining, no lag on page interactions.
    """
    from services.prediction_service import _ensure_trained
    return _ensure_trained(t)   # returns (lr, lstm, real_df)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Market Dashboard")

    try:
        # ── FIX: all metadata access is crash-proof ───────────────────────────
        info = get_stock_info(ticker)    # returns safe defaults on failure
        cur  = get_current_price(ticker) # returns 0.0 on failure
        df = _fetch(ticker)

        # 🔥 Safety checks (add THIS)
        if df is None or df.empty:
            st.error(f"No usable data for '{ticker}'. Try another stock.")
            st.stop()

        required_cols = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_cols):
            st.error("Stock data format invalid. Try another stock.")
            st.stop()

        if len(df) < 2:
            st.warning("Not enough data to display chart.")
            st.stop()

        # ── KPI row ──────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        # FIX: str() + safe slice prevents crash when name is None
        c1.metric("Company",  str(info.get("name", ticker))[:24])
        c2.metric("Sector",   str(info.get("sector", "N/A")))
        c3.metric("Exchange", str(info.get("exchange", "N/A")))
        c4.metric("Current Price", fmt_inr(cur) if cur > 0 else "N/A")

        # FIX: guard against df with < 2 rows
        if len(df) >= 2:
            chg = float(df["Close"].iloc[-1] - df["Close"].iloc[-2])
            pct = chg / float(df["Close"].iloc[-2]) * 100 if df["Close"].iloc[-2] != 0 else 0.0
            c5.metric("Day Change", fmt_inr(chg), f"{pct:+.2f}%")
        else:
            c5.metric("Day Change", "N/A")

        # Data context banner
        data_yrs = get_data_years(df)
        st.caption(f"📅 Historical data: **{data_yrs:.1f} years** ({len(df):,} trading days)")
        st.divider()

        # ── Chart controls ────────────────────────────────────────────────────
        col_period, col_type = st.columns([2, 3])
        with col_period:
            # Limit available periods to what data actually covers
            all_periods  = ["1M","3M","6M","1Y","2Y","5Y","10Y","15Y"]
            period_days  = {"1M":22,"3M":66,"6M":130,"1Y":252,"2Y":504,
                            "5Y":1260,"10Y":2520,"15Y":3780}
            avail        = [p for p in all_periods if period_days[p] <= len(df)]
            if not avail:
                avail = ["1M"]
            default_p    = "6M" if "6M" in avail else avail[-1]
            period       = st.select_slider("Period", avail, value=default_p)
        with col_type:
            chart_type = st.radio("Chart", ["Candlestick","Line"], horizontal=True)

        ndays = period_days[period]
        df_sl = df.tail(ndays)

        if chart_type == "Candlestick":
            st.plotly_chart(plot_price_history(df_sl, ticker), use_container_width=True)
        else:
            import plotly.express as px
            fig = px.line(df_sl, y="Close", template="plotly_dark",
                          title=f"{ticker} Closing Price")
            fig.update_layout(paper_bgcolor="#0E0E1A", plot_bgcolor="#1E1E2E")
            st.plotly_chart(fig, use_container_width=True)

        # ── Stats ─────────────────────────────────────────────────────────────
        st.subheader("📌 Quick Stats")
        s1, s2, s3, s4, s5 = st.columns(5)
        w52 = df.tail(min(252, len(df)))
        s1.metric("52W High",    fmt_inr(float(w52["High"].max())))
        s2.metric("52W Low",     fmt_inr(float(w52["Low"].min())))
        s3.metric("30D Avg Vol", f"{df.tail(min(30,len(df)))['Volume'].mean():,.0f}")
        s4.metric("Data Years",  f"{data_yrs:.1f} yrs")
        # FIX: guard against < 22 rows (new/recently listed stocks)
        if len(df) >= 22:
            monthly_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1) * 100
            s5.metric("1M Return", f"{monthly_ret:+.2f}%")
        else:
            s5.metric("1M Return", "N/A")

    except Exception as e:
        st.error(f"Failed to load dashboard: {e}")
        st.info("Try a different stock or check your internet connection.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.title("🔮 Price Prediction")
    st.caption("LSTM deep-learning model + Linear Regression baseline")

    ca, cb, cc = st.columns(3)
    with ca:
        days = st.slider("Forecast horizon (days)", 1, 30, PREDICTION_DAYS)
    with cb:
        if IS_CLOUD:
            model_type = st.selectbox("Model", ["LR"])
            st.info("⚡ LSTM model is available in local version only.")
        else:
            model_type = st.selectbox("Model", ["LSTM","LR","Both"])
    with cc:
        run = st.button("▶ Run Forecast", type="primary", use_container_width=True)

    if run:
        # Clear cache for this ticker so re-training picks up fresh data
        _predict.clear()

    if run or st.session_state.get("last_pred_ticker") == ticker:
        with st.spinner(f"Generating {days}-day forecast …"):
            try:
                # Force LR on cloud (extra safety)
                if IS_CLOUD:
                    model_type = "LR"

                res = _predict(ticker, days, model_type)
                df  = res["historical"]

                # ── Forecast chart ────────────────────────────────────────────
                st.plotly_chart(
                    plot_prediction(
                        df.tail(120), res["dates"], res["predictions"],
                        res.get("lr_predictions"), ticker,
                    ),
                    use_container_width=True,
                )

                # ── Forecast table ────────────────────────────────────────────
                st.subheader("📋 Forecast Table")
                tbl = {"Date": res["dates"],
                       f"{model_type if model_type != 'Both' else 'LSTM'} Prediction":
                           [fmt_inr(v) for v in res["predictions"]]}
                if res.get("lr_predictions"):
                    tbl["LR Prediction"] = [fmt_inr(v) for v in res["lr_predictions"]]
                st.dataframe(pd.DataFrame(tbl), use_container_width=True)

                st.session_state["last_pred_ticker"] = ticker

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Tip: Click 'Run Forecast' again. On first run, training may take ~1 min.")

    # ── In-sample actual vs predicted ─────────────────────────────────────────
    with st.expander("📉 Show In-Sample: Actual vs Predicted"):
        try:
            with st.spinner("Loading in-sample predictions …"):
                hp  = _history_preds(ticker)
                df2 = hp["df"]
            st.plotly_chart(plot_actual_vs_predicted(df2, ticker),
                            use_container_width=True)
        except Exception as e:
            st.error(f"Could not load in-sample chart: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRADE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💹 Trade":
    st.title("💹 Trading Simulator")

    try:
        price   = get_current_price(ticker)
        balance = get_portfolio_summary(uid)["cash_balance"]
    except Exception as e:
        st.error(str(e)); st.stop()

    max_buy = int(balance // price) if price > 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", fmt_inr(price))
    m2.metric("Cash Balance",  fmt_inr(balance))
    m3.metric("Max Buyable",   f"{max_buy} shares")

    st.divider()
    tab_buy, tab_sell = st.tabs(["🟢 Buy", "🔴 Sell"])

    # ── Buy tab ───────────────────────────────────────────────────────────────
    with tab_buy:
        qty_b = st.number_input("Quantity to BUY", min_value=1,
                                 max_value=max(1, max_buy), value=1, step=1,
                                 key="buy_qty")
        st.info(f"Estimated cost: **{fmt_inr(qty_b * price)}**")
        if st.button(f"Buy {qty_b} × {ticker}", type="primary",
                     use_container_width=True):
            res = buy_stock(uid, ticker, float(qty_b))
            if res["success"]:
                st.success(res["message"])
                st.rerun()
            else:
                st.error(res["message"])

    # ── Sell tab ──────────────────────────────────────────────────────────────
    with tab_sell:
        pos  = get_portfolio_summary(uid)
        held = next((h["Qty"] for h in pos["holdings"] if h["Ticker"] == ticker), 0)
        st.caption(f"You hold **{held}** share(s) of **{ticker}**")
        qty_s = st.number_input("Quantity to SELL", min_value=1,
                                 max_value=max(1, int(held)), value=1, step=1,
                                 key="sell_qty")
        pnl_est = (price - next(
            (h["Avg Cost"] for h in pos["holdings"] if h["Ticker"] == ticker),
            price,
        )) * qty_s
        st.info(
            f"Estimated proceeds: **{fmt_inr(qty_s * price)}**  |  "
            f"Est. P&L: **{fmt_inr(pnl_est)}**"
        )
        if st.button(f"Sell {qty_s} × {ticker}", type="secondary",
                     use_container_width=True):
            res = sell_stock(uid, ticker, float(qty_s))
            if res["success"]:
                st.success(res["message"])
                st.rerun()
            else:
                st.error(res["message"])

    # ── Recent transactions ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Recent Transactions")
    txns = get_transaction_history(uid, 20)
    if txns:
        df_t = pd.DataFrame(txns)[["timestamp","ticker","action","quantity","price","total"]]
        df_t["price"] = df_t["price"].apply(fmt_inr)
        df_t["total"] = df_t["total"].apply(fmt_inr)
        st.dataframe(df_t, use_container_width=True, hide_index=True)
    else:
        st.info("No transactions yet. Buy your first stock above! 🚀")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Portfolio":
    st.title(f"💼 {user['username']}'s Portfolio")
    summary = get_portfolio_summary(uid)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cash Balance",   fmt_inr(summary["cash_balance"]))
    k2.metric("Holdings Value", fmt_inr(summary["total_value"]))
    k3.metric("Net Worth",      fmt_inr(summary["net_worth"]))
    inv = max(summary["total_invested"], 1)
    k4.metric(
        "Unrealised P&L",
        fmt_inr(summary["total_pnl"]),
        delta=f"{summary['total_pnl'] / inv * 100:+.2f}%",
    )

    st.divider()

    if summary["holdings"]:
        col_tbl, col_vis = st.columns([3, 2])

        with col_tbl:
            st.subheader("Holdings")
            df_h = pd.DataFrame(summary["holdings"])

            def _colour(val: object) -> str:
                """Green/red CSS for numeric P&L values; neutral for non-numeric."""
                try:
                    return "color: #00C49F" if float(val) >= 0 else "color: #FF6B6B"
                except (TypeError, ValueError):
                    return ""

            def _detect_pnl_cols(df: "pd.DataFrame") -> "list[str]":
                """
                Dynamically detect P&L columns regardless of exact naming.
                Handles: 'P&L', 'P&L %', 'PnL', 'PnL %', 'Profit/Loss', 'gain', etc.
                Returns only columns that actually exist in df.
                """
                found = []
                for col in df.columns:
                    normalised = col.lower().replace(" ", "").replace("&", "").replace("/", "")
                    if any(k in normalised for k in ["pnl", "p&l", "pl", "profitloss", "gain", "loss"]):
                        found.append(col)
                return found

            # ── Robust styling: safe under ALL edge cases ─────────────────────
            # Scenario 1 — Normal: P&L columns present        → styled green/red
            # Scenario 2 — Renamed cols (PnL, gain, etc.)     → detected dynamically
            # Scenario 3 — Empty df                           → no crash, empty table
            # Scenario 4 — P&L cols completely missing        → unstyled fallback
            # Compatibility: uses .map() (pandas >= 2.1) with automatic fallback
            #                to .applymap() for older pandas installs
            if not df_h.empty:
                style_cols = _detect_pnl_cols(df_h)
                if style_cols:
                    _apply = (
                        df_h.style.map        # pandas >= 2.1 (applymap deprecated)
                        if hasattr(df_h.style, "map")
                        else df_h.style.applymap  # pandas < 2.1 fallback
                    )
                    display_df = _apply(_colour, subset=style_cols)
                else:
                    display_df = df_h.style   # P&L cols not found → unstyled
            else:
                display_df = df_h.style       # empty df → show empty table cleanly

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

        with col_vis:
            vis_tab = st.radio("View", ["Allocation", "P&L by Stock"],
                               horizontal=True, key="port_vis")
            if vis_tab == "Allocation":
                st.plotly_chart(plot_portfolio_pie(summary["holdings"]),
                                use_container_width=True)
            else:
                st.plotly_chart(plot_portfolio_pnl(summary["holdings"]),
                                use_container_width=True)

        # ── Portfolio growth (net worth over time from transactions) ──────────
        st.divider()
        st.subheader("📈 Activity Timeline")
        all_txns = get_transaction_history(uid, 200)
        if len(all_txns) >= 2:
            df_txn = pd.DataFrame(all_txns)
            df_txn["timestamp"] = pd.to_datetime(df_txn["timestamp"])
            df_txn = df_txn.sort_values("timestamp")
            # Running total spent/received
            df_txn["signed_total"] = df_txn.apply(
                lambda r: -r["total"] if r["action"] == "BUY" else r["total"],
                axis=1,
            )
            df_txn["cumulative_pnl"] = df_txn["signed_total"].cumsum()

            import plotly.express as px
            fig = px.line(df_txn, x="timestamp", y="cumulative_pnl",
                          template="plotly_dark",
                          labels={"cumulative_pnl": "Cumulative Cash Flow (₹)",
                                  "timestamp": "Date"},
                          title="Cumulative Cash Flow from Trades")
            fig.update_traces(line_color="#FFD700", line_width=2)
            fig.update_layout(paper_bgcolor="#0E0E1A", plot_bgcolor="#1E1E2E",
                              height=320, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Your portfolio is empty. Go to **Trade** to buy your first stock! 🚀")

    # ── Full transaction history ───────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Full Transaction History")
    all_txns2 = get_transaction_history(uid, 200)
    if all_txns2:
        st.dataframe(pd.DataFrame(all_txns2), use_container_width=True, hide_index=True)
    else:
        st.info("No transactions recorded yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Insights":
    st.title("🔍 Market Insights")

    try:
        df = _fetch(ticker)

        # 🔥 Safety checks (add THIS)
        if df is None or df.empty:
            st.error(f"No usable data for '{ticker}'. Try another stock.")
            st.stop()

        required_cols = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_cols):
            st.error("Stock data format invalid. Try another stock.")
            st.stop()

        if len(df) < 2:
            st.warning("Not enough data to display chart.")
            st.stop()
            
        risk = get_risk_score(df)

        # ── Risk banner ───────────────────────────────────────────────────────
        badge_cls = {"Low": "badge-green", "Medium": "badge-yellow",
                     "High": "badge-red"}.get(risk["risk_level"], "badge-green")
        st.markdown(
            f"<span class='{badge_cls}'>Risk Level: "
            f"{risk['risk_badge']} {risk['risk_level']}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Daily Volatility",   f"{risk['daily_std']} %")
        r2.metric("Annual Volatility",  f"{risk['annual_vol']} %")
        r3.metric("Sharpe (approx)",    str(risk["sharpe_approx"]))
        r4.metric("Risk Category",      f"{risk['risk_badge']} {risk['risk_level']}")

        st.divider()

        # ── Moving averages ────────────────────────────────────────────────────
        st.subheader("📈 Moving Averages")
        st.plotly_chart(plot_moving_averages(df, ticker), use_container_width=True)

        # ── Returns distribution ───────────────────────────────────────────────
        st.subheader("📊 Daily Returns Distribution")
        st.plotly_chart(plot_returns_dist(df), use_container_width=True)

        # ── Model comparison ─────────────────────────────────────────────────
        st.divider()
        st.subheader("🤖 Model Comparison")

        if st.button("Run Model Evaluation (≈1–2 min)", type="secondary"):
            with st.spinner("Training both models for evaluation …"):
                _metrics.clear()
                metrics = _metrics(ticker)

            # ── Data context ─────────────────────────────────────────────────
            d_yrs  = metrics["LR"].get("data_years", "?")
            d_rows = metrics["LR"].get("data_rows",  "?")

            st.success(f"📅 **Trained on {d_yrs} years of data** ({d_rows:,} trading days)")

            col_lr, col_lstm = st.columns(2)

            # ── Linear Regression Metrics ────────────────────────────────────
            col_lr.metric("LR RMSE", metrics["LR"]["RMSE"])
            col_lr.metric("LR MAE", metrics["LR"]["MAE"])
            col_lr.metric("LR MAPE", f"{metrics['LR'].get('MAPE', 'N/A')} %")
            col_lr.metric("LR Accuracy", f"{metrics['LR'].get('Accuracy', 'N/A')} %")

            # ✅ NEW
            col_lr.metric(
                "LR Direction Accuracy",
                f"{metrics['LR'].get('Direction Accuracy', 'N/A')} %"
            )

            # ── LSTM Metrics ─────────────────────────────────────────────────
            col_lstm.metric("LSTM RMSE", metrics["LSTM"]["RMSE"])
            col_lstm.metric("LSTM MAE", metrics["LSTM"]["MAE"])
            col_lstm.metric("LSTM MAPE", f"{metrics['LSTM'].get('MAPE', 'N/A')} %")
            col_lstm.metric("LSTM Accuracy", f"{metrics['LSTM'].get('Accuracy', 'N/A')} %")

            # ✅ NEW
            col_lstm.metric(
                "LSTM Direction Accuracy",
                f"{metrics['LSTM'].get('Direction Accuracy', 'N/A')} %"
            )

            # ── Better comparison logic (REAL) ────────────────────────────────
            lr_dir   = metrics["LR"].get("Direction Accuracy", 0)
            lstm_dir = metrics["LSTM"].get("Direction Accuracy", 0)

            winner = "LSTM 🏆" if lstm_dir >= lr_dir else "Linear Regression 🏆"

            st.info(
                f"**Better Model (Direction Accuracy):** {winner}  |  "
                f"LR: {lr_dir}%  vs  LSTM: {lstm_dir}%  |  "
                f"Trained on: {d_yrs} yrs"
            )

            # ── Chart ────────────────────────────────────────────────────────
            st.plotly_chart(
                plot_model_comparison(metrics["LR"], metrics["LSTM"]),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Could not load insights: {e}")

