"""
utils/helpers.py
Shared formatters + all Plotly chart builders.
No Streamlit imports here — pure visualisation logic.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = "#7EB8F7"
GOLD   = "#FFD700"
GREEN  = "#00C49F"
RED    = "#FF6B6B"
PURPLE = "#B48EF7"
BG     = "#0E0E1A"
CARD   = "#1E1E2E"


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

def fmt_inr(amount: float) -> str:
    return f"₹{amount:,.2f}"


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def pnl_colour(val: float) -> str:
    return GREEN if val >= 0 else RED


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_history(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick + volume sub-plot."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color=GREEN, decreasing_line_color=RED,
    ), row=1, col=1)

    bar_colours = [GREEN if c >= o else RED
                   for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=bar_colours, name="Volume", opacity=0.55,
    ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Price History",
        xaxis_rangeslider_visible=False,
        template="plotly_dark", height=520,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor=BG, plot_bgcolor=CARD,
    )
    return fig


def plot_prediction(
    df: pd.DataFrame,
    future_dates: List[str],
    lstm_preds: List[float],
    lr_preds: List[float] | None = None,
    ticker: str = "",
) -> go.Figure:
    """Historical close + LSTM (+ optional LR) forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Historical Close",
        line=dict(color=BLUE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=lstm_preds,
        mode="lines+markers", name="LSTM Forecast",
        line=dict(color=GOLD, width=2.5, dash="dash"),
        marker=dict(size=7, symbol="circle"),
    ))
    if lr_preds is not None:
        fig.add_trace(go.Scatter(
            x=future_dates, y=lr_preds,
            mode="lines+markers", name="LR Forecast",
            line=dict(color=RED, width=2, dash="dot"),
            marker=dict(size=6),
        ))
    fig.update_layout(
        title=f"{ticker} — Price Forecast",
        xaxis_title="Date", yaxis_title="Price (₹)",
        template="plotly_dark", height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor=BG, plot_bgcolor=CARD,
    )
    return fig


def plot_actual_vs_predicted(df_with_preds: pd.DataFrame, ticker: str) -> go.Figure:
    """Actual close overlaid with in-sample model predictions (aligned series)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_with_preds.index, y=df_with_preds["Close"],
        mode="lines", name="Actual", line=dict(color=BLUE, width=1.8),
    ))
    if "LSTM_Pred" in df_with_preds.columns:
        fig.add_trace(go.Scatter(
            x=df_with_preds.index, y=df_with_preds["LSTM_Pred"],
            mode="lines", name="LSTM (in-sample)",
            line=dict(color=GOLD, width=1.5, dash="dot"),
        ))
    if "LR_Pred" in df_with_preds.columns:
        fig.add_trace(go.Scatter(
            x=df_with_preds.index, y=df_with_preds["LR_Pred"],
            mode="lines", name="LR (in-sample)",
            line=dict(color=GREEN, width=1.5, dash="dash"),
        ))
    fig.update_layout(
        title=f"{ticker} — Actual vs Predicted (in-sample)",
        template="plotly_dark", height=420,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor=BG, plot_bgcolor=CARD,
    )
    return fig


def plot_portfolio_pie(holdings: list[dict]) -> go.Figure:
    if not holdings:
        return go.Figure()
    fig = go.Figure(go.Pie(
        labels=[h["Ticker"] for h in holdings],
        values=[h["Value"]  for h in holdings],
        hole=0.48,
        textinfo="label+percent",
        marker=dict(line=dict(color=BG, width=2)),
    ))
    fig.update_layout(
        title="Portfolio Allocation",
        template="plotly_dark", height=380,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor=BG,
    )
    return fig


def plot_portfolio_pnl(holdings: list[dict]) -> go.Figure:
    """Horizontal bar chart of P&L per holding."""
    if not holdings:
        return go.Figure()
    tickers = [h["Ticker"] for h in holdings]
    pnls    = [h["P&L"]    for h in holdings]
    colours = [GREEN if p >= 0 else RED for p in pnls]

    fig = go.Figure(go.Bar(
        x=pnls, y=tickers, orientation="h",
        marker_color=colours, text=[fmt_inr(p) for p in pnls],
        textposition="outside",
    ))
    fig.update_layout(
        title="Unrealised P&L by Stock",
        template="plotly_dark", height=max(280, 60 * len(holdings)),
        margin=dict(l=10, r=80, t=45, b=10),
        paper_bgcolor=BG, plot_bgcolor=CARD,
        xaxis_title="P&L (₹)",
    )
    return fig


def plot_model_comparison(lr_metrics: dict, lstm_metrics: dict) -> go.Figure:
    """
    Side-by-side grouped bar chart for RMSE, MAE and Accuracy.
    Accuracy is plotted on a secondary y-axis (0-100 %) so it is not
    dwarfed by the RMSE/MAE scale.
    """
    from plotly.subplots import make_subplots

    # Error metrics (lower is better) — primary axis
    error_metrics = ["RMSE", "MAE"]
    lr_err   = [lr_metrics.get(m, 0)   for m in error_metrics]
    lstm_err = [lstm_metrics.get(m, 0) for m in error_metrics]

    # Accuracy (higher is better) — secondary axis
    lr_acc   = lr_metrics.get("Accuracy",   0)
    lstm_acc = lstm_metrics.get("Accuracy", 0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Error bars — primary y
    fig.add_trace(go.Bar(
        name="LR — Error", x=error_metrics, y=lr_err,
        marker_color=BLUE, offsetgroup=0,
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        name="LSTM — Error", x=error_metrics, y=lstm_err,
        marker_color=GOLD, offsetgroup=1,
    ), secondary_y=False)

    # Accuracy bars — secondary y  (plotted at x position "Accuracy")
    fig.add_trace(go.Bar(
        name="LR — Accuracy (%)",   x=["Accuracy"], y=[lr_acc],
        marker_color=GREEN, offsetgroup=0,
    ), secondary_y=True)
    fig.add_trace(go.Bar(
        name="LSTM — Accuracy (%)", x=["Accuracy"], y=[lstm_acc],
        marker_color=PURPLE, offsetgroup=1,
    ), secondary_y=True)

    fig.update_layout(
        barmode="group",
        title="Model Comparison — RMSE, MAE & Accuracy",
        template="plotly_dark", height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=BG, plot_bgcolor=CARD,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Error (₹ / units)", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy (%)", secondary_y=True,
                     range=[0, 105])
    return fig


def plot_moving_averages(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"],
                              name="Close", line=dict(color=BLUE, width=1.5)))
    for window, colour, dash in [
        (7,  GOLD,   "dot"),
        (21, GREEN,  "dash"),
        (50, RED,    "longdash"),
    ]:
        ma = df["Close"].rolling(window).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma, name=f"MA {window}",
            line=dict(color=colour, dash=dash, width=1.4),
        ))
    fig.update_layout(
        title=f"{ticker} — Moving Averages",
        template="plotly_dark", height=420,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor=BG, plot_bgcolor=CARD,
    )
    return fig


def plot_returns_dist(df: pd.DataFrame) -> go.Figure:
    returns = df["Close"].pct_change().dropna() * 100
    fig = px.histogram(
        returns, nbins=60, template="plotly_dark",
        labels={"value": "Daily Return (%)"},
        color_discrete_sequence=[BLUE],
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=BG, plot_bgcolor=CARD,
    )
    return fig
