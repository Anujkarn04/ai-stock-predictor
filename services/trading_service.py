"""
services/trading_service.py
Multi-user virtual trading logic.
All functions accept user_id as first argument — no globals.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.db import (
    get_balance, set_balance,
    get_portfolio, get_position, upsert_position,
    log_transaction, get_transactions,
)
from data.fetch_data import get_current_price


# ─────────────────────────────────────────────────────────────────────────────
# Buy
# ─────────────────────────────────────────────────────────────────────────────

def buy_stock(user_id: int, ticker: str, quantity: float) -> dict:
    """
    Purchase *quantity* shares of *ticker* at current market price.
    Returns {"success": bool, "message": str, ...extra...}
    """
    if quantity <= 0:
        return {"success": False, "message": "Quantity must be positive."}

    price = get_current_price(ticker)
    if price <= 0:
        return {"success": False, "message": f"Could not fetch price for {ticker}."}

    total   = price * quantity
    balance = get_balance(user_id)

    if total > balance:
        return {
            "success": False,
            "message": (
                f"Insufficient funds. "
                f"Need ₹{total:,.2f}, have ₹{balance:,.2f}."
            ),
        }

    # Weighted-average cost basis
    pos = get_position(user_id, ticker)
    if pos and pos["quantity"] > 0:
        new_qty  = pos["quantity"] + quantity
        new_cost = (pos["avg_cost"] * pos["quantity"] + price * quantity) / new_qty
    else:
        new_qty, new_cost = quantity, price

    upsert_position(user_id, ticker, new_qty, new_cost)
    set_balance(user_id, balance - total)
    log_transaction(user_id, ticker, "BUY", quantity, price, total)

    return {
        "success":  True,
        "message":  f"✅ Bought {quantity} share(s) of {ticker} @ ₹{price:,.2f}",
        "price":    price,
        "total":    total,
        "balance":  get_balance(user_id),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sell
# ─────────────────────────────────────────────────────────────────────────────

def sell_stock(user_id: int, ticker: str, quantity: float) -> dict:
    """Sell *quantity* shares of *ticker* at current market price."""
    if quantity <= 0:
        return {"success": False, "message": "Quantity must be positive."}

    pos = get_position(user_id, ticker)
    if not pos or pos["quantity"] < quantity:
        held = pos["quantity"] if pos else 0
        return {
            "success": False,
            "message": f"Not enough shares. You hold {held} of {ticker}.",
        }

    price = get_current_price(ticker)
    if price <= 0:
        return {"success": False, "message": f"Could not fetch price for {ticker}."}

    total   = price * quantity
    balance = get_balance(user_id)
    new_qty = pos["quantity"] - quantity

    upsert_position(user_id, ticker, new_qty, pos["avg_cost"])
    set_balance(user_id, balance + total)
    log_transaction(user_id, ticker, "SELL", quantity, price, total)

    pnl = (price - pos["avg_cost"]) * quantity
    return {
        "success": True,
        "message": (
            f"✅ Sold {quantity} share(s) of {ticker} @ ₹{price:,.2f}  |  "
            f"P&L: ₹{pnl:+,.2f}"
        ),
        "price":   price,
        "total":   total,
        "pnl":     pnl,
        "balance": get_balance(user_id),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio summary
# ─────────────────────────────────────────────────────────────────────────────

def get_portfolio_summary(user_id: int) -> dict:
    """
    Return full portfolio valuation for *user_id*.
    """
    holdings       = get_portfolio(user_id)
    rows           = []
    total_invested = 0.0
    total_value    = 0.0

    for h in holdings:
        if h["quantity"] <= 0:
            continue
        try:
            cur_price = get_current_price(h["ticker"])
        except Exception:
            cur_price = h["avg_cost"]   # fallback to cost basis

        invested = h["avg_cost"] * h["quantity"]
        value    = cur_price    * h["quantity"]
        pnl      = value - invested
        pct      = (pnl / invested * 100) if invested > 0 else 0.0

        rows.append({
            "Ticker":        h["ticker"],
            "Qty":           h["quantity"],
            "Avg Cost":      round(h["avg_cost"],  2),
            "Current Price": round(cur_price,      2),
            "Invested":      round(invested,       2),
            "Value":         round(value,          2),
            "P&L":           round(pnl,            2),
            "P&L %":         round(pct,            2),
        })
        total_invested += invested
        total_value    += value

    balance = get_balance(user_id)
    return {
        "holdings":       rows,
        "total_invested": round(total_invested, 2),
        "total_value":    round(total_value,    2),
        "total_pnl":      round(total_value - total_invested, 2),
        "cash_balance":   round(balance, 2),
        "net_worth":      round(balance + total_value, 2),
    }


def get_transaction_history(user_id: int, limit: int = 50) -> list[dict]:
    return get_transactions(user_id, limit)
