"""
database/db.py
Multi-user SQLite persistence layer.

Tables
------
users        – id, username, password_hash, balance, created_at
portfolio    – id, user_id, ticker, quantity, avg_cost
transactions – id, user_id, ticker, action, quantity, price, total, timestamp
"""

import sqlite3
import hashlib
import hmac
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATABASE_PATH, INITIAL_BALANCE


# ─────────────────────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Schema bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables (idempotent). Call once at app startup."""
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT    NOT NULL,
                balance       REAL    NOT NULL DEFAULT 10000.0,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                ticker   TEXT    NOT NULL,
                quantity REAL    NOT NULL DEFAULT 0,
                avg_cost REAL    NOT NULL DEFAULT 0,
                UNIQUE(user_id, ticker)
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                ticker    TEXT    NOT NULL,
                action    TEXT    NOT NULL,
                quantity  REAL    NOT NULL,
                price     REAL    NOT NULL,
                total     REAL    NOT NULL,
                timestamp TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Password helpers  (PBKDF2-HMAC-SHA256, no bcrypt dependency needed)
# ─────────────────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """Return  salt_hex:dk_hex  (salt is 16 random bytes, 260k iterations)."""
    salt = os.urandom(16)
    dk   = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return salt.hex() + ":" + dk.hex()


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, dk_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        dk   = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> dict:
    """
    Register a new user.
    Returns {"success": bool, "message": str, "user_id": int | None}
    """
    username = username.strip()
    if len(username) < 3:
        return {"success": False, "message": "Username must be at least 3 characters.", "user_id": None}
    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters.", "user_id": None}

    pw_hash = _hash_password(password)
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO users (username, password_hash, balance) VALUES (?,?,?)",
                (username, pw_hash, INITIAL_BALANCE),
            )
            return {"success": True, "message": "Account created! Please log in.", "user_id": cur.lastrowid}
    except sqlite3.IntegrityError:
        return {"success": False, "message": f"Username '{username}' is already taken.", "user_id": None}


def login_user(username: str, password: str) -> dict:
    """
    Validate credentials.
    Returns {"success": bool, "message": str, "user": dict | None}
    """
    with _conn() as c:
        row = c.execute(
            "SELECT id, username, password_hash, balance, created_at FROM users WHERE username=?",
            (username.strip(),),
        ).fetchone()

    if row is None:
        return {"success": False, "message": "Username not found.", "user": None}
    if not _verify_password(password, row["password_hash"]):
        return {"success": False, "message": "Incorrect password.", "user": None}

    return {
        "success": True,
        "message": "Login successful!",
        "user": {
            "id":         row["id"],
            "username":   row["username"],
            "balance":    row["balance"],
            "created_at": row["created_at"],
        },
    }


def get_user_by_id(user_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT id, username, balance, created_at FROM users WHERE id=?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Balance
# ─────────────────────────────────────────────────────────────────────────────

def get_balance(user_id: int) -> float:
    with _conn() as c:
        row = c.execute("SELECT balance FROM users WHERE id=?", (user_id,)).fetchone()
    return float(row["balance"]) if row else 0.0


def set_balance(user_id: int, amount: float):
    with _conn() as c:
        c.execute("UPDATE users SET balance=? WHERE id=?", (amount, user_id))


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────────────────────────────────────

def get_portfolio(user_id: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT ticker, quantity, avg_cost FROM portfolio "
            "WHERE user_id=? AND quantity > 0",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_position(user_id: int, ticker: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT ticker, quantity, avg_cost FROM portfolio "
            "WHERE user_id=? AND ticker=?",
            (user_id, ticker),
        ).fetchone()
    return dict(row) if row else None


def upsert_position(user_id: int, ticker: str, quantity: float, avg_cost: float):
    with _conn() as c:
        c.execute("""
            INSERT INTO portfolio (user_id, ticker, quantity, avg_cost)
            VALUES (?,?,?,?)
            ON CONFLICT(user_id, ticker)
            DO UPDATE SET quantity=excluded.quantity, avg_cost=excluded.avg_cost
        """, (user_id, ticker, quantity, avg_cost))


# ─────────────────────────────────────────────────────────────────────────────
# Transactions
# ─────────────────────────────────────────────────────────────────────────────

def log_transaction(user_id: int, ticker: str, action: str,
                    quantity: float, price: float, total: float):
    with _conn() as c:
        c.execute("""
            INSERT INTO transactions (user_id, ticker, action, quantity, price, total)
            VALUES (?,?,?,?,?,?)
        """, (user_id, ticker, action, quantity, price, total))


def get_transactions(user_id: int, limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT id, ticker, action, quantity, price, total, timestamp
            FROM   transactions
            WHERE  user_id=?
            ORDER  BY id DESC
            LIMIT  ?
        """, (user_id, limit)).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Account reset (per user)
# ─────────────────────────────────────────────────────────────────────────────

def reset_account(user_id: int):
    with _conn() as c:
        c.execute("DELETE FROM transactions WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM portfolio     WHERE user_id=?", (user_id,))
        c.execute("UPDATE users SET balance=? WHERE id=?", (INITIAL_BALANCE, user_id))
