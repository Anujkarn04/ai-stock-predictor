"""
services/auth_service.py
Streamlit session management + login / register wrappers.

All Streamlit session state is managed here — app.py just calls these helpers.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from database.db import login_user, register_user, get_user_by_id


# ─────────────────────────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)


def current_user() -> dict | None:
    """Return the logged-in user dict, or None."""
    if not is_logged_in():
        return None
    # Re-fetch from DB on every call to keep balance fresh
    uid = st.session_state.get("user_id")
    return get_user_by_id(uid) if uid else None


def current_user_id() -> int | None:
    return st.session_state.get("user_id")


def do_logout():
    for key in ("logged_in", "user_id", "username", "auth_page"):
        st.session_state.pop(key, None)
    st.rerun()


def _set_session(user: dict):
    st.session_state["logged_in"] = True
    st.session_state["user_id"]   = user["id"]
    st.session_state["username"]  = user["username"]


# ─────────────────────────────────────────────────────────────────────────────
# Login / Register UI  (rendered inside app.py when not authenticated)
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.auth-card {
    max-width: 440px; margin: 10px auto;; padding: 40px 44px;
    background: #1e1e2e; border: 1px solid #2a2a3e;
    border-radius: 18px; box-shadow: 0 8px 32px rgba(0,0,0,.45);
}
.auth-title { font-size: 1.7rem; font-weight: 700; color: #FFD700;
              text-align: center; margin-bottom: 6px; }
.auth-sub   { text-align: center; color: #888; font-size: .9rem;
              margin-bottom: 24px; }
.tab-link   { cursor: pointer; font-size: .9rem; }
</style>
"""


def render_auth_page():
    """
    Render the login / register page.
    Returns True once the user is successfully authenticated (triggers rerun).
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    # Choose tab via session state
    if "auth_page" not in st.session_state:
        st.session_state["auth_page"] = "login"

    # ── Center column ────────────────────────────────────────────────────────
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">📈 AI Stock Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Virtual trading · ML forecasts · Real data</div>', unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        # ── LOGIN ────────────────────────────────────────────────────────────
        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", key="li_user")
                password = st.text_input("Password", type="password", key="li_pass")
                submitted = st.form_submit_button("Login", use_container_width=True,
                                                  type="primary")
            if submitted:
                if not username or not password:
                    st.error("Please fill in both fields.")
                else:
                    res = login_user(username, password)
                    if res["success"]:
                        _set_session(res["user"])
                        st.success(f"Welcome back, **{res['user']['username']}**! 🎉")
                        st.rerun()
                    else:
                        st.error(res["message"])

        # ── REGISTER ─────────────────────────────────────────────────────────
        with tab_register:
            with st.form("register_form", clear_on_submit=True):
                new_user = st.text_input("Choose a username", key="reg_user")
                new_pass = st.text_input("Password (min 6 chars)", type="password",
                                         key="reg_pass")
                confirm  = st.text_input("Confirm password", type="password",
                                         key="reg_confirm")
                submitted = st.form_submit_button("Create Account",
                                                  use_container_width=True,
                                                  type="primary")
            if submitted:
                if not new_user or not new_pass:
                    st.error("Please fill in all fields.")
                elif new_pass != confirm:
                    st.error("Passwords do not match.")
                else:
                    res = register_user(new_user, new_pass)
                    if res["success"]:
                        st.success(res["message"] + " Please log in.")
                    else:
                        st.error(res["message"])

        st.markdown("</div>", unsafe_allow_html=True)
