"""
utils/helper.py - Session state management, history utilities, auth helpers
"""
import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import HISTORY_CSV, HISTORY_DIR


# ─────────────────────────────────────────────
# HISTORY COLUMNS
# ─────────────────────────────────────────────
HISTORY_COLS = [
    "timestamp", "module", "input_summary",
    "prediction", "confidence", "username"
]


def init_session():
    """Initialize all session state defaults."""
    defaults = {
        "authenticated": False,
        "username": "",
        "total_predictions": 0,
        "fake_account_count": 0,
        "bot_count": 0,
        "fake_image_count": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def require_login():
    """Redirect to login page if not authenticated. Returns True if authenticated."""
    if not st.session_state.get("authenticated", False):
        st.warning("🔐 Please log in first.")
        st.stop()
    return True


def save_prediction(module: str, input_summary: str, prediction: str, confidence: float):
    """Append one prediction record to the CSV history file."""
    os.makedirs(HISTORY_DIR, exist_ok=True)

    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": module,
        "input_summary": input_summary,
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "username": st.session_state.get("username", "unknown"),
    }])

    if os.path.exists(HISTORY_CSV):
        row.to_csv(HISTORY_CSV, mode="a", header=False, index=False)
    else:
        row.to_csv(HISTORY_CSV, index=False)

    # Update counters
    st.session_state["total_predictions"] = st.session_state.get("total_predictions", 0) + 1
    if module == "Fake Account" and prediction == "Fake":
        st.session_state["fake_account_count"] = st.session_state.get("fake_account_count", 0) + 1
    if module == "Bot Detection" and prediction == "Bot":
        st.session_state["bot_count"] = st.session_state.get("bot_count", 0) + 1
    if module == "Fake Image" and prediction != "Real":
        st.session_state["fake_image_count"] = st.session_state.get("fake_image_count", 0) + 1


def load_history() -> pd.DataFrame:
    """Load prediction history from CSV. Returns empty DataFrame if none exists."""
    if os.path.exists(HISTORY_CSV):
        try:
            df = pd.read_csv(HISTORY_CSV)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=HISTORY_COLS)


def get_dashboard_stats(df: pd.DataFrame) -> dict:
    """Compute aggregate stats from the history DataFrame."""
    if df.empty:
        return {
            "total": 0, "fake_accounts": 0, "bots": 0,
            "fake_images": 0, "real_accounts": 0, "humans": 0, "real_images": 0
        }
    fake_df = df[df["module"] == "Fake Account"]
    bot_df = df[df["module"] == "Bot Detection"]
    img_df = df[df["module"] == "Fake Image"]

    return {
        "total": len(df),
        "fake_accounts": len(fake_df[fake_df["prediction"] == "Fake"]),
        "real_accounts": len(fake_df[fake_df["prediction"] == "Real"]),
        "bots": len(bot_df[bot_df["prediction"] == "Bot"]),
        "humans": len(bot_df[bot_df["prediction"] == "Human"]),
        "fake_images": len(img_df[img_df["prediction"] != "Real"]),
        "real_images": len(img_df[img_df["prediction"] == "Real"]),
    }


def confidence_badge(score: float, label: str) -> str:
    """Return an HTML badge coloured by confidence level."""
    if score >= 0.85:
        color = "#e53935"
        emoji = "🔴"
    elif score >= 0.65:
        color = "#fb8c00"
        emoji = "🟡"
    else:
        color = "#43a047"
        emoji = "🟢"
    pct = round(score * 100, 1)
    return f"""
    <div style='display:inline-block; background:{color}22; border:1px solid {color};
                border-radius:8px; padding:6px 16px; margin:4px;'>
        <span style='color:{color}; font-weight:700; font-size:1.1rem;'>{emoji} {label}</span><br>
        <span style='color:#ccc; font-size:0.85rem;'>Confidence: {pct}%</span>
    </div>"""
