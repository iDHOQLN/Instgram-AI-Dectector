"""
app.py — Main entry point for Instagram AI Detector
Run with: streamlit run app.py
"""
import streamlit as st
import sys
import os

# ── Path setup ───────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import configure_page, inject_custom_css, sidebar_branding
from config.config import APP_TITLE, APP_ICON, VERSION
from utils.helper import init_session
from login import show_login_page

# ── Page Config (MUST be first Streamlit call) ────────────────────────────────
configure_page(title=APP_TITLE, icon=APP_ICON, layout="wide")

# ── Session Init ──────────────────────────────────────────────────────────────
init_session()

# ── CSS Injection ─────────────────────────────────────────────────────────────
inject_custom_css()

# ── Authentication Gate ───────────────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    show_login_page()
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
sidebar_branding()

with st.sidebar:
    st.markdown("---")

    # User info
    st.markdown(f"""
    <div style='background:#1A1D27; border:1px solid #2A2D3E; border-radius:12px;
                padding:12px 16px; margin-bottom:12px;'>
        <span style='color:#aaa; font-size:0.8rem;'>Logged in as</span><br>
        <span style='color:#E91E8C; font-weight:700;'>👤 {st.session_state.get("username","")}</span>
    </div>
    """, unsafe_allow_html=True)

    # Stats quick view
    st.markdown(f"""
    <div style='background:#1A1D27; border:1px solid #2A2D3E; border-radius:12px; padding:12px 16px;'>
        <p style='color:#aaa; font-size:0.8rem; margin:0 0 8px 0;'>Session Stats</p>
        <table style='width:100%; font-size:0.82rem; color:#ccc;'>
            <tr><td>🔍 Total</td><td style='text-align:right; color:#E91E8C; font-weight:700;'>
                {st.session_state.get("total_predictions",0)}</td></tr>
            <tr><td>👤 Fakes</td><td style='text-align:right; color:#E53935; font-weight:700;'>
                {st.session_state.get("fake_account_count",0)}</td></tr>
            <tr><td>🤖 Bots</td><td style='text-align:right; color:#FB8C00; font-weight:700;'>
                {st.session_state.get("bot_count",0)}</td></tr>
            <tr><td>🖼 Fake Imgs</td><td style='text-align:right; color:#3D5AF1; font-weight:700;'>
                {st.session_state.get("fake_image_count",0)}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Logout
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown(f"""
    <div style='text-align:center; margin-top:20px;'>
        <p style='color:#444; font-size:0.7rem;'>v{VERSION} | AI & DS Project</p>
    </div>
    """, unsafe_allow_html=True)

# ── Home / Landing ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 30px 0 10px 0;'>
    <h1 style='font-size:2.8rem; font-weight:900; margin:0;
               background: linear-gradient(135deg, #E91E8C 0%, #3D5AF1 50%, #00BCD4 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🔍 Instagram AI Detector
    </h1>
    <p style='color:#888; font-size:1.05rem; margin:10px 0 0 0;'>
        AI-powered fake account, bot & image detection system
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Clickable Card CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Make card-nav buttons transparent — card HTML is the visible element */
div[data-testid="stButton"].card-btn > button {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-top: -8px !important;
    width: 100% !important;
    cursor: pointer !important;
    font-size: 0 !important;   /* hide label text */
    height: 36px !important;
    opacity: 0.01 !important;  /* nearly invisible overlay */
}
/* Elevate card on hover */
.module-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 35px rgba(255,255,255,0.10) !important;
}
.secondary-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(255,255,255,0.07) !important;
}
.module-card, .secondary-card {
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
</style>
""", unsafe_allow_html=True)

# ── Navigation Cards ───────────────────────────────────────────────────────────
st.markdown("### 🗂 Navigate to a Module")

col1, col2, col3, col4 = st.columns(4)

modules = [
    ("👤", "Fake Account Detection", "Detect fake Instagram accounts using ML",
     "#E91E8C", "pages/Fake_Account.py"),
    ("🤖", "Bot Detection", "Identify automated bot accounts using AI",
     "#3D5AF1", "pages/Bot_Detection.py"),
    ("🖼", "Fake Image Detection", "Detect AI-generated profile images using CNN",
     "#00BCD4", "pages/Fake_Image.py"),
    ("📊", "Analytics Dashboard", "View charts, trends, and prediction stats",
     "#43A047", "pages/Dashboard.py"),
]

for col, (icon, title, desc, color, page) in zip([col1, col2, col3, col4], modules):
    with col:
        st.markdown(f"""
        <div class='module-card' style='
            background:linear-gradient(135deg,#1A1D27,#12141E);
            border:2px solid {color}55; border-radius:20px; padding:24px 16px;
            text-align:center; box-shadow:0 4px 20px {color}22;
            min-height:180px; cursor:pointer;'>
            <div style='font-size:2.8rem; margin-bottom:12px;'>{icon}</div>
            <div style='color:{color}; font-weight:800; font-size:1rem; margin-bottom:8px;'>{title}</div>
            <div style='color:#888; font-size:0.82rem; line-height:1.4;'>{desc}</div>
            <div style='margin-top:14px; display:inline-block; padding:5px 14px;
                        border:1px solid {color}88; border-radius:20px;
                        color:{color}; font-size:0.75rem; font-weight:600;'>Open →</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Open {title}", key=f"btn_{page}", use_container_width=True):
            st.switch_page(page)

st.markdown("<br>", unsafe_allow_html=True)

# ── Secondary links row ────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
secondary = [
    ("📜", "Prediction History", "View and manage all past predictions", "#7B1FA2", "pages/History.py"),
    ("📥", "Reports",           "Download CSV and PDF reports",          "#FB8C00", "pages/Dashboard.py"),
    ("ℹ",  "About",             "Project info and technical details",    "#00BCD4", "pages/About.py"),
]
for col, (icon, title, desc, color, page) in zip([c1, c2, c3], secondary):
    with col:
        st.markdown(f"""
        <div class='secondary-card' style='
            background:linear-gradient(135deg,#1A1D27,#12141E);
            border:1px solid {color}55; border-radius:16px; padding:18px;
            text-align:center; box-shadow:0 2px 10px {color}22; cursor:pointer;'>
            <div style='font-size:2rem; margin-bottom:8px;'>{icon}</div>
            <div style='color:{color}; font-weight:700; font-size:0.95rem; margin-bottom:6px;'>{title}</div>
            <div style='color:#888; font-size:0.8rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Go to {title}", key=f"btn_sec_{page}", use_container_width=True):
            st.switch_page(page)

st.markdown("---")

# ── How it works ───────────────────────────────────────────────────────────────
with st.expander("📖 How It Works", expanded=False):
    st.markdown("""
    | Module | Algorithm | Features Used | Output |
    |--------|-----------|--------------|--------|
    | 👤 Fake Account | Random Forest + LR | Followers, Following, Posts, Bio, Engagement | Real / Fake |
    | 🤖 Bot Detection | Random Forest + Isolation Forest | Posting freq, Engagement, Activity patterns | Human / Bot |
    | 🖼 Image Detection | MobileNetV2 CNN | Pixel patterns, textures, artefacts | Real / AI-Gen / Edited |
    """)

    st.markdown("""
    **Workflow:**
    1. Input account features or upload a profile image
    2. The system preprocesses and scales your input
    3. The trained AI model generates predictions with confidence scores
    4. Results are shown with interactive gauges and charts
    5. All predictions are saved to history for report generation
    """)

st.markdown(f"""
<div style='text-align:center; margin-top:30px; color:#444; font-size:0.8rem;'>
    Instagram AI Detector v{VERSION}
</div>
""", unsafe_allow_html=True)
