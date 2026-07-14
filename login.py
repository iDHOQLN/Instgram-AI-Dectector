"""
login.py - Authentication page for Instagram AI Detector
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import USERS, APP_TITLE
from config.settings import inject_custom_css


def show_login_page():
    inject_custom_css()

    # ── Hero banner ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 40px 20px 20px 20px;'>
        <div style='font-size: 4rem; margin-bottom: 10px;'>🔍</div>
        <h1 style='font-size:2.4rem; font-weight:900;
                   background: linear-gradient(135deg, #E91E8C, #3D5AF1);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   margin:0;'>Instagram AI Detector</h1>
        <p style='color:#aaa; font-size:1rem; margin-top:8px;'>
            Fake Account • Bot Detection • AI Image Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Login card ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1A1D27,#12141E);
                    border:1px solid #E91E8C44; border-radius:20px; padding:32px 28px;
                    box-shadow: 0 8px 40px rgba(233,30,140,0.15);'>
            <h3 style='color:#E91E8C; text-align:center; margin-bottom:8px;'>🔐 Sign In</h3>
            <p style='color:#777; text-align:center; font-size:0.85rem; margin-bottom:20px;'>
                Access the AI Detection Dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Username", placeholder="Enter username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("🚀 Login", use_container_width=True)

            if submitted:
                if username in USERS and USERS[username] == password:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.success(f"✅ Welcome back, **{username}**!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password. Please try again.")

        # Demo credentials hint
        st.markdown("""
        <div style='background:#1A1D2722; border:1px solid #3D5AF133;
                    border-radius:10px; padding:12px; margin-top:12px; text-align:center;'>
            <p style='color:#777; font-size:0.8rem; margin:0;'>
                <b style='color:#3D5AF1;'>Demo Accounts:</b><br>
                admin / admin123 &nbsp;|&nbsp; demo / demo123
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Features preview ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("👤", "Fake Account", "Detect fake Instagram profiles with ML"),
        ("🤖", "Bot Detection", "Identify automated bot accounts"),
        ("🖼", "Image Analysis", "Detect AI-generated profile photos"),
        ("📊", "Analytics", "Interactive dashboards & reports"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(f"""
            <div style='background:#1A1D27; border:1px solid #2A2D3E;
                        border-radius:16px; padding:20px; text-align:center;
                        transition:0.3s; height:140px;'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='color:#E91E8C; font-weight:700; margin:6px 0 4px;'>{title}</div>
                <div style='color:#888; font-size:0.8rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
