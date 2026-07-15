"""
pages/Home.py — Home page (also accessible from the sidebar)
"""
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import configure_page, inject_custom_css, sidebar_branding
from utils.helper import init_session, require_login

configure_page("Home | Instagram AI Detector", "🏠")
init_session()
inject_custom_css()
sidebar_branding()
require_login()

st.markdown("""
<h1 style='background:linear-gradient(135deg,#E91E8C,#3D5AF1);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           font-size:2.5rem; font-weight:900;'>
    🏠 Welcome to Instagram AI Detector
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='color:#aaa; font-size:1.05rem;'>
    An intelligent AI-powered system to detect fake accounts, bots, and manipulated profile images.
    Use the sidebar to navigate between modules.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ### 🎯 Project Objectives
    - **Fake Account Detection** — Classify accounts as Real or Fake using ML
    - **Bot Detection** — Identify automated bots via behavioral analysis
    - **Image Authentication** — Detect AI-generated / edited profile photos
    - **Analytics Dashboard** — Real-time charts and prediction trends
    - **Report Generation** — Export results as CSV or PDF
    """)

with col2:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1A1D27,#12141E);
                border:1px solid #E91E8C44; border-radius:16px; padding:20px; text-align:center;'>
        <div style='font-size:3rem;'>🤖</div>
        <div style='color:#E91E8C; font-weight:800; font-size:1.1rem; margin:8px 0;'>AI-Powered</div>
        <div style='color:#888; font-size:0.85rem;'>
            Random Forest<br>MobileNetV2 CNN<br>Isolation Forest
        </div>
    </div>
    """, unsafe_allow_html=True)
