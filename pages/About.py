"""
pages/About.py — Project information, architecture, and team details
"""
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import configure_page, inject_custom_css, sidebar_branding
from utils.helper import init_session, require_login
from config.config import VERSION

configure_page("About | Instagram AI Detector", "ℹ")
init_session()
inject_custom_css()
sidebar_branding()
require_login()

st.markdown("""
<h1 style='background:linear-gradient(135deg,#00BCD4,#E91E8C);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           font-size:2.2rem; font-weight:900;'>
    ℹ About This Project
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Overview ────────────────────────────────────────────────────────────────────
st.markdown("""
### 📋 Project Overview

> **Instagram AI Detector** is a B.Tech Final Year Project in **Artificial Intelligence & Data Science**.
> It demonstrates real-world application of Machine Learning, Deep Learning, and Computer Vision
> to solve the critical problem of fake account detection on social media platforms.

**Problem Statement:**
Millions of Instagram users interact daily, but the platform contains fake accounts, bots, and
AI-generated profile images used for scams, spam, and misinformation. Manual detection is
infeasible at scale. This system automates detection using AI.
""")

st.markdown("---")

# ── Technology Stack ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🛠 Technology Stack

    | Category | Technology |
    |---|---|
    | **Web Framework** | Streamlit |
    | **ML Library** | Scikit-learn |
    | **Deep Learning** | TensorFlow / Keras |
    | **CNN Architecture** | MobileNetV2 |
    | **Image Processing** | OpenCV, Pillow |
    | **Data Processing** | Pandas, NumPy |
    | **Visualization** | Plotly, Matplotlib |
    | **Model Saving** | Joblib |
    | **Report Generation** | FPDF2 |
    | **Language** | Python 3.13+ |
    """)

with col2:
    st.markdown("""
    ### 🤖 ML Algorithms Used

    **Fake Account Detection:**
    - ✅ Logistic Regression
    - ✅ Random Forest *(Primary)*
    - ✅ Gradient Boosting

    **Bot Detection:**
    - ✅ Random Forest *(Primary)*
    - ✅ Gradient Boosting
    - ✅ Isolation Forest *(Anomaly)*

    **Fake Image Detection:**
    - ✅ MobileNetV2 (Transfer Learning)
    - ✅ Custom Dense Head
    - ✅ Fine-tuning Strategy
    """)

st.markdown("---")

# ── System Modules ────────────────────────────────────────────────────────────
st.markdown("### 🗂 System Modules")

modules_data = [
    ("🔐", "Login & Authentication", "Session-based auth with role management", "#E91E8C"),
    ("👤", "Fake Account Detection",
     "ML classifier using 13 account features → Real / Fake + Confidence", "#E53935"),
    ("🤖", "Bot Detection",
     "Behavioral pattern analysis using 11 features → Human / Bot + Confidence", "#FB8C00"),
    ("🖼", "Fake Image Detection",
     "CNN (MobileNetV2) classifies profile images → Real / AI-Generated / Edited", "#3D5AF1"),
    ("📊", "Analytics Dashboard",
     "Real-time Plotly charts, KPI metrics, and prediction trends", "#43A047"),
    ("📜", "History & Reports",
     "Full prediction log with CSV/PDF export capabilities", "#7B1FA2"),
]

cols = st.columns(3)
for i, (icon, title, desc, color) in enumerate(modules_data):
    with cols[i % 3]:
        st.markdown(f"""
        <div style='background:#1A1D27; border:1px solid {color}44; border-radius:16px;
                    padding:18px; margin-bottom:12px; min-height:130px;'>
            <div style='font-size:1.8rem; margin-bottom:8px;'>{icon}</div>
            <div style='color:{color}; font-weight:700; font-size:0.95rem; margin-bottom:6px;'>{title}</div>
            <div style='color:#888; font-size:0.82rem; line-height:1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Evaluation Metrics ────────────────────────────────────────────────────────
st.markdown("### 📏 Evaluation Metrics")
st.markdown("""
| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions / total predictions |
| **Precision** | True Positives / (True Positives + False Positives) |
| **Recall** | True Positives / (True Positives + False Negatives) |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve |
| **Confusion Matrix** | True/False Positive/Negative breakdown |
""")

st.markdown("---")

# ── ML Workflow ───────────────────────────────────────────────────────────────
st.markdown("### 🔄 ML Workflow")

steps = [
    "1️⃣ Data Collection / Generation",
    "2️⃣ Data Cleaning & EDA",
    "3️⃣ Feature Engineering",
    "4️⃣ Data Preprocessing & Scaling",
    "5️⃣ Train-Test Split (80/20)",
    "6️⃣ Model Training (Multiple Algorithms)",
    "7️⃣ Hyperparameter Tuning",
    "8️⃣ Model Evaluation (Accuracy, F1, ROC-AUC)",
    "9️⃣ Best Model Selection",
    "🔟 Model Saving (Joblib / Keras)",
    "1️⃣1️⃣ Deployment via Streamlit",
]

cols = st.columns(4)
for i, step in enumerate(steps):
    with cols[i % 4]:
        st.markdown(f"""
        <div style='background:#1A1D27; border:1px solid #2A2D3E; border-radius:10px;
                    padding:10px; margin-bottom:8px; font-size:0.82rem; color:#ccc;'>
            {step}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

st.markdown(f"""
<div style='text-align:center; padding:20px; color:#555; font-size:0.85rem;'>
    <b style='color:#E91E8C;'>Instagram AI Detector</b> v{VERSION}<br>
    B.Tech Final Year Project | AI & Data Science<br>
    Built with ❤️ using Python, Streamlit, TensorFlow & Scikit-learn
</div>
""", unsafe_allow_html=True)
