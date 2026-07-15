"""
pages/Dashboard.py — Analytics dashboard with charts and prediction stats
"""
import streamlit as st
import sys, os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import configure_page, inject_custom_css, sidebar_branding
from utils.helper import init_session, require_login, load_history, get_dashboard_stats
from utils.visualization import (
    pie_chart, bar_chart, timeline_chart, module_bar_chart, metric_card_html
)

configure_page("Analytics Dashboard | Instagram AI", "📊")
init_session()
inject_custom_css()
sidebar_branding()
require_login()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='background:linear-gradient(135deg,#43A047,#00BCD4);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           font-size:2.2rem; font-weight:900;'>
    📊 Analytics Dashboard
</h1>
<p style='color:#888; margin-top:-8px;'>
    Real-time prediction analytics and trends across all detection modules.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_history()
stats = get_dashboard_stats(df)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
st.markdown("### 🏆 Overall Statistics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card_html("Total Predictions", stats["total"], "🔍", "#E91E8C"),
                unsafe_allow_html=True)
with c2:
    st.markdown(metric_card_html("Fake Accounts", stats["fake_accounts"], "👤", "#E53935"),
                unsafe_allow_html=True)
with c3:
    st.markdown(metric_card_html("Bots Detected", stats["bots"], "🤖", "#FB8C00"),
                unsafe_allow_html=True)
with c4:
    st.markdown(metric_card_html("Fake Images", stats["fake_images"], "🖼", "#3D5AF1"),
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

c5, c6, c7 = st.columns(3)
with c5:
    st.markdown(metric_card_html("Real Accounts", stats["real_accounts"], "✅", "#43A047"),
                unsafe_allow_html=True)
with c6:
    st.markdown(metric_card_html("Human Accounts", stats["humans"], "👤", "#00BCD4"),
                unsafe_allow_html=True)
with c7:
    st.markdown(metric_card_html("Real Images", stats["real_images"], "🟢", "#7B1FA2"),
                unsafe_allow_html=True)

st.markdown("---")

if df.empty:
    st.info("📭 No predictions yet. Run detections to see analytics here.")
else:
    # ── Charts row 1 ──────────────────────────────────────────────────────────
    st.markdown("### 📈 Charts & Visualizations")
    ch1, ch2 = st.columns(2)

    with ch1:
        # Fake Account pie
        fa_df = df[df["module"] == "Fake Account"]
        if not fa_df.empty:
            counts = fa_df["prediction"].value_counts()
            fig = pie_chart(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                title="👤 Fake Account Distribution",
                colors=["#43A047", "#E53935"]
            )
            st.plotly_chart(fig, use_container_width=True)

    with ch2:
        # Bot detection pie
        bot_df = df[df["module"] == "Bot Detection"]
        if not bot_df.empty:
            counts = bot_df["prediction"].value_counts()
            fig = pie_chart(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                title="🤖 Bot vs Human Distribution",
                colors=["#43A047", "#FB8C00"]
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    ch3, ch4 = st.columns(2)

    with ch3:
        # Image detection pie
        img_df = df[df["module"] == "Fake Image"]
        if not img_df.empty:
            counts = img_df["prediction"].value_counts()
            fig = pie_chart(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                title="🖼 Image Classification Distribution",
                colors=["#43A047", "#E91E8C", "#FB8C00"]
            )
            st.plotly_chart(fig, use_container_width=True)

    with ch4:
        fig = module_bar_chart(df)
        st.plotly_chart(fig, use_container_width=True)

    # ── Timeline ───────────────────────────────────────────────────────────────
    st.plotly_chart(timeline_chart(df), use_container_width=True)

    # ── Confidence distribution ────────────────────────────────────────────────
    st.markdown("### 📊 Confidence Score Distribution")
    conf_col1, conf_col2 = st.columns(2)

    with conf_col1:
        if not df.empty and "confidence" in df.columns:
            bins = [0, 50, 65, 80, 90, 100]
            labels_b = ["0-50%", "50-65%", "65-80%", "80-90%", "90-100%"]
            df["conf_bin"] = pd.cut(df["confidence"], bins=bins, labels=labels_b)
            bin_counts = df["conf_bin"].value_counts().sort_index()
            fig = bar_chart(
                x=bin_counts.index.tolist(),
                y=bin_counts.values.tolist(),
                title="Confidence Score Bins",
                color="#E91E8C"
            )
            st.plotly_chart(fig, use_container_width=True)

    with conf_col2:
        st.markdown("""
        <div style='background:#1A1D27; border:1px solid #2A2D3E; border-radius:16px;
                    padding:20px; margin-top:10px;'>
            <h4 style='color:#E91E8C; margin-bottom:12px;'>📌 Confidence Interpretation</h4>
            <table style='width:100%; color:#ccc; font-size:0.85rem;'>
                <tr><td>🔴 <b>90-100%</b></td><td style='color:#E53935;'>Very High Confidence</td></tr>
                <tr><td>🟠 <b>80-90%</b></td><td style='color:#FB8C00;'>High Confidence</td></tr>
                <tr><td>🟡 <b>65-80%</b></td><td style='color:#FDD835;'>Moderate Confidence</td></tr>
                <tr><td>🟢 <b>50-65%</b></td><td style='color:#66BB6A;'>Low Confidence</td></tr>
                <tr><td>⚪ <b>0-50%</b></td><td style='color:#888;'>Uncertain</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Recent Activity ────────────────────────────────────────────────────────
    st.markdown("### 🕐 Recent Activity (Last 10)")
    recent = df.tail(10).iloc[::-1][["timestamp", "module", "prediction", "confidence", "username"]]
    st.dataframe(recent, use_container_width=True, hide_index=True)
