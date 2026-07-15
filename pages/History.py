"""
pages/History.py — Prediction History viewer and report exporter
"""
import streamlit as st
import sys, os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import configure_page, inject_custom_css, sidebar_branding
from utils.helper import init_session, require_login, load_history, get_dashboard_stats
from utils.report_generator import generate_csv_bytes, generate_pdf_bytes

configure_page("Prediction History | Instagram AI", "📜")
init_session()
inject_custom_css()
sidebar_branding()
require_login()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='background:linear-gradient(135deg,#7B1FA2,#E91E8C);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           font-size:2.2rem; font-weight:900;'>
    📜 Prediction History
</h1>
<p style='color:#888; margin-top:-8px;'>
    View, filter, and export all past predictions.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

df = load_history()

if df.empty:
    st.info("📭 No predictions found. Run detections in the other modules to populate history.")
    st.stop()

# ── Filters ────────────────────────────────────────────────────────────────────
st.markdown("### 🔎 Filter Predictions")
fcol1, fcol2, fcol3 = st.columns(3)

with fcol1:
    module_filter = st.multiselect(
        "📦 Module",
        options=df["module"].unique().tolist(),
        default=df["module"].unique().tolist()
    )
with fcol2:
    prediction_filter = st.multiselect(
        "🏷 Prediction",
        options=df["prediction"].unique().tolist(),
        default=df["prediction"].unique().tolist()
    )
with fcol3:
    min_conf = st.slider("📊 Min Confidence %", 0.0, 100.0, 0.0, 1.0)

filtered = df[
    df["module"].isin(module_filter) &
    df["prediction"].isin(prediction_filter) &
    (df["confidence"] >= min_conf)
].copy()

# ── Table ──────────────────────────────────────────────────────────────────────
st.markdown(f"**Showing {len(filtered)} of {len(df)} records**")

def colour_row(row):
    pred = row.get("prediction", "")
    if pred in ["Fake", "Bot"]:
        return ["background-color: #E5393511"] * len(row)
    elif pred in ["AI Generated", "Edited"]:
        return ["background-color: #FB8C0011"] * len(row)
    else:
        return ["background-color: #43A04711"] * len(row)

if not filtered.empty:
    st.dataframe(
        filtered.sort_values("timestamp", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=400
    )

st.markdown("---")

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown("### 📥 Export Reports")
export_col1, export_col2 = st.columns(2)

stats = get_dashboard_stats(df)

with export_col1:
    csv_bytes = generate_csv_bytes(filtered)
    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv_bytes,
        file_name="instagram_ai_predictions.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download prediction history as CSV"
    )

with export_col2:
    try:
        pdf_bytes = generate_pdf_bytes(filtered, stats)
        if pdf_bytes:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name="instagram_ai_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download full prediction report as PDF"
            )
        else:
            st.warning("PDF generation requires fpdf2. Install with: pip install fpdf2")
    except Exception as e:
        st.warning(f"PDF unavailable: {e}")

# ── Summary by module ──────────────────────────────────────────────────────────
if not filtered.empty:
    st.markdown("### 📊 Summary by Module")
    summary = filtered.groupby(["module", "prediction"]).size().reset_index(name="count")
    st.dataframe(summary, use_container_width=True, hide_index=True)
