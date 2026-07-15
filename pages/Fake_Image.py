"""
pages/Fake_Image.py — Fake / AI-Generated Image Detection module
"""
import streamlit as st
import sys, os
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import configure_page, inject_custom_css, sidebar_branding
from utils.helper import init_session, require_login, save_prediction
from utils.visualization import gauge_chart, bar_chart, pie_chart
from utils.image_processing import load_image_from_upload, extract_image_features
from predict import predict_image, check_models_exist

configure_page("Fake Image Detection | Instagram AI", "🖼")
init_session()
inject_custom_css()
sidebar_branding()
require_login()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='background:linear-gradient(135deg,#00BCD4,#7B1FA2);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           font-size:2.2rem; font-weight:900;'>
    🖼 Fake Profile Image Detection
</h1>
<p style='color:#888; margin-top:-8px;'>
    Upload a profile image to detect if it is Real, AI-Generated, or Edited.
</p>
""", unsafe_allow_html=True)

# ── Model check ────────────────────────────────────────────────────────────────
status = check_models_exist()
if not status["image_detector"]:
    st.error("⚠️ Image model not found! Run `python train_image_model.py` first.")
    st.stop()

st.markdown("---")

col_upload, col_info = st.columns([1.5, 1])

with col_upload:
    st.markdown("#### 📤 Upload Profile Image")
    uploaded = st.file_uploader(
        "Drop or browse a profile image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported: JPG, PNG, WEBP | Max 10 MB"
    )

with col_info:
    st.markdown("""
    #### 🔬 What We Detect
    <div style='background:#1A1D27; border:1px solid #00BCD444; border-radius:16px; padding:18px;'>
        <div style='margin-bottom:12px;'>
            <span style='color:#43A047; font-weight:700;'>🟢 Real Images</span><br>
            <span style='color:#888; font-size:0.85rem;'>Natural photos with authentic lighting,
            textures, and imperfections.</span>
        </div>
        <div style='margin-bottom:12px;'>
            <span style='color:#E91E8C; font-weight:700;'>🔴 AI Generated</span><br>
            <span style='color:#888; font-size:0.85rem;'>Images created by GANs or diffusion
            models — often too perfect.</span>
        </div>
        <div>
            <span style='color:#FB8C00; font-weight:700;'>🟡 Edited / Manipulated</span><br>
            <span style='color:#888; font-size:0.85rem;'>Real photos with heavy filters,
            face-swapping, or digital manipulation.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded is not None:
    # ── Display image ─────────────────────────────────────────────────────────
    img_array = load_image_from_upload(uploaded)
    pil_img = Image.fromarray(img_array)

    st.markdown("---")
    img_col, result_col = st.columns([1, 1.5])

    with img_col:
        st.markdown("#### 🖼 Uploaded Image")
        st.image(pil_img, caption=f"File: {uploaded.name}", use_container_width=True)

        # Image properties
        h, w, c = img_array.shape
        st.markdown(f"""
        <div style='background:#1A1D27; border:1px solid #2A2D3E; border-radius:12px; padding:14px; margin-top:8px;'>
            <p style='color:#aaa; font-size:0.8rem; margin:0;'>
                📐 Resolution: <b style='color:#E91E8C;'>{w} × {h}</b><br>
                🎨 Channels: <b style='color:#E91E8C;'>{c}</b><br>
                📁 Format: <b style='color:#E91E8C;'>{uploaded.type}</b><br>
                💾 Size: <b style='color:#E91E8C;'>{uploaded.size/1024:.1f} KB</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with result_col:
        with st.spinner("🔄 Analysing image with CNN..."):
            try:
                result = predict_image(img_array)
                features = extract_image_features(img_array)
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.stop()

        label      = result["label"]
        confidence = result["confidence"]
        class_probs = result["class_probs"]

        # ── Result badge ──────────────────────────────────────────────────────
        color_map = {"Real": "#43A047", "AI Generated": "#E91E8C", "Edited": "#FB8C00"}
        emoji_map = {"Real": "✅", "AI Generated": "🤖", "Edited": "✂️"}
        color  = color_map.get(label, "#888")
        emoji  = emoji_map.get(label, "🔍")

        st.markdown(f"""
        <div style='background:{color}22; border:2px solid {color}; border-radius:16px;
                    padding:24px; text-align:center; margin-bottom:16px;'>
            <div style='font-size:3rem;'>{emoji}</div>
            <h2 style='color:{color}; font-size:2rem; font-weight:900; margin:8px 0;'>{label}</h2>
            <p style='color:#aaa;'>Confidence: <b style='color:{color};'>{confidence*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        save_prediction(
            module="Fake Image",
            input_summary=f"{uploaded.name} ({w}x{h})",
            prediction=label,
            confidence=confidence
        )

        # ── Class probabilities ────────────────────────────────────────────────
        st.markdown("#### 📊 Class Probabilities")
        for cls, prob in class_probs.items():
            col = color_map.get(cls, "#888")
            pct = prob * 100
            st.markdown(f"""
            <div style='margin:6px 0;'>
                <div style='display:flex; justify-content:space-between; color:#ccc; font-size:0.85rem;'>
                    <span>{emoji_map.get(cls,"")} {cls}</span>
                    <span style='color:{col}; font-weight:700;'>{pct:.1f}%</span>
                </div>
                <div style='background:#2A2D3E; border-radius:6px; height:8px; margin-top:4px;'>
                    <div style='background:{col}; width:{pct:.1f}%; height:8px;
                                border-radius:6px; transition:width 0.5s ease;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Image Feature Analysis ────────────────────────────────────────────────
    st.markdown("### 🔬 Image Feature Analysis")
    feat_cols = st.columns(5)
    feat_items = [
        ("Sharpness", features["sharpness"], "↑ Better = More natural", "#43A047"),
        ("Brightness", features["brightness"], "0-255 scale", "#3D5AF1"),
        ("Saturation", features["saturation"], "0-255 scale", "#E91E8C"),
        ("Noise Level", features["noise_level"], "↑ Higher = More authentic", "#FB8C00"),
        ("Edge Density %", features["edge_density"], "% of edge pixels", "#00BCD4"),
    ]
    for col, (name, val, hint, color) in zip(feat_cols, feat_items):
        with col:
            st.markdown(f"""
            <div style='background:#1A1D27; border:1px solid {color}44; border-radius:12px;
                        padding:14px; text-align:center;'>
                <div style='color:{color}; font-size:1.4rem; font-weight:900;'>{val}</div>
                <div style='color:#ccc; font-size:0.8rem; margin:4px 0;'>{name}</div>
                <div style='color:#666; font-size:0.7rem;'>{hint}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Gauge chart ───────────────────────────────────────────────────────────
    st.plotly_chart(gauge_chart(confidence, f"Detection Confidence ({label})"),
                    use_container_width=False)
else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:60px; background:#1A1D27;
                border:2px dashed #2A2D3E; border-radius:20px; margin:20px 0;'>
        <div style='font-size:4rem; margin-bottom:16px;'>🖼</div>
        <h3 style='color:#555;'>Upload a Profile Image to Begin</h3>
        <p style='color:#444; font-size:0.9rem;'>
            Supported formats: JPG, JPEG, PNG, WEBP<br>
            The AI will classify it as Real, AI Generated, or Edited.
        </p>
    </div>
    """, unsafe_allow_html=True)
