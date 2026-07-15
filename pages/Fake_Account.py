"""
pages/Fake_Account.py — Instagram Fake Account Detection module
"""
import streamlit as st
import sys, os
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import configure_page, inject_custom_css, sidebar_branding
from utils.helper import init_session, require_login, save_prediction, confidence_badge
from utils.visualization import gauge_chart, bar_chart
from predict import predict_fake_account, check_models_exist

configure_page("Fake Account Detection | Instagram AI", "👤")
init_session()
inject_custom_css()
sidebar_branding()
require_login()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='background:linear-gradient(135deg,#E91E8C,#3D5AF1);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           font-size:2.2rem; font-weight:900;'>
    👤 Fake Account Detection
</h1>
<p style='color:#888; margin-top:-8px;'>
    Enter Instagram account details below to detect if the account is Real or Fake.
</p>
""", unsafe_allow_html=True)

# ── Model check ────────────────────────────────────────────────────────────────
status = check_models_exist()
if not status["fake_account"]:
    st.error("⚠️ Model not found! Run `python train_fake_account.py` first.")
    st.stop()

st.markdown("---")

# ── Input Form ─────────────────────────────────────────────────────────────────
with st.form("fake_account_form"):
    st.markdown("#### 📋 Account Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        username = st.text_input("📛 Username", value="sample_user123", help="Instagram username")
        username_length = len(username)
        st.caption(f"Username length: {username_length}")

        full_name = st.text_input("👤 Full Name", value="Sample User", help="Display name")
        full_name_length = len(full_name)
        st.caption(f"Full name length: {full_name_length}")

        followers = st.number_input("👥 Followers", min_value=0, value=1200, step=10)
        following = st.number_input("➕ Following", min_value=0, value=350, step=10)

    with col2:
        posts = st.number_input("📸 Number of Posts", min_value=0, value=45, step=1)
        bio = st.text_area("📝 Biography", value="Travel lover | Coffee addict ☕ | Photographer 📷",
                           height=80)
        bio_length = len(bio)
        st.caption(f"Bio length: {bio_length}")
        engagement_rate = st.slider("📈 Engagement Rate", 0.0, 0.5, 0.05, 0.001,
                                     format="%.3f", help="Avg likes+comments / followers")

    with col3:
        has_external_url = st.selectbox("🔗 External URL in Bio", [0, 1],
                                         format_func=lambda x: "Yes" if x else "No")
        has_profile_pic = st.selectbox("🖼 Has Profile Picture", [1, 0],
                                        format_func=lambda x: "Yes" if x else "No")
        is_private = st.selectbox("🔒 Private Account", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No")
        is_verified = st.selectbox("✅ Verified Account", [0, 1],
                                    format_func=lambda x: "Yes" if x else "No")

    st.markdown("---")
    submit = st.form_submit_button("🔍 Analyse Account", use_container_width=True)

# ── Results ────────────────────────────────────────────────────────────────────
if submit:
    raw = {
        "username_length":  username_length,
        "full_name_length": full_name_length,
        "followers":        followers,
        "following":        following,
        "posts":            posts,
        "bio_length":       bio_length,
        "has_external_url": has_external_url,
        "has_profile_pic":  has_profile_pic,
        "is_private":       is_private,
        "is_verified":      is_verified,
        "engagement_rate":  engagement_rate,
    }

    with st.spinner("🔄 Analysing account with AI..."):
        try:
            result = predict_fake_account(raw)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    label      = result["label"]
    confidence = result["confidence"]
    prob_fake  = result["prob_fake"]
    prob_real  = result["prob_real"]

    # ── Result banner ─────────────────────────────────────────────────────────
    if label == "Fake":
        bg_color = "#E5393522"
        border   = "#E53935"
        emoji    = "⚠️"
        msg      = "This account shows strong indicators of being **FAKE**."
    else:
        bg_color = "#43A04722"
        border   = "#43A047"
        emoji    = "✅"
        msg      = "This account appears to be a **REAL** account."

    st.markdown(f"""
    <div style='background:{bg_color}; border:2px solid {border}; border-radius:16px;
                padding:24px; margin:16px 0; text-align:center;'>
        <div style='font-size:3rem;'>{emoji}</div>
        <h2 style='color:{border}; font-size:2rem; font-weight:900; margin:8px 0;'>{label} Account</h2>
        <p style='color:#ccc;'>{msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # Save to history
    save_prediction(
        module="Fake Account",
        input_summary=f"@{username} | Followers:{followers} | Following:{following} | Posts:{posts}",
        prediction=label,
        confidence=confidence
    )

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown("### 📊 Prediction Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏷 Prediction", label)
    c2.metric("📊 Confidence", f"{confidence*100:.1f}%")
    c3.metric("🔴 Fake Probability", f"{prob_fake*100:.1f}%")
    c4.metric("🟢 Real Probability", f"{prob_real*100:.1f}%")

    # ── Charts ────────────────────────────────────────────────────────────────
    col_g, col_b = st.columns(2)
    with col_g:
        st.plotly_chart(gauge_chart(confidence, f"{label} Confidence"), use_container_width=True)

    with col_b:
        fig = bar_chart(
            x=["Real", "Fake"],
            y=[round(prob_real*100,1), round(prob_fake*100,1)],
            title="Probability Distribution",
            color=["#43A047", "#E53935"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature breakdown ─────────────────────────────────────────────────────
    with st.expander("🔎 Feature Analysis"):
        feats = result["features"]
        feat_df = pd.DataFrame([{
            "Feature": k.replace("_", " ").title(),
            "Value": round(v, 4) if isinstance(v, float) else v
        } for k, v in feats.items()])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Risk Indicators ───────────────────────────────────────────────────────
    st.markdown("### 🚦 Risk Indicators")
    risk_cols = st.columns(5)
    indicators = [
        ("Follower/Following Ratio", feats.get("follower_following_ratio", 0), 0.5, False),
        ("Engagement Rate", feats.get("engagement_rate", 0), 0.03, False),
        ("Has Profile Picture", feats.get("has_profile_pic", 0), 0.5, False),
        ("Is Verified", feats.get("is_verified", 0), 0.5, False),
        ("Posts Per Follower", feats.get("posts_per_follower", 0), 0.01, False),
    ]
    for col, (name, val, threshold, invert) in zip(risk_cols, indicators):
        is_risky = (val < threshold and not invert) or (val > threshold and invert)
        color = "#E53935" if is_risky else "#43A047"
        icon  = "🔴" if is_risky else "🟢"
        with col:
            st.markdown(f"""
            <div style='background:#1A1D27; border:1px solid {color}44; border-radius:12px;
                        padding:14px; text-align:center;'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div style='color:#ccc; font-size:0.75rem; margin:4px 0;'>{name}</div>
                <div style='color:{color}; font-weight:700; font-size:0.9rem;'>{round(val,3)}</div>
            </div>
            """, unsafe_allow_html=True)
