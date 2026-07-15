"""
pages/Bot_Detection.py — Bot Account Detection module
"""
import streamlit as st
import sys, os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import configure_page, inject_custom_css, sidebar_branding
from utils.helper import init_session, require_login, save_prediction
from utils.visualization import gauge_chart, bar_chart, pie_chart
from predict import predict_bot, check_models_exist

configure_page("Bot Detection | Instagram AI", "🤖")
init_session()
inject_custom_css()
sidebar_branding()
require_login()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='background:linear-gradient(135deg,#3D5AF1,#00BCD4);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           font-size:2.2rem; font-weight:900;'>
    🤖 Bot Account Detection
</h1>
<p style='color:#888; margin-top:-8px;'>
    Analyse account behavioral patterns to identify automated bot accounts.
</p>
""", unsafe_allow_html=True)

# ── Model check ────────────────────────────────────────────────────────────────
status = check_models_exist()
if not status["bot_detector"]:
    st.markdown("""
    <div style='background:#1A1D27; border:2px solid #FB8C00; border-radius:16px;
                padding:32px; text-align:center; margin:20px 0;'>
        <div style='font-size:3rem; margin-bottom:12px;'>⚠️</div>
        <h2 style='color:#FB8C00; font-weight:900; margin:0 0 12px 0;'>Bot Detection Model Not Found</h2>
        <p style='color:#ccc; font-size:1rem; margin-bottom:20px;'>
            The trained model file <code style='background:#0E1117; padding:2px 8px;
            border-radius:6px; color:#FB8C00;'>model/bot_detector.pkl</code> is missing.<br>
            You need to train the model once before using this page.
        </p>
        <div style='background:#0E1117; border-radius:10px; padding:14px 20px;
                    display:inline-block; text-align:left; margin-bottom:16px;'>
            <p style='color:#888; font-size:0.8rem; margin:0 0 6px 0;'>Run this command in your terminal:</p>
            <code style='color:#43A047; font-size:1rem; font-weight:700;'>
                python train_bot_model.py
            </code>
        </div>
        <p style='color:#888; font-size:0.82rem; margin:0;'>
            Training takes ~30 seconds and auto-generates synthetic data if no CSV is found.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.info("💡 **Tip:** Run `python train_fake_account.py` and `python train_image_model.py` too, to enable all modules.")
    st.stop()

st.markdown("---")

# ── Two-panel layout ───────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("#### 📊 Behavioural Metrics")
    with st.form("bot_form"):

        tab1, tab2 = st.tabs(["📈 Activity", "👥 Account"])

        with tab1:
            posting_frequency = st.slider(
                "📤 Posting Frequency (posts/day)", 0.0, 100.0, 2.0, 0.5,
                help="Average number of posts per day")
            avg_likes = st.number_input("❤️ Average Likes per Post", 0, 100000, 150)
            avg_comments = st.number_input("💬 Average Comments per Post", 0, 10000, 12)
            avg_engagement = st.slider(
                "📊 Average Engagement Rate", 0.0, 1.0, 0.04, 0.001, format="%.3f")
            duplicate_comments_ratio = st.slider(
                "🔁 Duplicate Comments Ratio", 0.0, 1.0, 0.05, 0.01,
                help="Fraction of identical/template comments")
            active_hours_count = st.slider(
                "⏰ Active Hours per Day", 0, 24, 8,
                help="How many hours per day the account is active")

        with tab2:
            followers = st.number_input("👥 Followers", 0, 10000000, 850)
            following = st.number_input("➕ Following", 0, 10000000, 3200)
            account_age_days = st.number_input(
                "📅 Account Age (days)", 0, 10000, 720,
                help="Number of days since account creation")

        st.markdown("---")
        submit = st.form_submit_button("🔍 Detect Bot", use_container_width=True)

with right_col:
    st.markdown("#### 🔬 Bot Behaviour Indicators")
    st.markdown("""
    <div style='background:#1A1D27; border:1px solid #3D5AF144; border-radius:16px; padding:20px;'>
        <table style='width:100%; color:#ccc; font-size:0.85rem;'>
            <tr><td>🔴 High posting frequency</td><td style='color:#E53935;'>&gt; 20/day</td></tr>
            <tr><td>🔴 Very low engagement</td><td style='color:#E53935;'>&lt; 0.5%</td></tr>
            <tr><td>🔴 High duplicate comments</td><td style='color:#E53935;'>&gt; 40%</td></tr>
            <tr><td>🔴 Always active (24/7)</td><td style='color:#E53935;'>&gt; 20 hrs/day</td></tr>
            <tr><td>🔴 New account</td><td style='color:#E53935;'>&lt; 180 days</td></tr>
            <tr><td>🔴 High following ratio</td><td style='color:#E53935;'>Following ≫ Followers</td></tr>
            <tr style='height:8px;'><td colspan='2'></td></tr>
            <tr><td>🟢 Normal posting</td><td style='color:#43A047;'>0.5 – 5/day</td></tr>
            <tr><td>🟢 Good engagement</td><td style='color:#43A047;'>&gt; 2%</td></tr>
            <tr><td>🟢 Unique comments</td><td style='color:#43A047;'>&lt; 15%</td></tr>
            <tr><td>🟢 Normal hours</td><td style='color:#43A047;'>4 – 16 hrs/day</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ── Results ────────────────────────────────────────────────────────────────────
if submit:
    raw = {
        "posting_frequency":        posting_frequency,
        "avg_likes":                avg_likes,
        "avg_comments":             avg_comments,
        "followers":                followers,
        "following":                following,
        "avg_engagement":           avg_engagement,
        "duplicate_comments_ratio": duplicate_comments_ratio,
        "active_hours_count":       active_hours_count,
        "account_age_days":         account_age_days,
    }

    with st.spinner("🔄 Analysing behavioural patterns..."):
        try:
            result = predict_bot(raw)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    label      = result["label"]
    confidence = result["confidence"]
    prob_bot   = result["prob_bot"]
    prob_human = result["prob_human"]

    # ── Result banner ─────────────────────────────────────────────────────────
    if label == "Bot":
        bg_color = "#FB8C0022"
        border   = "#FB8C00"
        emoji    = "🤖"
        msg      = "This account shows strong **BOT** behavioral patterns."
    else:
        bg_color = "#43A04722"
        border   = "#43A047"
        emoji    = "👤"
        msg      = "This account shows **HUMAN** behavioral patterns."

    st.markdown(f"""
    <div style='background:{bg_color}; border:2px solid {border}; border-radius:16px;
                padding:24px; margin:16px 0; text-align:center;'>
        <div style='font-size:3rem;'>{emoji}</div>
        <h2 style='color:{border}; font-size:2rem; font-weight:900; margin:8px 0;'>{label}</h2>
        <p style='color:#ccc;'>{msg}</p>
    </div>
    """, unsafe_allow_html=True)

    save_prediction(
        module="Bot Detection",
        input_summary=f"PostFreq:{posting_frequency} | Followers:{followers} | Following:{following}",
        prediction=label,
        confidence=confidence
    )

    st.markdown("### 📊 Analysis Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏷 Prediction", label)
    c2.metric("📊 Confidence", f"{confidence*100:.1f}%")
    c3.metric("🤖 Bot Probability", f"{prob_bot*100:.1f}%")
    c4.metric("👤 Human Probability", f"{prob_human*100:.1f}%")

    col_g, col_p = st.columns(2)
    with col_g:
        st.plotly_chart(gauge_chart(confidence, f"{label} Confidence"), use_container_width=True)
    with col_p:
        fig = pie_chart(
            labels=["Human", "Bot"],
            values=[round(prob_human*100,1), round(prob_bot*100,1)],
            title="Bot vs Human Probability",
            colors=["#43A047", "#FB8C00"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature breakdown ─────────────────────────────────────────────────────
    with st.expander("🔎 Feature Details"):
        feats = result["features"]
        feat_df = pd.DataFrame([{
            "Feature": k.replace("_", " ").title(),
            "Value": round(v, 4) if isinstance(v, float) else v
        } for k, v in feats.items()])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Suspicious pattern flags ───────────────────────────────────────────────
    st.markdown("### 🚩 Suspicious Pattern Flags")
    flags = []
    if posting_frequency > 20: flags.append(("🔴", "Abnormally high posting frequency"))
    if avg_engagement < 0.005: flags.append(("🔴", "Extremely low engagement rate"))
    if duplicate_comments_ratio > 0.4: flags.append(("🔴", "High duplicate comments ratio"))
    if active_hours_count > 20: flags.append(("🔴", "Account active almost 24/7"))
    if account_age_days < 180: flags.append(("🔴", "Very new account"))
    if following > followers * 3: flags.append(("🔴", "Following far more than followers"))

    if flags:
        for icon, text in flags:
            st.markdown(f"- {icon} {text}")
    else:
        st.success("✅ No suspicious patterns detected.")
