"""
streamlit_app.py  —  Streamlit Frontend
========================================
Two pages:
  1. Predict  — fill form, call Flask API, show result
  2. Dashboard — charts + model stats pulled from Flask API

Run (after Flask is running):
    streamlit run streamlit_app.py
"""

import os
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE   = "http://127.0.0.1:5000"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
EVAL_IMG   = os.path.join(BASE_DIR, "static", "model_evaluation.png")

st.set_page_config(
    page_title="Car Insurance Claim Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white;
        padding: 1.4rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: .3rem 0 0; opacity: .85; font-size: .95rem; }

    .result-claim    { background:#fef2f2; border:2px solid #f87171;
                       border-radius:10px; padding:1.2rem 1.6rem; }
    .result-no-claim { background:#f0fdf4; border:2px solid #4ade80;
                       border-radius:10px; padding:1.2rem 1.6rem; }

    .kpi-box { background:#f7f8fa; border-radius:8px; padding:.9rem 1.2rem;
               border-left:4px solid #2d6a9f; }
    .kpi-val { font-size:1.8rem; font-weight:800; color:#1e3a5f; }
    .kpi-lbl { font-size:.75rem; color:#6b7280; text-transform:uppercase;
               letter-spacing:.5px; }

    div[data-testid="stSidebar"] { background:#1e3a5f; }
    div[data-testid="stSidebar"] * { color:#c9d8ea !important; }
    div[data-testid="stSidebar"] .css-1d391kg { background:#1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚗 Insurance Predictor")
    st.markdown("---")
    page = st.radio("Navigation", ["🔮 Predict", "📊 Dashboard"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Flask API**")
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=3)
        if r.ok:
            _s = r.json()
            st.success("API Connected")
            st.caption(f"Model: {_s['model_name']}")
            st.caption(f"Accuracy: {_s['accuracy']}%")
        else:
            st.error("API error")
    except Exception:
        st.error("Flask API offline\nRun: `python app.py`")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict":

    st.markdown("""
    <div class="main-header">
      <h1>🔮 Car Insurance Claim Predictor</h1>
      <p>Fill in the driver and vehicle details below to predict whether a claim will be filed.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.subheader("👤 Personal Information")
        c1, c2, c3, c4 = st.columns(4)
        age        = c1.selectbox("Age Group",   ["16-25", "26-39", "40-64", "65+"])
        gender     = c2.selectbox("Gender",       ["male", "female"])
        race       = c3.selectbox("Race",         ["majority", "minority"])
        education  = c4.selectbox("Education",    ["none", "high school", "university"])

        c5, c6, c7, c8 = st.columns(4)
        income     = c5.selectbox("Income Class", ["poverty", "working class", "middle class", "upper class"])
        married    = c6.selectbox("Married",      [0, 1], format_func=lambda x: "Yes" if x else "No")
        children   = c7.selectbox("Has Children", [0, 1], format_func=lambda x: "Yes" if x else "No")
        credit     = c8.number_input("Credit Score (0–1)", min_value=0.0, max_value=1.0,
                                     value=0.55, step=0.01, format="%.3f")

        st.subheader("🚦 Driving Profile")
        d1, d2, d3, d4, d5 = st.columns(5)
        driving_exp   = d1.selectbox("Driving Experience", ["0-9y", "10-19y", "20-29y", "30y+"])
        speeding      = d2.number_input("Speeding Violations", min_value=0, value=0, step=1)
        duis          = d3.number_input("DUI Offences",        min_value=0, value=0, step=1)
        past_acc      = d4.number_input("Past Accidents",      min_value=0, value=0, step=1)
        annual_miles  = d5.number_input("Annual Mileage",      min_value=0, value=12000, step=500)

        st.subheader("🚗 Vehicle Details")
        v1, v2, v3, v4 = st.columns(4)
        vehicle_type  = v1.selectbox("Vehicle Type",       ["sedan", "sports car"])
        vehicle_year  = v2.selectbox("Vehicle Year",       ["before 2015", "after 2015"])
        vehicle_own   = v3.selectbox("Owns Vehicle",       [0, 1], format_func=lambda x: "Yes" if x else "No")
        postal        = v4.number_input("Postal Code",     min_value=0, value=10238, step=1)

        submitted = st.form_submit_button("🔍 Predict Claim", use_container_width=True,
                                          type="primary")

    if submitted:
        payload = {
            "AGE": age, "GENDER": gender, "RACE": race,
            "DRIVING_EXPERIENCE": driving_exp, "EDUCATION": education,
            "INCOME": income, "CREDIT_SCORE": credit,
            "VEHICLE_OWNERSHIP": float(vehicle_own),
            "VEHICLE_YEAR": vehicle_year, "MARRIED": float(married),
            "CHILDREN": float(children), "POSTAL_CODE": float(postal),
            "ANNUAL_MILEAGE": float(annual_miles),
            "VEHICLE_TYPE": vehicle_type,
            "SPEEDING_VIOLATIONS": float(speeding),
            "DUIS": float(duis), "PAST_ACCIDENTS": float(past_acc),
        }
        try:
            with st.spinner("Analysing..."):
                resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            data = resp.json()
            if not data.get("success"):
                st.error(f"API Error: {data.get('error')}")
            else:
                is_claim = data["prediction"] == 1
                risk_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                risk_icon   = risk_colors.get(data["risk_level"], "⚪")

                box_class = "result-claim" if is_claim else "result-no-claim"
                icon      = "⚠️" if is_claim else "✅"
                st.markdown(f"""
                <div class="{box_class}">
                  <h2>{icon} Prediction: <strong>{data['label']}</strong></h2>
                  <p>The model predicts this driver <strong>{'WILL' if is_claim else 'WILL NOT'}</strong>
                     file an insurance claim.</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")
                m1, m2, m3 = st.columns(3)
                m1.metric("Claim Probability", f"{data['probability']}%")
                m2.metric("Risk Level",         f"{risk_icon} {data['risk_level']}")
                m3.metric("Predicted Outcome",  data["label"])

                st.progress(int(data["probability"]))

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Flask API. Make sure `python app.py` is running on port 5000.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":

    st.markdown("""
    <div class="main-header">
      <h1>📊 Analytics Dashboard</h1>
      <p>Dataset exploration and model performance overview.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        s = requests.get(f"{API_BASE}/stats", timeout=5).json()
    except Exception:
        st.error("Flask API offline. Run `python app.py` first.")
        st.stop()

    # ── KPI row ────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Records",  f"{s['total']:,}")
    k2.metric("Claims Filed",   f"{s['claim_count']:,}", f"{s['claim_pct']}%")
    k3.metric("No Claims",      f"{s['no_claim_count']:,}")
    k4.metric("Model Accuracy", f"{s['accuracy']}%")
    k5.metric("ROC-AUC",        str(s['auc']))

    st.divider()

    # ── Model info ────────────────────────────────────────────────────────────
    st.subheader(f"🤖 Best Model: {s['model_name']}")
    if os.path.exists(EVAL_IMG):
        st.image(EVAL_IMG, caption="Confusion Matrix & Feature Importance", use_container_width=True)

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    st.subheader("📈 Dataset Distributions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Claims vs No Claims**")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie([s["no_claim_count"], s["claim_count"]],
               labels=["No Claim", "Claim"],
               colors=["#22c55e", "#ef4444"],
               autopct="%1.1f%%", startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("Outcome Distribution", fontsize=11, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Age Group Distribution**")
        ages  = list(s["age_dist"].keys())
        acnts = list(s["age_dist"].values())
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(ages, acnts, color="#2d6a9f", edgecolor="white")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title("Customers by Age Group", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Claim Rate by Age Group (%)**")
        ages_r = list(s["claim_by_age"].keys())
        rates  = list(s["claim_by_age"].values())
        fig, ax = plt.subplots(figsize=(5, 4))
        colors  = ["#ef4444" if r >= 40 else "#f59e0b" if r >= 25 else "#22c55e" for r in rates]
        bars    = ax.bar(ages_r, rates, color=colors, edgecolor="white")
        ax.bar_label(bars, labels=[f"{r}%" for r in rates], padding=3, fontsize=9)
        ax.set_title("Claim Rate by Age", fontsize=11, fontweight="bold")
        ax.set_ylabel("Claim Rate (%)")
        ax.set_ylim(0, max(rates) + 10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.markdown("**Claim Rate by Driving Experience (%)**")
        exps   = list(s["claim_by_exp"].keys())
        erates = list(s["claim_by_exp"].values())
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(exps, erates, color="#7c5cd8", edgecolor="white")
        for i, v in enumerate(erates):
            ax.text(v + 0.5, i, f"{v}%", va="center", fontsize=9)
        ax.set_title("Claim Rate by Driving Experience", fontsize=11, fontweight="bold")
        ax.set_xlabel("Claim Rate (%)")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Charts row 3 ──────────────────────────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**Claim Rate by Income Class (%)**")
        incs   = list(s["claim_by_income"].keys())
        irates = list(s["claim_by_income"].values())
        fig, ax = plt.subplots(figsize=(5, 4))
        palette = ["#1e3a5f", "#2d6a9f", "#06b6d4", "#7c5cd8"]
        bars    = ax.bar(incs, irates, color=palette[:len(incs)], edgecolor="white")
        ax.bar_label(bars, labels=[f"{r}%" for r in irates], padding=3, fontsize=9)
        ax.set_title("Claim Rate by Income", fontsize=11, fontweight="bold")
        ax.set_ylabel("Claim Rate (%)")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col6:
        st.markdown("**Income Distribution**")
        inc_keys = list(s["income_dist"].keys())
        inc_vals = list(s["income_dist"].values())
        fig, ax  = plt.subplots(figsize=(5, 4))
        ax.pie(inc_vals, labels=inc_keys,
               colors=["#1e3a5f","#2d6a9f","#06b6d4","#7c5cd8"],
               autopct="%1.1f%%", startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("Income Class Distribution", fontsize=11, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Raw dataset preview ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Dataset Preview")
    df_preview = pd.read_csv(os.path.join(BASE_DIR, "Car_Insurance_Claim.csv"))
    st.dataframe(df_preview.head(50), use_container_width=True)
    st.caption(f"Showing first 50 of {len(df_preview):,} rows   |   {df_preview.shape[1]} columns")
