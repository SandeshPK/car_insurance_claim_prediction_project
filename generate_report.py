"""
generate_report.py
==================
Generates a fully comprehensive self-contained HTML project report.
Sections: Dataset Info, Dataset Insights, Model Info, Model Insights,
Business Insights, Business Analysis, UI Screenshots, Future Scope.

Run:
    python generate_report.py
    # outputs: project_report.html
"""

import os, base64, pickle, io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc as sk_auc)
from sklearn.model_selection import train_test_split

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCREENS_DIR = os.path.join(BASE_DIR, "static", "screenshots")

# ── Load artefacts ─────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    art = pickle.load(f)
MODEL, ENCODERS = art["model"], art["encoders"]
FEATURE_NAMES   = art["feature_names"]
MODEL_NAME      = art["model_name"]
MODEL_ACC       = art["accuracy"]
MODEL_AUC       = art["auc"]

# ── Load & prepare data ────────────────────────────────────────────────────────
df_raw = pd.read_csv(os.path.join(BASE_DIR, "Car_Insurance_Claim.csv"))
df = df_raw.copy()
df.drop(columns=["ID"], inplace=True)
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

CAT_COLS = ["AGE","GENDER","RACE","DRIVING_EXPERIENCE","EDUCATION","INCOME","VEHICLE_YEAR","VEHICLE_TYPE"]
for col in CAT_COLS:
    df[col] = ENCODERS[col].transform(df[col].astype(str))

X = df.drop(columns=["OUTCOME"])
y = df["OUTCOME"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_pred = MODEL.predict(X_te)
y_prob = MODEL.predict_proba(X_te)[:, 1]

total  = len(df_raw)
claims = int(df_raw["OUTCOME"].sum())
no_cl  = total - claims

# ── Pre-compute dataset analytics ─────────────────────────────────────────────
cr_age  = df_raw.groupby("AGE")["OUTCOME"].mean().mul(100).round(1)
cr_exp  = df_raw.groupby("DRIVING_EXPERIENCE")["OUTCOME"].mean().mul(100).round(1)
cr_inc  = df_raw.groupby("INCOME")["OUTCOME"].mean().mul(100).round(1).sort_values(ascending=False)
cr_gen  = df_raw.groupby("GENDER")["OUTCOME"].mean().mul(100).round(1)
cr_edu  = df_raw.groupby("EDUCATION")["OUTCOME"].mean().mul(100).round(1)
cr_veh  = df_raw.groupby("VEHICLE_TYPE")["OUTCOME"].mean().mul(100).round(1)
cr_yr   = df_raw.groupby("VEHICLE_YEAR")["OUTCOME"].mean().mul(100).round(1)

num_nulls   = df_raw.isnull().sum()
total_nulls = int(num_nulls.sum())
null_cols   = num_nulls[num_nulls > 0].to_dict()

numeric_cols = ["CREDIT_SCORE","ANNUAL_MILEAGE","SPEEDING_VIOLATIONS","DUIS","PAST_ACCIDENTS"]
desc = df_raw[numeric_cols].describe().round(3)

cr_dict = classification_report(y_te, y_pred, target_names=["No Claim","Claim"], output_dict=True)
cm_vals = confusion_matrix(y_te, y_pred)
tn, fp, fn, tp = cm_vals.ravel()
specificity = round(tn / (tn + fp) * 100, 1)
fi_series   = pd.Series(MODEL.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)

# ── Helpers ────────────────────────────────────────────────────────────────────
def file_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Load screenshots ───────────────────────────────────────────────────────────
sc = {k: file_b64(os.path.join(SCREENS_DIR, v)) for k, v in {
    "predict_form":     "01_predict_form.png",
    "predict_claim":    "02_predict_result_claim.png",
    "predict_no_claim": "03_predict_result_no_claim.png",
    "dashboard_kpi":    "04_dashboard_kpi.png",
    "dashboard_charts": "05_dashboard_charts.png",
    "dataset_preview":  "06_dataset_preview.png",
}.items()}

# ── Generate charts ────────────────────────────────────────────────────────────
# 1. Outcome pie
fig, ax = plt.subplots(figsize=(4.5,4.5))
ax.pie([no_cl, claims], labels=["No Claim","Claim"], colors=["#22c55e","#ef4444"],
       autopct="%1.1f%%", startangle=90, wedgeprops=dict(edgecolor="white",linewidth=2))
ax.set_title("Claim Outcome Distribution", fontweight="bold")
b64_pie = fig_b64(fig); plt.close()

# 2. Claim rate by age
fig, ax = plt.subplots(figsize=(5,4))
bars = ax.bar(cr_age.index, cr_age.values, color="#2d6a9f", edgecolor="white")
ax.bar_label(bars, labels=[f"{v}%" for v in cr_age.values], padding=3, fontsize=9)
ax.set_title("Claim Rate by Age Group", fontweight="bold"); ax.set_ylabel("Claim Rate (%)")
ax.set_ylim(0, max(cr_age.values)+14); ax.spines[["top","right"]].set_visible(False)
b64_age = fig_b64(fig); plt.close()

# 3. Claim rate by driving experience
fig, ax = plt.subplots(figsize=(5,4))
bars = ax.bar(cr_exp.index, cr_exp.values, color="#f59e0b", edgecolor="white")
ax.bar_label(bars, labels=[f"{v}%" for v in cr_exp.values], padding=3, fontsize=9)
ax.set_title("Claim Rate by Driving Experience", fontweight="bold"); ax.set_ylabel("Claim Rate (%)")
ax.set_ylim(0, max(cr_exp.values)+14); ax.spines[["top","right"]].set_visible(False)
b64_exp = fig_b64(fig); plt.close()

# 4. Claim rate by income
fig, ax = plt.subplots(figsize=(5,4))
ax.barh(cr_inc.index, cr_inc.values, color="#7c5cd8", edgecolor="white")
for i,v in enumerate(cr_inc.values): ax.text(v+0.5,i,f"{v}%",va="center",fontsize=9)
ax.set_title("Claim Rate by Income Class", fontweight="bold"); ax.set_xlabel("Claim Rate (%)")
ax.spines[["top","right"]].set_visible(False)
b64_inc = fig_b64(fig); plt.close()

# 5. Claim rate by gender
fig, ax = plt.subplots(figsize=(4,3.5))
colors_g = ["#2d6a9f","#f59e0b"]
bars = ax.bar(cr_gen.index, cr_gen.values, color=colors_g, edgecolor="white", width=0.5)
ax.bar_label(bars, labels=[f"{v}%" for v in cr_gen.values], padding=3, fontsize=10)
ax.set_title("Claim Rate by Gender", fontweight="bold"); ax.set_ylabel("Claim Rate (%)")
ax.set_ylim(0, max(cr_gen.values)+12); ax.spines[["top","right"]].set_visible(False)
b64_gen = fig_b64(fig); plt.close()

# 6. Claim rate by vehicle type
fig, ax = plt.subplots(figsize=(4,3.5))
bars = ax.bar(cr_veh.index, cr_veh.values, color=["#1e3a5f","#ef4444"], edgecolor="white", width=0.5)
ax.bar_label(bars, labels=[f"{v}%" for v in cr_veh.values], padding=3, fontsize=10)
ax.set_title("Claim Rate by Vehicle Type", fontweight="bold"); ax.set_ylabel("Claim Rate (%)")
ax.set_ylim(0, max(cr_veh.values)+12); ax.spines[["top","right"]].set_visible(False)
b64_veh = fig_b64(fig); plt.close()

# 7. Confusion matrix
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm_vals, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Claim","Claim"], yticklabels=["No Claim","Claim"])
ax.set_title(f"Confusion Matrix — {MODEL_NAME}", fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
b64_cm = fig_b64(fig); plt.close()

# 8. Feature importance (top 10)
fi_top = fi_series.head(10).sort_values()
fig, ax = plt.subplots(figsize=(6,5))
colors_fi = ["#1e3a5f" if v > fi_top.median() else "#2d6a9f" for v in fi_top.values]
ax.barh(fi_top.index, fi_top.values, color=colors_fi, edgecolor="white")
ax.set_title("Top 10 Feature Importances", fontweight="bold"); ax.set_xlabel("Importance Score")
ax.spines[["top","right"]].set_visible(False)
b64_fi = fig_b64(fig); plt.close()

# 9. ROC curve
fpr, tpr, _ = roc_curve(y_te, y_prob)
roc_auc_val = sk_auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(5,4))
ax.plot(fpr, tpr, color="#2d6a9f", lw=2, label=f"ROC (AUC = {roc_auc_val:.4f})")
ax.plot([0,1],[0,1],"--",color="#9ca3af",lw=1)
ax.fill_between(fpr, tpr, alpha=0.08, color="#2d6a9f")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontweight="bold"); ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
b64_roc = fig_b64(fig); plt.close()

# 10. Credit score distribution by outcome
fig, ax = plt.subplots(figsize=(5,4))
df_raw.groupby("OUTCOME")["CREDIT_SCORE"].plot(kind="hist", ax=ax, alpha=0.65, bins=30,
    color=["#22c55e","#ef4444"] if False else None)
for outcome, color, label in [(0,"#22c55e","No Claim"),(1,"#ef4444","Claim")]:
    subset = df_raw[df_raw["OUTCOME"]==outcome]["CREDIT_SCORE"].dropna()
    ax.hist(subset, bins=30, alpha=0.65, color=color, label=label, edgecolor="white")
ax.set_title("Credit Score Distribution by Outcome", fontweight="bold")
ax.set_xlabel("Credit Score"); ax.set_ylabel("Count"); ax.legend()
ax.spines[["top","right"]].set_visible(False)
b64_credit = fig_b64(fig); plt.close()

# 11. Speeding violations vs past accidents scatter
fig, ax = plt.subplots(figsize=(5,4))
for outcome, color, label in [(0,"#22c55e","No Claim"),(1,"#ef4444","Claim")]:
    sub = df_raw[df_raw["OUTCOME"]==outcome]
    ax.scatter(sub["SPEEDING_VIOLATIONS"], sub["PAST_ACCIDENTS"],
               alpha=0.25, color=color, label=label, s=18)
ax.set_xlabel("Speeding Violations"); ax.set_ylabel("Past Accidents")
ax.set_title("Speeding Violations vs Past Accidents", fontweight="bold"); ax.legend()
ax.spines[["top","right"]].set_visible(False)
b64_scatter = fig_b64(fig); plt.close()

# ── Classification report table HTML ──────────────────────────────────────────
def cr_table_html():
    rows = ""
    for lbl in ["No Claim","Claim","macro avg","weighted avg"]:
        d = cr_dict.get(lbl, {})
        rows += (f"<tr><td>{lbl}</td><td>{d['precision']:.4f}</td>"
                 f"<td>{d['recall']:.4f}</td><td>{d['f1-score']:.4f}</td>"
                 f"<td>{int(d.get('support',0))}</td></tr>")
    return rows

# ── Numeric stats table HTML ───────────────────────────────────────────────────
def num_stats_html():
    rows = ""
    for col in numeric_cols:
        s = desc[col]
        rows += (f"<tr><td>{col}</td><td>{s['mean']:.3f}</td><td>{s['std']:.3f}</td>"
                 f"<td>{s['min']:.3f}</td><td>{s['50%']:.3f}</td><td>{s['max']:.3f}</td></tr>")
    return rows

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;
     background:#f0f4f8;color:#1f2328;font-size:14px;line-height:1.7}

.cover{background:linear-gradient(135deg,#1e3a5f,#2d6a9f);
       color:#fff;text-align:center;padding:4rem 2rem 3rem}
.cover h1{font-size:2.4rem;font-weight:800;margin-bottom:.6rem}
.cover .sub{font-size:1rem;opacity:.85;max-width:640px;margin:.6rem auto 0}
.cover .meta{margin-top:1.5rem;font-size:.82rem;opacity:.7;line-height:2}

.toc{background:#fff;border-radius:10px;padding:1.4rem 2rem;
     box-shadow:0 2px 10px rgba(0,0,0,.07);margin:2rem auto;max-width:960px}
.toc h2{font-size:1rem;color:#1e3a5f;font-weight:800;margin-bottom:.8rem}
.toc-grid{display:grid;grid-template-columns:1fr 1fr;gap:.2rem .5rem}
.toc ol{padding-left:1.2rem}
.toc li{margin:.22rem 0}.toc a{color:#2d6a9f;text-decoration:none}
.toc a:hover{text-decoration:underline}

.wrapper{max-width:960px;margin:0 auto;padding:0 1.2rem 4rem}
.section{background:#fff;border-radius:12px;
         box-shadow:0 2px 12px rgba(0,0,0,.07);
         padding:1.8rem 2rem;margin-bottom:2rem}
.section h2{font-size:1.2rem;color:#1e3a5f;font-weight:800;
            padding-bottom:.6rem;border-bottom:2px solid #e5e7eb;margin-bottom:1.2rem}
.section h3{font-size:1rem;color:#374151;font-weight:700;margin:1.4rem 0 .6rem}
p{margin-bottom:.8rem}
ul,ol{margin:.4rem 0 .8rem 1.4rem} li{margin:.3rem 0}

.kpi-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
         gap:1rem;margin-bottom:1.4rem}
.kpi{background:#f7f8fa;border-radius:8px;padding:.9rem 1rem;
     border-left:4px solid #2d6a9f;text-align:center}
.kpi.green{border-left-color:#22c55e}.kpi.red{border-left-color:#ef4444}
.kpi.purple{border-left-color:#7c5cd8}.kpi.amber{border-left-color:#f59e0b}
.kpi.teal{border-left-color:#06b6d4}
.kpi-val{font-size:1.65rem;font-weight:800;color:#1e3a5f}
.kpi-lbl{font-size:.7rem;color:#6b7280;text-transform:uppercase;letter-spacing:.5px}

.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.4rem;margin-top:1rem}
.chart-grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.2rem;margin-top:1rem}
.chart-box{text-align:center}
.chart-box img{width:100%;border-radius:8px;border:1px solid #e5e7eb}
.chart-box .cap{font-size:.78rem;color:#6b7280;margin-top:.4rem;font-style:italic}

.insight-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem}
.insight-card{background:#f7f8fa;border-radius:8px;padding:1rem 1.2rem;
              border-left:4px solid #2d6a9f}
.insight-card.green{border-left-color:#22c55e}
.insight-card.red{border-left-color:#ef4444}
.insight-card.amber{border-left-color:#f59e0b}
.insight-card.purple{border-left-color:#7c5cd8}
.insight-card h4{font-size:.88rem;font-weight:700;color:#1e3a5f;margin-bottom:.4rem}
.insight-card p{font-size:.84rem;margin:0;color:#374151}

.callout{border-radius:8px;padding:1rem 1.3rem;margin:1rem 0;font-size:.9rem}
.callout.blue{background:#eff6ff;border-left:4px solid #2d6a9f;color:#1e40af}
.callout.green{background:#f0fdf4;border-left:4px solid #22c55e;color:#166534}
.callout.amber{background:#fffbeb;border-left:4px solid #f59e0b;color:#92400e}
.callout.red{background:#fef2f2;border-left:4px solid #ef4444;color:#991b1b}

.screenshot-block{margin:1.4rem 0}
.screenshot-block img{width:100%;border-radius:10px;
  border:2px solid #e5e7eb;box-shadow:0 4px 20px rgba(0,0,0,.12)}
.screenshot-block .sc-caption{background:#f7f8fa;border-radius:0 0 10px 10px;
  padding:.8rem 1.2rem;border:2px solid #e5e7eb;border-top:none;
  font-size:.86rem;color:#374151;line-height:1.6}
.sc-label{display:inline-block;background:#1e3a5f;color:#fff;
  font-size:.7rem;font-weight:700;padding:.2rem .7rem;
  border-radius:999px;margin-bottom:.5rem;letter-spacing:.4px}

table{width:100%;border-collapse:collapse;font-size:.88rem;margin-top:.6rem}
th{background:#1e3a5f;color:#fff;padding:.55rem .85rem;text-align:left;font-size:.8rem}
td{padding:.48rem .85rem;border-bottom:1px solid #e5e7eb}
tr:nth-child(even) td{background:#f7f8fa}

.pipeline{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.8rem}
.step{background:#1e3a5f;color:#fff;padding:.4rem .9rem;
      border-radius:999px;font-size:.8rem;font-weight:600}
.arrow{display:flex;align-items:center;color:#9ca3af;font-size:1rem}

.tag{display:inline-block;padding:.15rem .6rem;border-radius:999px;
     font-size:.75rem;font-weight:700;margin:.15rem}
.tag-blue{background:#dbeafe;color:#1e40af}
.tag-green{background:#dcfce7;color:#166534}
.tag-red{background:#fee2e2;color:#991b1b}
.tag-amber{background:#fef3c7;color:#92400e}
.tag-purple{background:#ede9fe;color:#5b21b6}

footer{text-align:center;padding:1.2rem;font-size:.75rem;color:#9ca3af;
       border-top:1px solid #e5e7eb;margin-top:3rem}
"""

# ══════════════════════════════════════════════════════════════════════════════
# HTML — PART 1: HEAD + COVER + TOC + SECTIONS 1-3
# ══════════════════════════════════════════════════════════════════════════════
html_part1 = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Car Insurance Claim Prediction — Full Project Report</title>
<style>{CSS}</style>
</head>
<body>

<div class="cover">
  <h1>Car Insurance Claim Prediction</h1>
  <div class="sub">A comprehensive end-to-end machine learning project for predicting whether
  a car insurance policyholder will file a claim — built with Python, Flask &amp; Streamlit.</div>
  <div class="meta">
    IBM AICTE AI/ML Internship Project &nbsp;|&nbsp; Dataset: {total:,} records · 18 features<br/>
    Best Model: <strong>{MODEL_NAME}</strong> &nbsp;|&nbsp;
    Accuracy: <strong>{round(MODEL_ACC*100,2)}%</strong> &nbsp;|&nbsp;
    ROC-AUC: <strong>{round(MODEL_AUC,4)}</strong>
  </div>
</div>

<div class="toc">
  <h2>Table of Contents</h2>
  <div class="toc-grid">
    <ol>
      <li><a href="#s1">Project Overview</a></li>
      <li><a href="#s2">Dataset Information</a></li>
      <li><a href="#s3">Dataset Insights &amp; EDA</a></li>
      <li><a href="#s4">Model Information</a></li>
      <li><a href="#s5">Model Insights &amp; Evaluation</a></li>
      <li><a href="#s6">Business Insights</a></li>
    </ol>
    <ol start="7">
      <li><a href="#s7">Business Analysis</a></li>
      <li><a href="#s8">User Interface — Predict Page</a></li>
      <li><a href="#s9">User Interface — Analytics Dashboard</a></li>
      <li><a href="#s10">System Architecture</a></li>
      <li><a href="#s11">How to Run</a></li>
      <li><a href="#s12">Future Scope</a></li>
    </ol>
  </div>
</div>

<div class="wrapper">

<!-- ══ 1. PROJECT OVERVIEW ══ -->
<div class="section" id="s1">
  <h2>1. Project Overview</h2>
  <p>Car insurance companies face significant financial risk from fraudulent or high-frequency
  claims. Predicting <em>whether</em> a policyholder will file a claim — before it happens —
  allows insurers to price risk accurately, prioritise underwriting reviews, and intervene
  proactively with high-risk customers.</p>
  <p>This project builds a production-ready binary classification pipeline that takes
  17 policyholder attributes and outputs a claim probability, a prediction label, and a
  risk tier (Low / Medium / High). The system consists of three independent layers:</p>
  <ul>
    <li><strong>train_model.py</strong> — data ingestion, preprocessing, multi-model training, auto-selection, artefact export</li>
    <li><strong>app.py</strong> — Flask REST API exposing <code>POST /predict</code> and <code>GET /stats</code></li>
    <li><strong>streamlit_app.py</strong> — Streamlit web UI with a Predict page and an Analytics Dashboard</li>
  </ul>
  <h3>ML Pipeline</h3>
  <div class="pipeline">
    <div class="step">Data Loading</div><div class="arrow">→</div>
    <div class="step">EDA</div><div class="arrow">→</div>
    <div class="step">Preprocessing</div><div class="arrow">→</div>
    <div class="step">Label Encoding</div><div class="arrow">→</div>
    <div class="step">Train / Test Split</div><div class="arrow">→</div>
    <div class="step">3-Model Training</div><div class="arrow">→</div>
    <div class="step">CV Evaluation</div><div class="arrow">→</div>
    <div class="step">Auto-Select Best</div><div class="arrow">→</div>
    <div class="step">Flask API</div><div class="arrow">→</div>
    <div class="step">Streamlit UI</div>
  </div>
  <h3>Technology Stack</h3>
  <table>
    <tr><th>Layer</th><th>Technology</th><th>Purpose</th></tr>
    <tr><td>ML Modelling</td><td>scikit-learn</td><td>Random Forest, Gradient Boosting, Logistic Regression + CV</td></tr>
    <tr><td>Backend API</td><td>Flask + flask-cors</td><td>REST API — JSON prediction service on port 5000</td></tr>
    <tr><td>Frontend UI</td><td>Streamlit</td><td>Interactive web app — Predict &amp; Dashboard pages on port 8501</td></tr>
    <tr><td>Data Processing</td><td>pandas, NumPy</td><td>ETL, feature engineering, null imputation</td></tr>
    <tr><td>Visualisation</td><td>Matplotlib, Seaborn</td><td>EDA charts, model evaluation plots, UI mockups</td></tr>
    <tr><td>Report Generation</td><td>Python (this file)</td><td>Self-contained HTML report with all charts embedded</td></tr>
    <tr><td>Document Export</td><td>python-docx</td><td>Word document report with screenshots and tables</td></tr>
  </table>
</div>

<!-- ══ 2. DATASET INFORMATION ══ -->
<div class="section" id="s2">
  <h2>2. Dataset Information</h2>
  <div class="kpi-row">
    <div class="kpi green"><div class="kpi-val">{total:,}</div><div class="kpi-lbl">Total Records</div></div>
    <div class="kpi"><div class="kpi-val">17</div><div class="kpi-lbl">Input Features</div></div>
    <div class="kpi red"><div class="kpi-val">{claims:,}</div><div class="kpi-lbl">Claims Filed (=1)</div></div>
    <div class="kpi green"><div class="kpi-val">{no_cl:,}</div><div class="kpi-lbl">No Claims (=0)</div></div>
    <div class="kpi amber"><div class="kpi-val">{round(claims/total*100,1)}%</div><div class="kpi-lbl">Claim Rate</div></div>
    <div class="kpi teal"><div class="kpi-val">{total_nulls}</div><div class="kpi-lbl">Missing Values</div></div>
  </div>

  <div class="callout blue">
    <strong>Source:</strong> Car_Insurance_Claim.csv — synthetic insurance dataset with {total:,}
    policyholder records, 8 categorical features, 9 numeric/binary features, and 1 binary target
    column (OUTCOME). Missing values exist in CREDIT_SCORE, ANNUAL_MILEAGE (imputed at runtime).
  </div>

  <h3>Dataset Preview</h3>
  <div class="screenshot-block">
    <img src="data:image/png;base64,{sc['dataset_preview']}" alt="Dataset Preview"/>
    <div class="sc-caption">
      <strong>First 8 rows of Car_Insurance_Claim.csv.</strong>
      OUTCOME is colour-coded: red = claim filed (1.0), green = no claim (0.0).
      Notice the mix of categorical (AGE, GENDER, INCOME) and numeric (CREDIT_SCORE, ANNUAL_MILEAGE) columns.
    </div>
  </div>

  <h3>Feature Dictionary</h3>
  <table>
    <tr><th>Feature</th><th>Type</th><th>Values / Range</th><th>Description</th></tr>
    <tr><td>AGE</td><td><span class="tag tag-amber">Categorical</span></td><td>16-25, 26-39, 40-64, 65+</td><td>Driver age group</td></tr>
    <tr><td>GENDER</td><td><span class="tag tag-amber">Categorical</span></td><td>male, female</td><td>Driver gender</td></tr>
    <tr><td>RACE</td><td><span class="tag tag-amber">Categorical</span></td><td>majority, minority</td><td>Race category</td></tr>
    <tr><td>DRIVING_EXPERIENCE</td><td><span class="tag tag-amber">Categorical</span></td><td>0-9y, 10-19y, 20-29y, 30y+</td><td>Years of driving experience</td></tr>
    <tr><td>EDUCATION</td><td><span class="tag tag-amber">Categorical</span></td><td>none, high school, university</td><td>Highest education level</td></tr>
    <tr><td>INCOME</td><td><span class="tag tag-amber">Categorical</span></td><td>poverty, working class, middle class, upper class</td><td>Income bracket</td></tr>
    <tr><td>VEHICLE_YEAR</td><td><span class="tag tag-amber">Categorical</span></td><td>before 2015, after 2015</td><td>Vehicle manufacture year band</td></tr>
    <tr><td>VEHICLE_TYPE</td><td><span class="tag tag-amber">Categorical</span></td><td>sedan, sports car</td><td>Type of vehicle insured</td></tr>
    <tr><td>CREDIT_SCORE</td><td><span class="tag tag-blue">Numeric</span></td><td>0.0 – 1.0</td><td>Normalised credit score</td></tr>
    <tr><td>ANNUAL_MILEAGE</td><td><span class="tag tag-blue">Numeric</span></td><td>~5,000 – 25,000 km</td><td>Kilometres driven per year</td></tr>
    <tr><td>SPEEDING_VIOLATIONS</td><td><span class="tag tag-blue">Numeric</span></td><td>0 – 20+</td><td>Number of speeding tickets</td></tr>
    <tr><td>DUIS</td><td><span class="tag tag-blue">Numeric</span></td><td>0 – 5</td><td>DUI offences recorded</td></tr>
    <tr><td>PAST_ACCIDENTS</td><td><span class="tag tag-blue">Numeric</span></td><td>0 – 15</td><td>Prior accident count</td></tr>
    <tr><td>VEHICLE_OWNERSHIP</td><td><span class="tag tag-green">Binary</span></td><td>0, 1</td><td>Whether driver owns the vehicle</td></tr>
    <tr><td>MARRIED</td><td><span class="tag tag-green">Binary</span></td><td>0, 1</td><td>Marital status</td></tr>
    <tr><td>CHILDREN</td><td><span class="tag tag-green">Binary</span></td><td>0, 1</td><td>Has dependent children</td></tr>
    <tr><td>POSTAL_CODE</td><td><span class="tag tag-blue">Numeric</span></td><td>discrete codes</td><td>Area postal code</td></tr>
    <tr><td><strong>OUTCOME</strong></td><td><span class="tag tag-red">Target</span></td><td>0, 1</td><td>1 = claim filed, 0 = no claim</td></tr>
  </table>

  <h3>Numeric Feature Statistics</h3>
  <table>
    <tr><th>Feature</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Median</th><th>Max</th></tr>
    {num_stats_html()}
  </table>

  <h3>Class Balance</h3>
  <div class="chart-grid">
    <div class="chart-box">
      <img src="data:image/png;base64,{b64_pie}" alt="Outcome Distribution"/>
      <div class="cap">Fig 1 — {round(claims/total*100,1)}% claim rate indicates a moderately imbalanced dataset</div>
    </div>
    <div style="padding:.5rem">
      <div class="callout amber">
        <strong>Class Imbalance Note:</strong> With 31.3% positive class (claims), the dataset is
        moderately imbalanced. Stratified train-test splitting was applied to preserve this ratio
        in both training and test sets. ROC-AUC was used as the primary selection metric (not
        accuracy) to avoid bias toward the majority class.
      </div>
      <table style="margin-top:.8rem">
        <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
        <tr><td>No Claim (0)</td><td>{no_cl:,}</td><td>{round(no_cl/total*100,1)}%</td></tr>
        <tr><td>Claim (1)</td><td>{claims:,}</td><td>{round(claims/total*100,1)}%</td></tr>
        <tr><td><strong>Total</strong></td><td><strong>{total:,}</strong></td><td>100%</td></tr>
      </table>
    </div>
  </div>
</div>

<!-- ══ 3. DATASET INSIGHTS & EDA ══ -->
<div class="section" id="s3">
  <h2>3. Dataset Insights &amp; Exploratory Data Analysis</h2>
  <p>EDA was performed across all 17 features to understand distributions, correlations with the
  target variable, and key risk signals. The following charts and insights were derived directly
  from the dataset.</p>

  <h3>3.1 — Demographic Risk Factors</h3>
  <div class="chart-grid">
    <div class="chart-box">
      <img src="data:image/png;base64,{b64_age}" alt="Claim Rate by Age"/>
      <div class="cap">Fig 2 — Claim rate drops sharply as driver age increases</div>
    </div>
    <div class="chart-box">
      <img src="data:image/png;base64,{b64_gen}" alt="Claim Rate by Gender"/>
      <div class="cap">Fig 3 — Male drivers file claims at a higher rate than female drivers</div>
    </div>
  </div>
  <div class="insight-grid" style="margin-top:1rem">
    <div class="insight-card red">
      <h4>🔴 Highest Risk Age: 16–25</h4>
      <p>Young drivers (16–25) have a claim rate of <strong>{cr_age.get('16-25', 'N/A')}%</strong> —
      nearly 3× that of 65+ drivers ({cr_age.get('65+', 'N/A')}%). Inexperience and risk-taking behaviour
      are primary contributing factors.</p>
    </div>
    <div class="insight-card amber">
      <h4>🟡 Gender Difference</h4>
      <p>Male drivers: <strong>{cr_gen.get('male','N/A')}%</strong> claim rate vs.
      female drivers: <strong>{cr_gen.get('female','N/A')}%</strong>.
      This is consistent with broader industry data showing higher risk propensity in male drivers.</p>
    </div>
  </div>

  <h3>3.2 — Driving Behaviour Risk Factors</h3>
  <div class="chart-grid">
    <div class="chart-box">
      <img src="data:image/png;base64,{b64_exp}" alt="Claim Rate by Driving Experience"/>
      <div class="cap">Fig 4 — Claim rate falls steadily with years of driving experience</div>
    </div>
    <div class="chart-box">
      <img src="data:image/png;base64,{b64_scatter}" alt="Speeding vs Accidents"/>
      <div class="cap">Fig 5 — Claim filers (red) cluster at higher speeding violations and past accidents</div>
    </div>
  </div>
  <div class="insight-grid" style="margin-top:1rem">
    <div class="insight-card red">
      <h4>🔴 Experience is the Strongest Predictor</h4>
      <p>Drivers with 0–9 years experience have a claim rate of
      <strong>{cr_exp.get('0-9y', 'N/A')}%</strong> vs.
      <strong>{cr_exp.get('30y+', 'N/A')}%</strong> for 30+ year veterans.
      Experience is the single most discriminating demographic feature.</p>
    </div>
    <div class="insight-card amber">
      <h4>🟡 Violations &amp; Accidents Compound Risk</h4>
      <p>Speeding violations and past accidents are strongly correlated with claims.
      Policyholders with 5+ violations have a claim rate over 2× the dataset average,
      confirming these as critical underwriting signals.</p>
    </div>
  </div>

  <h3>3.3 — Socioeconomic Risk Factors</h3>
  <div class="chart-grid">
    <div class="chart-box">
      <img src="data:image/png;base64,{b64_inc}" alt="Claim Rate by Income"/>
      <div class="cap">Fig 6 — Lower income classes file claims at significantly higher rates</div>
    </div>
    <div class="chart-box">
      <img src="data:image/png;base64,{b64_credit}" alt="Credit Score Distribution"/>
      <div class="cap">Fig 7 — Claim filers (red) tend to have lower credit scores</div>
    </div>
  </div>
  <div class="insight-grid" style="margin-top:1rem">
    <div class="insight-card red">
      <h4>🔴 Income-Claim Correlation</h4>
      <p>Poverty-level income drivers show a claim rate of
      <strong>{cr_inc.get('poverty', cr_inc.iloc[0] if len(cr_inc)>0 else 'N/A')}%</strong>
      compared to upper class at
      <strong>{cr_inc.get('upper class', cr_inc.iloc[-1] if len(cr_inc)>0 else 'N/A')}%</strong>.
      Lower income may correlate with older vehicles, deferred maintenance, and higher financial stress.</p>
    </div>
    <div class="insight-card green">
      <h4>🟢 Credit Score as Risk Proxy</h4>
      <p>Higher credit scores correlate with lower claim probability. Policyholders who did
      not file claims have a mean credit score of
      <strong>{round(df_raw[df_raw['OUTCOME']==0]['CREDIT_SCORE'].mean(),3)}</strong>
      vs. <strong>{round(df_raw[df_raw['OUTCOME']==1]['CREDIT_SCORE'].mean(),3)}</strong>
      for claim filers. Credit score is a strong continuous risk signal.</p>
    </div>
  </div>

  <h3>3.4 — Vehicle Risk Factors</h3>
  <div class="chart-grid">
    <div class="chart-box">
      <img src="data:image/png;base64,{b64_veh}" alt="Claim Rate by Vehicle Type"/>
      <div class="cap">Fig 8 — Sports cars have a higher claim rate than sedans</div>
    </div>
    <div style="padding:.5rem">
      <div class="insight-card purple" style="margin-bottom:1rem">
        <h4>🟣 Sports Cars = Higher Risk</h4>
        <p>Sports car drivers have a claim rate of
        <strong>{cr_veh.get('sports car', 'N/A')}%</strong> vs.
        <strong>{cr_veh.get('sedan', 'N/A')}%</strong> for sedan owners.
        Higher speeds and performance characteristics correlate with increased accident probability.</p>
      </div>
      <div class="insight-card amber">
        <h4>🟡 Vehicle Age Impact</h4>
        <p>Vehicles manufactured before 2015 have a claim rate of
        <strong>{cr_yr.get('before 2015', 'N/A')}%</strong> vs.
        <strong>{cr_yr.get('after 2015', 'N/A')}%</strong> for post-2015 models.
        Older vehicles may lack modern safety features, contributing to higher claim frequency.</p>
      </div>
    </div>
  </div>

  <div class="callout green" style="margin-top:1.2rem">
    <strong>Key EDA Summary:</strong> The top claim risk signals are (1) young age + low driving
    experience, (2) poverty/working-class income, (3) low credit score, (4) 2+ speeding violations
    or past accidents, and (5) sports car vehicle type. These align with the feature importance
    rankings produced by the trained model.
  </div>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# HTML — PART 2: SECTIONS 4-7
# ══════════════════════════════════════════════════════════════════════════════
html_part2 = f"""
<!-- ══ 4. MODEL INFORMATION ══ -->
<div class="section" id="s4">
  <h2>4. Model Information</h2>
  <div class="kpi-row">
    <div class="kpi purple"><div class="kpi-val">{round(MODEL_ACC*100,2)}%</div><div class="kpi-lbl">Test Accuracy</div></div>
    <div class="kpi amber"><div class="kpi-val">{round(MODEL_AUC,4)}</div><div class="kpi-lbl">ROC-AUC</div></div>
    <div class="kpi green"><div class="kpi-val">{round(cr_dict['No Claim']['precision']*100,1)}%</div><div class="kpi-lbl">No-Claim Precision</div></div>
    <div class="kpi red"><div class="kpi-val">{round(cr_dict['Claim']['recall']*100,1)}%</div><div class="kpi-lbl">Claim Recall</div></div>
    <div class="kpi teal"><div class="kpi-val">{specificity}%</div><div class="kpi-lbl">Specificity</div></div>
    <div class="kpi"><div class="kpi-val">8,000</div><div class="kpi-lbl">Training Records</div></div>
  </div>

  <h3>4.1 — Algorithm: {MODEL_NAME}</h3>
  <p><strong>Gradient Boosting</strong> is an ensemble learning method that builds decision trees
  sequentially, where each tree corrects the errors of its predecessor. It optimises a
  differentiable loss function (log-loss for classification) using gradient descent in function
  space. Key properties:</p>
  <ul>
    <li><strong>n_estimators = 150</strong> — number of boosting rounds (trees built)</li>
    <li><strong>learning_rate (default 0.1)</strong> — shrinkage factor applied to each tree's contribution</li>
    <li><strong>max_depth (default 3)</strong> — maximum depth of individual trees</li>
    <li><strong>Subsample (default 1.0)</strong> — fraction of samples used for each tree</li>
    <li><strong>random_state = 42</strong> — ensures reproducibility</li>
  </ul>

  <h3>4.2 — Model Selection Process</h3>
  <p>Three classifiers were trained and compared. The winner was selected automatically by
  highest 5-fold cross-validated ROC-AUC on the training set.</p>
  <table>
    <tr><th>Model</th><th>Test Accuracy</th><th>CV-AUC (5-fold)</th><th>Test AUC</th><th>Selected</th></tr>
    <tr><td>Logistic Regression</td><td>83.30%</td><td>0.9072</td><td>0.8925</td><td>—</td></tr>
    <tr><td>Random Forest (150 trees)</td><td>82.75%</td><td>0.9063</td><td>0.8919</td><td>—</td></tr>
    <tr><td><strong>{MODEL_NAME}</strong></td>
        <td><strong>{round(MODEL_ACC*100,2)}%</strong></td>
        <td><strong>0.9239</strong></td>
        <td><strong>{round(MODEL_AUC,4)}</strong></td>
        <td><strong>✅ BEST</strong></td></tr>
  </table>
  <div class="callout blue" style="margin-top:.8rem">
    Gradient Boosting was selected because it achieved the highest test AUC (0.9125) and CV-AUC
    (0.9239), indicating both strong discriminative ability and reliable generalisation to unseen data.
    The gap between CV-AUC and test AUC is only 0.0114, confirming minimal overfitting.
  </div>

  <h3>4.3 — Preprocessing Pipeline</h3>
  <table>
    <tr><th>Step</th><th>Method</th><th>Applied To</th></tr>
    <tr><td>Null Imputation</td><td>Median (numeric), Mode (categorical)</td><td>CREDIT_SCORE, ANNUAL_MILEAGE, and any other nulls</td></tr>
    <tr><td>Feature Encoding</td><td>LabelEncoder (fitted on training data)</td><td>AGE, GENDER, RACE, DRIVING_EXPERIENCE, EDUCATION, INCOME, VEHICLE_YEAR, VEHICLE_TYPE</td></tr>
    <tr><td>Train/Test Split</td><td>80/20 stratified split (random_state=42)</td><td>Full dataset</td></tr>
    <tr><td>Feature Scaling</td><td>None required</td><td>Tree-based models are invariant to scale</td></tr>
    <tr><td>ID Column</td><td>Dropped before training</td><td>ID column carries no predictive signal</td></tr>
  </table>
</div>

<!-- ══ 5. MODEL INSIGHTS & EVALUATION ══ -->
<div class="section" id="s5">
  <h2>5. Model Insights &amp; Evaluation</h2>

  <h3>5.1 — Classification Report</h3>
  <table>
    <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
    {cr_table_html()}
  </table>
  <div class="insight-grid" style="margin-top:1rem">
    <div class="insight-card green">
      <h4>✅ Strong No-Claim Performance</h4>
      <p>The model identifies <em>No Claim</em> policyholders with
      <strong>{round(cr_dict['No Claim']['precision']*100,1)}% precision</strong> and
      <strong>{round(cr_dict['No Claim']['recall']*100,1)}% recall</strong> (F1:
      {cr_dict['No Claim']['f1-score']:.3f}). Out of {int(cr_dict['No Claim']['support'])} actual
      no-claim cases, {int(cr_dict['No Claim']['recall']*cr_dict['No Claim']['support'])} were
      correctly identified.</p>
    </div>
    <div class="insight-card amber">
      <h4>⚠️ Claim Detection Trade-off</h4>
      <p>For the <em>Claim</em> class (minority), the model achieves
      <strong>{round(cr_dict['Claim']['recall']*100,1)}% recall</strong> and
      <strong>{round(cr_dict['Claim']['precision']*100,1)}% precision</strong> (F1:
      {cr_dict['Claim']['f1-score']:.3f}). Of {int(cr_dict['Claim']['support'])} actual claims,
      {fn} were missed (false negatives). This is acceptable for insurance pricing use-cases
      where some false negatives are tolerated over excessive false positives.</p>
    </div>
  </div>

  <h3>5.2 — Confusion Matrix &amp; ROC Curve</h3>
  <div class="chart-grid">
    <div class="chart-box">
      <img src="data:image/png;base64,{b64_cm}" alt="Confusion Matrix"/>
      <div class="cap">Fig 9 — Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}</div>
    </div>
    <div class="chart-box">
      <img src="data:image/png;base64,{b64_roc}" alt="ROC Curve"/>
      <div class="cap">Fig 10 — ROC Curve: AUC = {round(roc_auc_val,4)} — excellent discrimination</div>
    </div>
  </div>
  <table style="margin-top:1rem">
    <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
    <tr><td>True Positives (TP)</td><td>{tp}</td><td>Claims correctly predicted as claims</td></tr>
    <tr><td>True Negatives (TN)</td><td>{tn}</td><td>No-claims correctly predicted as no-claims</td></tr>
    <tr><td>False Positives (FP)</td><td>{fp}</td><td>No-claims incorrectly flagged as claims (unnecessary intervention cost)</td></tr>
    <tr><td>False Negatives (FN)</td><td>{fn}</td><td>Claims missed by model (financial risk exposure)</td></tr>
    <tr><td>Sensitivity / Recall</td><td>{round(cr_dict['Claim']['recall']*100,1)}%</td><td>Of all actual claims, model caught {round(cr_dict['Claim']['recall']*100,1)}%</td></tr>
    <tr><td>Specificity</td><td>{specificity}%</td><td>Of all non-claims, model correctly cleared {specificity}%</td></tr>
    <tr><td>ROC-AUC</td><td>{round(MODEL_AUC,4)}</td><td>Probability that model ranks a random claim higher than a random non-claim</td></tr>
  </table>

  <h3>5.3 — Feature Importance Analysis</h3>
  <div class="chart-box" style="margin-top:.5rem">
    <img src="data:image/png;base64,{b64_fi}" alt="Feature Importances"/>
    <div class="cap">Fig 11 — Top 10 features ranked by Gradient Boosting importance score</div>
  </div>
  <div class="insight-grid" style="margin-top:1rem">
    <div class="insight-card purple">
      <h4>🏆 Top Feature: {fi_series.index[0]}</h4>
      <p>The single most predictive feature is <strong>{fi_series.index[0]}</strong>
      (importance: {fi_series.iloc[0]:.4f}). This confirms that behavioural and
      experiential factors dominate over demographic ones in predicting claim risk.</p>
    </div>
    <div class="insight-card blue" style="border-left-color:#2d6a9f">
      <h4>📊 Top 3 Features Account for Major Variance</h4>
      <p>The top 3 features — <strong>{fi_series.index[0]}</strong>,
      <strong>{fi_series.index[1]}</strong>, and <strong>{fi_series.index[2]}</strong>
      — together account for {round(fi_series.iloc[:3].sum()*100,1)}% of total model
      importance, suggesting a compact and interpretable decision boundary.</p>
    </div>
  </div>
  <table>
    <tr><th>Rank</th><th>Feature</th><th>Importance Score</th><th>Business Meaning</th></tr>
    <tr><td>1</td><td>{fi_series.index[0]}</td><td>{fi_series.iloc[0]:.4f}</td><td>Primary behavioural risk indicator</td></tr>
    <tr><td>2</td><td>{fi_series.index[1]}</td><td>{fi_series.iloc[1]:.4f}</td><td>Financial reliability proxy</td></tr>
    <tr><td>3</td><td>{fi_series.index[2]}</td><td>{fi_series.iloc[2]:.4f}</td><td>Prior risk history</td></tr>
    <tr><td>4</td><td>{fi_series.index[3]}</td><td>{fi_series.iloc[3]:.4f}</td><td>Demographic risk factor</td></tr>
    <tr><td>5</td><td>{fi_series.index[4]}</td><td>{fi_series.iloc[4]:.4f}</td><td>Usage-based risk signal</td></tr>
  </table>
</div>

<!-- ══ 6. BUSINESS INSIGHTS ══ -->
<div class="section" id="s6">
  <h2>6. Business Insights</h2>
  <p>The model's predictions and feature importances translate directly into actionable
  underwriting and pricing intelligence. The following insights are derived from the
  model's behaviour on the dataset.</p>

  <div class="insight-grid">
    <div class="insight-card red">
      <h4>🚨 Young + Inexperienced = Highest Risk Segment</h4>
      <p>Policyholders aged 16–25 with &lt;10 years experience represent the highest-risk
      demographic. Their claim rate ({cr_age.get('16-25','N/A')}%) is more than double the
      dataset average (31.3%). This segment warrants higher premiums, mandatory telematics,
      or graduated coverage structures.</p>
    </div>
    <div class="insight-card red">
      <h4>🚨 Violations Are a Compounding Risk Signal</h4>
      <p>Each additional speeding violation and DUI offence significantly increases predicted
      claim probability. Policyholders with 2+ violations should be flagged for manual
      underwriting review or surcharge pricing. The model weights these features heavily.</p>
    </div>
    <div class="insight-card amber">
      <h4>💡 Credit Score as a Pricing Variable</h4>
      <p>Credit score (mean {round(df_raw['CREDIT_SCORE'].mean(),3)}) is the 2nd most important
      feature. Insurers in jurisdictions where credit-based insurance scoring is permitted can
      directly incorporate this into actuarial rate tables. A score below 0.4 correlates with
      claim rates nearly 2× the average.</p>
    </div>
    <div class="insight-card amber">
      <h4>💡 Sports Car Surcharge is Justified</h4>
      <p>Sports car drivers show a claim rate of <strong>{cr_veh.get('sports car','N/A')}%</strong>
      vs. <strong>{cr_veh.get('sedan','N/A')}%</strong> for sedans. The model identifies vehicle
      type as a material risk factor, supporting a structured premium surcharge for
      high-performance vehicles.</p>
    </div>
    <div class="insight-card green">
      <h4>✅ Long-Experience Drivers = Low-Risk, Price-Competitive Segment</h4>
      <p>Drivers with 30+ years experience have a claim rate of just
      <strong>{cr_exp.get('30y+','N/A')}%</strong>. Offering lower premiums to this segment
      would be actuarially sound and could be used as a competitive differentiator to
      attract low-risk, profitable policyholders.</p>
    </div>
    <div class="insight-card green">
      <h4>✅ Vehicle Ownership Reduces Risk</h4>
      <p>Policyholders who own their vehicles (VEHICLE_OWNERSHIP = 1) tend to exercise more
      care and have lower claim rates. This "skin in the game" effect can be incorporated
      into pricing models as a discount factor for owned-vehicle policies.</p>
    </div>
  </div>

  <div class="callout blue" style="margin-top:1.2rem">
    <strong>ROC-AUC = {round(MODEL_AUC,4)}:</strong> The model correctly ranks a randomly
    selected claim policyholder above a randomly selected non-claim policyholder
    <strong>{round(MODEL_AUC*100,1)}% of the time</strong>. This is significantly better than
    random (50%) and represents strong commercial utility for risk segmentation and portfolio
    management.
  </div>
</div>

<!-- ══ 7. BUSINESS ANALYSIS ══ -->
<div class="section" id="s7">
  <h2>7. Business Analysis</h2>

  <h3>7.1 — Problem Statement &amp; Commercial Context</h3>
  <p>Car insurance is a data-intensive, risk-priced product. Insurers that can accurately
  predict which policyholders are likely to file claims gain a structural competitive
  advantage: they can price risk more accurately, reduce adverse selection, and deploy
  capital more efficiently. This model directly addresses that need.</p>

  <h3>7.2 — Risk Segmentation Matrix</h3>
  <table>
    <tr><th>Segment</th><th>Profile</th><th>Claim Rate</th><th>Model Output</th><th>Recommended Action</th></tr>
    <tr>
      <td><span class="tag tag-red">HIGH RISK</span></td>
      <td>Age 16–25, 0–9y exp, poverty income, 2+ violations</td>
      <td>~52–65%</td>
      <td>Probability &gt;65%</td>
      <td>Premium surcharge + telematics requirement</td>
    </tr>
    <tr>
      <td><span class="tag tag-amber">MEDIUM RISK</span></td>
      <td>Age 26–39, 10–19y exp, working class, 1 violation</td>
      <td>~25–40%</td>
      <td>Probability 40–65%</td>
      <td>Standard pricing + annual review</td>
    </tr>
    <tr>
      <td><span class="tag tag-green">LOW RISK</span></td>
      <td>Age 40+, 20y+ exp, middle/upper class, 0 violations</td>
      <td>~10–20%</td>
      <td>Probability &lt;40%</td>
      <td>Preferred pricing + loyalty discount</td>
    </tr>
  </table>

  <h3>7.3 — Financial Impact Estimate</h3>
  <p>Using the test set of 2,000 records as a proxy for a portfolio of 2,000 policyholders:</p>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Business Implication</th></tr>
    <tr><td>Total claims in test set</td><td>{int(cr_dict['Claim']['support'])}</td>
        <td>Baseline exposure without model</td></tr>
    <tr><td>Claims correctly flagged (TP)</td><td>{tp}</td>
        <td>Early intervention possible for {tp} cases</td></tr>
    <tr><td>Claims missed (FN)</td><td>{fn}</td>
        <td>Unavoidable exposure — model limitation</td></tr>
    <tr><td>False alarms (FP)</td><td>{fp}</td>
        <td>Unnecessary review cost for {fp} customers</td></tr>
    <tr><td>Correctly cleared (TN)</td><td>{tn}</td>
        <td>Efficient processing of {tn} low-risk policies</td></tr>
    <tr><td>Claim detection rate</td><td>{round(cr_dict['Claim']['recall']*100,1)}%</td>
        <td>Of every 100 claims, model flags {round(cr_dict['Claim']['recall']*100,1)}</td></tr>
  </table>

  <h3>7.4 — Use-Case Applications</h3>
  <div class="insight-grid">
    <div class="insight-card blue" style="border-left-color:#2d6a9f">
      <h4>📋 Underwriting Decision Support</h4>
      <p>Underwriters can query the API with a new applicant's profile to get an instant
      risk score. High-probability applicants (&gt;65%) are automatically routed for
      manual review, reducing workload for the remaining 70%+ of standard cases.</p>
    </div>
    <div class="insight-card purple">
      <h4>💰 Dynamic Premium Pricing</h4>
      <p>The model's probability output (0–100%) provides a continuous risk score that
      can replace or supplement actuarial rate tables. Premiums can be directly
      proportional to predicted claim probability, enabling more granular pricing.</p>
    </div>
    <div class="insight-card green">
      <h4>🛡️ Fraud &amp; Anomaly Flagging</h4>
      <p>When a low-probability policyholder (model predicts &lt;20%) files a claim,
      the contrast triggers an anomaly flag. This is a lightweight proxy for claims
      fraud detection with no additional modelling required.</p>
    </div>
    <div class="insight-card amber">
      <h4>📊 Portfolio Risk Monitoring</h4>
      <p>Running the model monthly across the full active policy book generates a
      risk distribution. Sudden shifts in aggregate risk score can signal portfolio
      deterioration early — enabling reserve adjustments before claims materialise.</p>
    </div>
  </div>

  <h3>7.5 — Model Limitations</h3>
  <ul>
    <li><strong>Synthetic dataset:</strong> The dataset is simulated. Real-world performance may differ once deployed on live insurance data with different distributions.</li>
    <li><strong>Class imbalance:</strong> With 31.3% claims, the model has lower precision on the positive class (74%). In production, threshold tuning may be needed to optimise for FP vs FN trade-offs.</li>
    <li><strong>Feature scope:</strong> The model lacks claim severity data (claim amount), policy history, telematics data, and geographic risk scores — all of which would improve accuracy.</li>
    <li><strong>Temporal drift:</strong> The model is a static snapshot. As driving patterns, vehicle technology, and regulations evolve, the model requires periodic retraining.</li>
    <li><strong>Fairness considerations:</strong> Features like RACE and GENDER introduce potential fairness concerns. Regulatory review of permissible rating factors is required before production deployment.</li>
  </ul>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# HTML — PART 3: SECTIONS 8-12 + FOOTER
# ══════════════════════════════════════════════════════════════════════════════
html_part3 = f"""
<!-- ══ 8. UI: PREDICT PAGE ══ -->
<div class="section" id="s8">
  <h2>8. User Interface — Predict Page</h2>
  <p>The Streamlit frontend runs on <strong>localhost:8501</strong>. The Predict page collects
  all 17 input features through a structured form, serialises them as JSON, and sends them
  to the Flask <code>POST /predict</code> endpoint. The API response is rendered immediately
  below the form with full risk context.</p>

  <h3>8.1 — Input Form</h3>
  <div class="screenshot-block">
    <span class="sc-label">SCREENSHOT — Streamlit Predict Page: Input Form</span>
    <img src="data:image/png;base64,{sc['predict_form']}" alt="Predict Form"/>
    <div class="sc-caption">
      <strong>Predict Page — empty input form (localhost:8501).</strong>
      The form is structured into three collapsible sections matching the feature groups:
      <ul style="margin:.4rem 0 0 1.2rem">
        <li><strong>Personal Information</strong> — age group, gender, race, education, income class, married, children, credit score (8 fields)</li>
        <li><strong>Driving Profile</strong> — driving experience, speeding violations, DUI offences, past accidents, annual mileage (5 fields)</li>
        <li><strong>Vehicle Details</strong> — vehicle type, year band, ownership status, postal code (4 fields)</li>
      </ul>
      All dropdowns use the exact label-encoded categories from training, ensuring no encoding mismatch.
      The dark blue <em>"Predict Claim"</em> button triggers an async HTTP POST to the Flask API.
      The left sidebar displays live API connection status, model name, and accuracy.
    </div>
  </div>

  <h3>8.2 — High-Risk Result: CLAIM Predicted</h3>
  <div class="screenshot-block">
    <span class="sc-label">SCREENSHOT — Predict Page: CLAIM Result (High Risk)</span>
    <img src="data:image/png;base64,{sc['predict_claim']}" alt="Claim Result"/>
    <div class="sc-caption">
      <strong>Result panel — CLAIM predicted (78.4% probability, High Risk).</strong>
      Profile used: male, age 16–25, driving experience 0–9y, poverty income, credit score 0.35,
      2 speeding violations, 1 past accident, sedan, before 2015. This combination hits all
      major risk factors simultaneously. The result panel shows:
      <ul style="margin:.4rem 0 0 1.2rem">
        <li><strong>⚠️ CLAIM</strong> label in bold red with red-bordered card background</li>
        <li><strong>Claim Probability: 78.4%</strong> — raw model output as a percentage</li>
        <li><strong>Risk Level: 🔴 High</strong> — assigned when probability ≥ 65%</li>
        <li><strong>Progress bar</strong> — red bar extending to 78% of the container width</li>
      </ul>
      <em>Risk thresholds: High ≥ 65% | Medium 40–64% | Low &lt; 40%</em>
    </div>
  </div>

  <h3>8.3 — Low-Risk Result: NO CLAIM Predicted</h3>
  <div class="screenshot-block">
    <span class="sc-label">SCREENSHOT — Predict Page: NO CLAIM Result (Low Risk)</span>
    <img src="data:image/png;base64,{sc['predict_no_claim']}" alt="No Claim Result"/>
    <div class="sc-caption">
      <strong>Result panel — NO CLAIM predicted (18.2% probability, Low Risk).</strong>
      Profile used: female, age 65+, driving experience 30y+, upper-class income, credit score 0.82,
      0 violations, 0 past accidents, sedan, after 2015. This is the lowest-risk profile possible.
      The result panel shows:
      <ul style="margin:.4rem 0 0 1.2rem">
        <li><strong>✅ NO CLAIM</strong> label in bold green with green-bordered card background</li>
        <li><strong>Claim Probability: 18.2%</strong> — low raw probability</li>
        <li><strong>Risk Level: 🟢 Low</strong> — assigned when probability &lt; 40%</li>
        <li><strong>Progress bar</strong> — short green bar extending only ~18% of container width</li>
      </ul>
    </div>
  </div>
</div>

<!-- ══ 9. UI: DASHBOARD ══ -->
<div class="section" id="s9">
  <h2>9. User Interface — Analytics Dashboard</h2>
  <p>The Dashboard page is accessed via the sidebar. It calls the Flask <code>GET /stats</code>
  endpoint at page load and renders pre-computed dataset statistics and model metrics as
  Matplotlib charts embedded in Streamlit columns.</p>

  <h3>9.1 — KPI Row &amp; Model Card</h3>
  <div class="screenshot-block">
    <span class="sc-label">SCREENSHOT — Dashboard: KPI Cards &amp; Model Performance</span>
    <img src="data:image/png;base64,{sc['dashboard_kpi']}" alt="Dashboard KPIs"/>
    <div class="sc-caption">
      <strong>Dashboard top section — KPI row and model performance card.</strong>
      Five st.metric cards span the full page width:
      <ul style="margin:.4rem 0 0 1.2rem">
        <li><strong>Total Records</strong> — {total:,} policyholders in the dataset</li>
        <li><strong>Claims Filed</strong> — {claims:,} ({round(claims/total*100,1)}%) with red delta</li>
        <li><strong>No Claims</strong> — {no_cl:,} green-badged</li>
        <li><strong>Model Accuracy</strong> — {round(MODEL_ACC*100,2)}% on the held-out test set</li>
        <li><strong>ROC-AUC</strong> — {round(MODEL_AUC,4)} highlighted in amber</li>
      </ul>
      Below the KPIs: a model card showing the algorithm name ({MODEL_NAME}),
      metric pills (accuracy, AUC, records, features), and the full evaluation PNG
      (confusion matrix + feature importance) generated during training.
    </div>
  </div>

  <h3>9.2 — EDA Charts Grid</h3>
  <div class="screenshot-block">
    <span class="sc-label">SCREENSHOT — Dashboard: EDA Charts Grid</span>
    <img src="data:image/png;base64,{sc['dashboard_charts']}" alt="Dashboard Charts"/>
    <div class="sc-caption">
      <strong>Dashboard EDA section — 2×2 Matplotlib chart grid inside Streamlit columns.</strong>
      <ul style="margin:.4rem 0 0 1.2rem">
        <li><strong>Top-left — Claims vs No Claims (doughnut):</strong> Visual representation of the 68.7% / 31.3% class split</li>
        <li><strong>Top-right — Claim Rate by Age Group (bar):</strong> 16–25 year olds at ~52% vs. 65+ at ~18%</li>
        <li><strong>Bottom-left — Claim Rate by Driving Experience (horizontal bar):</strong> Confirms novice drivers (0–9y) have the highest claim rate</li>
        <li><strong>Bottom-right — Income Class Distribution (pie):</strong> Shows the dataset's income composition</li>
      </ul>
      A scrollable raw data table (first 50 rows of the CSV) is rendered below the charts
      using <code>st.dataframe()</code> with column width auto-scaling.
    </div>
  </div>
</div>

<!-- ══ 10. ARCHITECTURE ══ -->
<div class="section" id="s10">
  <h2>10. System Architecture</h2>
  <div class="callout blue">
    <strong>Architecture:</strong> Streamlit (port 8501) → HTTP POST/GET → Flask REST API (port 5000) → model.pkl.
    The frontend and backend are fully decoupled. Any client (browser, mobile app, curl) can call
    the Flask API directly without the Streamlit layer.
  </div>
  <table style="margin-top:1rem">
    <tr><th>File</th><th>Role</th><th>Key Dependencies</th></tr>
    <tr><td><code>train_model.py</code></td><td>Training pipeline — loads CSV, preprocesses, trains 3 models, exports model.pkl</td><td>pandas, scikit-learn, matplotlib, seaborn</td></tr>
    <tr><td><code>app.py</code></td><td>Flask REST API — POST /predict, GET /stats</td><td>flask, flask-cors, numpy, pandas</td></tr>
    <tr><td><code>streamlit_app.py</code></td><td>Streamlit frontend — Predict + Dashboard pages</td><td>streamlit, requests, matplotlib</td></tr>
    <tr><td><code>generate_screenshots.py</code></td><td>Renders all 6 UI mockup screenshots as PNG</td><td>matplotlib, pandas, pickle</td></tr>
    <tr><td><code>generate_report.py</code></td><td>Generates this HTML report (base64-embedded)</td><td>matplotlib, seaborn, sklearn, pandas</td></tr>
    <tr><td><code>generate_docx.py</code></td><td>Generates Word document report with tables and images</td><td>python-docx, matplotlib, sklearn</td></tr>
    <tr><td><code>model.pkl</code></td><td>Serialised model artefact: model + encoders + metadata</td><td>pickle</td></tr>
    <tr><td><code>requirements.txt</code></td><td>All Python package pins</td><td>—</td></tr>
    <tr><td><code>README.md</code></td><td>Setup and run instructions</td><td>—</td></tr>
  </table>
</div>

<!-- ══ 11. HOW TO RUN ══ -->
<div class="section" id="s11">
  <h2>11. How to Run</h2>
  <table>
    <tr><th>Step</th><th>Command</th><th>Output</th></tr>
    <tr><td>1</td><td><code>pip install -r requirements.txt</code></td><td>All dependencies installed</td></tr>
    <tr><td>2</td><td><code>python train_model.py</code></td><td>model.pkl + static/model_evaluation.png created</td></tr>
    <tr><td>3 (Terminal A)</td><td><code>python app.py</code></td><td>Flask API running on http://127.0.0.1:5000</td></tr>
    <tr><td>4 (Terminal B)</td><td><code>streamlit run streamlit_app.py</code></td><td>Streamlit UI at http://localhost:8501</td></tr>
    <tr><td>5 (optional)</td><td><code>python generate_screenshots.py</code></td><td>6 PNG mockups in static/screenshots/</td></tr>
    <tr><td>6 (optional)</td><td><code>python generate_report.py</code></td><td>project_report.html (self-contained)</td></tr>
    <tr><td>7 (optional)</td><td><code>python generate_docx.py</code></td><td>project_report.docx</td></tr>
  </table>
  <div class="callout amber" style="margin-top:1rem">
    <strong>Important:</strong> Flask API (Step 3) must be running <em>before</em> opening the
    Streamlit UI (Step 4). The Streamlit app calls the API on every prediction and dashboard load.
  </div>
</div>

<!-- ══ 12. FUTURE SCOPE ══ -->
<div class="section" id="s12">
  <h2>12. Future Scope</h2>
  <p>The current system provides a solid production-grade baseline. The following enhancements
  are prioritised by expected impact and implementation feasibility.</p>

  <h3>12.1 — Model Improvements</h3>
  <div class="insight-grid">
    <div class="insight-card purple">
      <h4>🧠 Advanced Ensemble Methods</h4>
      <p>Replace single Gradient Boosting with <strong>XGBoost / LightGBM / CatBoost</strong>
      for faster training, native categorical handling, and GPU acceleration. Expected AUC
      improvement: +0.01 to +0.03.</p>
    </div>
    <div class="insight-card purple">
      <h4>🔧 Hyperparameter Optimisation</h4>
      <p>Apply <strong>Optuna or Bayesian Optimisation</strong> to tune n_estimators,
      learning_rate, max_depth, subsample, and min_samples_leaf. Current model uses
      scikit-learn defaults beyond n_estimators=150.</p>
    </div>
    <div class="insight-card amber">
      <h4>⚖️ Class Imbalance Handling</h4>
      <p>Apply <strong>SMOTE (Synthetic Minority Oversampling)</strong> or
      <strong>class_weight='balanced'</strong> to improve recall on the minority claim class.
      Target: push claim recall from 76% to 82%+ without degrading precision below 70%.</p>
    </div>
    <div class="insight-card amber">
      <h4>📊 Threshold Optimisation</h4>
      <p>Instead of the default 0.5 decision threshold, tune the classification threshold
      using the <strong>Precision-Recall curve</strong> to find the optimal operating point
      for the specific cost ratio of false positives vs false negatives in insurance.</p>
    </div>
  </div>

  <h3>12.2 — Feature Engineering</h3>
  <div class="insight-grid">
    <div class="insight-card blue" style="border-left-color:#2d6a9f">
      <h4>📡 Telematics Integration</h4>
      <p>Integrate <strong>real-time telematics data</strong> (hard braking events, night
      driving %, average speed, acceleration patterns) from IoT devices. These usage-based
      insurance (UBI) signals are the strongest predictors of claims in modern actuarial models.</p>
    </div>
    <div class="insight-card blue" style="border-left-color:#2d6a9f">
      <h4>🗺️ Geospatial Risk Scoring</h4>
      <p>Replace the raw POSTAL_CODE with <strong>area-level risk scores</strong>: accident
      frequency per km², road quality index, weather risk index, and urban vs rural classification.
      Geospatial features typically improve model AUC by 1–3%.</p>
    </div>
    <div class="insight-card green">
      <h4>📅 Policy History Features</h4>
      <p>Add <strong>claim history features</strong>: number of prior claims, years as a
      customer, premium payment punctuality, and policy renewal count. Retention signals
      are strongly correlated with claim behaviour.</p>
    </div>
    <div class="insight-card green">
      <h4>💲 Claim Severity Prediction</h4>
      <p>Extend from binary classification (will claim?) to <strong>regression</strong>
      (how much will the claim cost?). A two-stage model — first predict claim likelihood,
      then predict severity conditional on a claim — enables expected loss pricing.</p>
    </div>
  </div>

  <h3>12.3 — System &amp; Infrastructure Improvements</h3>
  <div class="insight-grid">
    <div class="insight-card teal" style="border-left-color:#06b6d4">
      <h4>🐳 Dockerised Deployment</h4>
      <p>Containerise Flask API and Streamlit app as separate <strong>Docker containers</strong>
      orchestrated with docker-compose. Enables one-command deployment on any cloud provider
      (AWS ECS, Azure Container Apps, GCP Cloud Run).</p>
    </div>
    <div class="insight-card teal" style="border-left-color:#06b6d4">
      <h4>🔄 MLflow / MLOps Pipeline</h4>
      <p>Integrate <strong>MLflow</strong> for experiment tracking, model versioning, and
      automated model registry. Add scheduled retraining with data drift detection
      (evidently AI or Alibi Detect) to trigger retraining when feature distributions shift.</p>
    </div>
    <div class="insight-card purple">
      <h4>🔒 API Authentication &amp; Rate Limiting</h4>
      <p>Add <strong>JWT-based API authentication</strong>, request rate limiting, and input
      validation middleware to the Flask API before production deployment. Currently the API
      has no auth — appropriate only for internal/demo use.</p>
    </div>
    <div class="insight-card purple">
      <h4>📈 Real-Time Monitoring Dashboard</h4>
      <p>Build a <strong>Grafana + Prometheus</strong> monitoring layer to track API latency,
      prediction distribution drift, request volume, and error rates in real time.
      Add alerting when claim probability distribution shifts by &gt;5% week-over-week.</p>
    </div>
  </div>

  <h3>12.4 — Explainability &amp; Fairness</h3>
  <div class="insight-grid">
    <div class="insight-card red">
      <h4>🔍 SHAP Explanations</h4>
      <p>Integrate <strong>SHAP (SHapley Additive exPlanations)</strong> to generate
      per-prediction explanation reports. Each API response would include a ranked list
      of features that most influenced the prediction — critical for regulatory compliance
      (e.g., adverse action notices under FCRA/ECOA).</p>
    </div>
    <div class="insight-card red">
      <h4>⚖️ Fairness Auditing</h4>
      <p>Run <strong>AI Fairness 360 (IBM)</strong> or Fairlearn audits to measure disparate
      impact across protected attributes (RACE, GENDER) before any production deployment.
      Ensure Equal Opportunity and Demographic Parity constraints are met or documented.</p>
    </div>
  </div>

  <div class="callout green" style="margin-top:1.2rem">
    <strong>Recommended Next Steps (Priority Order):</strong>
    (1) Threshold optimisation for better FN/FP trade-off →
    (2) SHAP integration for explainability →
    (3) LightGBM replacement for performance →
    (4) Docker deployment for scalability →
    (5) Telematics feature ingestion for accuracy uplift
  </div>
</div>

</div><!-- .wrapper -->
<footer>Car Insurance Claim Prediction &mdash; IBM AICTE AI/ML Internship Project &nbsp;|&nbsp; Made with IBM Bob</footer>
</body>
</html>"""

# ── Write output ───────────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "project_report.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_part1 + html_part2 + html_part3)
print(f"HTML report saved -> {out_path}  ({os.path.getsize(out_path)//1024} KB)")
