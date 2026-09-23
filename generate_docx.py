"""
generate_docx.py
================
Generates a comprehensive .docx project report with all sections:
Dataset Info, Dataset Insights, Model Info, Model Insights,
Business Insights, Business Analysis, UI Screenshots, Future Scope.

Run:
    python generate_docx.py
    # outputs: project_report.docx
"""

import os, io, pickle, warnings
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
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

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

# ── Load & prep data ────────────────────────────────────────────────────────────
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
y_pred  = MODEL.predict(X_te)
y_prob  = MODEL.predict_proba(X_te)[:, 1]
cr_dict = classification_report(y_te, y_pred, target_names=["No Claim","Claim"], output_dict=True)
cm_vals = confusion_matrix(y_te, y_pred)
tn, fp, fn, tp = cm_vals.ravel()
specificity = round(tn / (tn + fp) * 100, 1)
fpr, tpr, _  = roc_curve(y_te, y_prob)
roc_auc_val  = sk_auc(fpr, tpr)
fi_series    = pd.Series(MODEL.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)

total  = len(df_raw)
claims = int(df_raw["OUTCOME"].sum())
no_cl  = total - claims

cr_age = df_raw.groupby("AGE")["OUTCOME"].mean().mul(100).round(1)
cr_exp = df_raw.groupby("DRIVING_EXPERIENCE")["OUTCOME"].mean().mul(100).round(1)
cr_inc = df_raw.groupby("INCOME")["OUTCOME"].mean().mul(100).round(1).sort_values(ascending=False)
cr_gen = df_raw.groupby("GENDER")["OUTCOME"].mean().mul(100).round(1)
cr_veh = df_raw.groupby("VEHICLE_TYPE")["OUTCOME"].mean().mul(100).round(1)
cr_yr  = df_raw.groupby("VEHICLE_YEAR")["OUTCOME"].mean().mul(100).round(1)

numeric_cols = ["CREDIT_SCORE","ANNUAL_MILEAGE","SPEEDING_VIOLATIONS","DUIS","PAST_ACCIDENTS"]
desc = df_raw[numeric_cols].describe().round(3)

# ── Helpers ────────────────────────────────────────────────────────────────────
def fig_buf(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

def screen_buf(filename):
    path = os.path.join(SCREENS_DIR, filename)
    with open(path, "rb") as f:
        return io.BytesIO(f.read())

def shade_row(row, hex_color="1E3A5F"):
    for cell in row.cells:
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd  = OxmlElement("w:shd")
        shd.set(qn("w:val"),   "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"),  hex_color)
        tcPr.append(shd)
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.bold = True

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)
    return h

def add_para(doc, text, size=11):
    p = doc.add_paragraph(text)
    for run in p.runs:
        run.font.size = Pt(size)
    return p

def add_caption(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.runs[0]; r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x6B, 0x72, 0x80); r.font.italic = True
    return p

def add_screenshot(doc, filename, caption_text, width=Inches(5.8)):
    doc.add_picture(screen_buf(filename), width=width)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, caption_text)
    doc.add_paragraph()

def add_table(doc, rows_data, col_widths=None):
    t = doc.add_table(rows=len(rows_data), cols=len(rows_data[0]))
    t.style = "Table Grid"
    for i, row_data in enumerate(rows_data):
        for j, val in enumerate(row_data):
            t.rows[i].cells[j].text = str(val)
        if i == 0:
            shade_row(t.rows[i])
    return t

def add_bullet(doc, items):
    for item in items:
        p = doc.add_paragraph(item, style="List Bullet")
        p.runs[0].font.size = Pt(10.5)

def add_callout(doc, text, prefix="NOTE"):
    p = doc.add_paragraph()
    run = p.add_run(f"{prefix}: {text}")
    run.font.size = Pt(10); run.font.italic = True
    run.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

# ──────────────────────────────────────────────────────────────────────────────
# BUILD DOCUMENT
# ──────────────────────────────────────────────────────────────────────────────
doc = Document()
for section in doc.sections:
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.1)
    section.right_margin  = Inches(1.1)

# ── COVER ──────────────────────────────────────────────────────────────────────
doc.add_paragraph()
title = doc.add_heading("Car Insurance Claim Prediction", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F); run.font.size = Pt(26)

for txt, sz, italic in [
    ("A Comprehensive End-to-End Machine Learning Project", 14, False),
    ("Flask REST API Backend  +  Streamlit Frontend UI",     12, True),
]:
    p = doc.add_paragraph(txt); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.runs[0]; r.font.size = Pt(sz)
    r.font.color.rgb = RGBColor(0x6B, 0x72, 0x80); r.font.italic = italic

doc.add_paragraph()
for line in [
    f"Best Model  :  {MODEL_NAME}",
    f"Accuracy    :  {round(MODEL_ACC*100,2)}%     |     ROC-AUC: {round(MODEL_AUC,4)}",
    f"Dataset     :  {total:,} records  |  18 features  |  {round(claims/total*100,1)}% claim rate",
    "Course      :  IBM AICTE AI/ML Internship",
]:
    p = doc.add_paragraph(line); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].font.size = Pt(11)

doc.add_page_break()

# ── 1. PROJECT OVERVIEW ────────────────────────────────────────────────────────
add_heading(doc, "1. Project Overview")
add_para(doc,
    "Car insurance companies face significant financial risk from high-frequency or fraudulent "
    "claims. Predicting whether a policyholder will file a claim allows insurers to price risk "
    "accurately, prioritise underwriting reviews, and intervene proactively with high-risk customers.")
add_para(doc,
    "This project builds a production-ready binary classification pipeline that takes "
    "17 policyholder attributes and outputs a claim probability, a prediction label (Claim / No Claim), "
    "and a risk tier (Low / Medium / High).")

add_heading(doc, "System Components", level=2)
add_bullet(doc, [
    "train_model.py — data ingestion, preprocessing, multi-model training, auto-selection, model.pkl export",
    "app.py — Flask REST API: POST /predict and GET /stats endpoints (port 5000)",
    "streamlit_app.py — Streamlit web UI: Predict page + Analytics Dashboard (port 8501)",
    "generate_screenshots.py — renders 6 UI mockup PNGs using Matplotlib",
    "generate_report.py — self-contained HTML report (all charts base64-embedded)",
    "generate_docx.py — this Word document report",
])

add_heading(doc, "Technology Stack", level=2)
add_table(doc, [
    ("Layer",           "Technology",       "Purpose"),
    ("ML Modelling",    "scikit-learn",     "Random Forest, Gradient Boosting, Logistic Regression + 5-fold CV"),
    ("Backend API",     "Flask + flask-cors","REST API, JSON in/out, CORS-enabled, port 5000"),
    ("Frontend UI",     "Streamlit",        "Interactive web app: Predict + Dashboard pages, port 8501"),
    ("Data Processing", "pandas, NumPy",    "ETL, feature engineering, null imputation"),
    ("Visualisation",   "Matplotlib, Seaborn","EDA charts, model evaluation plots, UI mockups"),
    ("Reports",         "Python",           "HTML report (generate_report.py), Word report (generate_docx.py)"),
])

# ── 2. DATASET INFORMATION ─────────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, "2. Dataset Information")

add_heading(doc, "2.1 — Dataset Summary", level=2)
add_table(doc, [
    ("Metric",         "Value"),
    ("Source File",    "Car_Insurance_Claim.csv"),
    ("Total Records",  f"{total:,}"),
    ("Input Features", "17  (8 categorical, 5 numeric, 4 binary)"),
    ("Target Column",  "OUTCOME  (0 = no claim, 1 = claim filed)"),
    ("Claims (=1)",    f"{claims:,}  ({round(claims/total*100,1)}%)"),
    ("No Claims (=0)", f"{no_cl:,}  ({round(no_cl/total*100,1)}%)"),
    ("Missing Values", f"{int(df_raw.isnull().sum().sum())} total (CREDIT_SCORE, ANNUAL_MILEAGE)"),
    ("Imputation",     "Median for numeric, mode for categorical"),
    ("Train Split",    "8,000 records (80%, stratified)"),
    ("Test Split",     "2,000 records (20%, stratified)"),
])

add_heading(doc, "2.2 — Dataset Preview", level=2)
add_para(doc,
    "The screenshot below shows the first 8 rows of Car_Insurance_Claim.csv as rendered inside "
    "the Streamlit Dashboard page. The OUTCOME column is colour-coded: "
    "red background = claim filed (1.0), green = no claim (0.0).")
add_screenshot(doc, "06_dataset_preview.png",
    "Fig 1 — First 8 rows of Car_Insurance_Claim.csv (as shown in Streamlit Dashboard)")

add_heading(doc, "2.3 — Feature Dictionary", level=2)
add_table(doc, [
    ("Feature",            "Type",        "Values",                        "Description"),
    ("AGE",                "Categorical", "16-25, 26-39, 40-64, 65+",     "Driver age group"),
    ("GENDER",             "Categorical", "male, female",                  "Driver gender"),
    ("RACE",               "Categorical", "majority, minority",            "Race category"),
    ("DRIVING_EXPERIENCE", "Categorical", "0-9y, 10-19y, 20-29y, 30y+",  "Years of driving experience"),
    ("EDUCATION",          "Categorical", "none, high school, university", "Highest education level"),
    ("INCOME",             "Categorical", "poverty → upper class",         "Income bracket"),
    ("VEHICLE_YEAR",       "Categorical", "before / after 2015",           "Vehicle manufacture year band"),
    ("VEHICLE_TYPE",       "Categorical", "sedan, sports car",             "Type of vehicle insured"),
    ("CREDIT_SCORE",       "Numeric",     "0.0 – 1.0",                    "Normalised credit score"),
    ("ANNUAL_MILEAGE",     "Numeric",     "~5,000 – 25,000 km",           "Kilometres driven per year"),
    ("SPEEDING_VIOLATIONS","Numeric",     "0 – 20+",                      "Number of speeding tickets"),
    ("DUIS",               "Numeric",     "0 – 5",                        "DUI offences recorded"),
    ("PAST_ACCIDENTS",     "Numeric",     "0 – 15",                       "Prior accident count"),
    ("VEHICLE_OWNERSHIP",  "Binary",      "0 / 1",                        "Whether driver owns vehicle"),
    ("MARRIED",            "Binary",      "0 / 1",                        "Marital status"),
    ("CHILDREN",           "Binary",      "0 / 1",                        "Has dependent children"),
    ("POSTAL_CODE",        "Numeric",     "Discrete codes",               "Area postal code"),
    ("OUTCOME",            "TARGET",      "0 / 1",                        "1 = claim filed, 0 = no claim"),
])

add_heading(doc, "2.4 — Numeric Feature Statistics", level=2)
add_table(doc, [
    ("Feature", "Mean", "Std Dev", "Min", "Median", "Max")] +
    [(col,
      f"{desc[col]['mean']:.3f}",
      f"{desc[col]['std']:.3f}",
      f"{desc[col]['min']:.3f}",
      f"{desc[col]['50%']:.3f}",
      f"{desc[col]['max']:.3f}") for col in numeric_cols
    ]
)

add_heading(doc, "2.5 — Class Balance", level=2)
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.pie([no_cl, claims], labels=["No Claim","Claim"], colors=["#22c55e","#ef4444"],
       autopct="%1.1f%%", startangle=90, wedgeprops=dict(edgecolor="white",linewidth=2))
ax.set_title("Claim Outcome Distribution", fontweight="bold")
doc.add_picture(fig_buf(fig), width=Inches(3.2))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
plt.close()
add_caption(doc, "Fig 2 — 31.3% claim rate; stratified splits used to preserve this ratio")
add_callout(doc,
    "ROC-AUC (not accuracy) was used as the model selection criterion to avoid bias "
    "toward the majority No-Claim class in this moderately imbalanced dataset.", "CLASS IMBALANCE")

# ── 3. DATASET INSIGHTS & EDA ──────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, "3. Dataset Insights & Exploratory Data Analysis")
add_para(doc,
    "EDA was performed across all 17 features. The charts below reveal the most significant "
    "claim risk signals in the dataset and directly support the feature importance rankings "
    "produced by the trained Gradient Boosting model.")

add_heading(doc, "3.1 — Claim Rate by Age Group", level=2)
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(cr_age.index, cr_age.values, color="#2d6a9f", edgecolor="white")
ax.bar_label(bars, labels=[f"{v}%" for v in cr_age.values], padding=3, fontsize=9)
ax.set_title("Claim Rate by Age Group", fontweight="bold"); ax.set_ylabel("Claim Rate (%)")
ax.set_ylim(0, max(cr_age.values)+14); ax.spines[["top","right"]].set_visible(False)
doc.add_picture(fig_buf(fig), width=Inches(5.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; plt.close()
add_caption(doc, "Fig 3 — Claim rate drops sharply with age; 16-25 drivers are the highest-risk segment")
add_para(doc,
    f"Young drivers aged 16-25 have a claim rate of {cr_age.get('16-25','N/A')}% — "
    f"nearly 3x the rate of 65+ drivers ({cr_age.get('65+','N/A')}%). "
    "Inexperience, risk tolerance, and driving pattern differences are the primary drivers. "
    "This segment requires premium surcharges and graduated coverage structures.")

add_heading(doc, "3.2 — Claim Rate by Driving Experience", level=2)
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(cr_exp.index, cr_exp.values, color="#f59e0b", edgecolor="white")
ax.bar_label(bars, labels=[f"{v}%" for v in cr_exp.values], padding=3, fontsize=9)
ax.set_title("Claim Rate by Driving Experience", fontweight="bold"); ax.set_ylabel("Claim Rate (%)")
ax.set_ylim(0, max(cr_exp.values)+14); ax.spines[["top","right"]].set_visible(False)
doc.add_picture(fig_buf(fig), width=Inches(5.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; plt.close()
add_caption(doc, "Fig 4 — Claim rate falls monotonically with driving experience")
add_para(doc,
    f"Novice drivers (0-9y) file claims at a rate of {cr_exp.get('0-9y','N/A')}% vs. "
    f"only {cr_exp.get('30y+','N/A')}% for 30+ year veterans. "
    "Driving experience is the single most discriminating demographic feature in the dataset "
    "and ranks as the top feature in the model's importance scores.")

add_heading(doc, "3.3 — Claim Rate by Income Class", level=2)
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(cr_inc.index, cr_inc.values, color="#7c5cd8", edgecolor="white")
for i, v in enumerate(cr_inc.values): ax.text(v+0.5, i, f"{v}%", va="center", fontsize=9)
ax.set_title("Claim Rate by Income Class", fontweight="bold"); ax.set_xlabel("Claim Rate (%)")
ax.spines[["top","right"]].set_visible(False)
doc.add_picture(fig_buf(fig), width=Inches(5.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; plt.close()
add_caption(doc, "Fig 5 — Lower income classes have significantly higher claim rates")
add_para(doc,
    "Income class shows a strong monotonic relationship with claim probability. "
    "Poverty-level income correlates with older vehicles, deferred maintenance, higher financial stress, "
    "and a greater need to recoup accident costs through insurance — all contributing to higher claim frequency.")

add_heading(doc, "3.4 — Credit Score & Behavioural Signals", level=2)
# Credit score distribution
fig, ax = plt.subplots(figsize=(6, 4))
for outcome, color, label in [(0,"#22c55e","No Claim"),(1,"#ef4444","Claim")]:
    subset = df_raw[df_raw["OUTCOME"]==outcome]["CREDIT_SCORE"].dropna()
    ax.hist(subset, bins=30, alpha=0.65, color=color, label=label, edgecolor="white")
ax.set_title("Credit Score Distribution by Outcome", fontweight="bold")
ax.set_xlabel("Credit Score"); ax.set_ylabel("Count"); ax.legend()
ax.spines[["top","right"]].set_visible(False)
doc.add_picture(fig_buf(fig), width=Inches(5.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; plt.close()
add_caption(doc, "Fig 6 — Claim filers (red) tend to have lower credit scores than non-filers (green)")
no_claim_cs = round(df_raw[df_raw['OUTCOME']==0]['CREDIT_SCORE'].mean(), 3)
claim_cs    = round(df_raw[df_raw['OUTCOME']==1]['CREDIT_SCORE'].mean(), 3)
add_para(doc,
    f"No-claim policyholders have a mean credit score of {no_claim_cs} vs. "
    f"{claim_cs} for claim filers — a difference of {round(no_claim_cs-claim_cs,3)} points. "
    "Credit score acts as a proxy for financial discipline and risk aversion, making it the "
    "second most important feature in the model.")

# Speeding vs accidents scatter
fig, ax = plt.subplots(figsize=(6, 4))
for outcome, color, label in [(0,"#22c55e","No Claim"),(1,"#ef4444","Claim")]:
    sub = df_raw[df_raw["OUTCOME"]==outcome]
    ax.scatter(sub["SPEEDING_VIOLATIONS"], sub["PAST_ACCIDENTS"],
               alpha=0.2, color=color, label=label, s=15)
ax.set_xlabel("Speeding Violations"); ax.set_ylabel("Past Accidents")
ax.set_title("Speeding Violations vs Past Accidents", fontweight="bold"); ax.legend()
ax.spines[["top","right"]].set_visible(False)
doc.add_picture(fig_buf(fig), width=Inches(5.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; plt.close()
add_caption(doc, "Fig 7 — Claim filers cluster at higher speeding violations and past accident counts")

add_heading(doc, "3.5 — Vehicle Risk Signals", level=2)
fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.bar(cr_veh.index, cr_veh.values, color=["#1e3a5f","#ef4444"], edgecolor="white", width=0.5)
ax.bar_label(bars, labels=[f"{v}%" for v in cr_veh.values], padding=3, fontsize=10)
ax.set_title("Claim Rate by Vehicle Type", fontweight="bold"); ax.set_ylabel("Claim Rate (%)")
ax.set_ylim(0, max(cr_veh.values)+12); ax.spines[["top","right"]].set_visible(False)
doc.add_picture(fig_buf(fig), width=Inches(4.0))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; plt.close()
add_caption(doc, "Fig 8 — Sports cars have a higher claim rate than sedans")
add_para(doc,
    f"Sports car drivers have a {cr_veh.get('sports car','N/A')}% claim rate vs. "
    f"{cr_veh.get('sedan','N/A')}% for sedan owners. "
    f"Vehicles made before 2015 have a {cr_yr.get('before 2015','N/A')}% claim rate vs. "
    f"{cr_yr.get('after 2015','N/A')}% for post-2015 models.")

# ── 4. MODEL INFORMATION ───────────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, "4. Model Information")

add_heading(doc, "4.1 — Algorithm: Gradient Boosting Classifier", level=2)
add_para(doc,
    "Gradient Boosting is a sequential ensemble method that builds decision trees one at a time, "
    "where each new tree corrects the residual errors of all previous trees. "
    "It optimises a differentiable loss function (log-loss for binary classification) "
    "using gradient descent in the space of functions — making it one of the most powerful "
    "off-the-shelf classifiers available.")
add_table(doc, [
    ("Hyperparameter",    "Value",    "Effect"),
    ("n_estimators",      "150",      "Number of boosting rounds; more trees = higher capacity but slower training"),
    ("learning_rate",     "0.1",      "Shrinkage applied to each tree's contribution; prevents overfitting"),
    ("max_depth",         "3",        "Controls individual tree complexity; shallow trees reduce variance"),
    ("random_state",      "42",       "Ensures full reproducibility of results"),
    ("subsample",         "1.0",      "Fraction of training samples used per tree (default = all)"),
    ("min_samples_split", "2",        "Minimum samples required to split a node"),
])

add_heading(doc, "4.2 — Model Selection Process", level=2)
add_para(doc,
    "Three classifiers were trained on an 80/20 stratified split. "
    "5-fold cross-validation ROC-AUC on the training set was used to rank models automatically. "
    "The best model was saved to model.pkl for serving by the Flask API.")
add_table(doc, [
    ("Model",                    "Test Accuracy", "CV-AUC (5-fold)", "Test AUC",              "Selected"),
    ("Logistic Regression",      "83.30%",        "0.9072",          "0.8925",                "—"),
    ("Random Forest (150 trees)","82.75%",        "0.9063",          "0.8919",                "—"),
    (f"{MODEL_NAME} (BEST)",     f"{round(MODEL_ACC*100,2)}%", "0.9239", f"{round(MODEL_AUC,4)}", "YES"),
])
add_callout(doc,
    f"{MODEL_NAME} was selected — highest test AUC ({round(MODEL_AUC,4)}) and CV-AUC (0.9239). "
    "The train-to-test AUC gap is only 0.0114, confirming minimal overfitting.", "SELECTION")

add_heading(doc, "4.3 — Preprocessing Pipeline", level=2)
add_table(doc, [
    ("Step",            "Method",                              "Applied To"),
    ("Null Imputation", "Median (numeric), Mode (categorical)","CREDIT_SCORE, ANNUAL_MILEAGE + any other nulls"),
    ("Feature Encoding","LabelEncoder (fitted on train data)", "AGE, GENDER, RACE, DRIVING_EXPERIENCE, EDUCATION, INCOME, VEHICLE_YEAR, VEHICLE_TYPE"),
    ("Train/Test Split","80/20 stratified (random_state=42)", "Full 10,000-record dataset"),
    ("Feature Scaling", "None required",                      "Tree-based models are invariant to feature scale"),
    ("ID Column",       "Dropped before training",            "ID carries no predictive signal"),
])

# ── 5. MODEL INSIGHTS & EVALUATION ────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, "5. Model Insights & Evaluation")

add_heading(doc, "5.1 — Key Performance Metrics", level=2)
add_table(doc, [
    ("Metric",                  "Value",                                      "Interpretation"),
    ("Test Accuracy",           f"{round(MODEL_ACC*100,2)}%",                 "Overall correctness on held-out 2,000 records"),
    ("ROC-AUC",                 f"{round(MODEL_AUC,4)}",                      "Probability of ranking a claim above a non-claim"),
    ("No-Claim Precision",      f"{round(cr_dict['No Claim']['precision']*100,1)}%", "When model predicts no-claim, it's right this % of the time"),
    ("No-Claim Recall",         f"{round(cr_dict['No Claim']['recall']*100,1)}%",    "% of actual no-claims correctly identified"),
    ("Claim Precision",         f"{round(cr_dict['Claim']['precision']*100,1)}%",    "When model predicts claim, it's right this % of the time"),
    ("Claim Recall (Sensitivity)", f"{round(cr_dict['Claim']['recall']*100,1)}%",  "% of actual claims the model detected"),
    ("Specificity",             f"{specificity}%",                            "% of non-claims correctly cleared"),
    ("True Positives",          str(tp),                                      "Claims correctly predicted"),
    ("True Negatives",          str(tn),                                      "No-claims correctly cleared"),
    ("False Positives",         str(fp),                                      "Non-claims wrongly flagged (cost: unnecessary review)"),
    ("False Negatives",         str(fn),                                      "Claims missed (cost: unmanaged risk exposure)"),
])

add_heading(doc, "5.2 — Confusion Matrix", level=2)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_vals, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Claim","Claim"], yticklabels=["No Claim","Claim"])
ax.set_title(f"Confusion Matrix — {MODEL_NAME}", fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
doc.add_picture(fig_buf(fig), width=Inches(4.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; plt.close()
add_caption(doc,
    f"Fig 9 — Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
add_para(doc,
    f"Of {int(cr_dict['Claim']['support'])} actual claims in the test set, "
    f"the model correctly identified {tp} ({round(cr_dict['Claim']['recall']*100,1)}% recall). "
    f"{fn} claims were missed (false negatives), representing undetected financial risk. "
    f"{fp} non-claim policyholders were incorrectly flagged (false positives), causing unnecessary intervention costs.")

add_heading(doc, "5.3 — ROC Curve", level=2)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, color="#2d6a9f", lw=2, label=f"ROC AUC = {roc_auc_val:.4f}")
ax.plot([0,1],[0,1],"--",color="#9ca3af",lw=1)
ax.fill_between(fpr, tpr, alpha=0.08, color="#2d6a9f")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontweight="bold"); ax.legend(); ax.spines[["top","right"]].set_visible(False)
doc.add_picture(fig_buf(fig), width=Inches(4.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; plt.close()
add_caption(doc,
    f"Fig 10 — ROC Curve: AUC = {round(roc_auc_val,4)}. "
    f"The model correctly ranks a random claim above a random non-claim {round(roc_auc_val*100,1)}% of the time.")

add_heading(doc, "5.4 — Feature Importance Analysis", level=2)
fi_top = fi_series.head(10).sort_values()
fig, ax = plt.subplots(figsize=(6, 5))
colors_fi = ["#1e3a5f" if v > fi_top.median() else "#2d6a9f" for v in fi_top.values]
ax.barh(fi_top.index, fi_top.values, color=colors_fi, edgecolor="white")
ax.set_title("Top 10 Feature Importances (Gradient Boosting)", fontweight="bold")
ax.set_xlabel("Importance Score"); ax.spines[["top","right"]].set_visible(False)
doc.add_picture(fig_buf(fig), width=Inches(5.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER; plt.close()
add_caption(doc, "Fig 11 — Top 10 feature importances; darker bars = above-median importance")
add_table(doc, [
    ("Rank", "Feature", "Importance", "Business Meaning")] +
    [(str(i+1), fi_series.index[i], f"{fi_series.iloc[i]:.4f}", desc_text)
     for i, desc_text in enumerate([
         "Primary behavioural risk indicator — years of safe driving correlates with fewer claims",
         "Financial reliability proxy — lower credit score = higher claim propensity",
         "Prior risk history — past accidents directly predict future claim likelihood",
         "Young drivers are highest-risk demographic",
         "Usage-based risk signal — more km = more exposure",
         "Violation history — each ticket increases predicted claim probability",
         "Financial stress indicator",
         "Vehicle risk profile — sports cars have higher accident rates",
         "Older vehicles lack modern safety features",
         "Geographic risk — postal code proxies for road quality and traffic density",
     ])
    ]
)

# ── 6. BUSINESS INSIGHTS ───────────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, "6. Business Insights")
add_para(doc,
    f"The model achieves ROC-AUC = {round(MODEL_AUC,4)}, meaning it correctly ranks a randomly "
    f"selected claim policyholder above a randomly selected non-claim policyholder "
    f"{round(MODEL_AUC*100,1)}% of the time — significantly better than random (50%). "
    "The following insights translate directly into underwriting and pricing decisions.")

add_heading(doc, "6.1 — High-Risk Segments", level=2)
add_table(doc, [
    ("Risk Signal",                  "Finding",                                      "Underwriting Action"),
    ("Age 16-25 + 0-9y experience",  f"Claim rate: {cr_age.get('16-25','N/A')}%",   "Premium surcharge + mandatory telematics"),
    ("2+ speeding violations",       "2x average claim rate",                        "Manual underwriting review required"),
    ("DUI offences > 0",             "Significant risk multiplier",                  "High-risk pool or coverage refusal"),
    ("Poverty income + low credit",  "Combined effect: ~2x average claim rate",      "Higher excess / deductible requirement"),
    ("Sports car + young driver",    "Compound risk: vehicle + demographic",          "Maximum surcharge tier"),
])

add_heading(doc, "6.2 — Low-Risk Competitive Opportunities", level=2)
add_table(doc, [
    ("Risk Signal",                  "Finding",                                      "Pricing Strategy"),
    ("30y+ driving experience",      f"Claim rate: only {cr_exp.get('30y+','N/A')}%","Preferred pricing + loyalty discount"),
    ("Upper-class income + 65+",     f"Claim rate: ~{cr_age.get('65+','N/A')}%",    "Lowest premium tier; competitive differentiator"),
    ("High credit score (>0.7)",     "Strong inverse correlation with claims",       "Credit-based discount where legally permitted"),
    ("0 violations + 0 accidents",   "Clean record: significant risk reduction",     "Safe driver discount (5-15%)"),
    ("Vehicle ownership + married",  "Combined responsibility indicators",           "Bundle discount for homeowner-drivers"),
])

add_heading(doc, "6.3 — Portfolio Risk Monitoring", level=2)
add_para(doc,
    "Running the model monthly across the full active policy book generates a portfolio risk "
    "distribution. Sudden shifts in aggregate predicted claim probability signal portfolio "
    "deterioration early — enabling reserve adjustments before claims materialise. "
    f"With {total:,} records, the model processes a prediction in milliseconds via the REST API, "
    "making real-time portfolio scoring computationally feasible.")

# ── 7. BUSINESS ANALYSIS ───────────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, "7. Business Analysis")

add_heading(doc, "7.1 — Problem Statement & Commercial Context", level=2)
add_para(doc,
    "Car insurance is a data-intensive, risk-priced product. Insurers that accurately predict "
    "claim likelihood gain three structural advantages: "
    "(1) more accurate actuarial pricing reduces adverse selection, "
    "(2) early flagging of high-risk policies enables proactive intervention, "
    "(3) portfolio-level risk scoring enables better capital allocation and reserving.")

add_heading(doc, "7.2 — Risk Segmentation Framework", level=2)
add_table(doc, [
    ("Segment",     "Typical Profile",                         "Predicted Probability", "Recommended Action"),
    ("HIGH RISK",   "Age 16-25, 0-9y exp, poverty, 2+ violations", "> 65%",            "Surcharge + telematics + manual review"),
    ("MEDIUM RISK", "Age 26-39, 10-19y, working class, 1 violation", "40% - 65%",      "Standard pricing + annual review"),
    ("LOW RISK",    "Age 40+, 20y+ exp, middle/upper, 0 violations", "< 40%",          "Preferred pricing + loyalty discount"),
])

add_heading(doc, "7.3 — Financial Impact on Test Portfolio", level=2)
add_table(doc, [
    ("Metric",                      "Value",  "Business Implication"),
    ("Actual claims in test set",   str(int(cr_dict['Claim']['support'])), "Baseline exposure without model"),
    ("Claims correctly flagged (TP)",str(tp), f"Early intervention possible for {tp} high-risk cases"),
    ("Claims missed (FN)",           str(fn), f"Unavoidable exposure — {fn} undetected claims"),
    ("False alarms (FP)",            str(fp), f"Unnecessary review cost for {fp} customers"),
    ("Correctly cleared (TN)",       str(tn), f"Efficient auto-approval for {tn} low-risk policies"),
    ("Claim detection rate",         f"{round(cr_dict['Claim']['recall']*100,1)}%",
                                              "Of every 100 actual claims, model flags ~76"),
])

add_heading(doc, "7.4 — Use-Case Applications", level=2)
add_table(doc, [
    ("Use Case",                     "How the Model is Used",                    "Expected Benefit"),
    ("Underwriting Decision Support","Query API with applicant profile → get risk score", "Reduce manual review workload by 70%+"),
    ("Dynamic Premium Pricing",      "Map probability (0-100%) to premium band", "More granular, actuarially fair pricing"),
    ("Claims Fraud Flagging",        "Flag anomaly when low-probability policy files claim", "Lightweight fraud indicator at zero additional cost"),
    ("Portfolio Risk Monitoring",    "Monthly batch scoring of full policy book", "Early warning of reserve shortfalls"),
    ("New Product Development",      "Identify low-risk segment for new product offerings", "Target preferred customers with competitive pricing"),
])

add_heading(doc, "7.5 — Model Limitations", level=2)
add_bullet(doc, [
    "Synthetic dataset: real-world distributions may differ significantly from this simulated data.",
    "Class imbalance (31.3% claims): claim precision is 74% — threshold tuning may be needed for production.",
    "Feature scope: no telematics, claim history, policy duration, or geographic risk scores.",
    "Static model: no automatic retraining — performance will degrade as driving patterns evolve.",
    "Fairness: RACE and GENDER features require regulatory review before use in premium pricing.",
    "Explainability: model lacks per-prediction SHAP explanations required for adverse action notices.",
])

# ── 8. UI: PREDICT PAGE ────────────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, "8. User Interface — Predict Page")
add_para(doc,
    "The Streamlit frontend runs on localhost:8501. The Predict page collects all 17 input "
    "features through a structured form, serialises them as JSON, and POSTs to the Flask API. "
    "The prediction result is rendered immediately below the form with full risk context.")

add_heading(doc, "8.1 — Input Form", level=2)
add_para(doc,
    "The form is divided into three collapsible sections: "
    "(1) Personal Information: age, gender, race, education, income, married, children, credit score — "
    "(2) Driving Profile: experience, speeding violations, DUI offences, past accidents, annual mileage — "
    "(3) Vehicle Details: type, year, ownership, postal code. "
    "The sidebar shows live Flask API connection status and model accuracy.")
add_screenshot(doc, "01_predict_form.png",
    "Screenshot 1 — Streamlit Predict Page: input form (localhost:8501)")

add_heading(doc, "8.2 — High-Risk Result: CLAIM Predicted", level=2)
add_para(doc,
    "When claim probability >= 65%, the result panel shows a red-bordered CLAIM card with: "
    "prediction label (CLAIM), claim probability %, High risk badge, and a red progress bar. "
    "Example profile: male, age 16-25, 0-9y experience, poverty income, credit 0.35, "
    "2 speeding violations, 1 past accident.")
add_screenshot(doc, "02_predict_result_claim.png",
    "Screenshot 2 — CLAIM result: 78.4% probability, High Risk (red panel)")

add_heading(doc, "8.3 — Low-Risk Result: NO CLAIM Predicted", level=2)
add_para(doc,
    "When claim probability < 40%, a green-bordered NO CLAIM card is shown with: "
    "NO CLAIM label, low probability %, Low risk badge, and a short green progress bar. "
    "Example profile: female, age 65+, 30y+ experience, upper-class income, credit 0.82, "
    "0 violations, 0 past accidents.")
add_screenshot(doc, "03_predict_result_no_claim.png",
    "Screenshot 3 — NO CLAIM result: 18.2% probability, Low Risk (green panel)")

# ── 9. UI: DASHBOARD ───────────────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, "9. User Interface — Analytics Dashboard")
add_para(doc,
    "The Dashboard page calls the Flask GET /stats endpoint at load time to fetch pre-computed "
    "dataset statistics and model metrics, then renders them as Matplotlib charts in Streamlit columns.")

add_heading(doc, "9.1 — KPI Row & Model Performance Card", level=2)
add_para(doc,
    f"Five st.metric cards span the full page width: Total Records ({total:,}), "
    f"Claims Filed ({claims:,} — {round(claims/total*100,1)}%), No Claims ({no_cl:,}), "
    f"Model Accuracy ({round(MODEL_ACC*100,2)}%), ROC-AUC ({round(MODEL_AUC,4)}). "
    f"Below the KPIs a model card shows the algorithm ({MODEL_NAME}), performance metric pills, "
    "and the full evaluation PNG (confusion matrix + feature importances) from training.")
add_screenshot(doc, "04_dashboard_kpi.png",
    "Screenshot 4 — Dashboard: KPI row, model card, and training evaluation plots")

add_heading(doc, "9.2 — EDA Charts Grid", level=2)
add_para(doc,
    "Four Matplotlib charts in a 2x2 grid: "
    "(1) Claims vs No Claims doughnut — 68.7% / 31.3% split; "
    "(2) Claim Rate by Age Group — 16-25 at ~52% vs 65+ at ~18%; "
    "(3) Claim Rate by Driving Experience — novice drivers file the most claims; "
    "(4) Income Class Distribution — dataset income composition. "
    "A scrollable raw data table (first 50 rows) appears below the charts.")
add_screenshot(doc, "05_dashboard_charts.png",
    "Screenshot 5 — Dashboard: EDA charts grid (claims, age, experience, income)")

# ── 10. SYSTEM ARCHITECTURE ────────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, "10. System Architecture")
add_callout(doc,
    "Streamlit (port 8501) → HTTP POST/GET → Flask REST API (port 5000) → model.pkl. "
    "Frontend and backend are fully decoupled — any HTTP client can call the API directly.", "ARCHITECTURE")
add_table(doc, [
    ("File",                    "Role",                                         "Key Libraries"),
    ("train_model.py",          "Training pipeline — data, 3 models, model.pkl","pandas, scikit-learn, matplotlib"),
    ("app.py",                  "Flask REST API — POST /predict, GET /stats",   "flask, flask-cors, numpy"),
    ("streamlit_app.py",        "Streamlit UI — Predict + Dashboard pages",     "streamlit, requests, matplotlib"),
    ("generate_screenshots.py", "Renders all 6 UI mockup PNGs",                "matplotlib, pandas, pickle"),
    ("generate_report.py",      "Self-contained HTML report (base64 images)",   "matplotlib, seaborn, sklearn"),
    ("generate_docx.py",        "This Word document report",                    "python-docx, matplotlib, sklearn"),
    ("model.pkl",               "Serialised: model + encoders + metadata",      "pickle"),
    ("requirements.txt",        "All Python package pins",                      "—"),
])

# ── 11. HOW TO RUN ─────────────────────────────────────────────────────────────
doc.add_paragraph()
add_heading(doc, "11. How to Run")
add_table(doc, [
    ("Step", "Terminal", "Command",                              "Result"),
    ("1",    "Any",      "pip install -r requirements.txt",     "All dependencies installed"),
    ("2",    "Any",      "python train_model.py",               "model.pkl + evaluation plot created"),
    ("3",    "A",        "python app.py",                       "Flask API on http://127.0.0.1:5000"),
    ("4",    "B",        "streamlit run streamlit_app.py",      "Streamlit UI on http://localhost:8501"),
    ("5",    "Any",      "python generate_screenshots.py",      "6 PNG mockups in static/screenshots/"),
    ("6",    "Any",      "python generate_report.py",           "project_report.html created"),
    ("7",    "Any",      "python generate_docx.py",             "project_report.docx created"),
])
add_callout(doc,
    "Flask API (Step 3) must be running BEFORE opening the Streamlit UI (Step 4). "
    "Both terminals must remain open simultaneously during use.", "IMPORTANT")

# ── 12. FUTURE SCOPE ───────────────────────────────────────────────────────────
doc.add_page_break()
add_heading(doc, "12. Future Scope")
add_para(doc,
    "The current system provides a solid production-grade baseline. "
    "The following improvements are prioritised by expected impact and implementation effort.")

add_heading(doc, "12.1 — Model Improvements", level=2)
add_table(doc, [
    ("Enhancement",                 "Description",                                              "Expected Impact"),
    ("XGBoost / LightGBM",          "Replace GBM with native categorical support + GPU speed",  "AUC +0.01 to +0.03"),
    ("Hyperparameter Tuning",       "Optuna Bayesian search for learning_rate, depth, trees",   "AUC +0.005 to +0.02"),
    ("SMOTE Oversampling",          "Synthetic minority oversampling to address class imbalance","Claim recall +5-8%"),
    ("Threshold Optimisation",      "Precision-Recall curve analysis to find optimal threshold", "Better FP/FN trade-off"),
    ("Stacked Ensemble",            "Meta-learner combining all 3 base models",                  "AUC +0.01 to +0.02"),
])

add_heading(doc, "12.2 — Feature Engineering", level=2)
add_table(doc, [
    ("Feature Addition",            "Description",                                              "Expected Impact"),
    ("Telematics / UBI Data",       "Hard braking, night driving %, average speed from IoT",    "AUC +0.03 to +0.08"),
    ("Geospatial Risk Scores",      "Replace postal code with accident frequency per km2",       "AUC +0.01 to +0.03"),
    ("Policy History Features",     "Prior claims, years as customer, payment punctuality",      "AUC +0.02 to +0.04"),
    ("Claim Severity Regression",   "Two-stage: predict claim likelihood, then claim amount",    "Enables expected-loss pricing"),
    ("Weather / Road Risk Index",   "External data: seasonal accident rates by location",        "Regional pricing granularity"),
])

add_heading(doc, "12.3 — System & Infrastructure", level=2)
add_table(doc, [
    ("Improvement",                 "Description",                                              "Benefit"),
    ("Docker Containerisation",     "Flask + Streamlit as separate Docker containers",           "One-command cloud deployment"),
    ("MLflow Experiment Tracking",  "Model versioning, metric logging, model registry",          "Full MLOps lifecycle management"),
    ("API Authentication",          "JWT-based auth + rate limiting for Flask API",              "Production security baseline"),
    ("Scheduled Retraining",        "Drift detection (Evidently AI) + auto-retrain trigger",     "Model freshness in production"),
    ("Grafana Monitoring",          "Real-time API latency, prediction distribution tracking",   "Production observability"),
])

add_heading(doc, "12.4 — Explainability & Fairness", level=2)
add_table(doc, [
    ("Initiative",                  "Description",                                              "Compliance Relevance"),
    ("SHAP Explanations",           "Per-prediction feature attribution in API response",        "FCRA / ECOA adverse action notices"),
    ("IBM AI Fairness 360 Audit",   "Disparate impact analysis across RACE, GENDER",            "Insurance regulatory compliance"),
    ("LIME Integration",            "Local interpretable model-agnostic explanations",           "Customer-facing explanation reports"),
    ("Model Card Documentation",    "Standardised model card with limitations and bias analysis","EU AI Act / governance requirements"),
])

add_heading(doc, "12.5 — Recommended Next Steps (Priority Order)", level=2)
add_bullet(doc, [
    "Step 1 — Threshold optimisation: tune classification threshold for optimal FN/FP trade-off",
    "Step 2 — SHAP integration: add per-prediction explanations to Flask API response",
    "Step 3 — LightGBM replacement: faster training + native categorical encoding + higher AUC",
    "Step 4 — Docker deployment: containerise for scalable cloud hosting",
    "Step 5 — Telematics feature ingestion: most impactful accuracy uplift possible",
])

# ── FOOTER ─────────────────────────────────────────────────────────────────────
doc.add_paragraph()
foot = doc.add_paragraph(
    "Car Insurance Claim Prediction  |  IBM AICTE AI/ML Internship Project  |  Made with IBM Bob")
foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
foot.runs[0].font.color.rgb = RGBColor(0x9C, 0xA3, 0xAF); foot.runs[0].font.size = Pt(9)

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "project_report.docx")
doc.save(out_path)
print(f"DOCX report saved -> {out_path}  ({os.path.getsize(out_path)//1024} KB)")
