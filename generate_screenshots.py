"""
generate_screenshots.py
========================
Renders pixel-faithful mockup screenshots of every Streamlit UI page
using Matplotlib. No browser or running server needed.

Outputs (all saved to static/screenshots/):
  01_predict_form.png       — Predict page: empty form
  02_predict_result_claim.png    — Predict page: "Claim" result
  03_predict_result_no_claim.png — Predict page: "No Claim" result
  04_dashboard_kpi.png      — Dashboard page: KPI + model info
  05_dashboard_charts.png   — Dashboard page: EDA charts
  06_sidebar.png            — Sidebar navigation panel

Run:
    python generate_screenshots.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCREENS_DIR = os.path.join(BASE_DIR, "static", "screenshots")
os.makedirs(SCREENS_DIR, exist_ok=True)

# ── Load model & data ──────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    art = pickle.load(f)
MODEL         = art["model"]
ENCODERS      = art["encoders"]
FEATURE_NAMES = art["feature_names"]
MODEL_NAME    = art["model_name"]
MODEL_ACC     = art["accuracy"]
MODEL_AUC     = art["auc"]

df_raw = pd.read_csv(os.path.join(BASE_DIR, "Car_Insurance_Claim.csv"))
total  = len(df_raw)
claims = int(df_raw["OUTCOME"].sum())

# ── Shared drawing helpers ─────────────────────────────────────────────────────
NAV_BG      = "#1e3a5f"
PAGE_BG     = "#f0f4f8"
CARD_BG     = "#ffffff"
ACCENT      = "#2d6a9f"
GREEN       = "#22c55e"
RED         = "#ef4444"
PURPLE      = "#7c5cd8"
AMBER       = "#f59e0b"
TEXT        = "#1f2328"
MUTED       = "#6b7280"
BORDER      = "#e5e7eb"

def draw_browser_chrome(fig, ax_full, title="Car Insurance Claim Predictor — Streamlit"):
    """Draw a browser tab bar at the very top of the figure."""
    ax_full.set_xlim(0, 1); ax_full.set_ylim(0, 1)
    ax_full.axis("off")
    # browser bar
    ax_full.add_patch(FancyBboxPatch((0, 0.965), 1, 0.035,
        boxstyle="square,pad=0", fc="#e5e7eb", ec="none", zorder=5))
    # address bar
    ax_full.add_patch(FancyBboxPatch((0.18, 0.967), 0.64, 0.027,
        boxstyle="round,pad=0.004", fc="white", ec="#d1d5db", lw=0.8, zorder=6))
    ax_full.text(0.50, 0.9805, "localhost:8501", ha="center", va="center",
                 fontsize=7.5, color=MUTED, zorder=7)
    # three dots (close/min/max)
    for xi, col in [(0.025, "#ef4444"), (0.045, "#f59e0b"), (0.065, GREEN)]:
        circ = plt.Circle((xi, 0.9805), 0.008, color=col, zorder=7)
        ax_full.add_patch(circ)

def nav_sidebar(ax, active="predict"):
    """Draw the left sidebar."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
        boxstyle="square,pad=0", fc=NAV_BG, ec="none"))
    ax.text(0.5, 0.93, "🚗 Insurance", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(0.5, 0.87, "Predictor", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.axhline(0.83, color="#2d6a9f", lw=0.8, xmin=0.05, xmax=0.95)
    # nav items
    items = [("🔮 Predict", "predict", 0.75), ("📊 Dashboard", "dashboard", 0.67)]
    for label, key, y in items:
        is_active = (key == active)
        if is_active:
            ax.add_patch(FancyBboxPatch((0.05, y - 0.03), 0.90, 0.055,
                boxstyle="round,pad=0.01", fc="#2d6a9f", ec="none", alpha=0.6))
        ax.text(0.12, y + 0.005, label, ha="left", va="center",
                fontsize=9.5, color="white",
                fontweight="bold" if is_active else "normal")
    ax.axhline(0.60, color="#2d6a9f", lw=0.8, xmin=0.05, xmax=0.95)
    ax.text(0.5, 0.55, "Flask API", ha="center", va="center",
            fontsize=8, color="#c9d8ea", fontweight="bold")
    # API status badge
    ax.add_patch(FancyBboxPatch((0.1, 0.47), 0.80, 0.055,
        boxstyle="round,pad=0.01", fc="#166534", ec="none"))
    ax.text(0.5, 0.498, "✓  API Connected", ha="center", va="center",
            fontsize=8, color="white")
    ax.text(0.5, 0.43, f"Model: {MODEL_NAME}", ha="center", va="center",
            fontsize=7.5, color="#c9d8ea")
    ax.text(0.5, 0.39, f"Accuracy: {round(MODEL_ACC*100,1)}%", ha="center", va="center",
            fontsize=7.5, color="#c9d8ea")

def section_header(ax, x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.008", fc=CARD_BG, ec=BORDER, lw=0.8))
    ax.text(x + 0.015, y + h/2, text, ha="left", va="center",
            fontsize=10, fontweight="bold", color=NAV_BG)

def form_field(ax, x, y, w, h, label, value, is_select=False):
    ax.text(x, y + h + 0.008, label.upper(), ha="left", va="bottom",
            fontsize=6.2, color=MUTED, fontweight="bold")
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.005", fc="#f9fafb", ec="#d1d5db", lw=0.8))
    ax.text(x + 0.01, y + h/2, value, ha="left", va="center",
            fontsize=7.5, color=TEXT)
    if is_select:
        ax.text(x + w - 0.012, y + h/2, "▾", ha="right", va="center",
                fontsize=7, color=MUTED)

def kpi_card(ax, x, y, w, h, value, label, color=ACCENT):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.01", fc="#f7f8fa", ec=BORDER, lw=0.8))
    ax.add_patch(plt.Rectangle((x, y), 0.006, h, fc=color, ec="none"))
    ax.text(x + w/2, y + h*0.62, str(value), ha="center", va="center",
            fontsize=13, fontweight="bold", color=NAV_BG)
    ax.text(x + w/2, y + h*0.22, label, ha="center", va="center",
            fontsize=6.5, color=MUTED, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 1 — Predict Page (form)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering 01_predict_form.png ...")
fig = plt.figure(figsize=(14, 9), facecolor=PAGE_BG)

# Layout: sidebar (18%) | main (82%)
ax_browser = fig.add_axes([0, 0, 1, 1], zorder=0)
ax_browser.set_facecolor(PAGE_BG); ax_browser.axis("off")
draw_browser_chrome(fig, ax_browser)

ax_side = fig.add_axes([0, 0, 0.18, 0.965])
nav_sidebar(ax_side, active="predict")

ax = fig.add_axes([0.19, 0.02, 0.80, 0.935])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_facecolor(PAGE_BG)

# Hero banner
ax.add_patch(FancyBboxPatch((0, 0.90), 1.0, 0.098,
    boxstyle="round,pad=0.01", fc=NAV_BG, ec="none"))
ax.text(0.5, 0.955, "🔮  Car Insurance Claim Predictor", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text(0.5, 0.920, "Fill in driver and vehicle details below to predict whether a claim will be filed.",
        ha="center", va="center", fontsize=8.5, color="#c9d8ea")

# Main form card
ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.875,
    boxstyle="round,pad=0.01", fc=CARD_BG, ec=BORDER, lw=0.8))

# Section: Personal Info
section_header(ax, 0.03, 0.775, 0.94, 0.028, "👤  Personal Information")

field_data_row1 = [
    ("Age Group",    "16-25",         True),
    ("Gender",       "male",          True),
    ("Race",         "majority",      True),
    ("Education",    "high school",   True),
]
xs = [0.04, 0.27, 0.50, 0.73]
for (lbl, val, sel), x in zip(field_data_row1, xs):
    form_field(ax, x, 0.715, 0.21, 0.040, lbl, val, is_select=sel)

field_data_row2 = [
    ("Income Class",  "poverty",   True),
    ("Married",       "No",        True),
    ("Has Children",  "No",        True),
    ("Credit Score",  "0.350",     False),
]
for (lbl, val, sel), x in zip(field_data_row2, xs):
    form_field(ax, x, 0.645, 0.21, 0.040, lbl, val, is_select=sel)

# Section: Driving Profile
section_header(ax, 0.03, 0.600, 0.94, 0.028, "🚦  Driving Profile")

drive_fields = [
    ("Driving Experience", "0-9y",  True),
    ("Speeding Violations","2",     False),
    ("DUI Offences",       "0",     False),
    ("Past Accidents",     "1",     False),
    ("Annual Mileage",     "16000", False),
]
xs5 = [0.04, 0.23, 0.42, 0.61, 0.80]
for (lbl, val, sel), x in zip(drive_fields, xs5):
    form_field(ax, x, 0.540, 0.17, 0.040, lbl, val, is_select=sel)

# Section: Vehicle Details
section_header(ax, 0.03, 0.496, 0.94, 0.028, "🚗  Vehicle Details")

veh_fields = [
    ("Vehicle Type",  "sedan",        True),
    ("Vehicle Year",  "before 2015",  True),
    ("Owns Vehicle",  "No",           True),
    ("Postal Code",   "10238",        False),
]
for (lbl, val, sel), x in zip(veh_fields, xs):
    form_field(ax, x, 0.436, 0.21, 0.040, lbl, val, is_select=sel)

# Predict button
ax.add_patch(FancyBboxPatch((0.03, 0.380), 0.94, 0.042,
    boxstyle="round,pad=0.008", fc=NAV_BG, ec="none"))
ax.text(0.5, 0.402, "🔍  Predict Claim", ha="center", va="center",
        fontsize=11, fontweight="bold", color="white")

plt.savefig(os.path.join(SCREENS_DIR, "01_predict_form.png"),
            dpi=130, bbox_inches="tight", facecolor=PAGE_BG)
plt.close()
print("  saved.")

# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 2 — Predict Page: CLAIM result
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering 02_predict_result_claim.png ...")
fig = plt.figure(figsize=(14, 9), facecolor=PAGE_BG)
ax_browser = fig.add_axes([0, 0, 1, 1], zorder=0)
ax_browser.set_facecolor(PAGE_BG); ax_browser.axis("off")
draw_browser_chrome(fig, ax_browser)
ax_side = fig.add_axes([0, 0, 0.18, 0.965])
nav_sidebar(ax_side, active="predict")
ax = fig.add_axes([0.19, 0.02, 0.80, 0.935])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_facecolor(PAGE_BG)

# Hero
ax.add_patch(FancyBboxPatch((0, 0.90), 1.0, 0.098,
    boxstyle="round,pad=0.01", fc=NAV_BG, ec="none"))
ax.text(0.5, 0.955, "🔮  Car Insurance Claim Predictor", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text(0.5, 0.920, "Fill in driver and vehicle details below to predict whether a claim will be filed.",
        ha="center", va="center", fontsize=8.5, color="#c9d8ea")

# Compact form (greyed out / submitted state)
ax.add_patch(FancyBboxPatch((0.01, 0.46), 0.98, 0.42,
    boxstyle="round,pad=0.01", fc=CARD_BG, ec=BORDER, lw=0.8, alpha=0.6))
ax.text(0.5, 0.78, "[ Form submitted — see prediction below ]",
        ha="center", va="center", fontsize=8, color=MUTED, style="italic")

# ── RESULT BOX: Claim ──
ax.add_patch(FancyBboxPatch((0.01, 0.255), 0.98, 0.185,
    boxstyle="round,pad=0.012", fc="#fef2f2", ec="#f87171", lw=2.0))
ax.text(0.06, 0.420, "⚠️", fontsize=22, va="center")
ax.text(0.16, 0.425, "Prediction:", fontsize=10, color=MUTED, va="center")
ax.text(0.16, 0.395, "CLAIM", fontsize=19, fontweight="bold", color="#dc2626", va="center")
ax.text(0.16, 0.372, "The model predicts this driver WILL file an insurance claim.",
        fontsize=8.5, color=TEXT, va="center")

# Three metric columns
for x, val, lbl in [(0.03, "78.4%", "Claim Probability"),
                    (0.37, "🔴 High", "Risk Level"),
                    (0.71, "Claim", "Predicted Outcome")]:
    ax.add_patch(FancyBboxPatch((x, 0.263), 0.295, 0.075,
        boxstyle="round,pad=0.008", fc="white", ec=BORDER, lw=0.8))
    ax.text(x + 0.148, 0.305, val, ha="center", va="center",
            fontsize=13, fontweight="bold", color="#dc2626" if "%" in val or "Claim" in val else TEXT)
    ax.text(x + 0.148, 0.275, lbl, ha="center", va="center",
            fontsize=7, color=MUTED, fontweight="bold")

# Progress bar
ax.add_patch(FancyBboxPatch((0.01, 0.248), 0.98, 0.012,
    boxstyle="square,pad=0", fc=BORDER, ec="none"))
ax.add_patch(FancyBboxPatch((0.01, 0.248), 0.98*0.784, 0.012,
    boxstyle="square,pad=0", fc="#ef4444", ec="none"))

plt.savefig(os.path.join(SCREENS_DIR, "02_predict_result_claim.png"),
            dpi=130, bbox_inches="tight", facecolor=PAGE_BG)
plt.close()
print("  saved.")

# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 3 — Predict Page: NO CLAIM result
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering 03_predict_result_no_claim.png ...")
fig = plt.figure(figsize=(14, 9), facecolor=PAGE_BG)
ax_browser = fig.add_axes([0, 0, 1, 1], zorder=0)
ax_browser.set_facecolor(PAGE_BG); ax_browser.axis("off")
draw_browser_chrome(fig, ax_browser)
ax_side = fig.add_axes([0, 0, 0.18, 0.965])
nav_sidebar(ax_side, active="predict")
ax = fig.add_axes([0.19, 0.02, 0.80, 0.935])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_facecolor(PAGE_BG)

ax.add_patch(FancyBboxPatch((0, 0.90), 1.0, 0.098,
    boxstyle="round,pad=0.01", fc=NAV_BG, ec="none"))
ax.text(0.5, 0.955, "🔮  Car Insurance Claim Predictor", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text(0.5, 0.920, "Fill in driver and vehicle details below to predict whether a claim will be filed.",
        ha="center", va="center", fontsize=8.5, color="#c9d8ea")

ax.add_patch(FancyBboxPatch((0.01, 0.46), 0.98, 0.42,
    boxstyle="round,pad=0.01", fc=CARD_BG, ec=BORDER, lw=0.8, alpha=0.6))
ax.text(0.5, 0.78, "[ Form submitted — see prediction below ]",
        ha="center", va="center", fontsize=8, color=MUTED, style="italic")

# ── RESULT BOX: No Claim ──
ax.add_patch(FancyBboxPatch((0.01, 0.255), 0.98, 0.185,
    boxstyle="round,pad=0.012", fc="#f0fdf4", ec="#4ade80", lw=2.0))
ax.text(0.06, 0.420, "✅", fontsize=22, va="center")
ax.text(0.16, 0.425, "Prediction:", fontsize=10, color=MUTED, va="center")
ax.text(0.16, 0.395, "NO CLAIM", fontsize=19, fontweight="bold", color="#16a34a", va="center")
ax.text(0.16, 0.372, "The model predicts this driver WILL NOT file an insurance claim.",
        fontsize=8.5, color=TEXT, va="center")

for x, val, lbl, col in [(0.03, "18.2%", "Claim Probability", "#16a34a"),
                          (0.37, "🟢 Low", "Risk Level", TEXT),
                          (0.71, "No Claim", "Predicted Outcome", "#16a34a")]:
    ax.add_patch(FancyBboxPatch((x, 0.263), 0.295, 0.075,
        boxstyle="round,pad=0.008", fc="white", ec=BORDER, lw=0.8))
    ax.text(x + 0.148, 0.305, val, ha="center", va="center",
            fontsize=13, fontweight="bold", color=col)
    ax.text(x + 0.148, 0.275, lbl, ha="center", va="center",
            fontsize=7, color=MUTED, fontweight="bold")

ax.add_patch(FancyBboxPatch((0.01, 0.248), 0.98, 0.012,
    boxstyle="square,pad=0", fc=BORDER, ec="none"))
ax.add_patch(FancyBboxPatch((0.01, 0.248), 0.98*0.182, 0.012,
    boxstyle="square,pad=0", fc=GREEN, ec="none"))

plt.savefig(os.path.join(SCREENS_DIR, "03_predict_result_no_claim.png"),
            dpi=130, bbox_inches="tight", facecolor=PAGE_BG)
plt.close()
print("  saved.")

# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 4 — Dashboard: KPI + model card
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering 04_dashboard_kpi.png ...")
fig = plt.figure(figsize=(14, 9), facecolor=PAGE_BG)
ax_browser = fig.add_axes([0, 0, 1, 1], zorder=0)
ax_browser.set_facecolor(PAGE_BG); ax_browser.axis("off")
draw_browser_chrome(fig, ax_browser)
ax_side = fig.add_axes([0, 0, 0.18, 0.965])
nav_sidebar(ax_side, active="dashboard")
ax = fig.add_axes([0.19, 0.02, 0.80, 0.935])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_facecolor(PAGE_BG)

# Hero
ax.add_patch(FancyBboxPatch((0, 0.90), 1.0, 0.098,
    boxstyle="round,pad=0.01", fc=NAV_BG, ec="none"))
ax.text(0.5, 0.955, "📊  Analytics Dashboard", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text(0.5, 0.920, "Dataset exploration and model performance overview.",
        ha="center", va="center", fontsize=8.5, color="#c9d8ea")

# KPI row
kpi_items = [
    (f"{total:,}",                      "Total Records",    ACCENT),
    (f"{claims:,}",                     "Claims Filed",     RED),
    (f"{total-claims:,}",               "No Claims",        GREEN),
    (f"{round(MODEL_ACC*100,2)}%",      "Model Accuracy",   PURPLE),
    (f"{round(MODEL_AUC,4)}",           "ROC-AUC",          AMBER),
]
kpi_w = 0.178
for i, (val, lbl, col) in enumerate(kpi_items):
    kpi_card(ax, 0.02 + i*(kpi_w + 0.015), 0.805, kpi_w, 0.075, val, lbl, col)

# Model card
ax.add_patch(FancyBboxPatch((0.02, 0.545), 0.96, 0.240,
    boxstyle="round,pad=0.01", fc=CARD_BG, ec=BORDER, lw=0.8))
ax.text(0.035, 0.768, f"🤖  Best Model:  {MODEL_NAME}", ha="left", va="center",
        fontsize=11, fontweight="bold", color=NAV_BG)
ax.axhline(0.755, color=BORDER, lw=0.8, xmin=0.03, xmax=0.97)

# Mini metric pills inside model card
pills = [
    (f"{round(MODEL_ACC*100,2)}%", "Accuracy"),
    (f"{round(MODEL_AUC,4)}",      "ROC-AUC"),
    (f"{total:,}",                 "Training Records"),
    ("18",                         "Features Used"),
]
pw = 0.19
for i, (v, l) in enumerate(pills):
    px = 0.04 + i * (pw + 0.03)
    ax.add_patch(FancyBboxPatch((px, 0.590), pw, 0.130,
        boxstyle="round,pad=0.01", fc="#f0f4f8", ec=BORDER, lw=0.6))
    ax.text(px + pw/2, 0.670, v,  ha="center", va="center",
            fontsize=15, fontweight="bold", color=NAV_BG)
    ax.text(px + pw/2, 0.612, l, ha="center", va="center",
            fontsize=7, color=MUTED, fontweight="bold")

# eval image thumbnail
eval_img_path = os.path.join(BASE_DIR, "static", "model_evaluation.png")
eval_img = plt.imread(eval_img_path)
ax_eval = fig.add_axes([0.19 + 0.80*0.02, 0.02 + 0.935*0.05, 0.80*0.96, 0.935*0.47])
ax_eval.imshow(eval_img, aspect="auto")
ax_eval.set_title("Confusion Matrix & Feature Importance (from training)",
                   fontsize=8, color=MUTED, pad=4)
ax_eval.axis("off")

plt.savefig(os.path.join(SCREENS_DIR, "04_dashboard_kpi.png"),
            dpi=130, bbox_inches="tight", facecolor=PAGE_BG)
plt.close()
print("  saved.")

# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 5 — Dashboard: EDA Charts grid
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering 05_dashboard_charts.png ...")

fig = plt.figure(figsize=(14, 10), facecolor=PAGE_BG)
fig.suptitle("📊  Analytics Dashboard  —  Dataset Distributions",
             fontsize=13, fontweight="bold", color=NAV_BG, y=0.98)

axes_pos = [
    [0.04,  0.54, 0.43, 0.40],  # top-left
    [0.53,  0.54, 0.43, 0.40],  # top-right
    [0.04,  0.06, 0.43, 0.40],  # bottom-left
    [0.53,  0.06, 0.43, 0.40],  # bottom-right
]

# Chart A — Outcome pie
ax0 = fig.add_axes(axes_pos[0])
ax0.set_facecolor(CARD_BG)
wedges, texts, auto = ax0.pie(
    [total - claims, claims],
    labels=["No Claim", "Claim"],
    colors=[GREEN, RED],
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=9))
ax0.set_title("Claims vs No Claims", fontsize=11, fontweight="bold", color=NAV_BG, pad=8)

# Chart B — Claim rate by age
ax1 = fig.add_axes(axes_pos[1])
ax1.set_facecolor(CARD_BG)
cr_age = df_raw.groupby("AGE")["OUTCOME"].mean().mul(100).round(1)
bars = ax1.bar(cr_age.index, cr_age.values, color=ACCENT, edgecolor="white", width=0.6)
ax1.bar_label(bars, labels=[f"{v}%" for v in cr_age.values], padding=3, fontsize=8.5)
ax1.set_title("Claim Rate by Age Group", fontsize=11, fontweight="bold", color=NAV_BG, pad=8)
ax1.set_ylabel("Claim Rate (%)", fontsize=8.5)
ax1.set_ylim(0, max(cr_age.values) + 12)
ax1.tick_params(axis="both", labelsize=8)
ax1.set_facecolor(CARD_BG)
ax1.spines[["top","right"]].set_visible(False)

# Chart C — Claim rate by driving experience
ax2 = fig.add_axes(axes_pos[2])
ax2.set_facecolor(CARD_BG)
cr_exp = df_raw.groupby("DRIVING_EXPERIENCE")["OUTCOME"].mean().mul(100).round(1)
ax2.barh(cr_exp.index, cr_exp.values, color=PURPLE, edgecolor="white", height=0.6)
for i, v in enumerate(cr_exp.values):
    ax2.text(v + 0.8, i, f"{v}%", va="center", fontsize=8.5)
ax2.set_title("Claim Rate by Driving Experience", fontsize=11, fontweight="bold", color=NAV_BG, pad=8)
ax2.set_xlabel("Claim Rate (%)", fontsize=8.5)
ax2.tick_params(axis="both", labelsize=8)
ax2.spines[["top","right"]].set_visible(False)

# Chart D — Income distribution
ax3 = fig.add_axes(axes_pos[3])
ax3.set_facecolor(CARD_BG)
inc = df_raw["INCOME"].value_counts()
inc_colors = [NAV_BG, ACCENT, "#06b6d4", PURPLE]
ax3.pie(inc.values, labels=inc.index,
        colors=inc_colors[:len(inc)],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=8.5))
ax3.set_title("Income Class Distribution", fontsize=11, fontweight="bold", color=NAV_BG, pad=8)

plt.savefig(os.path.join(SCREENS_DIR, "05_dashboard_charts.png"),
            dpi=130, bbox_inches="tight", facecolor=PAGE_BG)
plt.close()
print("  saved.")

# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT 6 — Dataset preview table
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering 06_dataset_preview.png ...")
fig, ax = plt.subplots(figsize=(14, 5), facecolor=CARD_BG)
ax.set_facecolor(CARD_BG)
ax.axis("off")
ax.set_title("📋  Dataset Preview  —  Car_Insurance_Claim.csv  (first 8 rows)",
             fontsize=11, fontweight="bold", color=NAV_BG, pad=10, loc="left")

preview = df_raw.head(8)
cols_show = ["ID","AGE","GENDER","DRIVING_EXPERIENCE","INCOME",
             "CREDIT_SCORE","SPEEDING_VIOLATIONS","PAST_ACCIDENTS","OUTCOME"]
preview = preview[cols_show].copy()
preview["CREDIT_SCORE"] = preview["CREDIT_SCORE"].round(3)

col_widths = [0.09, 0.07, 0.07, 0.12, 0.11, 0.10, 0.13, 0.11, 0.08]
xs = [sum(col_widths[:i]) + 0.02 for i in range(len(col_widths))]

# Header row
for j, (col, x) in enumerate(zip(cols_show, xs)):
    ax.add_patch(FancyBboxPatch((x, 0.82), col_widths[j]-0.005, 0.10,
        boxstyle="square,pad=0", fc=NAV_BG, ec="white", lw=0.5,
        transform=ax.transAxes))
    ax.text(x + (col_widths[j]-0.005)/2, 0.87, col.replace("_","\n"),
            ha="center", va="center", fontsize=6.5, fontweight="bold",
            color="white", transform=ax.transAxes)

# Data rows
row_colors = [CARD_BG, "#f7f8fa"]
for i, (_, row) in enumerate(preview.iterrows()):
    yb = 0.82 - (i+1)*0.095
    for j, (col, x) in enumerate(zip(cols_show, xs)):
        ax.add_patch(FancyBboxPatch((x, yb), col_widths[j]-0.005, 0.090,
            boxstyle="square,pad=0", fc=row_colors[i % 2], ec=BORDER, lw=0.3,
            transform=ax.transAxes))
        val = str(row[col])
        if col == "OUTCOME":
            fc = "#fee2e2" if val == "1.0" else "#dcfce7"
            ax.add_patch(FancyBboxPatch((x+0.005, yb+0.015),
                col_widths[j]-0.015, 0.058,
                boxstyle="round,pad=0.005", fc=fc, ec="none",
                transform=ax.transAxes))
        ax.text(x + (col_widths[j]-0.005)/2, yb + 0.042, val,
                ha="center", va="center", fontsize=6.8, color=TEXT,
                transform=ax.transAxes)

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(SCREENS_DIR, "06_dataset_preview.png"),
            dpi=130, bbox_inches="tight", facecolor=CARD_BG)
plt.close()
print("  saved.")

print(f"\nAll screenshots saved to: {SCREENS_DIR}")
print("Files:")
for f in sorted(os.listdir(SCREENS_DIR)):
    path = os.path.join(SCREENS_DIR, f)
    print(f"  {f}  ({os.path.getsize(path)//1024} KB)")
