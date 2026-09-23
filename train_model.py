"""
train_model.py
--------------
Trains a Random Forest classifier on the Car Insurance Claim dataset,
evaluates it, and persists the model + label encoders to disk so the
Flask app can load them without re-training.

Run once before starting the web application:
    python train_model.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Car_Insurance_Claim.csv")

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['OUTCOME'].value_counts()}\n")

# ── 2. Pre-process ────────────────────────────────────────────────────────────
# Drop the ID column — it carries no predictive signal
df.drop(columns=["ID"], inplace=True)

# Fill numeric nulls with median; categorical nulls with mode
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical columns and save encoders for inference
CATEGORICAL_COLS = [
    "AGE", "GENDER", "RACE", "DRIVING_EXPERIENCE",
    "EDUCATION", "INCOME", "VEHICLE_YEAR", "VEHICLE_TYPE",
]

encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ── 3. Feature / target split ─────────────────────────────────────────────────
X = df.drop(columns=["OUTCOME"])
y = df["OUTCOME"].astype(int)

FEATURE_NAMES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train multiple models and pick the best ────────────────────────────────
candidates = {
    "Random Forest":        RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
for name, clf in candidates.items():
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    results[name] = {"model": clf, "accuracy": acc, "auc": auc, "cv_mean": cv_scores.mean()}
    print(f"{name:25s}  Accuracy={acc:.4f}  AUC={auc:.4f}  CV-AUC={cv_scores.mean():.4f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name}  (AUC={results[best_name]['auc']:.4f})\n")

# ── 5. Detailed evaluation ────────────────────────────────────────────────────
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Claim", "Claim"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Claim", "Claim"],
            yticklabels=["No Claim", "Claim"])
axes[0].set_title(f"Confusion Matrix — {best_name}")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Feature importance (available for tree-based models)
if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
    fi.plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Feature Importances")
    axes[1].set_xlabel("Importance")

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "static", "model_evaluation.png")
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"Evaluation plot saved -> {plot_path}")

# ── 6. Persist artefacts ──────────────────────────────────────────────────────
artefacts = {
    "model":         best_model,
    "encoders":      encoders,
    "feature_names": FEATURE_NAMES,
    "model_name":    best_name,
    "accuracy":      results[best_name]["accuracy"],
    "auc":           results[best_name]["auc"],
}

artefact_path = os.path.join(BASE_DIR, "model.pkl")
with open(artefact_path, "wb") as f:
    pickle.dump(artefacts, f)

print(f"Model artefacts saved -> {artefact_path}")
print("\nTraining complete. You can now run:  python app.py")
