"""
app.py  —  Flask REST API backend
==================================
Exposes two endpoints consumed by the Streamlit frontend.

  POST /predict   – accepts JSON, returns prediction + probability
  GET  /stats     – returns dataset & model summary statistics

Run:
    python app.py
    # server starts on http://127.0.0.1:5000
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App ────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)   # allow Streamlit (different port) to call this API

# ── Load model artefacts ───────────────────────────────────────────────────────
ARTEFACT_PATH = os.path.join(BASE_DIR, "model.pkl")

if not os.path.exists(ARTEFACT_PATH):
    raise FileNotFoundError("model.pkl not found. Run `python train_model.py` first.")

with open(ARTEFACT_PATH, "rb") as f:
    _art = pickle.load(f)

MODEL         = _art["model"]
ENCODERS      = _art["encoders"]
FEATURE_NAMES = _art["feature_names"]
MODEL_NAME    = _art["model_name"]
MODEL_ACC     = _art["accuracy"]
MODEL_AUC     = _art["auc"]

# ── Dataset stats (built once at startup) ─────────────────────────────────────
_df = pd.read_csv(os.path.join(BASE_DIR, "Car_Insurance_Claim.csv"))

def _build_stats():
    df    = _df.copy()
    total = len(df)
    claim = int(df["OUTCOME"].sum())
    return {
        "total":              total,
        "claim_count":        claim,
        "no_claim_count":     total - claim,
        "claim_pct":          round(claim / total * 100, 1),
        "age_dist":           df["AGE"].value_counts().to_dict(),
        "income_dist":        df["INCOME"].value_counts().to_dict(),
        "vehicle_type_dist":  df["VEHICLE_TYPE"].value_counts().to_dict(),
        "gender_dist":        df["GENDER"].value_counts().to_dict(),
        "education_dist":     df["EDUCATION"].value_counts().to_dict(),
        "claim_by_age":       df.groupby("AGE")["OUTCOME"].mean().mul(100).round(1).to_dict(),
        "claim_by_exp":       df.groupby("DRIVING_EXPERIENCE")["OUTCOME"].mean().mul(100).round(1).to_dict(),
        "claim_by_income":    df.groupby("INCOME")["OUTCOME"].mean().mul(100).round(1).to_dict(),
        "model_name":         MODEL_NAME,
        "accuracy":           round(MODEL_ACC * 100, 2),
        "auc":                round(MODEL_AUC, 4),
        "features":           FEATURE_NAMES,
    }

STATS = _build_stats()

# ── Helper: encode one input row ──────────────────────────────────────────────
def _encode(payload: dict) -> np.ndarray:
    row = {}
    for feat in FEATURE_NAMES:
        val = payload.get(feat, "")
        if feat in ENCODERS:
            le = ENCODERS[feat]
            row[feat] = int(le.transform([val])[0]) if val in le.classes_ else 0
        else:
            try:
                row[feat] = float(val)
            except (ValueError, TypeError):
                row[feat] = 0.0
    return np.array([row[f] for f in FEATURE_NAMES]).reshape(1, -1)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body with feature key/value pairs.
    Returns { prediction, label, probability, risk_level }.
    """
    try:
        payload     = request.get_json(force=True)
        X           = _encode(payload)
        prediction  = int(MODEL.predict(X)[0])
        probability = float(MODEL.predict_proba(X)[0][1])
        risk        = ("High" if probability >= 0.65 else
                       "Medium" if probability >= 0.40 else "Low")
        return jsonify({
            "success":     True,
            "prediction":  prediction,
            "label":       "Claim" if prediction == 1 else "No Claim",
            "probability": round(probability * 100, 2),
            "risk_level":  risk,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/stats", methods=["GET"])
def stats():
    """Returns dataset and model performance statistics."""
    return jsonify(STATS)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Model  : {MODEL_NAME}")
    print(f"Acc    : {MODEL_ACC:.4f}   AUC: {MODEL_AUC:.4f}")
    print("Flask API running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
