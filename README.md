# Car Insurance Claim Prediction

**IBM AICTE AI/ML Internship Project**

A production-grade binary classification system that predicts whether a car insurance policyholder will file a claim, built on a Flask REST API backend and a Streamlit analytics frontend.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Information](#2-dataset-information)
3. [Dataset Insights](#3-dataset-insights)
4. [Model Information](#4-model-information)
5. [Model Insights](#5-model-insights)
6. [Business Insights](#6-business-insights)
7. [Business Analysis](#7-business-analysis)
8. [Project Structure](#8-project-structure)
9. [Tech Stack](#9-tech-stack)
10. [How to Run](#10-how-to-run)
11. [API Reference](#11-api-reference)
12. [Future Scope](#12-future-scope)

---

## 1. Project Overview

Car insurance companies operate on thin margins where adverse claim selection directly erodes profitability. This project addresses the core actuarial problem: **predicting claim likelihood before policy issuance**, enabling data-driven underwriting, risk-tiered pricing, and proactive portfolio management.

The system ingests 17 policyholder attributes — spanning demographics, driving behaviour, vehicle characteristics, and financial indicators — and returns a **claim probability score (0–100%)**, a binary prediction, and a **risk tier (Low / Medium / High)** via a REST API consumed by a Streamlit web interface.

**Architecture:**
```
Streamlit UI (port 8501)
       ↕  HTTP POST /predict  |  GET /stats
Flask REST API (port 5000)
       ↕  loads
model.pkl  ←  train_model.py  ←  Car_Insurance_Claim.csv
```

---

## 2. Dataset Information

| Attribute           | Value                                              |
|---------------------|----------------------------------------------------|
| Source file         | `Car_Insurance_Claim.csv`                          |
| Total records       | 10,000                                             |
| Input features      | 17 (8 categorical, 5 numeric, 4 binary)            |
| Target variable     | `OUTCOME` — binary (0 = No Claim, 1 = Claim Filed) |
| Class distribution  | 68.7% No Claim · 31.3% Claim                       |
| Missing values      | Present in `CREDIT_SCORE`, `ANNUAL_MILEAGE` (imputed at runtime) |
| Train / test split  | 80% / 20% stratified by target class               |

**Feature Summary:**

| Feature              | Type        | Description                                   |
|----------------------|-------------|-----------------------------------------------|
| `AGE`                | Categorical | Driver age group: 16-25, 26-39, 40-64, 65+   |
| `GENDER`             | Categorical | male / female                                 |
| `RACE`               | Categorical | majority / minority                           |
| `DRIVING_EXPERIENCE` | Categorical | 0-9y, 10-19y, 20-29y, 30y+                   |
| `EDUCATION`          | Categorical | none / high school / university               |
| `INCOME`             | Categorical | poverty / working class / middle class / upper class |
| `VEHICLE_TYPE`       | Categorical | sedan / sports car                            |
| `VEHICLE_YEAR`       | Categorical | before 2015 / after 2015                      |
| `CREDIT_SCORE`       | Numeric     | Normalised score 0.0–1.0                      |
| `ANNUAL_MILEAGE`     | Numeric     | Kilometres driven per year                    |
| `SPEEDING_VIOLATIONS`| Numeric     | Count of speeding tickets                     |
| `DUIS`               | Numeric     | Count of DUI offences                         |
| `PAST_ACCIDENTS`     | Numeric     | Count of prior accidents                      |
| `VEHICLE_OWNERSHIP`  | Binary      | 1 = driver owns vehicle                       |
| `MARRIED`            | Binary      | 1 = married                                   |
| `CHILDREN`           | Binary      | 1 = has dependents                            |
| `POSTAL_CODE`        | Numeric     | Area-level geographic proxy                   |

> **Class Imbalance Note:** The 31.3% positive class rate constitutes moderate imbalance. ROC-AUC was used as the primary model selection criterion over accuracy to prevent bias toward the majority class. Stratified splitting preserved this ratio across both training and evaluation sets.

---

## 3. Dataset Insights

The following patterns were identified through exploratory data analysis and directly corroborate the model's learned feature importance rankings.

### 3.1 Demographic Risk Signals

| Risk Factor           | Claim Rate     | Insight                                                                 |
|-----------------------|----------------|-------------------------------------------------------------------------|
| Age 16–25             | ~52%           | Highest-risk demographic; 3× the rate of 65+ drivers (~18%)            |
| Age 65+               | ~18%           | Lowest-risk demographic; extended experience and conservative driving    |
| Male drivers          | Marginally higher | Consistent with industry actuarial data on gender risk propensity    |
| Driving experience 0–9y | ~50%+        | Lack of hazard anticipation and emergency response skills               |
| Driving experience 30y+ | ~14%         | Risk falls monotonically with experience accumulation                   |

**Key finding:** Age and driving experience are strongly co-correlated. Their combined effect on predicted claim probability is non-linear — a 16-25 year old with 0-9 years experience represents the maximum-risk demographic segment.

### 3.2 Behavioural & Financial Risk Signals

| Risk Factor                  | Direction | Insight                                                              |
|------------------------------|-----------|----------------------------------------------------------------------|
| Speeding violations (2+)     | Positive  | Each additional violation significantly elevates claim probability   |
| DUI offences (>0)            | Positive  | Strong compound risk multiplier; indicates chronic risk-taking       |
| Past accidents (>0)          | Positive  | Prior claims are one of the strongest leading indicators of future claims |
| Credit score (high)          | Negative  | Proxy for financial discipline; mean score 0.59 (no claim) vs 0.54 (claim) |
| Income class (poverty)       | Positive  | Correlates with older vehicles, deferred maintenance, higher financial stress |

**Key finding:** Behavioural features (violations, DUIs, accidents) are the most discriminating predictors. A policyholder with 2+ speeding violations has a claim rate more than double the dataset average. Credit score, while not a direct measure of driving ability, serves as a reliable financial risk proxy.

### 3.3 Vehicle Risk Signals

| Risk Factor         | Claim Rate  | Insight                                                    |
|---------------------|-------------|------------------------------------------------------------|
| Sports car          | Higher      | High-performance vehicles correlate with increased accident frequency |
| Sedan               | Lower       | More conservative driving profiles associated with sedans  |
| Vehicle pre-2015    | Higher      | Older vehicles lack modern ADAS safety features (AEB, lane assist) |
| Vehicle post-2015   | Lower       | Modern safety technology measurably reduces claim frequency |

---

## 4. Model Information

Three candidate models were trained, evaluated via 5-fold cross-validation, and benchmarked on a held-out 20% test set. The best-performing model was auto-selected and serialised to `model.pkl`.

### 4.1 Model Comparison

| Model                | Test Accuracy | CV-AUC (5-fold) | Test AUC | Selected |
|----------------------|---------------|-----------------|----------|----------|
| Logistic Regression  | 83.30%        | 0.9072          | 0.8925   | —        |
| Random Forest        | 82.75%        | 0.9063          | 0.8919   | —        |
| **Gradient Boosting**| **84.25%**    | **0.9239**      | **0.9125**| ✅ Best  |

**Selection rationale:** Gradient Boosting was selected on highest test AUC (0.9125) and CV-AUC (0.9239). The train-to-test AUC delta of 0.0114 confirms minimal overfitting and robust generalisation. The sequential boosting mechanism's ability to model non-linear feature interactions was particularly suited to this dataset's compound risk structure.

### 4.2 Algorithm: Gradient Boosting Classifier

Gradient Boosting constructs an additive ensemble of weak learners (shallow decision trees) sequentially. Each tree fits the pseudo-residuals of the ensemble accumulated to that point, optimising log-loss via gradient descent in function space.

**Key hyperparameters used:**

| Parameter       | Value | Rationale                                          |
|-----------------|-------|----------------------------------------------------|
| `n_estimators`  | 150   | Sufficient rounds to converge without over-training |
| `learning_rate` | 0.1   | Standard shrinkage; balances bias-variance trade-off |
| `max_depth`     | 3     | Shallow trees reduce variance; prevent feature memorisation |
| `random_state`  | 42    | Full reproducibility of training runs              |

### 4.3 Preprocessing Pipeline

| Step             | Method                                  | Target Columns                                           |
|------------------|-----------------------------------------|----------------------------------------------------------|
| Null Imputation  | Median (numeric), Mode (categorical)    | `CREDIT_SCORE`, `ANNUAL_MILEAGE`, and any other nulls    |
| Feature Encoding | `LabelEncoder` (fitted on training data only) | All 8 categorical features                         |
| Train/Test Split | 80/20 stratified (`random_state=42`)    | Full dataset                                             |
| Feature Scaling  | None                                    | Tree-based models are invariant to feature magnitude     |
| ID Column        | Dropped pre-training                    | `ID` carries zero predictive information                 |

---

## 5. Model Insights

### 5.1 Performance Metrics (Test Set)

| Metric                     | Value   | Interpretation                                                      |
|----------------------------|---------|---------------------------------------------------------------------|
| Accuracy                   | 84.25%  | Overall correctness on 2,000 held-out records                       |
| ROC-AUC                    | 0.9125  | 91.3% probability of correctly ranking a claim above a non-claim    |
| No-Claim Precision         | 89%     | When predicting No Claim, the model is correct 89% of the time      |
| No-Claim Recall            | 88%     | 88% of all actual no-claim cases correctly identified               |
| Claim Precision            | 74%     | When predicting Claim, correct 74% of the time                      |
| Claim Recall (Sensitivity) | 76%     | 76% of all actual claims detected; 24% false negative rate          |
| Specificity                | ~88%    | 88% of non-claim policyholders correctly cleared                    |
| False Positives            | ~165    | Non-claims incorrectly flagged (unnecessary intervention cost)       |
| False Negatives            | ~150    | Claims missed (unmanaged financial risk exposure)                   |

### 5.2 Feature Importance (Top 5)

The Gradient Boosting model's feature importances reflect actuarially well-understood risk factors:

| Rank | Feature              | Importance | Interpretation                                                  |
|------|----------------------|------------|-----------------------------------------------------------------|
| 1    | DRIVING_EXPERIENCE   | Highest    | Primary behavioural risk indicator; years of risk-free exposure  |
| 2    | CREDIT_SCORE         | High       | Financial reliability proxy; inverse correlation with claim rate |
| 3    | PAST_ACCIDENTS       | High       | Strongest retrospective risk signal; history predicts future    |
| 4    | AGE                  | Moderate   | Correlated with experience; young drivers are highest-risk      |
| 5    | ANNUAL_MILEAGE       | Moderate   | Usage-based exposure; more km = greater accident probability     |

> **Insight:** The top 3 features — driving experience, credit score, and accident history — collectively capture both prospective and retrospective risk. This aligns with modern insurance pricing frameworks where behavioural indicators outperform demographic proxies.

### 5.3 Class-Level Trade-off Analysis

The model exhibits an intentional asymmetry: stronger precision/recall on the majority No-Claim class versus the minority Claim class. For the insurance use case, the relevant trade-off is:

- **False Negatives (missed claims):** Represent undetected financial risk — the insurer pays unexpected claims without pricing for them.
- **False Positives (unnecessary flags):** Represent operational cost — legitimate low-risk policyholders subjected to manual review or premium surcharges, risking adverse customer experience.

At the default 0.5 threshold, the model optimises for balanced F1. Threshold tuning (see [Future Scope](#12-future-scope)) can shift this balance depending on the insurer's FP/FN cost ratio.

---

## 6. Business Insights

### 6.1 High-Risk Segments (Actionable)

| Segment                          | Claim Rate  | Predicted Probability | Recommended Action                         |
|----------------------------------|-------------|------------------------|--------------------------------------------|
| Age 16–25 + 0–9y experience      | ~52–65%     | > 65%                 | Premium surcharge + mandatory telematics   |
| 2+ speeding violations           | 2× average  | > 55%                 | Manual underwriting referral               |
| DUI offence on record            | High        | Elevated              | High-risk pool assignment or declination   |
| Poverty income + credit < 0.4   | ~45–55%     | > 50%                 | Higher excess/deductible requirement       |
| Sports car + young male driver   | Compound    | > 65%                 | Maximum surcharge tier                     |

### 6.2 Low-Risk Competitive Opportunities

| Segment                          | Claim Rate  | Strategic Opportunity                              |
|----------------------------------|-------------|----------------------------------------------------|
| 30y+ driving experience          | ~14%        | Preferred pricing tier; loyalty discount programme |
| Age 65+ with clean record        | ~18%        | Competitive rate to attract low-risk, high-tenure customers |
| Credit score > 0.7 + no violations | < 20%    | Credit-based discount (where jurisdiction permits) |
| Married vehicle owner, 0 accidents | Low       | Bundle and safe-driver discount stacking           |

### 6.3 ROC-AUC Business Interpretation

An AUC of **0.9125** means that if an insurer randomly selects one policyholder who filed a claim and one who did not, the model correctly assigns a higher risk score to the claim filer **91.3% of the time**. This discriminative power directly translates to premium differentiation accuracy — the model can rank a portfolio by risk with high confidence, enabling actuarially sound tiered pricing.

---

## 7. Business Analysis

### 7.1 Use-Case Applications

| Use Case                     | Implementation                                          | Expected Benefit                              |
|------------------------------|---------------------------------------------------------|-----------------------------------------------|
| Underwriting Decision Support| Query `POST /predict` at point of application           | Automate ~70% of standard risk assessments    |
| Dynamic Premium Pricing      | Map probability output (0–100%) to premium band        | More granular, actuarially fair pricing model |
| Claims Fraud Flagging        | Flag anomaly when low-probability policy files a claim  | Lightweight fraud signal at zero added cost   |
| Portfolio Risk Monitoring    | Monthly batch scoring across the active policy book    | Early detection of reserve shortfalls         |
| Product Segmentation         | Identify low-risk clusters for new product offerings   | Targeted growth in profitable segments        |

### 7.2 Financial Impact (Test Portfolio Proxy)

Based on the 2,000-record held-out test set as a proxy portfolio:

- **627 actual claims detected:** Model flagged **477 (76%)** for early intervention
- **1,373 non-claims:** Model correctly cleared **1,208 (88%)** without manual review
- **~150 missed claims (FN):** Represent unpriced risk exposure requiring reserve allocation
- **~165 false positives:** Unnecessary review cost; manageable with threshold calibration

### 7.3 Known Limitations

| Limitation                | Detail                                                                    |
|---------------------------|---------------------------------------------------------------------------|
| Synthetic dataset         | Simulated data may not fully replicate real-world claim distributions      |
| No claim severity         | Model predicts *whether* a claim occurs, not *how much* it will cost      |
| Static model              | No automated retraining; feature drift will degrade performance over time  |
| Threshold not calibrated  | Default 0.5 threshold may not be optimal for a specific FP/FN cost ratio  |
| Fairness not audited      | `RACE` and `GENDER` features require regulatory review before production deployment |
| No SHAP explanations      | Per-prediction attribution required for adverse action compliance (FCRA/ECOA) |

---

## 8. Project Structure

```
car_insurance_claim_prediction_project/
│
├── Car_Insurance_Claim.csv        # Source dataset — 10,000 records, 18 features
├── train_model.py                 # Training pipeline — preprocessing, 3-model CV, model.pkl export
├── app.py                         # Flask REST API — POST /predict, GET /stats (port 5000)
├── streamlit_app.py               # Streamlit frontend — Predict page + Dashboard (port 8501)
│
├── generate_screenshots.py        # Renders 6 UI mockup PNGs via Matplotlib
├── generate_report.py             # Generates project_report.html (self-contained, base64)
├── generate_docx.py               # Generates project_report.docx (Word with charts)
│
├── model.pkl                      # Serialised artefact: model + encoders + metadata
├── project_report.html            # Full HTML report (auto-generated)
├── project_report.docx            # Full Word report (auto-generated)
├── requirements.txt               # Pinned Python dependencies
├── README.md                      # This file
│
└── static/
    ├── model_evaluation.png       # Confusion matrix + feature importance (from training)
    └── screenshots/               # UI mockup PNGs (from generate_screenshots.py)
```

---

## 9. Tech Stack

| Layer              | Technology              | Version  | Purpose                                             |
|--------------------|-------------------------|----------|-----------------------------------------------------|
| ML Modelling       | scikit-learn            | 1.5.1    | GradientBoosting, RandomForest, LogisticRegression  |
| Backend API        | Flask + flask-cors      | 3.0.3    | REST API — JSON in/out, CORS-enabled, port 5000     |
| Frontend UI        | Streamlit               | 1.36.0   | Interactive web app — Predict + Dashboard pages     |
| Data Processing    | pandas, NumPy           | 2.2.2 / 1.26.4 | ETL, imputation, feature encoding            |
| Visualisation      | Matplotlib, Seaborn     | 3.9.1 / 0.13.2 | EDA charts, evaluation plots, UI mockups     |
| Report — HTML      | Python (base64 embed)   | —        | Self-contained portable HTML report                 |
| Report — DOCX      | python-docx             | 1.1.2    | Word document with embedded charts and tables       |

---

## 10. How to Run

### Prerequisites
- Python **3.9+**
- pip

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the Model
```bash
python train_model.py
```
Outputs `model.pkl` and `static/model_evaluation.png`.

Expected console output:
```
Random Forest          Accuracy=0.8275  AUC=0.8919  CV-AUC=0.9063
Gradient Boosting      Accuracy=0.8425  AUC=0.9125  CV-AUC=0.9239  ← BEST
Logistic Regression    Accuracy=0.8330  AUC=0.8925  CV-AUC=0.9072
```

### Step 3 — Start Flask API (Terminal A)
```bash
python app.py
```
API available at `http://127.0.0.1:5000`

### Step 4 — Start Streamlit UI (Terminal B)
```bash
streamlit run streamlit_app.py
```
UI available at `http://localhost:8501`

> Flask API must be running before launching the Streamlit UI.

### Step 5 — Generate Reports (Optional)
```bash
python generate_screenshots.py   # Renders 6 UI mockup PNGs
python generate_report.py        # Generates project_report.html
python generate_docx.py          # Generates project_report.docx
```

---

## 11. API Reference

### `POST /predict`

Accepts a JSON payload of policyholder features. Returns a prediction, probability, and risk tier.

**Request body:**
```json
{
  "AGE": "16-25",
  "GENDER": "male",
  "RACE": "majority",
  "DRIVING_EXPERIENCE": "0-9y",
  "EDUCATION": "high school",
  "INCOME": "poverty",
  "CREDIT_SCORE": 0.35,
  "VEHICLE_OWNERSHIP": 0,
  "VEHICLE_YEAR": "before 2015",
  "MARRIED": 0,
  "CHILDREN": 0,
  "POSTAL_CODE": 10238,
  "ANNUAL_MILEAGE": 16000,
  "VEHICLE_TYPE": "sedan",
  "SPEEDING_VIOLATIONS": 2,
  "DUIS": 0,
  "PAST_ACCIDENTS": 1
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 1,
  "label": "Claim",
  "probability": 78.4,
  "risk_level": "High"
}
```

**Risk tier thresholds:**

| Tier   | Probability Range | Interpretation                         |
|--------|-------------------|----------------------------------------|
| High   | ≥ 65%             | Refer to manual underwriting           |
| Medium | 40–64%            | Standard review; monitor renewal       |
| Low    | < 40%             | Auto-approve; eligible for discounts   |

### `GET /stats`

Returns pre-computed dataset statistics and model performance metrics used by the Streamlit Dashboard.

---

## 12. Future Scope

The following enhancements are prioritised by expected actuarial impact and technical feasibility. They represent the natural evolution from a prototype to a production-grade insurance risk scoring system.

### 12.1 Model Improvements

| Enhancement                  | Description                                                                 | Expected Impact         |
|------------------------------|-----------------------------------------------------------------------------|-------------------------|
| **XGBoost / LightGBM**       | Native categorical encoding, GPU acceleration, regularisation terms        | AUC +0.01 to +0.03      |
| **Hyperparameter Optimisation** | Optuna Bayesian search over learning rate, depth, estimators, subsample | AUC +0.005 to +0.02     |
| **SMOTE Oversampling**        | Synthetic minority oversampling to address 31.3% class imbalance           | Claim recall +5–8%      |
| **Threshold Calibration**     | Precision-Recall curve analysis to tune decision threshold for specific FP/FN cost ratio | Operational optimisation |
| **Stacked Ensemble**          | Meta-learner combining GBM, RF, and LR predictions                         | AUC +0.01 to +0.02      |
| **Isotonic Regression Calibration** | Post-hoc probability calibration for well-specified output scores  | Actuarial pricing quality |

### 12.2 Feature Engineering

| Feature Addition              | Description                                                                  | Expected Impact         |
|-------------------------------|------------------------------------------------------------------------------|-------------------------|
| **Telematics / UBI Data**     | Hard braking events, night driving %, average speed, acceleration patterns   | AUC +0.03 to +0.08      |
| **Geospatial Risk Scores**    | Replace postal code with accident frequency/km², road quality, weather risk  | AUC +0.01 to +0.03      |
| **Policy History Features**   | Prior claims count, customer tenure, payment punctuality, renewal count      | AUC +0.02 to +0.04      |
| **Claim Severity Regression** | Two-stage model: P(claim) × E[cost | claim] = expected loss for pricing     | Enables per-policy premium |
| **External Risk Indices**     | Seasonal accident rates, local crime index, traffic density by postal zone   | Regional pricing granularity |

### 12.3 Infrastructure & MLOps

| Improvement                   | Description                                                                  | Benefit                          |
|-------------------------------|------------------------------------------------------------------------------|----------------------------------|
| **Docker + docker-compose**   | Containerise Flask API and Streamlit as separate services                    | One-command cloud deployment     |
| **MLflow Experiment Tracking**| Model versioning, metric logging, parameter registry, artifact lineage       | Full ML lifecycle management     |
| **Scheduled Retraining**      | Evidently AI / Alibi Detect drift monitoring + auto-retrain trigger          | Model freshness in production    |
| **API Authentication**        | JWT-based auth, rate limiting, input validation on Flask endpoints           | Production security baseline     |
| **Grafana / Prometheus**      | Real-time API latency, prediction distribution drift, request volume alerts  | Production observability         |
| **CI/CD Pipeline**            | GitHub Actions: lint → test → train → validate AUC threshold → deploy       | Automated, auditable releases    |

### 12.4 Explainability & Fairness

| Initiative                    | Description                                                                  | Compliance Relevance             |
|-------------------------------|------------------------------------------------------------------------------|----------------------------------|
| **SHAP Explanations**         | Per-prediction Shapley values in API response for each top contributing feature | FCRA / ECOA adverse action notices |
| **IBM AI Fairness 360 Audit** | Disparate impact, equal opportunity, and demographic parity analysis across `RACE`, `GENDER` | Insurance regulatory compliance |
| **LIME Integration**          | Local surrogate model explanations for non-technical stakeholder reporting   | Customer-facing explanation reports |
| **Model Card**                | Standardised model card documenting intended use, limitations, bias analysis | EU AI Act / internal governance  |

### 12.5 Prioritised Roadmap

```
Phase 1 (Immediate):
  → Threshold calibration for FP/FN cost optimisation
  → SHAP integration for per-prediction explainability

Phase 2 (Near-term):
  → LightGBM migration for performance and native categoricals
  → SMOTE / class-weight rebalancing for improved claim recall
  → Docker containerisation for deployment portability

Phase 3 (Strategic):
  → Telematics feature pipeline (highest accuracy uplift)
  → Two-stage expected-loss model for actuarial pricing
  → MLflow + drift monitoring for production MLOps
  → IBM AI Fairness 360 audit for regulatory readiness
```

---

## Requirements

```
flask==3.0.3
flask-cors==4.0.1
streamlit==1.36.0
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
matplotlib==3.9.1
seaborn==0.13.2
python-docx==1.1.2
```

```bash
pip install -r requirements.txt
```

---

*IBM AICTE AI/ML Internship Project · Car Insurance Claim Prediction*
