# 🚗 Car Insurance Claim Prediction

> **IBM AICTE AI/ML Internship Project**  
> End-to-end machine learning system for predicting whether a car insurance policyholder will file a claim.

---

## 📋 Project Structure

```
car_insurance_project/
├── Car_Insurance_Claim.csv   # Source dataset (10,000 records, 18 features)
├── train_model.py            # ML training pipeline — outputs model.pkl
├── app.py                    # Flask REST API backend (port 5000)
├── streamlit_app.py          # Streamlit frontend UI (port 8501)
├── generate_report.py        # Generates project_report.html
├── generate_docx.py          # Generates project_report.docx
├── requirements.txt          # Python dependencies
├── model.pkl                 # Trained model (auto-generated)
├── static/
│   └── model_evaluation.png  # Confusion matrix + feature importance plot
└── README.md                 # This file
```

---

## 🧠 Tech Stack

| Layer              | Technology                                  |
|--------------------|---------------------------------------------|
| ML Modelling       | scikit-learn (GBM, Random Forest, Logistic) |
| Backend API        | **Flask** + flask-cors                      |
| Frontend UI        | **Streamlit**                               |
| Data Processing    | pandas, NumPy                               |
| Visualisation      | Matplotlib, Seaborn                         |
| Report Generation  | Python (HTML + python-docx)                 |

---

## 🚀 How to Run

### Prerequisites
- Python 3.9 or higher
- pip

---

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess `Car_Insurance_Claim.csv`
- Train 3 classifiers (Gradient Boosting, Random Forest, Logistic Regression)
- Auto-select the best model by ROC-AUC
- Save `model.pkl` and `static/model_evaluation.png`

Expected output:
```
Random Forest          Accuracy=0.8275  AUC=0.8919
Gradient Boosting      Accuracy=0.8425  AUC=0.9125  ← BEST
Logistic Regression    Accuracy=0.8330  AUC=0.8925
```

---

### Step 3 — Start the Flask API

Open a terminal and run:

```bash
python app.py
```

Flask API will start on **http://127.0.0.1:5000**

Available endpoints:

| Method | Endpoint    | Description                         |
|--------|-------------|-------------------------------------|
| POST   | `/predict`  | Returns claim prediction + risk     |
| GET    | `/stats`    | Returns dataset & model statistics  |

Example `/predict` request:
```json
{
  "AGE": "16-25",
  "GENDER": "male",
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
  "PAST_ACCIDENTS": 1,
  "RACE": "majority"
}
```

Example response:
```json
{
  "success": true,
  "prediction": 1,
  "label": "Claim",
  "probability": 78.4,
  "risk_level": "High"
}
```

---

### Step 4 — Start the Streamlit Frontend

Open a **second** terminal and run:

```bash
streamlit run streamlit_app.py
```

Streamlit UI will open at **http://localhost:8501**

The UI has two pages:
- **🔮 Predict** — Fill in driver/vehicle details and get an instant claim prediction
- **📊 Dashboard** — View dataset EDA charts and model performance metrics

> **Note:** Flask API (Step 3) must be running before you open the Streamlit UI.

---

### Step 5 — Generate HTML Report (Optional)

```bash
python generate_report.py
```

Opens `project_report.html` in your browser — a self-contained report with all charts embedded.

---

### Step 6 — Generate Word Document Report (Optional)

```bash
python generate_docx.py
```

Saves `project_report.docx` with full project documentation, charts, and evaluation tables.

---

## 📊 Model Performance

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 83.30%   | 0.8925  |
| Random Forest       | 82.75%   | 0.8919  |
| **Gradient Boosting** | **84.25%** | **0.9125** ✅ |

---

## 📁 Dataset

**File:** `Car_Insurance_Claim.csv`  
**Records:** 10,000  
**Target:** `OUTCOME` (0 = No Claim, 1 = Claim filed)  
**Class Balance:** ~68.7% No Claim / 31.3% Claim

Key features: `AGE`, `DRIVING_EXPERIENCE`, `CREDIT_SCORE`, `PAST_ACCIDENTS`, `SPEEDING_VIOLATIONS`, `INCOME`, `ANNUAL_MILEAGE`

---

## 📦 Requirements

```
flask
flask-cors
streamlit
numpy
pandas
scikit-learn
matplotlib
seaborn
python-docx
```

Install all with:
```bash
pip install -r requirements.txt
```
