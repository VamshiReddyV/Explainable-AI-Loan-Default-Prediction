# LoanPredict AI: Explainable AI Loan Default Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.46.0-red.svg)](https://github.com/shap/shap)
[![LIME](https://img.shields.io/badge/LIME-0.2.0-purple.svg)](https://github.com/marcotcr/lime)

A production-grade, transparent Machine Learning credit underwriting platform designed to assess loan default probability and provide mathematically sound, auditable explanations for every prediction using **SHAP (Shapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**.

---

## 🌟 Key Features

1. **Executive Underwriting Dashboard**
   - Live KPI cards: Total Predictions (1,248), Approved Loans (812), Defaulted Loans (436), Default Rate (34.94%), and Model Accuracy (89.67%).
   - Interactive Chart.js visualizations: Donut chart for approval breakdown, 5-bucket risk distribution histogram, and monthly accuracy trend.
   - Dual-tab Explainability module: Switch seamlessly between SHAP feature attributions and LIME local surrogate rules for recent applicants.

2. **5-Step Loan Application Wizard**
   - Step 1: Personal Details (Name, Age, Gender, Marital Status, Dependents)
   - Step 2: Employment & Income Details (Type, Experience, Monthly/Annual Income auto-calculated)
   - Step 3: Loan Details (Loan Amount, Purpose, Term, Interest Rate, Active Loans)
   - Step 4: Financial & Credit Details (CIBIL Credit Score validation: 300 - 900)
   - Step 5: Review & Submit (Instant confirmation before running AI underwriting inference)

3. **Dual-Method Explainable AI (XAI)**
   - **SHAP (TreeExplainer):** Global and local feature attributions rooted in cooperative game theory.
   - **LIME (LimeTabularExplainer):** Local condition thresholds that formulate adverse action notices (e.g., *Credit Score &le; 600 increases risk by +0.38*).

4. **Prediction History & Audit Trail**
   - Full historical log of all assessments stored in thread-safe SQLite.
   - Live search by applicant name or loan purpose.
   - Filter by status (All, Approved, Default).
   - One-click CSV export (`/api/export-csv`).

5. **Regulatory Governance & Compliance**
   - Built to satisfy **FCRA (Fair Credit Reporting Act)** adverse action requirements and **GDPR Article 22 (Right to Explanation)**.
   - Bias auditing across sensitive applicant demographics.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Applicant Form / REST API] -->|JSON Payload| B[Flask Server (app.py)]
    B --> C[Data Preprocessing & Scaling]
    C --> D[Random Forest Classifier (200 Trees)]
    D -->|Class & Probability| E[Prediction Engine]
    D -->|Model & Data| F[SHAP TreeExplainer]
    D -->|Model & Scaled Data| G[LIME Tabular Explainer]
    F -->|Feature Attributions| H[XAI Aggregator]
    G -->|Condition Rules| H
    E --> I[SQLite Database (Audit Log)]
    H --> I
    I --> J[Executive Dashboard & Reports]
```

---

## 🚀 Quickstart Guide

### 1. Clone & Navigate to Repository
```bash
cd Explainable-AI-Loan-Default-Prediction
```

### 2. Environment Setup
All required dependencies (`Flask`, `joblib`, `scikit-learn`, `shap`, `lime`, `pandas`, `numpy`, `matplotlib`, `pytest`) are listed in `requirements.txt`.

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Platform

#### Option A: One-Click Windows Launcher
Double-click `run.bat` or run:
```cmd
run.bat
```

#### Option B: Convenience Python Launcher
```bash
python run.py
```

#### Option C: Standard Flask Run
```bash
python app.py
```

Open your browser and navigate to:
**`http://127.0.0.1:5000`**

---

## 🧪 Running Automated Tests

Run the test suite with `pytest`:
```bash
python -m pytest tests/ -v
```

---

## 📂 Project Structure

```
Explainable-AI-Loan-Default-Prediction/
├── app.py                     # Main Flask web application
├── run.py                     # Cross-platform application launcher
├── run.bat                    # Windows batch launcher
├── config.py                  # App configuration & hyperparameters
├── requirements.txt           # Python dependencies
├── data/
│   └── raw/
│       └── loan_data.csv      # Synthetic training dataset (5,000 samples)
├── instance/
│   └── predictions.db         # SQLite persistent audit store
├── models/
│   ├── loan_default_model.pkl # Trained Random Forest ensemble
│   ├── scaler.pkl             # Fitted StandardScaler
│   ├── feature_columns.pkl    # Feature ordering schema
│   └── categorical_mappings.pkl # Categorical mappings
├── src/
│   ├── data_preprocessing.py  # Data cleaning and encoding
│   ├── feature_engineering.py # DTI and financial ratios
│   ├── train_model.py         # Model training pipeline
│   ├── evaluate_model.py      # Confusion matrix & ROC evaluation
│   ├── explain_model.py       # SHAP & LIME explanation engines
│   └── database.py            # SQLite ORM & seed utilities
├── static/
│   ├── css/
│   │   └── style.css          # Core CSS design system
│   └── js/
│       └── main.js            # Wizard, Chart.js charts & AJAX
├── templates/
│   ├── base.html              # Base layout with responsive sidebar
│   ├── dashboard.html         # Executive overview & dynamic charts
│   ├── new_prediction.html    # 5-step loan prediction wizard
│   ├── history.html           # Historical log & CSV export
│   ├── explainability.html    # In-depth SHAP vs LIME studio
│   ├── performance.html       # Model metrics & confusion matrix
│   ├── about.html             # Project & compliance documentation
│   └── prediction_result.html # Individual applicant XAI report
└── tests/
    ├── test_model.py          # Model and preprocessing unit tests
    └── test_explanation.py    # SHAP and LIME unit tests
```

---

## 📊 REST API Endpoints

### 1. Predict Loan Default Risk
- **Route:** `POST /api/predict`
- **Headers:** `Content-Type: application/json`
- **Payload:**
```json
{
  "full_name": "Rahul Sharma",
  "age": 32,
  "gender": "Male",
  "marital_status": "Married",
  "number_of_dependents": 2,
  "employment_type": "Salaried",
  "employment_experience": 4,
  "monthly_income": 48000,
  "credit_score": 580,
  "loan_amount": 250000,
  "loan_purpose": "Personal",
  "loan_term": 24,
  "interest_rate": 12.5,
  "existing_loans": 2
}
```
- **Response:**
```json
{
  "success": true,
  "prediction_id": 1,
  "prediction": "Default",
  "default_probability": 72.43,
  "risk_level": "High Risk",
  "shap_explanation": [
    {"feature": "Credit_Score", "display_name": "Credit Score", "shap_value": 0.42},
    {"feature": "Loan_Amount", "display_name": "Loan Amount", "shap_value": 0.31}
  ],
  "lime_explanation": [
    {"condition": "Credit Score <= 600.00", "weight": 0.38, "direction": "increases"}
  ]
}
```

### 2. Export Audit Log
- **Route:** `GET /api/export-csv`
- **Description:** Returns a downloadable CSV file of all processed predictions.
