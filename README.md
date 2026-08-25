# EMIPredict AI - Intelligent Financial Risk Assessment Platform

## Overview
EMIPredict AI is a financial risk assessment and automated loan decisioning platform built for banks, credit agencies, and fintech companies. It leverages machine learning to solve two critical underwriting problems simultaneously:
1. **Classification:** Predicts loan EMI eligibility status (`Eligible`, `High_Risk`, `Not_Eligible`).
2. **Regression:** Predicts the borrower's personalized maximum safe monthly EMI capacity (`max_monthly_emi` in INR).

---

## Project Structure
```
AribaProject/
│
├── EMIPredict_AI_Deep_Analysis.ipynb    # Deep Analysis, EDA, MLflow Tracking & Modeling
├── app.py                               # Interactive Multi-Page Streamlit Web Application
├── requirements.txt                     # Production Python Dependencies
├── README.md                            # Comprehensive Technical & Deployment Documentation
│
├── models/                              # Serialized Production Artifacts
│   ├── best_cls_model.pkl               # Selected XGBoost Classification Model
│   ├── best_reg_model.pkl               # Selected Random Forest Regression Model
│   ├── scaler.pkl                       # Fitted StandardScaler
│   ├── feature_cols.pkl                 # Feature Alignment Column List
│   ├── label_map.pkl                    # Target Label Encodings
│   └── model_metadata.json              # Evaluation Metrics & Model Registry Metadata
│
├── .streamlit/
│   └── config.toml                      # Streamlit Theme and UI Configuration
│
├── mlflow.db                            # SQLite Experiment Tracking Database
└── emi_prediction_dataset (1).csv       # Dataset (400,000 Financial Records)
```

---

## Technical Performance Benchmarks

| Objective | Target Benchmark | Achieved Metric | Status |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | $> 90.00\%$ | **$98.30\%$** (XGBoost) | **Passed** |
| **Classification ROC-AUC** | High Discriminative Ability | **$0.9977$** | **Passed** |
| **Regression RMSE** | $< \text{INR } 2,000$ | **$\text{INR } 701.95$** (Random Forest) | **Passed** |
| **Regression $R^2$ Score** | $> 0.90$ | **$0.9918$** | **Passed** |

---

## Key Features of the Streamlit Application

### 1. Real-Time Risk Assessment & Safe EMI Prediction
- Interactive form capturing demographic, income, employment, living expense, and credit details.
- Instant calculation of banking ratios: Debt-to-Income (DTI), Expense-to-Income, and Composite Financial Stress Index.
- Color-coded eligibility badge with model confidence scores and personalized maximum safe monthly EMI recommendation.

### 2. Interactive Data Analytics (EDA) Dashboard
- Dynamic filtering by lending scenario, employment sector, and monthly income range.
- Interactive Plotly visualizations for loan scenario distributions, salary vs. eligibility histograms, and demographic risk breakdowns.

### 3. Model Performance & MLflow Leaderboard
- Model comparison tables across all trained classification and regression algorithms.
- MLflow experiment tracking details with version-controlled model registry metadata.

### 4. Customer Data Management (CRUD)
- **Create:** Add new loan applicant profiles.
- **Read & Search:** Filter and inspect existing customer records.
- **Update:** Modify existing records and re-assess financial capacity.
- **Delete:** Remove outdated applicant files.
- **Export:** Download records as CSV.

---

## How to Run Locally

### 1. Activate Environment
```bash
conda activate emipredict
```

### 2. Run Deep Analysis Notebook
Open `EMIPredict_AI_Deep_Analysis.ipynb` in VS Code / Jupyter Lab and select kernel `Python (emipredict)`.

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```

### 4. Launch MLflow Experiment Dashboard (Optional)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Streamlit Cloud Deployment Guide
1. Push the project repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**, select your repository, branch, and set main file path to `app.py`.
4. Deploy the application.
