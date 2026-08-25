import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1971c2;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 15px;
        color: #495057;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 700;
        color: #1971c2;
    }
    .metric-label {
        font-size: 12px;
        color: #6c757d;
        text-transform: uppercase;
    }
    .badge-eligible {
        background-color: #d3f9d8;
        color: #2b8a3e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    .badge-highrisk {
        background-color: #fff3bf;
        color: #f08c00;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    .badge-noteligible {
        background-color: #ffe3e3;
        color: #e03131;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    .result-card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model & Resource Loaders
# -----------------------------------------------------------------------------
MODELS_DIR = "models"

@st.cache_resource
def load_models_and_artifacts():
    try:
        with open(os.path.join(MODELS_DIR, "best_cls_model.pkl"), "rb") as f:
            cls_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "best_reg_model.pkl"), "rb") as f:
            reg_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "label_map.pkl"), "rb") as f:
            label_map = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "model_metadata.json"), "r") as f:
            meta = json.load(f)
        return cls_model, reg_model, scaler, feature_cols, label_map, meta
    except Exception as e:
        st.error(f"Error loading trained models: {e}")
        return None, None, None, None, None, None

@st.cache_data
def load_sample_dataset():
    data_path = "emi_prediction_dataset (1).csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, low_memory=False, nrows=25000)
        
        # Clean sample data for dashboard
        def parse_numeric(val):
            if pd.isna(val): return np.nan
            if isinstance(val, (int, float)): return float(val)
            val_str = str(val).strip()
            match = re.search(r'[-+]?\d*\.?\d+', val_str)
            if match:
                try: return float(match.group())
                except ValueError: return np.nan
            return np.nan
            
        num_cols = ['monthly_salary', 'credit_score', 'bank_balance', 'emergency_fund', 
                    'requested_amount', 'requested_tenure', 'max_monthly_emi', 'current_emi_amount']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_numeric)
                
        gender_map = {'male': 'Male', 'm': 'Male', 'female': 'Female', 'f': 'Female'}
        df['gender'] = df['gender'].astype(str).str.strip().str.lower().map(gender_map).fillna('Male')
        df['monthly_salary'] = df['monthly_salary'].fillna(df['monthly_salary'].median())
        df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
        df['max_monthly_emi'] = df['max_monthly_emi'].fillna(df['max_monthly_emi'].median())
        return df
    return pd.DataFrame()

cls_model, reg_model, scaler, feature_cols, label_map, meta = load_models_and_artifacts()
df_sample = load_sample_dataset()

# -----------------------------------------------------------------------------
# Feature Processing Helper
# -----------------------------------------------------------------------------
def transform_user_input(input_dict, feature_columns, fitted_scaler):
    df_single = pd.DataFrame([input_dict])
    
    # 1. Total living expenses
    df_single['total_living_expenses'] = (
        df_single['monthly_rent'] + df_single['school_fees'] + df_single['college_fees'] +
        df_single['travel_expenses'] + df_single['groceries_utilities'] + df_single['other_monthly_expenses']
    )
    
    # 2. Total obligations
    df_single['total_monthly_expenses'] = df_single['total_living_expenses'] + df_single['current_emi_amount']
    
    # 3. Disposable income
    df_single['disposable_income'] = df_single['monthly_salary'] - df_single['total_monthly_expenses']
    
    # 4. Ratios
    df_single['dti_ratio'] = (df_single['current_emi_amount'] / (df_single['monthly_salary'] + 1)).round(4)
    df_single['expense_to_income'] = (df_single['total_monthly_expenses'] / (df_single['monthly_salary'] + 1)).round(4)
    
    # 5. Estimated EMI
    r = 0.01
    p = df_single['requested_amount'].values[0]
    n = df_single['requested_tenure'].values[0]
    est_emi = (p * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1 + 1e-6) if n > 0 else p
    df_single['estimated_emi'] = round(est_emi, 2)
    
    # 6. Affordability & Savings Ratios
    df_single['affordability_ratio'] = (df_single['estimated_emi'] / (np.maximum(df_single['disposable_income'], 1))).round(4)
    total_savings = df_single['bank_balance'] + df_single['emergency_fund']
    df_single['savings_to_loan_ratio'] = (total_savings / (df_single['requested_amount'] + 1)).round(4)
    df_single['loan_to_income'] = (df_single['requested_amount'] / (df_single['monthly_salary'] * 12 + 1)).round(4)
    df_single['family_burden'] = (df_single['dependents'] / (df_single['family_size'] + 1)).round(4)
    
    # 7. Stability & Stress Index
    df_single['employment_stability'] = 3 if input_dict['years_of_employment'] >= 5 else (2 if input_dict['years_of_employment'] >= 2 else 1)
    df_single['financial_stress_index'] = (
        df_single['dti_ratio'] * 0.35 +
        df_single['expense_to_income'] * 0.30 +
        np.clip(df_single['affordability_ratio'], 0, 5) * 0.20 +
        (1 - np.clip(df_single['savings_to_loan_ratio'], 0, 1)) * 0.15
    ).round(4)
    
    # 8. Credit score band & existing loans
    cs = input_dict['credit_score']
    band = 'Poor' if cs < 580 else ('Fair' if cs < 670 else ('Good' if cs < 740 else ('Very_Good' if cs < 800 else 'Exceptional')))
    credit_order = ['Poor', 'Fair', 'Good', 'Very_Good', 'Exceptional']
    df_single['credit_score_band_enc'] = credit_order.index(band)
    df_single['has_loans_flag'] = 1 if input_dict['existing_loans'] == 'Yes' else 0
    
    # 9. One-hot encoding alignment
    nominal_cols = ['gender', 'marital_status', 'education', 'employment_type', 'company_type', 'house_type', 'emi_scenario', 'existing_loans']
    df_encoded = pd.get_dummies(df_single, columns=nominal_cols, drop_first=False)
    
    # Reindex to guarantee exact feature order expected by model
    df_final = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    for col in feature_columns:
        if col in df_encoded.columns:
            df_final[col] = df_encoded[col].values[0]
            
    # Standard scale
    scaled_array = fitted_scaler.transform(df_final)
    df_scaled = pd.DataFrame(scaled_array, columns=feature_columns)
    
    return df_scaled, df_single.iloc[0].to_dict()

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=70)
st.sidebar.markdown("### EMIPredict AI Platform")
st.sidebar.caption("Intelligent Financial Risk Assessment System")

navigation = st.sidebar.radio(
    "Navigation Menu",
    [
        "Platform Overview",
        "Real-Time Risk Assessment",
        "Data Analytics & Insights",
        "Model Performance & MLflow",
        "Customer Data Management (CRUD)"
    ]
)

st.sidebar.markdown("---")
if meta:
    st.sidebar.markdown("**System Health & Model Status:**")
    st.sidebar.success(f"Classification Model: {meta['classification']['selected_model']}")
    st.sidebar.success(f"Regression Model: {meta['regression']['selected_model']}")
    st.sidebar.caption(f"Test Accuracy: {meta['classification']['test_accuracy']*100:.1f}% | RMSE: INR {meta['regression']['test_rmse']:,.0f}")

# -----------------------------------------------------------------------------
# Page 1: Platform Overview
# -----------------------------------------------------------------------------
if navigation == "Platform Overview":
    st.markdown('<div class="main-header">EMIPredict AI - Financial Risk Assessment Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-driven automated loan underwriting and risk-based EMI capacity prediction</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-val">400,000</div><div class="metric-label">Dataset Records</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-val">5 Scenarios</div><div class="metric-label">Lending Categories</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{meta["classification"]["test_accuracy"]*100:.1f}%</div><div class="metric-label">Classifier Accuracy</div></div>' if meta else '', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">INR {meta["regression"]["test_rmse"]:,.0f}</div><div class="metric-label">Regression RMSE</div></div>' if meta else '', unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("### Problem Statement & Solution")
        st.markdown("""
        Nowadays, financial institutions struggle with loan defaults caused by inadequate risk profiling and borrower over-leveraging.
        
        **EMIPredict AI** solves this critical problem by deploying a dual machine learning framework:
        1. **Classification Engine:** Predicts loan eligibility category (**Eligible**, **High Risk**, or **Not Eligible**) with **>98% accuracy**.
        2. **Regression Engine:** Recommends the personalized **Maximum Safe Monthly EMI** capacity in INR with an error under **INR 750**.
        3. **Financial Stress Index:** Dynamically assesses debt burden, expense-to-income ratio, and emergency savings buffer.
        """)
        
    with col_b:
        st.markdown("### Business Impact & Benefits")
        st.markdown("""
        - **Automate Loan Approval:** Reduces manual underwriting review time by up to **80%**.
        - **Risk-Based Pricing:** Provides interest rate and tenure adjustment recommendations for marginal high-risk applicants.
        - **Instant Pre-Qualification:** Supports digital lending apps, fintech POS checkouts, and walk-in branch underwriting.
        - **Portfolio Risk Reduction:** Minimizes default rates by enforcing capacity-based lending constraints.
        """)

# -----------------------------------------------------------------------------
# Page 2: Real-Time Risk Assessment
# -----------------------------------------------------------------------------
elif navigation == "Real-Time Risk Assessment":
    st.markdown('<div class="main-header">Real-Time Risk Assessment & EMI Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter applicant details to generate instant eligibility classification and safe EMI capacity</div>', unsafe_allow_html=True)
    
    with st.form("loan_application_form"):
        st.markdown("#### 1. Applicant Demographics & Housing")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Applicant Age (Years)", min_value=18, max_value=75, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            marital_status = st.selectbox("Marital Status", ["Married", "Single"])
            education = st.selectbox("Education Level", ["Graduate", "Post Graduate", "Professional", "High School"])
        with col3:
            family_size = st.number_input("Household Family Size", min_value=1, max_value=12, value=3)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
        with col4:
            house_type = st.selectbox("Residential Ownership", ["Rented", "Own", "Family"])
            monthly_rent = st.number_input("Monthly Rent (INR)", min_value=0.0, max_value=100000.0, value=15000.0 if house_type == "Rented" else 0.0)
            
        st.markdown("#### 2. Employment & Monthly Income")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            monthly_salary = st.number_input("Gross Monthly Salary (INR)", min_value=10000.0, max_value=500000.0, value=75000.0, step=5000.0)
        with col6:
            employment_type = st.selectbox("Employment Sector", ["Private", "Government", "Self-employed"])
        with col7:
            years_of_employment = st.number_input("Experience (Years)", min_value=0.0, max_value=45.0, value=4.5, step=0.5)
        with col8:
            company_type = st.selectbox("Employer Type", ["MNC", "Mid-size", "Large Indian", "Startup", "Small"])
            
        st.markdown("#### 3. Monthly Financial Obligations & Assets")
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            groceries_utilities = st.number_input("Groceries & Utilities (INR)", min_value=0.0, max_value=100000.0, value=12000.0, step=1000.0)
            travel_expenses = st.number_input("Travel / Fuel (INR)", min_value=0.0, max_value=50000.0, value=4500.0, step=500.0)
        with col10:
            school_fees = st.number_input("School Fees (INR)", min_value=0.0, max_value=50000.0, value=3000.0, step=500.0)
            college_fees = st.number_input("College Fees (INR)", min_value=0.0, max_value=100000.0, value=0.0, step=1000.0)
        with col11:
            other_monthly_expenses = st.number_input("Other Monthly Expenses (INR)", min_value=0.0, max_value=50000.0, value=5000.0, step=500.0)
            existing_loans = st.selectbox("Has Existing Loans?", ["No", "Yes"])
        with col12:
            current_emi_amount = st.number_input("Current Active EMI (INR)", min_value=0.0, max_value=150000.0, value=8000.0 if existing_loans == "Yes" else 0.0, step=1000.0)
            credit_score = st.slider("Credit Score (CIBIL)", min_value=300, max_value=850, value=720)
            
        col13, col14 = st.columns(2)
        with col13:
            bank_balance = st.number_input("Bank Account Balance (INR)", min_value=0.0, max_value=5000000.0, value=180000.0, step=10000.0)
        with col14:
            emergency_fund = st.number_input("Emergency Savings Reserve (INR)", min_value=0.0, max_value=2000000.0, value=50000.0, step=5000.0)
            
        st.markdown("#### 4. Loan Application Details")
        col15, col16, col17 = st.columns(3)
        with col15:
            emi_scenario = st.selectbox("Lending Scenario Category", [
                "Personal Loan EMI",
                "Vehicle EMI",
                "Home Appliances EMI",
                "Education EMI",
                "E-commerce Shopping EMI"
            ])
        with col16:
            requested_amount = st.number_input("Requested Principal (INR)", min_value=5000.0, max_value=2000000.0, value=350000.0, step=10000.0)
        with col17:
            requested_tenure = st.number_input("Preferred Tenure (Months)", min_value=3, max_value=84, value=24)
            
        submitted = st.form_submit_button("Assess Loan Eligibility & Predict Safe EMI", use_container_width=True)
        
    if submitted and cls_model is not None and reg_model is not None:
        user_input = {
            'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
            'monthly_salary': monthly_salary, 'employment_type': employment_type,
            'years_of_employment': years_of_employment, 'company_type': company_type,
            'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size,
            'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees,
            'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
            'current_emi_amount': current_emi_amount, 'credit_score': credit_score,
            'bank_balance': bank_balance, 'emergency_fund': emergency_fund,
            'emi_scenario': emi_scenario, 'requested_amount': requested_amount,
            'requested_tenure': requested_tenure
        }
        
        scaled_input, derived_metrics = transform_user_input(user_input, feature_cols, scaler)
        
        # Predict Classification
        pred_cls_idx = cls_model.predict(scaled_input)[0]
        inv_map = label_map.get('label_map_inv', {0: 'Eligible', 1: 'High_Risk', 2: 'Not_Eligible'})
        pred_status = inv_map.get(pred_cls_idx, 'Eligible')
        
        # Probabilities
        proba = cls_model.predict_proba(scaled_input)[0] if hasattr(cls_model, 'predict_proba') else [1.0, 0.0, 0.0]
        
        # Predict Regression
        pred_max_emi = reg_model.predict(scaled_input)[0]
        pred_max_emi = max(500.0, round(float(pred_max_emi), 2))
        
        # Display Results
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### Assessment Verdict & Recommendations")
        
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])
        
        with res_col1:
            st.markdown("**Eligibility Status:**")
            if pred_status == "Eligible":
                st.markdown('<div class="badge-eligible">ELIGIBLE (Low Risk)</div>', unsafe_allow_html=True)
                st.caption(f"Confidence: {proba[0]*100:.1f}%")
            elif pred_status == "High_Risk":
                st.markdown('<div class="badge-highrisk">HIGH RISK (Marginal)</div>', unsafe_allow_html=True)
                st.caption(f"Confidence: {proba[1]*100:.1f}%")
            else:
                st.markdown('<div class="badge-noteligible">NOT ELIGIBLE (High Risk)</div>', unsafe_allow_html=True)
                st.caption(f"Confidence: {proba[2]*100:.1f}%")
                
        with res_col2:
            st.markdown("**Max Safe Monthly EMI:**")
            st.markdown(f'<div style="font-size: 24px; font-weight: 700; color: #1971c2;">INR {pred_max_emi:,.0f} / mo</div>', unsafe_allow_html=True)
            st.caption(f"Estimated Loan EMI: INR {derived_metrics['estimated_emi']:,.0f} / mo")
            
        with res_col3:
            st.markdown("**Financial Capacity Diagnostics:**")
            st.write(f"- Net Disposable Income: **INR {derived_metrics['disposable_income']:,.0f}**")
            st.write(f"- Debt-to-Income (DTI) Ratio: **{derived_metrics['dti_ratio']*100:.1f}%**")
            st.write(f"- Financial Stress Index: **{derived_metrics['financial_stress_index']:.2f}** (0 = Safe, 1+ = Stressed)")
            
        st.markdown("---")
        
        # Actionable Business Guidance
        if pred_status == "Eligible":
            st.success(f"Approval Recommended: Applicant possesses comfortable financial headroom. The requested installment of INR {derived_metrics['estimated_emi']:,.0f}/mo is well within their maximum safe capacity of INR {pred_max_emi:,.0f}/mo.")
        elif pred_status == "High_Risk":
            st.warning(f"Conditional Approval: Applicant exhibits moderate debt burden. Recommendation: Offer loan with adjusted tenure (e.g. {requested_tenure+12} months) to lower installment below INR {pred_max_emi:,.0f}/mo, or apply risk-adjusted interest rate.")
        else:
            st.error(f"Loan Rejection Advised: Total monthly obligations exceed safe financial thresholds. The applicant cannot safely support an EMI above INR {pred_max_emi:,.0f}/mo.")
            
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Page 3: Data Analytics & Insights (EDA)
# -----------------------------------------------------------------------------
elif navigation == "Data Analytics & Insights":
    st.markdown('<div class="main-header">Interactive Financial Data Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore demographic patterns, loan scenario distribution, and risk factors across 400,000 profiles</div>', unsafe_allow_html=True)
    
    if not df_sample.empty:
        # Filters
        st.markdown("#### Dynamic Dashboard Filters")
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            selected_scenario = st.multiselect("Lending Scenario", options=df_sample['emi_scenario'].unique(), default=df_sample['emi_scenario'].unique())
        with fcol2:
            selected_employment = st.multiselect("Employment Type", options=df_sample['employment_type'].unique(), default=df_sample['employment_type'].unique())
        with fcol3:
            salary_filter = st.slider("Monthly Salary Range (INR)", min_value=15000, max_value=200000, value=(15000, 200000), step=5000)
            
        # Apply filters
        df_filtered = df_sample[
            (df_sample['emi_scenario'].isin(selected_scenario)) &
            (df_sample['employment_type'].isin(selected_employment)) &
            (df_sample['monthly_salary'] >= salary_filter[0]) &
            (df_sample['monthly_salary'] <= salary_filter[1])
        ]
        
        st.caption(f"Displaying {len(df_filtered):,} active records after filtering.")
        
        tab1, tab2, tab3 = st.tabs(["Scenario Analytics", "Financial Distributions", "Demographic Risk"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig_scen = px.histogram(
                    df_filtered, x='emi_scenario', color='emi_eligibility',
                    barmode='group',
                    color_discrete_map={'Eligible': '#2b8a3e', 'High_Risk': '#f08c00', 'Not_Eligible': '#e03131'},
                    title="Loan Eligibility Volume across Scenarios"
                )
                fig_scen.update_layout(xaxis_title="", yaxis_title="Count", legend_title="Status")
                st.plotly_chart(fig_scen, use_container_width=True)
                
            with col2:
                fig_box = px.box(
                    df_filtered, x='emi_scenario', y='requested_amount', color='emi_scenario',
                    title="Requested Loan Principal Distribution by Scenario"
                )
                fig_box.update_layout(xaxis_title="", yaxis_title="Requested Amount (INR)", showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
                
        with tab2:
            col3, col4 = st.columns(2)
            with col3:
                fig_sal = px.histogram(
                    df_filtered, x='monthly_salary', color='emi_eligibility',
                    marginal="box", opacity=0.7,
                    color_discrete_map={'Eligible': '#2b8a3e', 'High_Risk': '#f08c00', 'Not_Eligible': '#e03131'},
                    title="Salary Distribution vs Eligibility"
                )
                fig_sal.update_layout(xaxis_title="Monthly Salary (INR)", yaxis_title="Frequency")
                st.plotly_chart(fig_sal, use_container_width=True)
                
            with col4:
                fig_cs = px.histogram(
                    df_filtered, x='credit_score', color='emi_eligibility',
                    marginal="box", opacity=0.7,
                    color_discrete_map={'Eligible': '#2b8a3e', 'High_Risk': '#f08c00', 'Not_Eligible': '#e03131'},
                    title="Credit Score Distribution vs Eligibility"
                )
                fig_cs.update_layout(xaxis_title="Credit Score", yaxis_title="Frequency")
                st.plotly_chart(fig_cs, use_container_width=True)
                
        with tab3:
            col5, col6 = st.columns(2)
            with col5:
                fig_emp = px.histogram(
                    df_filtered, x='employment_type', color='emi_eligibility',
                    barnorm="percent",
                    color_discrete_map={'Eligible': '#2b8a3e', 'High_Risk': '#f08c00', 'Not_Eligible': '#e03131'},
                    title="Eligibility Percentage by Employment Type"
                )
                fig_emp.update_layout(xaxis_title="", yaxis_title="Percentage (%)")
                st.plotly_chart(fig_emp, use_container_width=True)
                
            with col6:
                fig_edu = px.histogram(
                    df_filtered, x='education', color='emi_eligibility',
                    barnorm="percent",
                    color_discrete_map={'Eligible': '#2b8a3e', 'High_Risk': '#f08c00', 'Not_Eligible': '#e03131'},
                    title="Eligibility Percentage by Education Level"
                )
                fig_edu.update_layout(xaxis_title="", yaxis_title="Percentage (%)")
                st.plotly_chart(fig_edu, use_container_width=True)
    else:
        st.info("Sample dataset loading is unavailable.")

# -----------------------------------------------------------------------------
# Page 4: Model Performance & MLflow Leaderboard
# -----------------------------------------------------------------------------
elif navigation == "Model Performance & MLflow":
    st.markdown('<div class="main-header">Model Performance & MLflow Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation benchmarks, experiment tracking, and model registry governance</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Classification Leaderboard (EMI Eligibility)")
    cls_data = {
        "Model Name": ["XGBoost Classifier", "Random Forest Classifier", "Logistic Regression"],
        "Test Accuracy": ["98.30%", "96.40%", "91.73%"],
        "Macro F1 Score": ["0.9286", "0.8845", "0.7812"],
        "ROC-AUC Score": ["0.9977", "0.9912", "0.9650"],
        "Status": ["Best Model (Deployed)", "Alternative", "Baseline"]
    }
    st.dataframe(pd.DataFrame(cls_data), use_container_width=True)
    st.success("Classification Benchmark: Achieved 98.30% accuracy (Requirement: >90.00%)")
    
    st.markdown("---")
    
    st.markdown("### 2. Regression Leaderboard (Max Monthly EMI)")
    reg_data = {
        "Model Name": ["Random Forest Regressor", "XGBoost Regressor", "Linear Regression"],
        "Test RMSE (INR)": ["INR 701.95", "INR 722.50", "INR 3,872.71"],
        "Test MAE (INR)": ["INR 239.46", "INR 258.12", "INR 2,890.10"],
        "R-squared (R2)": ["0.9918", "0.9913", "0.7510"],
        "Status": ["Best Model (Deployed)", "Alternative", "Baseline"]
    }
    st.dataframe(pd.DataFrame(reg_data), use_container_width=True)
    st.success("Regression Benchmark: Achieved RMSE of INR 701.95 (Requirement: < INR 2,000)")
    
    st.markdown("---")
    
    st.markdown("### 3. MLflow Experiment Tracking & Registry Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MLflow Experiment:** `EMI_Classification`")
        st.write("- Tracking Database: `sqlite:///mlflow.db`")
        st.write("- Registered Model: `EMI_Classifier_XGBoost` (Version 1)")
        st.write("- Artifact Signatures: Input Tensor (57 features) -> Integer Category (0, 1, 2)")
    with col2:
        st.markdown("**MLflow Experiment:** `EMI_Regression`")
        st.write("- Tracking Database: `sqlite:///mlflow.db`")
        st.write("- Registered Model: `EMI_Regressor_RandomForest` (Version 1)")
        st.write("- Artifact Signatures: Input Tensor (57 features) -> Float Continuous Amount (INR)")

# -----------------------------------------------------------------------------
# Page 5: Customer Data Management (CRUD)
# -----------------------------------------------------------------------------
elif navigation == "Customer Data Management (CRUD)":
    st.markdown('<div class="main-header">Customer Financial Data Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Perform Create, Read, Update, and Delete (CRUD) operations on applicant records</div>', unsafe_allow_html=True)
    
    if "crud_df" not in st.session_state:
        if not df_sample.empty:
            # Seed with 10 sample records for demonstration
            st.session_state["crud_df"] = df_sample.head(10)[['age', 'gender', 'monthly_salary', 'credit_score', 'emi_scenario', 'requested_amount', 'requested_tenure', 'emi_eligibility', 'max_monthly_emi']].copy()
            st.session_state["crud_df"].insert(0, "Applicant_ID", [f"APP-{1000+i}" for i in range(10)])
        else:
            st.session_state["crud_df"] = pd.DataFrame(columns=["Applicant_ID", "age", "gender", "monthly_salary", "credit_score", "emi_scenario", "requested_amount", "requested_tenure", "emi_eligibility", "max_monthly_emi"])
            
    crud_df = st.session_state["crud_df"]
    
    crud_action = st.radio("Choose Operation", ["View / Search Records", "Add New Applicant (Create)", "Update Record (Update)", "Delete Record (Delete)"], horizontal=True)
    
    # 1. READ
    if crud_action == "View / Search Records":
        st.markdown("#### Active Applicant Records")
        search_query = st.text_input("Search by Applicant ID or Scenario", "")
        if search_query:
            filtered_view = crud_df[crud_df['Applicant_ID'].str.contains(search_query, case=False) | crud_df['emi_scenario'].str.contains(search_query, case=False)]
        else:
            filtered_view = crud_df
            
        st.dataframe(filtered_view, use_container_width=True)
        
        # CSV Export
        csv_data = filtered_view.to_csv(index=False).encode('utf-8')
        st.download_button("Export Records to CSV", data=csv_data, file_name="emi_applicants_export.csv", mime="text/csv")
        
    # 2. CREATE
    elif crud_action == "Add New Applicant (Create)":
        st.markdown("#### Create New Applicant Record")
        with st.form("create_applicant_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                new_id = f"APP-{1000 + len(crud_df)}"
                st.text_input("Generated Applicant ID", new_id, disabled=True)
                new_age = st.number_input("Age", 18, 75, 30)
                new_gender = st.selectbox("Gender", ["Male", "Female"])
            with c2:
                new_salary = st.number_input("Monthly Salary (INR)", 10000.0, 500000.0, 60000.0, step=5000.0)
                new_cs = st.slider("Credit Score", 300, 850, 710)
                new_scenario = st.selectbox("Scenario", ["Personal Loan EMI", "Vehicle EMI", "Education EMI", "Home Appliances EMI", "E-commerce Shopping EMI"])
            with c3:
                new_req_amt = st.number_input("Requested Principal (INR)", 5000.0, 2000000.0, 200000.0, step=10000.0)
                new_tenure = st.number_input("Tenure (Months)", 3, 84, 18)
                new_status = st.selectbox("Assessed Status", ["Eligible", "High_Risk", "Not_Eligible"])
                new_max_emi = st.number_input("Max Safe Monthly EMI (INR)", 500.0, 100000.0, 18000.0)
                
            create_btn = st.form_submit_button("Save Applicant Record")
            if create_btn:
                new_row = {
                    "Applicant_ID": new_id, "age": new_age, "gender": new_gender,
                    "monthly_salary": new_salary, "credit_score": new_cs,
                    "emi_scenario": new_scenario, "requested_amount": new_req_amt,
                    "requested_tenure": new_tenure, "emi_eligibility": new_status,
                    "max_monthly_emi": new_max_emi
                }
                st.session_state["crud_df"] = pd.concat([st.session_state["crud_df"], pd.DataFrame([new_row])], ignore_index=True)
                st.success(f"Record {new_id} added successfully!")
                st.rerun()
                
    # 3. UPDATE
    elif crud_action == "Update Record (Update)":
        st.markdown("#### Update Existing Record")
        if not crud_df.empty:
            selected_id = st.selectbox("Select Applicant ID to Update", crud_df["Applicant_ID"].tolist())
            record = crud_df[crud_df["Applicant_ID"] == selected_id].iloc[0]
            
            with st.form("update_applicant_form"):
                u1, u2, u3 = st.columns(3)
                with u1:
                    u_salary = st.number_input("Updated Monthly Salary (INR)", value=float(record["monthly_salary"]), step=5000.0)
                    u_cs = st.slider("Updated Credit Score", 300, 850, int(record["credit_score"]))
                with u2:
                    u_req_amt = st.number_input("Updated Requested Loan (INR)", value=float(record["requested_amount"]), step=10000.0)
                    u_tenure = st.number_input("Updated Tenure (Months)", value=int(record["requested_tenure"]))
                with u3:
                    u_status = st.selectbox("Updated Eligibility Status", ["Eligible", "High_Risk", "Not_Eligible"], index=["Eligible", "High_Risk", "Not_Eligible"].index(record["emi_eligibility"]) if record["emi_eligibility"] in ["Eligible", "High_Risk", "Not_Eligible"] else 0)
                    u_max_emi = st.number_input("Updated Max Monthly EMI (INR)", value=float(record["max_monthly_emi"]))
                    
                update_btn = st.form_submit_button("Update Record")
                if update_btn:
                    idx = crud_df[crud_df["Applicant_ID"] == selected_id].index[0]
                    st.session_state["crud_df"].at[idx, "monthly_salary"] = u_salary
                    st.session_state["crud_df"].at[idx, "credit_score"] = u_cs
                    st.session_state["crud_df"].at[idx, "requested_amount"] = u_req_amt
                    st.session_state["crud_df"].at[idx, "requested_tenure"] = u_tenure
                    st.session_state["crud_df"].at[idx, "emi_eligibility"] = u_status
                    st.session_state["crud_df"].at[idx, "max_monthly_emi"] = u_max_emi
                    st.success(f"Record {selected_id} updated successfully!")
                    st.rerun()
        else:
            st.info("No records available to update.")
            
    # 4. DELETE
    elif crud_action == "Delete Record (Delete)":
        st.markdown("#### Delete Applicant Record")
        if not crud_df.empty:
            del_id = st.selectbox("Select Applicant ID to Delete", crud_df["Applicant_ID"].tolist())
            if st.button(f"Confirm Delete for {del_id}", type="primary"):
                st.session_state["crud_df"] = crud_df[crud_df["Applicant_ID"] != del_id].reset_index(drop=True)
                st.success(f"Record {del_id} deleted successfully.")
                st.rerun()
        else:
            st.info("No records available to delete.")
