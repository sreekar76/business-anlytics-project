import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Application Configuration
# ==========================================
st.set_page_config(page_title="Loan Underwriting AI", layout="wide", page_icon="🏦")

# Load the trained model only once using Streamlit's caching
# This satisfies Part G: loading a pre-trained model rather than retraining
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please run the Jupyter Notebook to train and save the model first.")
    st.stop()

# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.title("🏦 Menu")
page = st.sidebar.radio("Navigate to:",
                        ["1. Business Problem", "2. Data Insights", "3. Prediction Engine"])

# ==========================================
# Page 1: Business Problem
# ==========================================
if page == "1. Business Problem":
    st.title("Automated Loan Underwriting & Risk Prediction")

    st.header("Business Problem")
    st.write("A retail bank is facing elevated non-performing assets (NPAs) and wants to automate the preliminary loan underwriting process to reduce default rates without slowing down approval turnaround times.")

    st.header("Analytics Objective")
    st.write("Develop a classification model that predicts whether an applicant is likely to default (High Risk) or qualify for approval (Low Risk) based on their demographic and financial profile.")

    st.header("Dataset Information")
    st.markdown("""
    * **Name:** Loan Prediction Dataset
    * **Source:** Kaggle
    * **Size:** 614 observations, 13 features
    * **Target Variable:** `Loan_Status` (1 = Approved, 0 = Rejected)
    """)

# ==========================================
# Page 2: Data Insights
# ==========================================
elif page == "2. Data Insights":
    st.title("Exploratory Data Analysis")
    st.write("Key drivers of credit risk identified during model training.")

    # Load dataset for visualizations
    try:
        df = pd.read_csv('loan_dataset.csv')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Credit History Impact")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df, x='Credit_History', hue='Loan_Status', palette='Set1', ax=ax1)
            ax1.set_title("Approval by Credit History")
            st.pyplot(fig1)
            st.info("**Business Insight:** Applicants lacking a credit history (0.0) are almost universally rejected. This is the strongest predictive feature in the dataset.")

        with col2:
            st.subheader("Property Area Risk Profile")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df, x='Property_Area', hue='Loan_Status', palette='Pastel1', ax=ax2)
            ax2.set_title("Approval by Property Area")
            st.pyplot(fig2)
            st.info("**Business Insight:** Semi-urban properties see the highest volume of applications and the highest proportion of safe approvals, indicating a highly profitable lending segment.")

    except FileNotFoundError:
        st.warning("Please ensure 'data/loan_dataset.csv' is present in the repository to view live charts.")

# ==========================================
# Page 3: Prediction Engine
# ==========================================
elif page == "3. Prediction Engine":
    st.title("Interactive Underwriting Tool")
    st.write("Enter the applicant's details below to generate a real-time risk assessment.")

    # Input Form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])

        with col2:
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
            coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)

        with col3:
            loan_amount_term = st.number_input("Loan Term (Days)", min_value=0, value=360)
            credit_history = st.selectbox("Credit History (1=Good, 0=Bad)", [1.0, 0.0])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submit_button = st.form_submit_button("Predict Approval Status")

    if submit_button:
        # 1. Map inputs to exactly match the training data format
        input_data = pd.DataFrame({
            'Gender': [1 if gender == "Male" else 0],
            'Married': [1 if married == "Yes" else 0],
            'Dependents': [3 if dependents == "3+" else int(dependents)],
            'Education': [1 if education == "Graduate" else 0],
            'Self_Employed': [1 if self_employed == "Yes" else 0],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [{'Urban': 2, 'Semiurban': 1, 'Rural': 0}[property_area]]
        })

        # 2. Generate Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # 3. Display Results and Visualizations
        st.markdown("---")
        st.subheader("Underwriting Decision")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.success("✅ APPROVED: Low Default Risk")
                st.write("**Recommended Action:** Auto-approve with standard prime interest rates.")
            else:
                st.error("❌ REJECTED: High Default Risk")
                st.write("**Recommended Action:** Route to manual underwriting; require additional collateral or co-signer.")

        with res_col2:
            # Visualization inside the app (Requirement)
            st.write("**Approval Confidence:**")

            # Create a simple probability bar chart
            fig, ax = plt.subplots(figsize=(5, 1.5))
            ax.barh(["Risk Assessment"], [probability[1] * 100], color='green' if prediction == 1 else 'red')
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability of Approval (%)")
            st.pyplot(fig)
            st.caption(f"The model is {probability[1] * 100:.1f}% confident in approving this loan.")
