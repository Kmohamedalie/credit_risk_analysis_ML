import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="centered")

# --- LOAD MODELS ---
@st.cache_resource # Cache to prevent reloading on every click
def load_assets():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('tuned_svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return scaler, model
    except FileNotFoundError:
        return None, None

scaler, model = load_assets()

# --- APP UI ---
st.title("💰 Loan Default Prediction Portal")
st.markdown("""
This application uses a **Tuned SVM Model** to assess the likelihood of a loan applicant defaulting based on their financial profile.
""")
st.divider()

if scaler is None or model is None:
    st.error("⚠️ Model or Scaler files not found! Please ensure 'scaler.pkl' and 'tuned_svm_model.pkl' are in the app directory.")
else:
    # --- INPUT SECTION ---
    st.subheader("Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        
    with col2:
        years_at_job = st.number_input("Years at Current Job", min_value=0, max_value=50, value=5)
        existing_loans = st.number_input("Number of Existing Loans", min_value=0, max_value=20, value=1)
        dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", 0.0, 1.0, 0.35, step=0.01)

    # --- PREDICTION LOGIC ---
    if st.button("Analyze Risk Profile", type="primary", use_container_width=True):
        # Prepare Data
        input_df = pd.DataFrame([{
            'Age': age,
            'Annual_Income': annual_income,
            'Credit_Score': credit_score,
            'Years_at_Job': years_at_job,
            'Existing_Loans': existing_loans,
            'DTI_Ratio': dti_ratio
        }])

        # Scale and Predict
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prob_default = model.predict_proba(scaled_input)[0][1]

        # --- RESULTS DISPLAY ---
        st.divider()
        st.subheader("Risk Assessment Results")
        
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            if prediction == 1:
                st.error("### Prediction: HIGH RISK")
                st.write("The model suggests a high likelihood of default.")
            else:
                st.success("### Prediction: LOW RISK")
                st.write("The model suggests a low likelihood of default.")

        with m_col2:
            st.metric(label="Probability of Default", value=f"{prob_default:.2%}")
            st.progress(prob_default)

        # Insightful Tip
        if prob_default > 0.5:
            st.warning("**Note:** Applicants with higher DTI ratios and lower credit scores significantly increase the model's risk score.")