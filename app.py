import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Loan Risk Portal",
    page_icon="⚖️",
    layout="wide"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Loading the scaler [cite: 209]
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        # Loading the tuned SVM model [cite: 1]
        with open('tuned_svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return scaler, model
    except FileNotFoundError:
        return None, None

scaler, model = load_assets()

# --- SIDEBAR / INFO ---
with st.sidebar:
    st.title("Model Information")
    st.info("""
    **Algorithm:** Support Vector Machine (SVM) [cite: 1]
    **Kernel:** Linear [cite: 1]
    **Features:** Age, Income, Credit Score, Job Years, Loans, DTI [cite: 1, 209]
    """)
    st.write("---")
    st.write("This tool predicts the likelihood of a loan default based on historical training data.")

# --- MAIN UI ---
st.title("💰 Loan Default Prediction & Risk Analysis")
st.markdown("Enter the applicant's financial details below to generate a risk assessment.")

if scaler is None or model is None:
    st.error("⚠️ Required files (`scaler.pkl` or `tuned_svm_model.pkl`) are missing from the directory.")
else:
    # Create Tabs for a cleaner look
    tab1, tab2 = st.tabs(["📋 Applicant Assessment", "📊 Model Insights"])

    with tab1:
        # Input Form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
                credit_score = st.slider("Credit Score", 300, 850, 650)
            
            with col2:
                years_at_job = st.number_input("Years at Current Job", min_value=0, max_value=50, value=5)
                existing_loans = st.number_input("Number of Existing Loans", min_value=0, max_value=20, value=1)
                dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", 0.0, 1.0, 0.30, step=0.01)
            
            submit = st.form_submit_button("Run Risk Analysis", type="primary", use_container_width=True)

        if submit:
            # Prepare Input Data based on trained features [cite: 1, 209]
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
            # Probabilities available because probability=True in model [cite: 1]
            prob_default = model.predict_proba(scaled_input)[0][1]

            # Results Display
            st.subheader("Result")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prediction == 1:
                    st.error("### HIGH RISK")
                    st.write("Potential for Default")
                else:
                    st.success("### LOW RISK")
                    st.write("Likely to Repay")
            
            with res_col2:
                st.metric("Probability of Default", f"{prob_default:.2%}")
                st.progress(prob_default)

    with tab2:
        st.subheader("Feature Importance (Linear Coefficients)")
        st.write("This chart shows how much weight the model gives to each factor. Positive weights (red) increase default risk, while negative weights (green) decrease it.")
        
        # Extracting coefficients from the Linear SVM [cite: 1, 52]
        feature_names = ['Age', 'Annual Income', 'Credit Score', 'Years at Job', 'Existing Loans', 'DTI Ratio']
        weights = model.coef_[0]

        # Sorting for better visualization
        sorted_indices = np.argsort(weights)
        sorted_weights = weights[sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]

        fig = go.Figure(go.Bar(
            x=sorted_weights,
            y=sorted_features,
            orientation='h',
            marker_color=['#2ecc71' if w < 0 else '#e74c3c' for w in sorted_weights]
        ))

        fig.update_layout(
            xaxis_title="Coefficient Weight",
            yaxis_title="Feature",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        

        st.markdown("""
        **Key Takeaways:**
        - **Credit Score:** Usually has a strong negative weight (higher score = lower risk).
        - **DTI Ratio:** Usually has a positive weight (higher debt = higher risk).
        """)
