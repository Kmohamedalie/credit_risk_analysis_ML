import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import shap
from lime import lime_tabular
import streamlit.components.v1 as components

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Ca' Foscari Loan Analytics",
    page_icon="🏛️",
    layout="wide"
)

# --- CUSTOM STYLING (The "Professional" Look) ---
st.markdown("""
    <style>
    .stMetric { padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { border: none !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {  
        border: 1px solid #ddd; 
        padding: 10px 20px; 
        border-radius: 5px 5px 0 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
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

# --- SIDEBAR ---
with st.sidebar:
    # Adding the Ca' Foscari logo / Course Image
    try:
        logo = Image.open('VSM.jpg')
        st.image(logo, use_container_width=True)
    except:
        st.warning("VSM.jpg not found. Please upload to GitHub.")
    
    st.title("Model Specs")
    st.caption("Machine Learning for Credit Risk")
    st.write(f"**Core Model:** SVM")
    st.write(f"**Kernel:** {model.kernel if model else 'Linear'}")
    st.write(f"**Features:** {len(scaler.feature_names_in_) if scaler else 0}")
    st.divider()
    st.markdown("[Course Syllabus](https://www.unive.it/data/insegnamento/558646/programma)")
    st.markdown("[EDA-ML app](https://eda-ml.streamlit.app/)")
    st.markdown("[Demo loan dataset](https://github.com/Kmohamedalie/synthetic_loan_dataset/blob/master/loan_data.csv)")
    st.markdown("[Loan data generator app](https://loan-data-generator.streamlit.app/)")
    st.markdown("[Loan data generator Colab](https://colab.research.google.com/drive/1fmNamAKM2qlGlfv-PmlIVMZsSi8bD0Zg?usp=sharing)")

# --- MAIN CONTENT ---
st.title("💰 Loan Default Risk Portal")
st.markdown("### Decision Support System for Financial Risk Assessment")

if scaler is None or model is None:
    st.error("⚠️ System Offline: Model files missing.")
else:
    # Added the 4th tab for XAI
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Applicant", "📂 Batch Processing", "📈 Insights & About", "🧠 Explainable AI"])

    with tab1:
        col_in, col_res = st.columns([1.5, 1])
        
        with col_in:
            st.subheader("Input Parameters")
            with st.container(border=True):
                c1, c2 = st.columns(2)
                age = c1.number_input("Age", 18, 100, 35)
                income = c1.number_input("Annual Income ($)", 0, 1000000, 60000)
                credit = c1.slider("Credit Score", 300, 850, 650)
                
                job_yrs = c2.number_input("Years at Job", 0, 50, 5)
                loans = c2.number_input("Existing Loans", 0, 20, 1)
                dti = c2.slider("DTI Ratio", 0.0, 1.0, 0.35)

        with col_res:
            st.subheader("Assessment")
            input_df = pd.DataFrame([[age, income, credit, job_yrs, loans, dti]], 
                                    columns=['Age', 'Annual_Income', 'Credit_Score', 'Years_at_Job', 'Existing_Loans', 'DTI_Ratio'])
            
            scaled_data = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0] if 'scaled_input' in locals() else model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0][1]

            if prediction == 1:
                st.error("#### STATUS: HIGH RISK")
            else:
                st.success("#### STATUS: APPROVED")
            
            st.metric("Probability of Default", f"{probability:.2%}")
            st.progress(probability)

    with tab2:
        st.subheader("Batch Upload")
        uploaded_file = st.file_uploader("Upload CSV Data", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            # Scaling and prediction logic
            batch_scaled = scaler.transform(df[['Age', 'Annual_Income', 'Credit_Score', 'Years_at_Job', 'Existing_Loans', 'DTI_Ratio']])
            df['Risk Score'] = model.predict_proba(batch_scaled)[:, 1]
            df['Decision'] = np.where(model.predict(batch_scaled) == 1, 'Deny', 'Approve')
            st.dataframe(df, use_container_width=True)

    with tab3:
        st.subheader("About the Project")
        st.info("""
        Inspired by the **Research Method in Accounting and Finance** course at **Ca' Foscari University of Venice**.
        This project utilizes a Linear SVM model to classify credit risk based on six key financial metrics.
        """)
        
        # Feature Importance Chart
        weights = model.coef_[0]
        features = scaler.feature_names_in_
        fig = go.Figure(go.Bar(x=weights, y=features, orientation='h', marker_color=['#2ecc71' if w < 0 else '#e74c3c' for w in weights])) # Venetian Red , marker_color='#800000'
        fig.update_layout(title="Feature Influence on Default Risk", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Model Explainability (XAI)")
        st.write("Understand the exact drivers behind the model's decision for the applicant currently in the **Single Applicant** tab.")
        
        if 'scaled_data' in locals():
            
            # Dummy background using zeros (represents average since data is scaled)
            dummy_background = np.zeros((100, len(scaler.feature_names_in_)))
            
            # --- LIME EXPLANATION ---
            st.markdown("### 1. LIME (Local Interpretable Model-agnostic Explanations)")
            st.caption("LIME builds a mini-model around this specific applicant to explain the probability score.")
            
            explainer_lime = lime_tabular.LimeTabularExplainer(
                training_data=dummy_background,
                feature_names=scaler.feature_names_in_,
                class_names=['Approved', 'High Risk'],
                mode='classification',
                discretize_continuous=False
            )
            
            predict_fn = lambda x: model.predict_proba(x)
            exp = explainer_lime.explain_instance(scaled_data[0], predict_fn, num_features=6)
            
            # Render LIME as HTML in Streamlit
            components.html(exp.as_html(), height=350, scrolling=True)
            
            st.divider()
            
            # --- SHAP EXPLANATION ---
            st.markdown("### 2. SHAP (SHapley Additive exPlanations)")
            st.caption("SHAP breaks down how much each feature pushed the applicant's risk score higher (red) or lower (blue).")
            
            # Initialize SHAP LinearExplainer
            explainer_shap = shap.LinearExplainer(model, dummy_background)
            shap_values = explainer_shap.shap_values(scaled_data)
            
            expected_val = explainer_shap.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = expected_val[0] 
                
            shap_val = shap_values[0] 
            
            # Generate SHAP Force Plot via JavaScript
            shap.initjs()
            force_plot = shap.force_plot(
                expected_val, 
                shap_val, 
                scaled_data[0], 
                feature_names=scaler.feature_names_in_
            )
            
            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(shap_html, height=200)
            
        else:
            st.info("👈 Please enter applicant details in the **Single Applicant** tab first to see explanations.")
