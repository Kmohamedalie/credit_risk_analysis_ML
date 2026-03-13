import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

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
    st.write(f"**Core Model:** SVM [cite: 1]")
    st.write(f"**Kernel:** {model.kernel if model else 'Linear'} [cite: 1]")
    st.write(f"**Features:** {len(scaler.feature_names_in_) if scaler else 0} [cite: 1]")
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
    tab1, tab2, tab3 = st.tabs(["🎯 Single Applicant", "📂 Batch Processing", "📈 Insights & About"])

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
        This project utilizes a Linear SVM model [cite: 1] to classify credit risk based on six key financial metrics[cite: 1].
        """)
        
        # Feature Importance Chart
        weights = model.coef_[0]
        features = scaler.feature_names_in_
        fig = go.Figure(go.Bar(x=weights, y=features, orientation='h', marker_color=['#2ecc71' if w < 0 else '#e74c3c' for w in weights])) # Venetian Red , marker_color='#800000'
        fig.update_layout(title="Feature Influence on Default Risk", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)







