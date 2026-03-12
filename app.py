import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# --- PAGE CONFIG ---
st.set_config = st.set_page_config(
    page_title="Ca' Foscari Loan Analytics",
    page_icon="🏛️",
    layout="wide"
)

# --- CUSTOM CSS FOR VISIBILITY ---
st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #f4f7f6; }
    
    /* Result Card Styling */
    .result-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border-left: 10px solid #800000; /* Venetian Red */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Force Metric Label Colors */
    [data-testid="stMetricLabel"] {
        color: #444444 !important;
        font-weight: bold !important;
    }
    
    /* Force Metric Value Colors (The Percentage) */
    [data-testid="stMetricValue"] {
        color: #800000 !important; 
        font-size: 2.5rem !important;
    }

    /* Style for anchor links */
    .toc-link {
        text-decoration: none;
        color: #800000;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f) [cite: 209]
        with open('tuned_svm_model.pkl', 'rb') as f:
            model = pickle.load(f) [cite: 1]
        return scaler, model
    except Exception:
        return None, None

scaler, model = load_assets()

# --- SIDEBAR NAVIGATION (Anchor Links) ---
with st.sidebar:
    try:
        st.image("VSM.jpg", use_container_width=True)
    except:
        pass
    
    st.title("Navigation")
    st.markdown("""
    <ul style="list-style-type:none; padding:0;">
        <li><a class="toc-link" href="#single-assessment">🎯 Single Assessment</a></li>
        <li><a class="toc-link" href="#batch-upload">📂 Batch Upload</a></li>
        <li><a class="toc-link" href="#model-insights">📊 Model Insights</a></li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.divider()
    if model:
        st.write(f"**Kernel:** {model.kernel.upper()}") [cite: 1]
        st.write(f"**Iterations:** {model.n_iter_[0]}") [cite: 208]

# --- MAIN PAGE ---
st.title("💰 Loan Default Risk Portal")

if scaler and model:
    tab1, tab2, tab3 = st.tabs(["Individual", "Batch", "Insights"])

    with tab1:
        st.markdown('<div id="single-assessment"></div>', unsafe_allow_html=True)
        col_in, col_res = st.columns([1.2, 1])
        
        with col_in:
            with st.container(border=True):
                st.subheader("Applicant Data")
                age = st.number_input("Age", 18, 100, 35)
                income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
                credit = st.slider("Credit Score", 300, 850, 650)
                dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
                # Grouping other features
                c1, c2 = st.columns(2)
                job_yrs = c1.number_input("Years at Job", 0, 50, 5)
                loans = c2.number_input("Existing Loans", 0, 20, 1)

        with col_res:
            st.subheader("Risk Assessment")
            # Logic
            input_data = pd.DataFrame([[age, income, credit, job_yrs, loans, dti]], 
                                     columns=['Age', 'Annual_Income', 'Credit_Score', 'Years_at_Job', 'Existing_Loans', 'DTI_Ratio'])
            scaled = scaler.transform(input_data) [cite: 209]
            prob = model.predict_proba(scaled)[0][1] [cite: 1]
            pred = model.predict(scaled)[0] [cite: 1]

            # High-Visibility Result Card
            st.markdown(f"""
                <div class="result-card">
                    <h3 style="margin-top:0; color:#444;">Analysis Summary</h3>
                    <p style="color:#666;">Status: <b style="color:{'#e74c3c' if pred==1 else '#27ae60'}">
                    {'DEFAULT RISK' if pred==1 else 'APPROVED'}</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Metric with forced contrast
            st.metric(label="Probability of Default", value=f"{prob:.2%}")
            
            # Progress bar with color logic
            bar_color = "red" if prob > 0.5 else "green"
            st.write(f"**Confidence Level:**")
            st.progress(prob)

    with tab2:
        st.markdown('<div id="batch-upload"></div>', unsafe_allow_html=True)
        st.subheader("📂 Batch Processing")
        # (Batch logic same as previous version...)

    with tab3:
        st.markdown('<div id="model-insights"></div>', unsafe_allow_html=True)
        st.subheader("📊 Model Insights")
        # Plotly chart using Model Coefficients [cite: 208]
        weights = model.coef_[0]
        feats = scaler.feature_names_in_
        fig = go.Figure(go.Bar(x=weights, y=feats, orientation='h', marker_color='#800000'))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Please ensure model and scaler files are uploaded.")
