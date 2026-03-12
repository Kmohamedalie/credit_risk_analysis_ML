import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Loan Risk Analysis Portal",
    page_icon="💰",
    layout="wide"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Loading the scaler 
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        # Loading the tuned SVM model 
        with open('tuned_svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return scaler, model
    except FileNotFoundError:
        return None, None

scaler, model = load_assets()

# --- MAIN UI ---
st.title("💰 Professional Loan Default Risk Portal")
st.markdown("Assess individual applicants or process batch files using your **Tuned SVM Model**.")

if scaler is None or model is None:
    st.error("⚠️ Required files (`scaler.pkl` or `tuned_svm_model.pkl`) not found in the root directory.")
else:
    # Sidebar Info
    st.sidebar.header("Model Specifications")
    st.sidebar.text(f"Model Type: SVM (SVC)") # 
    st.sidebar.text(f"Kernel: Linear") # 
    st.sidebar.text(f"Features: {len(scaler.feature_names_in_)}") # 
    
    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Single Applicant", "📁 Batch Upload", "📊 Model Insights"])

    # --- TAB 1: SINGLE PREDICTION ---
    with tab1:
        st.subheader("Individual Risk Assessment")
        with st.form("single_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", 18, 100, 35)
                income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
                credit = st.slider("Credit Score", 300, 850, 650)
            with col2:
                job_yrs = st.number_input("Years at Job", 0, 50, 5)
                loans = st.number_input("Existing Loans", 0, 20, 1)
                dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
            
            submitted = st.form_submit_button("Predict Default Risk", type="primary")

        if submitted:
            # Create DataFrame with exact training feature names [cite: 1, 209]
            input_df = pd.DataFrame([[age, income, credit, job_yrs, loans, dti]], 
                                    columns=['Age', 'Annual_Income', 'Credit_Score', 
                                             'Years_at_Job', 'Existing_Loans', 'DTI_Ratio'])
            
            scaled_data = scaler.transform(input_df) # 
            prediction = model.predict(scaled_data)[0] # 
            probability = model.predict_proba(scaled_data)[0][1] # 

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                if prediction == 1:
                    st.error("### PREDICTION: DEFAULT")
                else:
                    st.success("### PREDICTION: NO DEFAULT")
            with c2:
                st.metric("Risk Probability", f"{probability:.2%}")
                st.progress(probability)

    # --- TAB 2: BATCH PROCESSING ---
    with tab2:
        st.subheader("Batch File Processing")
        st.write("Upload a CSV file containing the required features.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            # Ensure correct columns exist
            required = ['Age', 'Annual_Income', 'Credit_Score', 'Years_at_Job', 'Existing_Loans', 'DTI_Ratio']
            if all(col in data.columns for col in required):
                # Scale and Predict
                batch_scaled = scaler.transform(data[required])
                preds = model.predict(batch_scaled)
                probs = model.predict_proba(batch_scaled)[:, 1]
                
                # Append results
                data['Prediction'] = np.where(preds == 1, 'Default', 'No Default')
                data['Risk_Probability'] = probs
                
                st.write("### Analysis Results")
                st.dataframe(data)
                
                # Download button
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", csv, "loan_results.csv", "text/csv")
            else:
                st.error(f"CSV must contain these columns: {required}")

    # --- TAB 3: MODEL INSIGHTS ---
    with tab3:
        st.subheader("Feature Importance")
        st.write("This chart represents the coefficients ($w$) of the Linear SVM.")
        
        # Get feature names from scaler 
        features = scaler.feature_names_in_
        # Get coefficients from model 
        importance = model.coef_[0]

        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color=['#e74c3c' if x > 0 else '#2ecc71' for x in importance]
        ))
        
        fig.update_layout(xaxis_title="Influence on Risk (Coefficient)", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)
