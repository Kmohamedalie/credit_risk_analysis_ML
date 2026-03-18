# Credit Risk Analysis (Machine Learning)

A Streamlit-based **credit risk decision support tool** that predicts the likelihood of **loan default** using a trained **Linear SVM** model. The app supports both **single-applicant scoring** and **batch CSV processing**, and provides lightweight explainability by showing how each feature contributes to an individual prediction.


<img width="1902" height="855" alt="loan" src="https://github.com/user-attachments/assets/186cb452-2f57-49e4-adde-df94246da590" />



## What this project does
- **Predicts default risk** (probability + approve/deny-style decision)
- **Single Applicant** mode: enter applicant details and instantly receive a risk assessment
- **Batch Processing** mode: upload a CSV and generate risk scores + decisions for many applicants
- **Insights & Explainability**
  - Global influence: visualizes model feature weights
  - Individual explanation: shows feature-level contribution for a specific applicant (linear model math)

## Model & Features
- **Model:** Support Vector Machine (SVM), linear kernel
- **Pipeline:** input features → scaling (`scaler.pkl`) → prediction (`tuned_svm_model.pkl`)
- **Features used (6):**
  - `Age`
  - `Annual_Income`
  - `Credit_Score`
  - `Years_at_Job`
  - `Existing_Loans`
  - `DTI_Ratio`

## Run the app locally

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Start Streamlit
```bash
streamlit run app.py
```

## Data
The repository includes example datasets:
- `loan_data.csv` (sample inputs)
- `loan_results.csv` (example outputs/results)

## Notes
- Ensure these model artifacts exist in the project root before running:
  - `scaler.pkl`
  - `tuned_svm_model.pkl`
- The sidebar logo expects `VSM.jpg`.

## Background
Inspired by the **Research Method in Accounting and Finance** course at **Ca' Foscari University of Venice**.

---
**Disclaimer:** This project is for educational/demo purposes and should not be used as the sole basis
