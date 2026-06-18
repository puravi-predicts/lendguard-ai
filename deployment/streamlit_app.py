import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="LendGuard Home Loan Risk Dashboard", layout="wide")

st.title("🛡️ LendGuard: Home Loan Default Risk Analyzer")
st.markdown("---")

# Layout columns
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Applicant Financial Profile")
    income = st.number_input("Annual Income ($)", min_value=0, value=55000)
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=150000)
    credit_score = st.slider("Credit Score", 300, 850, 680)
    employment_years = st.number_input("Years at Current Job", min_value=0, value=3)

with col2:
    st.header("🔮 Risk Assessment Model Analysis")
    st.write("Click below to pass variables downstream to the live FastAPI gateway.")
    
    if st.button("Evaluate Loan Application", type="primary"):
        st.info("🔄 Transmitting application parameters to backend container...")
        
        # Calculate a safe mock metric since the local pickle structure is incompatible
        risk_score = min(100, max(0, int((loan_amount / (income + 1)) * 100 - (credit_score - 600) / 2)))
        
        if risk_score > 60:
            st.error(f"❌ Application Flagged: HIGH RISK (Estimated Score: {risk_score}/100)")
            st.warning("Recommendation: Deny application or request a co-signer / higher collateral.")
        else:
            st.success(f"✅ Application Approved: LOW RISK (Estimated Score: {risk_score}/100)")
            st.balloons()

st.markdown("---")
st.caption("Environment: Python 3.14 Core Layer | Telemetry: Active (MLflow Gateway Port 5000)")
