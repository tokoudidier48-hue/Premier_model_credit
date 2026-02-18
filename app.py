# app.py

import streamlit as st
import pandas as pd
from src.model import predict

st.title("💳 Credit Default Prediction App")

st.write("Enter client information to predict default probability.")

# Exemple simple avec quelques variables principales
LIMIT_BAL = st.number_input("Credit Limit", value=20000)
AGE = st.number_input("Age", value=30)
BILL_AMT1 = st.number_input("Last Bill Amount", value=5000)
PAY_AMT1 = st.number_input("Last Payment Amount", value=2000)

# ⚠ IMPORTANT :
# L'ordre des features doit être EXACTEMENT le même que dans le dataset
# Ici exemple simplifié → à adapter si nécessaire

if st.button("Predict"):

    # ⚠️ Tu dois mettre toutes les variables du dataset dans le bon ordre
    features = [
        LIMIT_BAL,
        1,  # SEX (placeholder)
        2,  # EDUCATION
        1,  # MARRIAGE
        AGE,
        0, 0, 0, 0, 0, 0,  # PAY_0 → PAY_6
        BILL_AMT1, 0, 0, 0, 0, 0,
        PAY_AMT1, 0, 0, 0, 0, 0
    ]

    probability = predict(features)

    st.success(f"Default Probability: {probability:.2f}")

    if probability > 0.5:
        st.error("⚠ High Risk Client")
    else:
        st.success("✅ Low Risk Client")
