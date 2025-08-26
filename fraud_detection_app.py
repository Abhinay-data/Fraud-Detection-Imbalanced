import streamlit as st
import pandas as pd
import joblib


model = joblib.load("fraud_detection_pipeline.pkl")
threshold = joblib.load("fraud_threshold.pkl")

st.title("Fraud Detection Prediction App")

st.markdown("Please enter the transaction details below and use the predict button")

st.divider()

transation_type = st.selectbox("Transaction Type",["PAYMENT","TRANSFER","CASH_OUT","CASH_IN","DEPOSIT"])
amount = st.number_input("Amount",min_value = 0.0, value = 1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)",min_value = 0.0, value = 10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)",min_value = 0.0, value = 9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)",min_value = 0.0, value = 0.0)
newbalanceDest = st.number_input("New Balance (Receiver)",min_value = 0.0, value = 0.0)


input_data = pd.DataFrame([{
    "type" : transation_type,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig":newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest":newbalanceDest
}])


if st.button("Predict Fraud"):
    proba = model.predict_proba(input_data)[:,1][0]
    prediction = 1 if proba >= threshold else 0

    st.write(f"**Fraud Probability:** {proba:.3f}")

    if prediction == 1:
        st.error("This transaction is likely FRAUD!")
    else:
        st.success("This transaction seems SAFE.")
