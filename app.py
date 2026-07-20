# 1 is good / lower risk
# 2 is bad / higher risk

import streamlit as st
import pandas as pd
import joblib

model = joblib.load('best_model.pkl')
encoder = {col : joblib.load(f"{col}_encoder.pkl") for col in ['Sex','Housing','Saving accounts','Checking account']}
st.title("Credit Risk Prediction")
st.write("Enter the details of the applicant to predict credit risk.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ['male', 'female'])
job = st.number_input("Job", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ['own', 'rent', 'free'])
saving_accounts = st.selectbox("Saving accounts", ['little', 'moderate', 'rich', 'quite rich'])
checking_account = st.selectbox("Checking account", ['little', 'moderate', 'rich'])
credit_amount = st.number_input("Credit amount", min_value=0, value=1000)
duration = st.number_input("Duration", min_value=1, value=12)

input_df = pd.DataFrame({
    'Age': [age],
    'Sex': [encoder['Sex'].transform([sex])[0]],
    'Job': [job],
    'Housing': [encoder['Housing'].transform([housing])[0]],
    'Saving accounts': [encoder['Saving accounts'].transform([saving_accounts])[0]],
    'Checking account': [encoder['Checking account'].transform([checking_account])[0]],
    'Credit amount': [credit_amount],
    'Duration': [duration]
})

if st.button('Predict Risk'):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success('The predicted credit risk is good')
    else:
        st.error('The predict credit is bad')

