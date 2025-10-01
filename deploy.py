import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
model = joblib.load('voting_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the original data to fit LabelEncoder
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop(['customerID'], axis=1)
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# Fit LabelEncoder for each categorical column
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Streamlit app
st.title('Customer Churn Prediction')
st.write('Enter customer details to predict churn:')

input_data = {}
input_data['gender'] = st.selectbox('Gender', options=label_encoders['gender'].classes_)
input_data['SeniorCitizen'] = st.selectbox('Senior Citizen', options=[0, 1])
input_data['Partner'] = st.selectbox('Partner', options=label_encoders['Partner'].classes_)
input_data['Dependents'] = st.selectbox('Dependents', options=label_encoders['Dependents'].classes_)
input_data['tenure'] = st.slider('Tenure (in months)', min_value=0, max_value=72, value=1, step=1)
input_data['PhoneService'] = st.selectbox('Phone Service', options=label_encoders['PhoneService'].classes_)
input_data['MultipleLines'] = st.selectbox('Multiple Lines', options=label_encoders['MultipleLines'].classes_)
input_data['InternetService'] = st.selectbox('Internet Service', options=label_encoders['InternetService'].classes_)
input_data['OnlineSecurity'] = st.selectbox('Online Security', options=label_encoders['OnlineSecurity'].classes_)
input_data['OnlineBackup'] = st.selectbox('Online Backup', options=label_encoders['OnlineBackup'].classes_)
input_data['DeviceProtection'] = st.selectbox('Device Protection', options=label_encoders['DeviceProtection'].classes_)
input_data['TechSupport'] = st.selectbox('Tech Support', options=label_encoders['TechSupport'].classes_)
input_data['StreamingTV'] = st.selectbox('Streaming TV', options=label_encoders['StreamingTV'].classes_)
input_data['StreamingMovies'] = st.selectbox('Streaming Movies', options=label_encoders['StreamingMovies'].classes_)
input_data['Contract'] = st.selectbox('Contract', options=label_encoders['Contract'].classes_)
input_data['PaperlessBilling'] = st.selectbox('Paperless Billing', options=label_encoders['PaperlessBilling'].classes_)
input_data['PaymentMethod'] = st.selectbox('Payment Method', options=label_encoders['PaymentMethod'].classes_)
input_data['MonthlyCharges'] = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=70.0, step=0.1)
input_data['TotalCharges'] = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=200.0, step=0.1)

input_df = pd.DataFrame([input_data])

# Apply label encoding for categorical columns
for col in categorical_cols:
    input_df[col] = label_encoders[col].transform([input_df[col][0]])

# Scale numeric columns
input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

if st.button('Predict Churn'):
    prediction = model.predict(input_df)
    churn_status = 'Churn' if prediction[0] == 1 else 'No Churn'
    st.write(f'The predicted customer status is: **{churn_status}**')
