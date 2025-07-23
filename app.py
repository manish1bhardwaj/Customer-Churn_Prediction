import streamlit as st
import pandas as pd
import pickle

# Load the model and encoders
with open("logistic_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data['model']
encoders = model_data['encoders']
features = model_data['features']

st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details below to predict churn:")

# --- INPUTS FROM USER ---

# Define all input fields (match raw categorical values)
input_dict = {
    "gender": st.selectbox("Gender", ["Male", "Female"]),
    "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
    "Partner": st.selectbox("Partner", ["Yes", "No"]),
    "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
    "tenure": st.number_input("Tenure (months)", min_value=0, max_value=72, value=12),
    "PhoneService": st.selectbox("Phone Service", ["Yes", "No"]),
    "MultipleLines": st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"]),
    "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": st.selectbox("Online Security", ["No internet service", "No", "Yes"]),
    "OnlineBackup": st.selectbox("Online Backup", ["No internet service", "No", "Yes"]),
    "DeviceProtection": st.selectbox("Device Protection", ["No internet service", "No", "Yes"]),
    "TechSupport": st.selectbox("Tech Support", ["No internet service", "No", "Yes"]),
    "StreamingTV": st.selectbox("Streaming TV", ["No internet service", "No", "Yes"]),
    "StreamingMovies": st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"]),
    "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": st.selectbox("Paperless Billing", ["Yes", "No"]),
    "PaymentMethod": st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]),
    "MonthlyCharges": st.number_input("Monthly Charges", min_value=0.0, value=70.0),
    "TotalCharges": st.number_input("Total Charges", min_value=0.0, value=800.0),
}

# --- ENCODING & FORMATTING INPUT ---

# Encode categorical values using saved LabelEncoders
encoded_input = {}
for col, val in input_dict.items():
    if col in encoders:
        encoded_input[col] = encoders[col].transform([val])[0]
    else:
        encoded_input[col] = val  # Numeric fields remain as is

# Format as dataframe in same column order as training
input_df = pd.DataFrame([[encoded_input[col] for col in features]], columns=features)

# --- PREDICT BUTTON ---
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
