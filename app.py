import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("📱 Customer Churn Prediction App")

# Inputs from user
gender = st.selectbox("Gender", ['Female', 'Male'])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner?", ['Yes', 'No'])
Dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
Contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
PaymentMethod = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 3000.0)

# Encoding same as training
mapper = {
    'Female': 0, 'Male': 1,
    'Yes': 1, 'No': 0,
    'No phone service': 2,
    'No internet service': 2,
    'DSL': 0, 'Fiber optic': 1, 'No': 2,
    'Month-to-month': 0, 'One year': 1, 'Two year': 2,
    'Electronic check': 0, 'Mailed check': 1,
    'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
}

input_data = [
    mapper[gender],
    SeniorCitizen,
    mapper[Partner],
    mapper[Dependents],
    tenure,
    mapper[PhoneService],
    mapper[MultipleLines],
    mapper[InternetService],
    mapper[OnlineSecurity],
    mapper[OnlineBackup],
    mapper[DeviceProtection],
    mapper[TechSupport],
    mapper[StreamingTV],
    mapper[StreamingMovies],
    mapper[Contract],
    mapper[PaperlessBilling],
    mapper[PaymentMethod],
    MonthlyCharges,
    TotalCharges
]

# Final missing feature padding (20th feature to match model)
# Double-check if your model had 'customerID' or any other column while training — if yes, remove it from training and retrain model.
# If not, this list is 19 — so add a dummy 0 to fix temp mismatch
# ⚠️ BUT ideally: retrain your model on 19 correct features. This dummy fix below just avoids the crash.

# input_data.append(0)  # <— Uncomment only if you're stuck and want a temp fix (Not ideal)

# Predict
if st.button("Predict Churn"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    if prediction == 1:
        st.warning("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
