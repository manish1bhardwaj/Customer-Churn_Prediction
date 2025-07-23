# 📊 Customer Churn Prediction - Streamlit Web App

This project predicts whether a telecom customer is likely to churn based on their service usage and contract details. The model is trained using Logistic Regression and deployed as an interactive web app using Streamlit.

---

## 🚀 Live Demo

🔗 [Try the App on Streamlit](https://manish1bhardwaj-customer-churn-prediction.streamlit.app/)

---

## 🧠 Problem Statement

Customer churn is a major challenge in the telecom industry. This project helps predict which customers are at risk of leaving so the business can take proactive retention steps.

---

## 📁 Dataset Overview

The dataset includes the following key features:

- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `tenure`, `PhoneService`, `MultipleLines`, `InternetService`
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`
- `StreamingTV`, `StreamingMovies`
- `Contract`, `PaperlessBilling`, `PaymentMethod`
- `MonthlyCharges`, `TotalCharges`
- **Target**: `Churn` (Yes/No)

---

## ⚙️ ML Model Details

- **Model**: Logistic Regression (`scikit-learn`)
- **Preprocessing**:
  - Label Encoding for categorical features
  - Missing value handling
  - Feature scaling if required
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 🖥️ Features of the Web App

- Simple UI to input customer data
- Predicts churn in real-time
- Displays result as:
  - ✅ Customer is likely to stay
  - ⚠️ Customer is likely to churn

---

## ▶️ Run Locally

Clone this repository and run the app locally using Streamlit:

```bash
git clone https://github.com/manish1bhardwaj/Customer-Churn_Prediction.git
cd Customer-Churn_Prediction
pip install -r requirements.txt
streamlit run app.py
