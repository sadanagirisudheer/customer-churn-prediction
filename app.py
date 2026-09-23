import streamlit as st
import pandas as pd
import joblib

# Load the deployment bundle (model + preprocessing tools, all in one file)
bundle = joblib.load('churn_deployment_bundle.pkl')
model = bundle['model']
scaler = bundle['scaler']
label_encoders = bundle['label_encoders']
feature_columns = bundle['feature_columns']
num_cols = bundle['num_cols']
multi_cols = bundle['multi_cols']
binary_cols = bundle['binary_cols']

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

# --- Collect raw user input (same fields as original dataset) ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                               "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)

if st.button("Predict Churn"):
    # Build a single-row dataframe matching the ORIGINAL raw column structure
    raw = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }])

    # Apply the SAME preprocessing steps used during training
    for col in binary_cols:
        raw[col] = label_encoders[col].transform(raw[col])

    raw_encoded = pd.get_dummies(raw, columns=multi_cols)
    # Align columns with training data (add any missing dummy columns as 0)
    for col in feature_columns:
        if col not in raw_encoded.columns:
            raw_encoded[col] = 0
    raw_encoded = raw_encoded[feature_columns]  # exact column order

    raw_encoded[num_cols] = scaler.transform(raw_encoded[num_cols])

    prediction = model.predict(raw_encoded)[0]
    probability = model.predict_proba(raw_encoded)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — Probability: {probability:.1%}")
    else:
        st.success(f"✅ Low Churn Risk — Probability: {probability:.1%}")
    st.progress(float(probability))