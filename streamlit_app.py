import streamlit as st
import httpx
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #6d5dfc;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #5b4cfc;
        border: 1px solid white;
    }
    .predict-box {
        padding: 20px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #6d5dfc !important;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("📡 Telco Customer Churn Prediction Dashboard")
st.markdown("Enter customer details below to predict the likelihood of churn.")

# Input Form
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographic")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    st.subheader("📶 Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("💳 Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure)

# Prediction Button
if st.button("🔍 Predict Churn Risk"):
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    with st.spinner("Analyzing..."):
        try:
            # Call FastAPI
            response = httpx.post("http://localhost:8000/predict", json=input_data, timeout=10.0)
            if response.status_code == 200:
                result = response.json()
                churn = result['churn']
                prob = result['probability']

                # Display Results
                st.markdown('<div class="predict-box">', unsafe_allow_html=True)
                if churn == "Yes":
                    st.error(f"🔴 **HIGH RISK**: This customer is likely to CHURN.")
                else:
                    st.success(f"🟢 **LOW RISK**: This customer is likely to STAY.")
                
                st.write(f"Confidence Level: **{prob*100:.2f}%**")
                st.progress(prob)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")
            st.info("Make sure the FastAPI server is running on http://localhost:8000")
