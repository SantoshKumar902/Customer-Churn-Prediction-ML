import joblib
import streamlit as st
import joblib
import numpy as np

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Predictor")
st.write("Fill in customer details to predict churn risk!")

# Input fields
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

# Encode inputs
def encode(val): return 1 if val == "Yes" else 0

senior = encode(senior_citizen)
part = encode(partner)
dep = encode(dependents)
phone = encode(phone_service)
paper = encode(paperless_billing)

contract_one = 1 if contract == "One year" else 0
contract_two = 1 if contract == "Two year" else 0
internet_fiber = 1 if internet_service == "Fiber optic" else 0
internet_no = 1 if internet_service == "No" else 0
pay_credit = 1 if payment_method == "Credit card (automatic)" else 0
pay_echeck = 1 if payment_method == "Electronic check" else 0
pay_mailed = 1 if payment_method == "Mailed check" else 0
sec_yes = 1 if online_security == "Yes" else 0
sec_no_int = 1 if online_security == "No internet service" else 0
tech_yes = 1 if tech_support == "Yes" else 0
tech_no_int = 1 if tech_support == "No internet service" else 0
lines_yes = 1 if multiple_lines == "Yes" else 0
lines_no_phone = 1 if multiple_lines == "No phone service" else 0

# Feature engineering
tenure_group = 0
if tenure <= 12: tenure_group = 0
elif tenure <= 24: tenure_group = 1
elif tenure <= 48: tenure_group = 2
elif tenure <= 60: tenure_group = 3
else: tenure_group = 4

high_monthly = 1 if monthly_charges > 65 else 0
high_charge = 1 if monthly_charges > 65 else 0

# Scale numerical features
scaled = scaler.transform([[tenure, monthly_charges, total_charges]])

# Build feature array (34 features)
features = np.array([[
    0,  # gender (neutral)
    senior, part, dep,
    scaled[0][0],  # tenure scaled
    phone, paper,
    scaled[0][1],  # monthly charges scaled
    scaled[0][2],  # total charges scaled
    0,  # churn placeholder
    tenure_group, high_monthly, high_charge,
    lines_no_phone, lines_yes,
    internet_fiber, internet_no,
    sec_no_int, sec_yes,
    0, 0,  # OnlineBackup
    0, 0,  # DeviceProtection
    tech_no_int, tech_yes,
    0, 0,  # StreamingTV
    0, 0,  # StreamingMovies
    contract_one, contract_two,
    pay_credit, pay_echeck, pay_mailed
]])

# Predict
if st.button("🔍 Predict Churn"):
    proba = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Churn Risk! ({proba*100:.1f}% probability)")
        st.write("**Recommendations:**")
        st.write("- Offer long-term contract discount")
        st.write("- Provide personalized retention offer")
        st.write("- Assign dedicated customer support")
    else:
        st.success(f"✅ Low Churn Risk ({proba*100:.1f}% probability)")
        st.write("Customer is likely to stay!")
