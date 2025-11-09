# Create a new file: app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load the trained model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load('optimized_churn_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor


model, preprocessor = load_model()

# Feature names after preprocessing (from our training)
feature_names = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
    'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check', 'TenureGroup_1-2 Years', 'TenureGroup_2-3 Years',
    'TenureGroup_3-4 Years', 'TenureGroup_4-5 Years', 'TenureGroup_5+ Years'
]

# App title and description
st.title("🏢 Telecom Customer Churn Prediction")
st.markdown("""
This app predicts the likelihood of a customer leaving the service (churning) 
based on their profile and usage patterns. Use the sidebar to input customer details.
""")

# Sidebar for user input
st.sidebar.header("📋 Customer Information")

# Customer demographics
st.sidebar.subheader("👤 Demographics")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

# Account information
st.sidebar.subheader("💳 Account Details")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Service information
st.sidebar.subheader("📡 Services")
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])

# Charges
st.sidebar.subheader("💰 Charges")
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 2000.0)

# Calculate total services
total_services = sum([
    1 if phone_service == "Yes" else 0,
    1 if multiple_lines == "Yes" else 0,
    1 if online_security == "Yes" else 0,
    1 if online_backup == "Yes" else 0,
    1 if device_protection == "Yes" else 0,
    1 if tech_support == "Yes" else 0,
    1 if streaming_tv == "Yes" else 0,
    1 if streaming_movies == "Yes" else 0
])


# Determine tenure group
def get_tenure_group(tenure):
    if tenure <= 12:
        return '0-1 Year'
    elif tenure <= 24:
        return '1-2 Years'
    elif tenure <= 36:
        return '2-3 Years'
    elif tenure <= 48:
        return '3-4 Years'
    elif tenure <= 60:
        return '4-5 Years'
    else:
        return '5+ Years'


tenure_group = get_tenure_group(tenure)

# Create input dataframe
input_data = pd.DataFrame({
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'TotalServices': [total_services],
    'gender': [gender],
    'Partner': [partner],
    'Dependents': [dependents],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'TenureGroup': [tenure_group]
})

# Display input summary
st.header("📊 Customer Profile Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Tenure", f"{tenure} months")
    st.metric("Monthly Charges", f"${monthly_charges:.2f}")
    st.metric("Total Services", total_services)

with col2:
    st.metric("Contract", contract)
    st.metric("Payment Method", payment_method)
    st.metric("Internet Service", internet_service)

with col3:
    st.metric("Senior Citizen", senior_citizen)
    st.metric("Paperless Billing", paperless_billing)
    st.metric("Total Charges", f"${total_charges:.2f}")

# Make prediction
if st.button("🔮 Predict Churn Risk", type="primary"):
    # Preprocess input
    processed_input = preprocessor.transform(input_data)

    # Make prediction
    churn_probability = model.predict_proba(processed_input)[0, 1]
    churn_prediction = model.predict(processed_input)[0]

    # Display results
    st.header("🎯 Prediction Results")

    # Create columns for results
    result_col1, result_col2 = st.columns(2)

    with result_col1:
        # Probability gauge
        st.subheader("Churn Probability")
        st.progress(float(churn_probability))
        st.metric("Risk Score", f"{churn_probability:.1%}")

        # Prediction
        if churn_prediction == 1:
            st.error("🚨 HIGH RISK: Customer likely to churn")
        else:
            st.success("✅ LOW RISK: Customer likely to stay")

    with result_col2:
        # Risk factors analysis
        st.subheader("🔍 Key Risk Factors")

        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append("📝 Month-to-month contract")
        if payment_method == "Electronic check":
            risk_factors.append("💳 Electronic check payment")
        if internet_service == "Fiber optic":
            risk_factors.append("📡 Fiber optic internet")
        if tenure < 12:
            risk_factors.append("🆕 New customer (<1 year)")
        if total_services < 2:
            risk_factors.append("🔧 Few additional services")

        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.write("✅ No major risk factors identified")

    # Recommendations
    st.header("💡 Retention Recommendations")

    if churn_prediction == 1:
        st.warning("This customer is at high risk of churning. Consider:")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            if contract == "Month-to-month":
                st.write("• **Offer contract upgrade** to 1-year with discount")
            if tenure < 12:
                st.write("• **New customer onboarding** call to improve experience")
            if total_services < 2:
                st.write("• **Bundle promotion** for additional services")

        with rec_col2:
            if payment_method == "Electronic check":
                st.write("• **Payment method incentive** for auto-pay setup")
            st.write("• **Loyalty discount** of 10-15% for 6 months")
            st.write("• **Proactive support** call to address issues")
    else:
        st.info("This customer has low churn risk. Focus on:")
        st.write("• **Upsell opportunities** for premium services")
        st.write("• **Referral program** enrollment")
        st.write("• **Satisfaction survey** to maintain high service quality")

    # Feature importance explanation
    st.header("📈 Churn Drivers Analysis")

    # Get feature contributions
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })

    # Map back to original features for interpretation
    top_features = feature_importance.reindex(
        feature_importance.coefficient.abs().sort_values(ascending=False).index).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if coef > 0 else 'green' for coef in top_features['coefficient']]
    ax.barh(range(len(top_features)), top_features['coefficient'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Impact on Churn Probability')
    ax.set_title('Top Factors Influencing Churn Prediction')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**About this model**: 
- **Accuracy**: 72.8% 
- **Recall**: 81.0% (catches 8/10 churners)
- **Precision**: 49.3% 
- **Business Impact**: Identifies at-risk customers for proactive retention
""")