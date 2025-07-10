import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AML Fraud Detection", layout="centered")

st.title("💸 AML Fraud Detection Live Testing")
st.write("Upload your transaction data below to check for potential fraudulent transactions using the trained model.")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("aml_fraud_detection_rf_model.pkl")
    except ModuleNotFoundError as e:
        st.warning(f"Could not load the pre-trained model due to an error: {e}. Using a dummy model instead.")
        # Create a dummy model that always predicts 0 (not fraud)
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)
        return DummyModel()

model = load_model()

# Upload CSV/Excel
uploaded_file = st.file_uploader("Upload Transaction CSV/Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully. Processing...")

    # High-risk countries
    high_risk_countries = ["Pakistan", "Afghanistan", "Syria", "North Korea"]

    # Generate receiver_country if not provided (mock India 85%, others 5%)
    if "receiver_country" not in df.columns:
        np.random.seed(42)
        df["receiver_country"] = np.random.choice(
            ["India", "Pakistan", "Syria", "Afghanistan"],
            size=len(df),
            p=[0.85, 0.05, 0.05, 0.05]
        )

    # Feature Engineering
    df["is_high_risk_country"] = df["receiver_country"].isin(high_risk_countries).astype(int)
    df["is_cross_city"] = (df["sender_location"] != df["receiver_location"]).astype(int)

    # Approximate weekly average expense
    user_weekly_avg = df.groupby("sender_account")["amount"].transform("mean") * 0.1
    df["is_large_transaction"] = (df["amount"] > 3 * user_weekly_avg).astype(int)

    # Prepare feature set
    features = ["is_high_risk_country", "is_cross_city", "is_large_transaction", "amount"]
    X = df[features]

    # Predict fraud
    df["predicted_is_fraud"] = model.predict(X)

    # Display output
    st.subheader("📊 Prediction Results")
    st.dataframe(df[["sender_account", "receiver_account", "amount", "sender_location", "receiver_location", "consent_status", "receiver_country", "is_high_risk_country", "is_cross_city", "is_large_transaction", "predicted_is_fraud"]])

    # Download predictions
    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df(df)
    st.download_button(
        "📥 Download Predictions CSV",
        csv,
        "aml_fraud_predictions.csv",
        "text/csv"
    )

    st.success("✅ Predictions complete. Review flagged transactions in the table above.")

else:
    st.info("📂 Please upload your transaction file to begin.")