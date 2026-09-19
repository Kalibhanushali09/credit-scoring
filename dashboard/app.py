import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import shap
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "xgb_model.pkl")
SAMPLE_PATH = os.path.join(ROOT_DIR, "data", "processed", "X_test_sample.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def score_payload(payload: dict) -> tuple[float, str]:
    model = load_model()
    data = pd.DataFrame([payload])
    expected_cols = model.get_booster().feature_names
    if expected_cols:
        for col in expected_cols:
            if col not in data.columns:
                data[col] = 0
        data = data[expected_cols]
    prob = float(model.predict_proba(data)[0][1])
    if prob < 0.2:
        category = "LOW"
    elif prob < 0.5:
        category = "MEDIUM"
    else:
        category = "HIGH"
    return round(prob, 4), category


def show_result(prob: float, category: str) -> None:
    st.metric("Default Probability", f"{prob:.1%}")
    if category == "LOW":
        st.success("LOW risk")
    elif category == "MEDIUM":
        st.warning("MEDIUM risk")
    else:
        st.error("HIGH risk")


st.title("Credit Risk Scoring Dashboard")
st.markdown("**Behavioral Risk Modeling for Thin-File Borrowers**")
st.metric("Model AUC", "0.773")

api_ok = False
try:
    health = requests.get(f"{API_URL}/", timeout=3)
    api_ok = health.ok
    if api_ok:
        st.sidebar.success(f"API connected: {API_URL}")
    else:
        st.sidebar.warning("API unavailable; scoring with local model")
except requests.RequestException:
    st.sidebar.warning("API unavailable; scoring with local model")

st.header("Single Borrower Prediction")

col1, col2 = st.columns(2)

with col1:
    ext_source_2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5)
    ext_source_3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5)
    amt_credit = st.number_input("AMT_CREDIT", value=500000)

with col2:
    amt_income = st.number_input("AMT_INCOME_TOTAL", value=150000)
    days_birth = st.number_input("DAYS_BIRTH", value=-12000)
    days_employed = st.number_input("DAYS_EMPLOYED", value=-2000)

if st.button("Predict"):
    payload = {
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "AMT_CREDIT": amt_credit,
        "AMT_INCOME_TOTAL": amt_income,
        "DAYS_BIRTH": int(days_birth),
        "DAYS_EMPLOYED": int(days_employed),
    }
    if api_ok:
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            show_result(result["default_probability"], result["risk_category"])
        except requests.RequestException:
            st.warning("API request failed; using local model")
            prob, category = score_payload(payload)
            show_result(prob, category)
    else:
        prob, category = score_payload(payload)
        show_result(prob, category)

st.header("Global Feature Importance")

if st.button("Generate SHAP Plot"):
    model = load_model()
    X_sample = pd.read_csv(SAMPLE_PATH)
    expected_cols = model.get_booster().feature_names
    if expected_cols:
        for col in expected_cols:
            if col not in X_sample.columns:
                X_sample[col] = 0
        X_sample = X_sample[expected_cols]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    fig, _ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig)
    plt.close(fig)
