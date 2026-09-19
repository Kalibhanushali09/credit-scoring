import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import shap
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_PATH = os.path.join(ROOT_DIR, "data", "processed", "X_test_sample.csv")
DEFAULT_API_URL = "https://credit-scoring-api-vert.vercel.app"


def get_api_url() -> str:
    try:
        secret_url = st.secrets.get("API_URL")
    except Exception:
        secret_url = None
    return str(secret_url or os.getenv("API_URL", DEFAULT_API_URL)).rstrip("/")


API_URL = get_api_url()


def call_api(path: str, method: str = "GET", payload: dict | None = None, timeout: int = 30):
    url = f"{API_URL}{path}"
    if method == "GET":
        return requests.get(url, timeout=timeout)
    return requests.post(url, json=payload, timeout=timeout)


st.title("Credit Risk Scoring Dashboard")
st.markdown("**Behavioral Risk Modeling for Thin-File Borrowers**")
st.metric("Model AUC", "0.773")

api_ok = False
api_info = {}
try:
    health = call_api("/", timeout=8)
    api_ok = health.ok
    if api_ok:
        api_info = health.json()
        st.sidebar.success(f"API connected: {API_URL}")
        if api_info.get("model"):
            st.sidebar.caption(f"Serving trained model: {api_info['model']}")
    else:
        st.sidebar.error(f"API error {health.status_code}: {API_URL}")
except requests.RequestException as exc:
    st.sidebar.error(f"API unreachable: {API_URL}")
    st.sidebar.caption(str(exc))

st.header("Single Borrower Prediction")
st.caption("Scores are requested from the FastAPI service, which loads models/xgb_model.pkl.")

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
    if not api_ok:
        st.error("Prediction is API-only. The scoring API is not available.")
    else:
        payload = {
            "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_3": ext_source_3,
            "AMT_CREDIT": amt_credit,
            "AMT_INCOME_TOTAL": amt_income,
            "DAYS_BIRTH": int(days_birth),
            "DAYS_EMPLOYED": int(days_employed),
        }
        try:
            response = call_api("/predict", method="POST", payload=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            st.metric("Default Probability", f"{result['default_probability']:.1%}")
            category = result["risk_category"]
            if category == "LOW":
                st.success("LOW risk")
            elif category == "MEDIUM":
                st.warning("MEDIUM risk")
            else:
                st.error("HIGH risk")
            st.caption("Prediction returned by FastAPI from the trained XGBoost model.")
        except requests.RequestException as exc:
            st.error(f"API prediction failed: {exc}")

st.header("Global Feature Importance")
st.caption("SHAP uses the same trained pickle the API serves.")

if st.button("Generate SHAP Plot"):
    if not api_ok:
        st.error("API must be available before generating explanations.")
    else:
        import joblib

        model = joblib.load(os.path.join(ROOT_DIR, "models", "xgb_model.pkl"))
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
