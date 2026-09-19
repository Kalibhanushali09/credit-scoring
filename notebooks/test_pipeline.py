import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np
import pandas as pd

from src.data.loader import (
    load_bureau,
    load_bureau_bal_app,
    load_credit_card_bal_app,
    load_installments,
    load_path,
    load_previous_app,
)
from src.explain.shap_explainer import explain_model
from src.features.sequence_features import (
    build_bureau_bal_app_features,
    build_bureau_features,
    build_credit_card_bal_app_features,
    build_installment_features,
    build_previous_app_features,
    build_sequences,
)
from src.models.baseline import baseline_model
from src.models.sequence_model import generate_embeddings
from src.models.train_xgb import train_xgb
from src.monitoring.drift_monitor import detect_drift

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

X, y = load_path("data/raw/application_train.csv")
bureau_df = load_bureau("data/raw/bureau.csv")
installments_df = load_installments("data/raw/installments_payments.csv")
prev_df = load_previous_app("data/raw/previous_application.csv")
credits_df = load_credit_card_bal_app("data/raw/credit_card_balance.csv")
bureau_bal_df = load_bureau_bal_app("data/raw/bureau_balance.csv")

sample_rows = os.environ.get("SAMPLE_ROWS")
if sample_rows:
    n = int(sample_rows)
    X = X.iloc[:n].copy()
    y = y.iloc[:n].copy()
    keep_ids = set(X["SK_ID_CURR"])
    bureau_df = bureau_df[bureau_df["SK_ID_CURR"].isin(keep_ids)]
    installments_df = installments_df[installments_df["SK_ID_CURR"].isin(keep_ids)]
    prev_df = prev_df[prev_df["SK_ID_CURR"].isin(keep_ids)]
    credits_df = credits_df[credits_df["SK_ID_CURR"].isin(keep_ids)]
    keep_bureau = set(bureau_df["SK_ID_BUREAU"])
    bureau_bal_df = bureau_bal_df[bureau_bal_df["SK_ID_BUREAU"].isin(keep_bureau)]

sequences, ids = build_sequences(installments_df)
embeddings = generate_embeddings(sequences)

embedding_cols = [f"LSTM_{i}" for i in range(embeddings.shape[1])]
embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)
embeddings_df["SK_ID_CURR"] = ids

print(X.shape)
print(y.value_counts())

_, baseline_auc = baseline_model(X, y)

bureau_agg = build_bureau_features(bureau_df)
installment_agg = build_installment_features(installments_df)
prev_agg = build_previous_app_features(prev_df)
cc_agg = build_credit_card_bal_app_features(credits_df)
bb_agg = build_bureau_bal_app_features(bureau_bal_df)
bureau_with_bal = bureau_df[["SK_ID_CURR", "SK_ID_BUREAU"]].merge(
    bb_agg, on="SK_ID_BUREAU", how="left"
)

bureau_bal_curr = (
    bureau_with_bal.groupby("SK_ID_CURR")
    .agg(
        {
            "MONTHS_BALANCE_count": ["sum"],
            "STATUS_OVERDUE_sum": ["sum"],
            "STATUS_OVERDUE_mean": ["mean"],
        }
    )
    .reset_index()
)
bureau_bal_curr.columns = [
    "_".join(col).strip("_") for col in bureau_bal_curr.columns
]

X = X.merge(bureau_agg, on="SK_ID_CURR", how="left")
X = X.merge(installment_agg, on="SK_ID_CURR", how="left")
X = X.merge(prev_agg, on="SK_ID_CURR", how="left")
X = X.merge(cc_agg, on="SK_ID_CURR", how="left")
X = X.merge(bureau_bal_curr, on="SK_ID_CURR", how="left")
X = X.merge(embeddings_df, on="SK_ID_CURR", how="left")

X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

model, xgb_auc, X_train, X_test = train_xgb(X, y)
X_test.sample(min(500, len(X_test)), random_state=42).to_csv(
    "data/processed/X_test_sample.csv", index=False
)
explain_model(model, X_train, X_test)

print(f"Baseline AUC: {baseline_auc:.3f}")
print(f"XGBoost AUC: {xgb_auc:.3f}")
print(f"Lift: {xgb_auc - baseline_auc:.3f}")

joblib.dump(model, "models/xgb_model.pkl")
print("Model saved to models/xgb_model.pkl")

train_sample = X_train.sample(min(500, len(X_train)), random_state=42)
new_sample = X_test.sample(min(200, len(X_test)), random_state=42)

drift_results = detect_drift(train_sample, new_sample)
drifted_features = drift_results[drift_results["drifted"] == True]

print(f"\nDrift detected in {len(drifted_features)} features:")
print(drifted_features[["feature", "psi", "ks_pvalue"]].head(10))

drift_results.to_csv("data/processed/drift_report.csv", index=False)
print("Drift report saved.")
