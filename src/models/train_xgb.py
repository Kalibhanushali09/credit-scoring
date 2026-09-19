import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _drop_id_columns(X):
    id_cols = [col for col in X.columns if str(col).startswith("SK_ID")]
    return X.drop(columns=id_cols, errors="ignore")


def _uri_to_local_path(uri: str) -> str:
    path = uri
    if path.startswith("file:"):
        path = path[5:]
        while path.startswith("/") and not os.path.exists(path) and len(path) > 3 and path[2] == ":":
            path = path.lstrip("/")
    return path


def _artifact_location_writable(uri: str) -> bool:
    if not uri:
        return False
    if uri.startswith("http://") or uri.startswith("https://") or uri.startswith("s3://"):
        return True
    path = _uri_to_local_path(uri)
    try:
        os.makedirs(path, exist_ok=True)
        return os.access(path, os.W_OK)
    except OSError:
        return False


def _configure_mlflow():
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "file:" + os.path.abspath("mlruns"),
    )
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "credit_scoring")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment and not _artifact_location_writable(experiment.artifact_location):
        client = mlflow.MlflowClient()
        client.rename_experiment(experiment.experiment_id, f"{experiment_name}_legacy")
        experiment = None
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)


def train_xgb(X, y):
    _configure_mlflow()

    X = _drop_id_columns(X)

    with mlflow.start_run():
        # shuffle=False keeps a later-period holdout for drift-style evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = (n_neg / n_pos) if n_pos else 1.0

        model = xgb.XGBClassifier(
            n_estimators=500,
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="auc",
            early_stopping_rounds=20,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        print("XGBoost AUC", test_auc)

        mlflow.log_param("n_estimators", 500)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))
        mlflow.log_metric("auc", test_auc)
        try:
            mlflow.xgboost.log_model(model, name="xgb_model")
        except TypeError:
            mlflow.xgboost.log_model(model, artifact_path="xgb_model")
        except Exception as exc:
            print(f"MLflow model artifact skipped: {exc}")

        return model, test_auc, X_train, X_test