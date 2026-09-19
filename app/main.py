import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictRequest, PredictResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "xgb_model.pkl")
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def align_features(data: pd.DataFrame, model) -> pd.DataFrame:
    expected_cols = model.get_booster().feature_names
    if not expected_cols:
        return data
    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0
    return data[expected_cols]


@app.get("/")
def health_check():
    model = get_model()
    n_features = len(model.get_booster().feature_names or [])
    return {
        "status": "ok",
        "model": "xgb_model.pkl",
        "model_auc": 0.773,
        "n_features": n_features,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    data = pd.DataFrame([request.model_dump()])
    data = align_features(data, model)
    prob = model.predict_proba(data)[0][1]

    if prob < 0.2:
        category = "LOW"
    elif prob < 0.5:
        category = "MEDIUM"
    else:
        category = "HIGH"

    return PredictResponse(
        default_probability=round(float(prob), 4),
        risk_category=category,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
