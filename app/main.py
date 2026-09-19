import os

import joblib
import numpy as np
import xgboost as xgb
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


def row_from_request(payload: dict, feature_names: list[str]) -> np.ndarray:
    return np.array([[payload.get(col, 0) for col in feature_names]], dtype=float)


@app.get("/")
def health_check():
    model = get_model()
    names = model.get_booster().feature_names or []
    return {
        "status": "ok",
        "model": "xgb_model.pkl",
        "model_auc": 0.773,
        "n_features": len(names),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        payload = request.model_dump()
        booster = model.get_booster()
        names = booster.feature_names or list(payload.keys())
        data = row_from_request(payload, names)
        dmat = xgb.DMatrix(data, feature_names=names)
        prob = float(booster.predict(dmat)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model predict failed: {exc}") from exc

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
