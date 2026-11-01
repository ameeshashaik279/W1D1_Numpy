
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from preprocessing.pipeline import scaler

ROOT = Path(__file__).resolve().parent

model = joblib.load(ROOT / "models" / "tuned_model.pkl")

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

app = FastAPI(
    title="MLflow Experiment Tracking Platform",
    version="1.0.0"
)


class PredictRequest(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": type(model).__name__
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        missing = [col for col in FEATURES if col not in req.features]

        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing features: {missing}"
            )

        df = pd.DataFrame(
            [[req.features[col] for col in FEATURES]],
            columns=FEATURES
        )

        X = scaler.transform(df)
        prediction = model.predict(X)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(X).max())

        return {
            "prediction": prediction.item()
            if hasattr(prediction, "item") else prediction,
            "confidence": confidence
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
