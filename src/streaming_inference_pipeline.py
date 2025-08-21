from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from src.model_inference import InferenceService

app = FastAPI()
inference = InferenceService("/mnt/data/rf_best_pipeline.joblib")  # update path

class SingleRequest(BaseModel):
    features: Dict[str, float]   # simple; for production use strict schema

@app.post("/predict")
def predict(request: SingleRequest, threshold: float = 0.5):
    df = pd.DataFrame([request.features])
    probs = inference.predict_proba(df)
    preds = (probs >= threshold).astype(int)
    return {"probability": float(probs[0]), "prediction": int(preds[0])}

## can Run via uvicorn streaming_inference_pipeline:app --reload --host 0.0.0.0 --port 8000