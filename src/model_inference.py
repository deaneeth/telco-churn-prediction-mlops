from typing import Any, Dict
import pandas as pd
import numpy as np
from joblib import load

class InferenceService:
    def __init__(self, pipeline_path: str):
        self.pipeline = load(pipeline_path)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # input validation could be added here (schema checks)
        return self.pipeline.predict_proba(X)[:,1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
