from pydantic import BaseModel
from typing import List

class PredictionResult(BaseModel):
    predicted_class: int               # 0-4
    predicted_label: str               # human-readable class name
    probabilities: List[float]         # length 5, sums to 1.0
    model_count: int                   # number of models used (10 for dual ensemble)
    confidence: float                  # max probability
    ensemble_strategy: str             # "simple_avg", etc.

class ErrorResponse(BaseModel):
    detail: str
