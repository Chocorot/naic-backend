from pydantic import BaseModel
from typing import List

class PredictionResult(BaseModel):
    prediction: int
    probabilities: List[float]
    model_count: int
    confidence: float

class ErrorResponse(BaseModel):
    detail: str
