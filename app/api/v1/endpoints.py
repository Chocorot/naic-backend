from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
import logging
from app.schemas.prediction import PredictionResult
from app.models.classifier import ensemble
from app.utils.image import process_image

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    Industry standard prediction endpoint.
    1. Validates file type.
    2. Preprocesses image.
    3. Runs ensemble inference.
    4. Returns formatted JSON matching frontend expectations.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image bytes
        content = await file.read()
        
        # Preprocess
        input_tensor = process_image(content)
        
        # Inference
        probs_tensor = ensemble.predict(input_tensor)
        
        # Format results
        probs = probs_tensor[0].tolist()
        prediction = int(torch.argmax(probs_tensor, dim=1).item())
        confidence = float(probs[prediction])
        
        return PredictionResult(
            prediction=prediction,
            probabilities=probs,
            model_count=len(ensemble.models),
            confidence=round(confidence, 4)
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during inference.")
