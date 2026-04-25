from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
import logging
from app.schemas.prediction import PredictionResult
from app.models.classifier import ensemble
from app.utils.image import process_image
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    Accepts a fundus image, returns the ensembled prediction across
    10 models (5 ConvNeXt folds + 5 EfficientNet folds).

    Response format:
        {
            "predicted_class": int,           # 0-4
            "predicted_label": str,           # human-readable class name
            "probabilities": list[float],     # length 5, sums to 1.0
            "model_count": int,               # 10
            "confidence": float,              # max probability
            "ensemble_strategy": str          # "simple_avg", etc.
        }
    """
    # ── Validate file type ───────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # ── Read image bytes ─────────────────────────────────────
        content = await file.read()

        # ── Size guard (50MB) ────────────────────────────────────
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image exceeds 50MB size limit.")

        # ── Preprocess (CLAHE + per-model transforms) ────────────
        inputs = process_image(content)

        # ── Ensemble inference ───────────────────────────────────
        probs_tensor = ensemble.predict(
            inputs["convnext_input"],
            inputs["efficientnet_input"],
        )

        # ── Format results ───────────────────────────────────────
        probs = probs_tensor[0].tolist()
        predicted_class = int(torch.argmax(probs_tensor, dim=1).item())
        confidence = float(probs[predicted_class])

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_label=settings.CLASS_NAMES[predicted_class],
            probabilities=probs,
            model_count=len(ensemble.models),
            confidence=round(confidence, 4),
            ensemble_strategy=settings.ENSEMBLE_STRATEGY,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during inference.")
