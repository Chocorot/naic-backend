import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import router as api_v1_router
from app.core.config import settings
from app.models.classifier import ensemble

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    All 10 models (5 ConvNeXt + 5 EfficientNet) are loaded into memory once
    and shared across requests.
    """
    logger.info("Initializing application lifespan...")
    logger.info(f"Ensemble strategy: {settings.ENSEMBLE_STRATEGY}")
    logger.info(f"ConvNeXt weights: {settings.CONVNEXT_WEIGHTS_DIR}")
    logger.info(f"EfficientNet weights: {settings.EFFICIENTNET_WEIGHTS_DIR}")

    ensemble.load_models()

    total = len(ensemble.models)
    logger.info(f"Startup complete — {total} models loaded and ready.")
    yield

    logger.info("Shutting down application lifespan...")
    ensemble.cleanup()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API routes
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring systems.
    """
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "models_loaded": len(ensemble.models),
        "convnext_folds": len(ensemble.convnext_models),
        "efficientnet_folds": len(ensemble.efficientnet_models),
        "ensemble_strategy": settings.ENSEMBLE_STRATEGY,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1 # Using 1 worker for ML to avoid memory issues with large models
    )
