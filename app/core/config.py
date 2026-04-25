import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "NAIC Image Classification API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "https://naic-2026-site.vercel.app", # Example production URL
    ]

    # ── Ensemble model settings ───────────────────────────────────────────────
    NUM_FOLDS: int = 5
    NUM_CLASSES: int = 5

    # ConvNeXt-Small (Exp 7)
    CONVNEXT_WEIGHTS_DIR: str = "convnext_best_weights"
    CONVNEXT_MODEL_NAME: str = "facebook/convnext-small-224"
    CONVNEXT_IMG_SIZE: int = 224
    CONVNEXT_FEATURE_DIM: int = 768
    CONVNEXT_ATTENTION: str = "eca"  # 'eca' | 'cbam' | 'none'

    # EfficientNet-B3 (Exp 5)
    EFFICIENTNET_WEIGHTS_DIR: str = "efficientnetb3_best_weights"
    EFFICIENTNET_MODEL_NAME: str = "google/efficientnet-b3"
    EFFICIENTNET_IMG_SIZE: int = 300
    EFFICIENTNET_ATTENTION: str = "eca"  # 'eca' | 'cbam' | 'none'

    # ── Ensemble strategy ─────────────────────────────────────────────────────
    # Options: "simple_avg" | "weighted_avg" | "per_class_weighted" | "rank_fusion"
    ENSEMBLE_STRATEGY: str = "simple_avg"

    # Global weights (for weighted_avg strategy)
    CONVNEXT_WEIGHT: float = 0.55
    EFFICIENTNET_WEIGHT: float = 0.45

    # Per-class weights (for per_class_weighted strategy)
    PER_CLASS_CONVNEXT_WEIGHTS: List[float] = [0.40, 0.65, 0.50, 0.60, 0.55]
    PER_CLASS_EFFICIENTNET_WEIGHTS: List[float] = [0.60, 0.35, 0.50, 0.40, 0.45]

    # ── Class name mapping ────────────────────────────────────────────────────
    CLASS_NAMES: List[str] = [
        "No DR",              # C0
        "Mild",               # C1
        "Moderate",           # C2
        "Severe",             # C3
        "Proliferative DR",   # C4
    ]

    # ── Legacy fallback settings ──────────────────────────────────────────────
    LEGACY_WEIGHTS_DIR: str = "weights"
    MODEL_PREFIX: str = "model_fold_"
    LEGACY_IMAGE_SIZE: int = 224

    # Device
    DEVICE: str = "cpu"  # Default to CPU; set to "cuda" for GPU

    class Config:
        case_sensitive = True

settings = Settings()
