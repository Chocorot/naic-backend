import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "NAIC Image Classification API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "https://naic-2026-site.vercel.app", # Example production URL
    ]

    # Model Settings
    WEIGHTS_DIR: str = "weights"
    NUM_FOLDS: int = 5
    MODEL_PREFIX: str = "model_fold_"
    IMAGE_SIZE: int = 224
    
    # Device
    DEVICE: str = "cpu" # Default to CPU as requested

    class Config:
        case_sensitive = True

settings = Settings()
