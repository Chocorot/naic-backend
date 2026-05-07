# NAIC Image Classification Ensemble API

A production-ready FastAPI server for medical image classification using a **10-model Dual-Architecture Ensemble**.

## Features
- **Dual-Architecture Ensemble**: Combines 5 ConvNeXt-Small folds and 5 EfficientNet-B3 folds (10 models total) for maximum robustness.
- **Attention Modules**: Both architectures are enhanced with ECA (Efficient Channel Attention) for improved feature focus.
- **Industry Standard Structure**: Modularized code for high maintainability and scalability.
- **Optimized Inference**: Uses `torch.inference_mode` and standardized preprocessing (CLAHE + architecture-specific transforms).
- **Flexible Ensemble Strategies**: Supports simple averaging, weighted averaging, and rank fusion (configurable via `.env`).
- **Production Lifespan**: Models are loaded once on startup and stored in memory for low-latency inference.

## Directory Structure
- `app/api/v1`: Route definitions and endpoint logic.
- `app/core`: Project configuration, constants, and `.env` loading.
- `app/models`: Model architectures (ConvNeXt, EfficientNet, Attention) and ensemble loading logic.
- `app/schemas`: Pydantic request/response validation.
- `app/utils`: Image processing helpers (CLAHE, transforms).
- `convnext_best_weights/`: Directory for ConvNeXt-Small fold weights.
- `efficientnetb3_best_weights/`: Directory for EfficientNet-B3 fold weights.

## Setup Instructions

### 1. Environment Configuration
Create a `.env` file in the root directory (copy from `.env.example` if available). Key variables:
```env
CONVNEXT_WEIGHTS_DIR="convnext_best_weights"
EFFICIENTNET_WEIGHTS_DIR="efficientnetb3_best_weights"
ENSEMBLE_STRATEGY="simple_avg"  # Options: simple_avg, weighted_avg, rank_fusion
DEVICE="cpu"                   # Use "cuda" if a GPU is available
```

### 2. Weight Preparation
Place your trained weight files (`.pth`) in their respective directories. The expected file naming convention for both architectures is:
- `best_model_fold_0.pth`
- `best_model_fold_1.pth`
- `best_model_fold_2.pth`
- `best_model_fold_3.pth`
- `best_model_fold_4.pth`

### 3. Environment Setup
```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Running the Server
**Development Mode:**
```bash
python main.py
```

**Production Mode (Gunicorn):**
```bash
gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

### 5. Testing
You can test the API locally using the provided script:
```bash
python scripts/test_predict.py path/to/your/image.jpg
```

## API Reference

### POST `/api/v1/predict`
Accepts an image file and returns ensembled classification results across all 10 models.

**Response Example:**
```json
{
  "prediction": 2,
  "predicted_class": 2,
  "predicted_label": "Moderate",
  "probabilities": [0.05, 0.15, 0.75, 0.03, 0.02],
  "model_count": 10,
  "confidence": 0.75,
  "ensemble_strategy": "simple_avg"
}
```

## Git Guidelines
- **Do not** commit weight files (`.pth`) or the `.env` file to the repository.
- Ensure `venv/`, `__pycache__/`, and weight directories are listed in your `.gitignore`.
