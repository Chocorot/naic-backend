# NAIC Image Classification Ensemble API

A production-ready FastAPI server for medical image classification using a 5-fold model ensemble.

## Features
- **Industry Standard Structure**: Modularized code for high maintainability.
- **Optimized Inference**: Uses `torch.inference_mode` and `AdaptiveAvgPool2d` for consistent results regardless of input size.
- **Production Lifespan**: Models are loaded once on startup and stored in memory.
- **Frontend Compatible**: Response format matches the expected types in `naic-2026-site`.
- **CORS Ready**: Configured for local development and common production origins.

## Directory Structure
- `app/api/v1`: Route definitions.
- `app/core`: Project configuration and constants.
- `app/models`: Model architecture and ensemble loading logic.
- `app/schemas`: Pydantic request/response validation.
- `app/utils`: Image processing helpers.
- `weights/`: **Folder for storing your model weight files.**

## Setup Instructions

### 1. Weight Preparation
Place your 5 PyTorch weight files (`.pth`) in the `weights/` directory. 
The expected names are:
- `model_fold_1.pth`
- `model_fold_2.pth`
- `model_fold_3.pth`
- `model_fold_4.pth`
- `model_fold_5.pth`

### 2. Environment Setup
```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

### 3. Running the Server
**Development Mode:**
```bash
python main.py
```

**Production Mode (Local):**
```bash
gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

**Production Mode (Docker):**
```bash
docker build -t naic-backend .
docker run -p 8080:8080 naic-backend
```

### 4. Deployment (Google Cloud Run)
We provide a `cloudbuild.yaml` for automated deployment:
```bash
gcloud builds submit --config cloudbuild.yaml
```

### 5. Testing
You can test the API locally using the provided script:
```bash
python scripts/test_predict.py path/to/your/image.jpg
```

## API Reference

### POST `/api/v1/predict`
Accepts an image file and returns classification results.

**Response Example:**
```json
{
  "prediction": 2,
  "probabilities": [0.05, 0.15, 0.75, 0.03, 0.02],
  "model_count": 5,
  "confidence": 0.75
}
```

## Git Guidelines
- Do **not** commit weight files (`.pth`) to the repository.
- Use `.gitignore` to exclude `venv/`, `__pycache__/`, and `weights/*.pth`.
