"""
Image Preprocessing Utilities

Pipeline (must match training exactly):
1. CLAHE preprocessing: crop_fundus() + Green-channel CLAHE (from notebook Cell 5)
2. Per-model transforms: separate resolution + normalization per architecture

ConvNeXt-Small: 224×224, HuggingFace ConvNeXt normalization
EfficientNet-B3: 300×300, HuggingFace EfficientNet normalization
"""
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import io
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLAHE Preprocessing — verbatim from notebook Cell 5
# ─────────────────────────────────────────────────────────────────────────────
def crop_fundus(image, tol=10):
    """Remove black borders around circular fundus image."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


def preprocess_fundus(image):
    """Auto-crop + Green Channel CLAHE pipeline."""
    cropped = crop_fundus(image)
    r, g, b = cv2.split(cropped)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_enhanced = clahe.apply(g)
    enhanced = cv2.merge((r, g_enhanced, b))
    return Image.fromarray(enhanced)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model transforms — built lazily to allow HuggingFace processor loading
# ─────────────────────────────────────────────────────────────────────────────
_convnext_transform = None
_efficientnet_transform = None


def get_convnext_transform():
    """Build ConvNeXt transform using HuggingFace processor normalization constants."""
    global _convnext_transform
    if _convnext_transform is None:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(settings.CONVNEXT_MODEL_NAME)
        _convnext_transform = transforms.Compose([
            transforms.Resize((settings.CONVNEXT_IMG_SIZE, settings.CONVNEXT_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])
        logger.info(f"ConvNeXt transform built: {settings.CONVNEXT_IMG_SIZE}x{settings.CONVNEXT_IMG_SIZE}")
    return _convnext_transform


def get_efficientnet_transform():
    """Build EfficientNet transform using HuggingFace processor normalization constants."""
    global _efficientnet_transform
    if _efficientnet_transform is None:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(settings.EFFICIENTNET_MODEL_NAME)
        _efficientnet_transform = transforms.Compose([
            transforms.Resize((settings.EFFICIENTNET_IMG_SIZE, settings.EFFICIENTNET_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])
        logger.info(f"EfficientNet transform built: {settings.EFFICIENTNET_IMG_SIZE}x{settings.EFFICIENTNET_IMG_SIZE}")
    return _efficientnet_transform


# ─────────────────────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────────────────────
def process_image(image_bytes: bytes) -> dict:
    """
    Converts raw image bytes into preprocessed tensors for both model families.

    Pipeline:
    1. Open image as RGB PIL
    2. Apply CLAHE preprocessing (crop + green-channel enhancement)
    3. Apply per-model transforms (resize + normalize)

    Returns:
        dict with 'convnext_input' and 'efficientnet_input' tensors,
        each with batch dimension (1, C, H, W).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # CLAHE preprocessing — must match training exactly
    preprocessed = preprocess_fundus(image)

    # Per-model transforms
    convnext_tensor = get_convnext_transform()(preprocessed).unsqueeze(0)
    efficientnet_tensor = get_efficientnet_transform()(preprocessed).unsqueeze(0)

    return {
        "convnext_input": convnext_tensor,
        "efficientnet_input": efficientnet_tensor,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy processing (for old single-model fallback)
# ─────────────────────────────────────────────────────────────────────────────
legacy_transform = transforms.Compose([
    transforms.Resize((settings.LEGACY_IMAGE_SIZE, settings.LEGACY_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def process_image_legacy(image_bytes: bytes) -> torch.Tensor:
    """Legacy preprocessing for the old ConvNeXt-only ensemble."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = legacy_transform(image)
    return tensor.unsqueeze(0)
