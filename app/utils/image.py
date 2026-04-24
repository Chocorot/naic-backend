from PIL import Image
import torchvision.transforms as transforms
import io
import torch
from app.core.config import settings

# Optimized transforms
transform = transforms.Compose([
    transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image_bytes: bytes) -> torch.Tensor:
    """
    Converts raw image bytes into a preprocessed tensor.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0) # Add batch dimension
