import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_parameter = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ConvNextSmall(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)
        self.embeddings.layernorm = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        
        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Module()
            if i > 0:
                stage.downsampling_layer = nn.Sequential(
                    LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2),
                )
            else:
                stage.downsampling_layer = nn.Identity()
            stage.layers = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.encoder.stages.append(stage)
            cur += depths[i]
        self.layernorm = nn.LayerNorm(dims[-1], eps=1e-6)

class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 5):
        super(ImageClassifier, self).__init__()
        self.convnext = ConvNextSmall()
        self.eca = ECA(kernel_size=5)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convnext.embeddings.patch_embeddings(x)
        x = self.convnext.embeddings.layernorm(x)
        for stage in self.convnext.encoder.stages:
            x = stage.downsampling_layer(x)
            x = stage.layers(x)
        x = self.convnext.layernorm(x.mean([-2, -1]))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.eca(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class ModelEnsemble:
    def __init__(self):
        self.models: List[nn.Module] = []
        self.device = torch.device(settings.DEVICE)
    def load_models(self):
        self.models = []
        for i in range(1, settings.NUM_FOLDS + 1):
            model_path = os.path.join(settings.WEIGHTS_DIR, f"{settings.MODEL_PREFIX}{i}.pth")
            if os.path.exists(model_path):
                try:
                    model = ImageClassifier()
                    state_dict = torch.load(model_path, map_location="cpu")
                    model.load_state_dict(state_dict)
                    model.to(self.device)
                    model.eval()
                    self.models.append(model)
                    logger.info(f"Successfully loaded model: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_path}: {e}")
            else:
                logger.warning(f"Model weight not found: {model_path}")
        if not self.models:
            logger.critical("No models loaded! The API will not be able to process requests.")
    @torch.inference_mode()
    def predict(self, input_tensor: torch.Tensor):
        if not self.models: raise RuntimeError("No models loaded")
        input_tensor = input_tensor.to(self.device)
        ensemble_logits = [model(input_tensor) for model in self.models]
        stacked_logits = torch.stack(ensemble_logits)
        mean_logits = torch.mean(stacked_logits, dim=0)
        return torch.softmax(mean_logits, dim=1)

ensemble = ModelEnsemble()
