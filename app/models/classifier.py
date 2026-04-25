"""
Model Classifier — Dual-Architecture Ensemble + Legacy Fallback

Supports two modes:
1. DualModelEnsemble: ConvNeXt-Small (5 folds) + EfficientNet-B3 (5 folds)
   Uses HuggingFace transformers backbones with ECA attention wrappers.
2. LegacyModelEnsemble: Original ConvNeXt-only (5 folds) — preserved as fallback.

Architecture classes must match training-time definitions exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from typing import List

from app.core.config import settings
from app.models.wrappers import (
    EfficientNetWithECA, EfficientNetWithCBAM,
    ConvNextWithECA, ConvNextWithCBAM,
)
from app.models.ensemble import apply_strategy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Flexible checkpoint loader (from notebook Cell 12)
# ─────────────────────────────────────────────────────────────────────────────
def load_state_dict_flexible(model, ckpt_path, device):
    """
    Load either a raw state_dict or a checkpoint containing 'model_state_dict'.
    Warns loudly on large key mismatches — strict=False can silently load garbage.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"  {len(missing)} missing keys loading {ckpt_path} (first 3: {missing[:3]})")
    if unexpected:
        logger.warning(f"  {len(unexpected)} unexpected keys loading {ckpt_path} (first 3: {unexpected[:3]})")
    if len(missing) > 5 or len(unexpected) > 5:
        raise RuntimeError(
            f"Large key mismatch loading {ckpt_path}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected — "
            f"architecture likely does not match checkpoint."
        )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model builder (from notebook Cell 12)
# ─────────────────────────────────────────────────────────────────────────────
def build_model(model_type: str, attention: str, device: torch.device):
    """
    Construct the correct architecture based on model_type + attention flag.
    Returns (model, is_wrapped) — is_wrapped affects how forward() is called.
    """
    from transformers import (
        ConvNextForImageClassification,
        EfficientNetForImageClassification,
    )

    if model_type == "efficientnet":
        base = EfficientNetForImageClassification.from_pretrained(
            settings.EFFICIENTNET_MODEL_NAME,
            num_labels=settings.NUM_CLASSES,
            ignore_mismatched_sizes=True,
        ).to(device)
        if attention == "eca":
            model = EfficientNetWithECA(base).to(device)
        elif attention == "cbam":
            model = EfficientNetWithCBAM(base).to(device)
        else:
            model = base

    elif model_type == "convnext":
        base = ConvNextForImageClassification.from_pretrained(
            settings.CONVNEXT_MODEL_NAME,
            num_labels=settings.NUM_CLASSES,
            ignore_mismatched_sizes=True,
        ).to(device)
        feature_dim = settings.CONVNEXT_FEATURE_DIM
        if attention == "eca":
            model = ConvNextWithECA(base, feature_dim).to(device)
        elif attention == "cbam":
            model = ConvNextWithCBAM(base, feature_dim).to(device)
        else:
            model = base
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    is_wrapped = (attention != "none")
    return model, is_wrapped


# ─────────────────────────────────────────────────────────────────────────────
# DualModelEnsemble — 2-model × 5-fold = 10-weight ensemble
# ─────────────────────────────────────────────────────────────────────────────
class DualModelEnsemble:
    """
    Loads and manages the dual-architecture ensemble:
    - 5 ConvNeXt-Small folds (HuggingFace backbone + ECA)
    - 5 EfficientNet-B3 folds (HuggingFace backbone + ECA)
    """
    def __init__(self):
        self.convnext_models: List[nn.Module] = []
        self.efficientnet_models: List[nn.Module] = []
        self.device = torch.device(settings.DEVICE)

    @property
    def models(self):
        """All loaded models (for health check compatibility)."""
        return self.convnext_models + self.efficientnet_models

    def load_models(self):
        """Load all fold checkpoints for both model families."""
        self.convnext_models = []
        self.efficientnet_models = []

        # ── Load ConvNeXt folds ──────────────────────────────────
        logger.info("Loading ConvNeXt-Small models...")
        for fold in range(settings.NUM_FOLDS):
            ckpt_path = os.path.join(
                settings.CONVNEXT_WEIGHTS_DIR,
                f"best_model_fold_{fold}.pth",
            )
            if not os.path.exists(ckpt_path):
                logger.warning(f"ConvNeXt fold {fold} not found: {ckpt_path}")
                continue
            try:
                model, _ = build_model("convnext", settings.CONVNEXT_ATTENTION, self.device)
                model = load_state_dict_flexible(model, ckpt_path, self.device)
                model.eval()
                self.convnext_models.append(model)
                logger.info(f"  ConvNeXt fold {fold} loaded")
            except Exception as e:
                logger.error(f"Failed to load ConvNeXt fold {fold}: {e}")

        # ── Load EfficientNet folds ──────────────────────────────
        logger.info("Loading EfficientNet-B3 models...")
        for fold in range(settings.NUM_FOLDS):
            ckpt_path = os.path.join(
                settings.EFFICIENTNET_WEIGHTS_DIR,
                f"best_model_fold_{fold}.pth",
            )
            if not os.path.exists(ckpt_path):
                logger.warning(f"EfficientNet fold {fold} not found: {ckpt_path}")
                continue
            try:
                model, _ = build_model("efficientnet", settings.EFFICIENTNET_ATTENTION, self.device)
                model = load_state_dict_flexible(model, ckpt_path, self.device)
                model.eval()
                self.efficientnet_models.append(model)
                logger.info(f"  EfficientNet fold {fold} loaded")
            except Exception as e:
                logger.error(f"Failed to load EfficientNet fold {fold}: {e}")

        total = len(self.convnext_models) + len(self.efficientnet_models)
        logger.info(f"All models loaded: {len(self.convnext_models)} ConvNeXt + "
                     f"{len(self.efficientnet_models)} EfficientNet = {total} total")

        if total == 0:
            logger.critical("No models loaded! The API will not be able to process requests.")

    @torch.inference_mode()
    def predict(self, convnext_input: torch.Tensor, efficientnet_input: torch.Tensor):
        """
        Run inference across all loaded folds, average per-model, then apply
        the ensemble strategy to combine both model families.

        Args:
            convnext_input:     (1, C, 224, 224) preprocessed tensor
            efficientnet_input: (1, C, 300, 300) preprocessed tensor

        Returns:
            final_probs: (1, 5) ensembled probability tensor
            metadata: dict with per-model info
        """
        if not self.convnext_models and not self.efficientnet_models:
            raise RuntimeError("No models loaded")

        convnext_input = convnext_input.to(self.device)
        efficientnet_input = efficientnet_input.to(self.device)

        # ── ConvNeXt inference across folds ───────────────────────
        convnext_probs = None
        if self.convnext_models:
            probs_list = []
            for model in self.convnext_models:
                logits = model(convnext_input)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs)
            convnext_probs = torch.cat(probs_list).mean(dim=0, keepdim=True)  # (1, 5)

        # ── EfficientNet inference across folds ───────────────────
        effnet_probs = None
        if self.efficientnet_models:
            probs_list = []
            for model in self.efficientnet_models:
                logits = model(efficientnet_input)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs)
            effnet_probs = torch.cat(probs_list).mean(dim=0, keepdim=True)  # (1, 5)

        # ── Apply ensemble strategy ──────────────────────────────
        if convnext_probs is not None and effnet_probs is not None:
            final_probs = apply_strategy(convnext_probs.cpu(), effnet_probs.cpu())
        elif convnext_probs is not None:
            final_probs = convnext_probs.cpu()
        else:
            final_probs = effnet_probs.cpu()

        return final_probs

    def cleanup(self):
        """Release model memory on shutdown."""
        self.convnext_models.clear()
        self.efficientnet_models.clear()
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Legacy single-model ensemble — preserved as fallback
# ─────────────────────────────────────────────────────────────────────────────

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

class LegacyLayerNorm(nn.Module):
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
        self.layernorm = LegacyLayerNorm(dim, eps=1e-6)
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

class LegacyECA(nn.Module):
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
        self.embeddings.layernorm = LegacyLayerNorm(dims[0], eps=1e-6, data_format="channels_first")

        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Module()
            if i > 0:
                stage.downsampling_layer = nn.Sequential(
                    LegacyLayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
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

class LegacyImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 5):
        super(LegacyImageClassifier, self).__init__()
        self.convnext = ConvNextSmall()
        self.eca = LegacyECA(kernel_size=5)
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


class LegacyModelEnsemble:
    """
    Original single-architecture ensemble (ConvNeXt-only).
    Preserved as fallback — uses the old custom ConvNeXt implementation
    and weights from the original weights/ directory.
    """
    def __init__(self):
        self.models: List[nn.Module] = []
        self.device = torch.device(settings.DEVICE)

    def load_models(self):
        self.models = []
        for i in range(1, settings.NUM_FOLDS + 1):
            model_path = os.path.join(settings.LEGACY_WEIGHTS_DIR, f"{settings.MODEL_PREFIX}{i}.pth")
            if os.path.exists(model_path):
                try:
                    model = LegacyImageClassifier()
                    state_dict = torch.load(model_path, map_location="cpu")
                    model.load_state_dict(state_dict)
                    model.to(self.device)
                    model.eval()
                    self.models.append(model)
                    logger.info(f"Successfully loaded legacy model: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load legacy model {model_path}: {e}")
            else:
                logger.warning(f"Legacy model weight not found: {model_path}")
        if not self.models:
            logger.warning("No legacy models loaded.")

    @torch.inference_mode()
    def predict(self, input_tensor: torch.Tensor):
        if not self.models:
            raise RuntimeError("No legacy models loaded")
        input_tensor = input_tensor.to(self.device)
        ensemble_logits = [model(input_tensor) for model in self.models]
        stacked_logits = torch.stack(ensemble_logits)
        mean_logits = torch.mean(stacked_logits, dim=0)
        return torch.softmax(mean_logits, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — used by main.py and endpoints
# ─────────────────────────────────────────────────────────────────────────────
ensemble = DualModelEnsemble()
