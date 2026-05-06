"""
Model Architecture Wrappers — EfficientNet-B3 + ConvNeXt-Small

Verbatim copy from ensemble_template.ipynb Cell 9.
CRITICAL: These wrappers must exactly match the training-time architecture
for checkpoints to load correctly.

Both use HuggingFace `transformers` backbones (not `torchvision.models`).
Using the wrong backbone would silently load random weights.

- EfficientNetWithECA / EfficientNetWithCBAM — from Exp 5 training
- ConvNextWithECA / ConvNextWithCBAM — from Exp 7 training
"""
import torch.nn as nn

from app.models.attention import ECA, CBAM


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet-B3 Wrappers — matches exp_5/exp_7 training architecture
# ─────────────────────────────────────────────────────────────────────────────
class EfficientNetWithECA(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.efficientnet = base_model.efficientnet
        feature_dim       = base_model.classifier.in_features
        self.eca          = ECA(feature_dim)
        self.dropout      = base_model.dropout
        self.classifier   = base_model.classifier

    def forward(self, pixel_values):
        features = self.efficientnet(pixel_values).last_hidden_state
        attended = self.eca(features)
        pooled   = attended.mean(dim=[-2, -1])
        pooled   = self.dropout(pooled)
        return self.classifier(pooled)


class EfficientNetWithCBAM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.efficientnet = base_model.efficientnet
        feature_dim       = base_model.classifier.in_features
        self.cbam         = CBAM(feature_dim)
        self.dropout      = base_model.dropout
        self.classifier   = base_model.classifier

    def forward(self, pixel_values):
        features = self.efficientnet(pixel_values).last_hidden_state
        attended = self.cbam(features)
        pooled   = attended.mean(dim=[-2, -1])
        pooled   = self.dropout(pooled)
        return self.classifier(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# ConvNeXt-Small Wrappers — matches convnext_best training architecture
# ─────────────────────────────────────────────────────────────────────────────
class ConvNextWithECA(nn.Module):
    def __init__(self, base_model, feature_dim):
        super().__init__()
        self.convnext   = base_model.convnext
        self.eca        = ECA(feature_dim)
        self.classifier = base_model.classifier

    def forward(self, pixel_values):
        features = self.convnext(pixel_values).last_hidden_state
        attended = self.eca(features)
        pooled   = attended.mean(dim=[-2, -1])
        return self.classifier(pooled)


class ConvNextWithCBAM(nn.Module):
    def __init__(self, base_model, feature_dim):
        super().__init__()
        self.convnext   = base_model.convnext
        self.cbam       = CBAM(feature_dim)
        self.classifier = base_model.classifier

    def forward(self, pixel_values):
        features = self.convnext(pixel_values).last_hidden_state
        attended = self.cbam(features)
        pooled   = attended.mean(dim=[-2, -1])
        return self.classifier(pooled)
