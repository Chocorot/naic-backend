"""
Attention Modules — ECA + CBAM

Verbatim copy from ensemble_template.ipynb Cell 8.
These must match the training-time architecture exactly for checkpoints to load.

- ECA: Efficient Channel Attention (Wang et al., 2020)
- CBAM: Channel + Spatial Attention (Woo et al., 2018)
"""
import math
import torch
import torch.nn as nn


# ── ECA: Efficient Channel Attention (Wang et al., 2020) ─────────────────────
class ECA(nn.Module):
    """Adaptive kernel size: k = |log2(C)/gamma + b/gamma|, forced odd."""
    def __init__(self, in_channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(in_channels)) / gamma + b / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(1, 2).unsqueeze(-1)
        return x * y


# ── CBAM: Channel + Spatial (Woo et al., 2018) ───────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
