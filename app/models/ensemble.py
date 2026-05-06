"""
Ensemble Strategy Module

Implements the fusion strategies from ensemble_template.ipynb Cell 14,
simplified for inference (no metric evaluation).

Strategies:
- simple_avg:          50/50 softmax average (default)
- weighted_avg:        Tunable global model weights
- per_class_weighted:  Class-specific weights per model
- rank_fusion:         Borda-count rank aggregation

Majority vote is intentionally omitted — it produces unstable probability
outputs unsuitable for UI display.
"""
import torch
from app.core.config import settings


def apply_strategy(
    convnext_probs: torch.Tensor,
    effnet_probs: torch.Tensor,
    strategy: str = None,
) -> torch.Tensor:
    """
    Combine per-model probabilities into a single prediction.
    Both inputs are shape (B, 5) and expected to sum to 1.0 per row.

    Returns (B, 5) — the ensembled probability matrix.
    """
    if strategy is None:
        strategy = settings.ENSEMBLE_STRATEGY

    if strategy == "simple_avg":
        return 0.5 * convnext_probs + 0.5 * effnet_probs

    elif strategy == "weighted_avg":
        return (settings.CONVNEXT_WEIGHT * convnext_probs +
                settings.EFFICIENTNET_WEIGHT * effnet_probs)

    elif strategy == "per_class_weighted":
        w_c = torch.tensor(settings.PER_CLASS_CONVNEXT_WEIGHTS).unsqueeze(0)       # (1, 5)
        w_e = torch.tensor(settings.PER_CLASS_EFFICIENTNET_WEIGHTS).unsqueeze(0)    # (1, 5)
        # Per-class weights sum to 1.0 per class → output still sums to 1.0 per row
        return w_c * convnext_probs + w_e * effnet_probs

    elif strategy == "rank_fusion":
        # Borda-count style — each model ranks classes, sum ranks
        c_ranks = convnext_probs.argsort(dim=1).argsort(dim=1).float()
        e_ranks = effnet_probs.argsort(dim=1).argsort(dim=1).float()
        summed = c_ranks + e_ranks
        return summed / summed.sum(dim=1, keepdim=True).clip(min=1e-12)

    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")
