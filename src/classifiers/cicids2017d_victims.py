"""Load CICIDS2017-DistriNet category checkpoints as attack victims.

The returned module consumes the saved RobustScaler-space 79-feature vectors and
returns five category logits (Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4). Model
parameters are frozen, but gradients still propagate from logits to the input so
latent attacks and Stage-B residual training can optimize adversarial samples.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.classifiers.models import get_model

CICIDS2017_NUM_FEATURES = 79
CICIDS2017_CATEGORY_CLASSES = 5


class _GradientSafeVictim(nn.Module):
    """Keep recurrent victims deterministic while allowing CUDA input gradients."""

    def __init__(self, model: nn.Module, *, disable_cudnn: bool) -> None:
        super().__init__()
        self.model = model
        self.disable_cudnn = disable_cudnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.disable_cudnn:
            return self.model(x)
        # cuDNN RNN backward requires training mode/reserve-space. Attacks require
        # deterministic eval semantics, so use PyTorch's native recurrent path.
        with torch.backends.cudnn.flags(enabled=False):
            return self.model(x)


def load_category_victim(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load, validate, freeze, and return one five-category victim checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"missing CICIDS2017 victim checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"state_dict", "model_type", "model_kwargs", "num_features", "num_classes"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"{path}: checkpoint missing keys {sorted(missing)}")
    if int(checkpoint["num_features"]) != CICIDS2017_NUM_FEATURES:
        raise ValueError(
            f"{path}: expected {CICIDS2017_NUM_FEATURES} features, "
            f"got {checkpoint['num_features']}"
        )
    if int(checkpoint["num_classes"]) != CICIDS2017_CATEGORY_CLASSES:
        raise ValueError(
            f"{path}: expected {CICIDS2017_CATEGORY_CLASSES} category classes, "
            f"got {checkpoint['num_classes']}"
        )

    model = get_model(
        str(checkpoint["model_type"]),
        num_features=CICIDS2017_NUM_FEATURES,
        num_classes=CICIDS2017_CATEGORY_CLASSES,
        **dict(checkpoint["model_kwargs"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    has_recurrent_layer = any(isinstance(module, nn.LSTM) for module in model.modules())
    return _GradientSafeVictim(model, disable_cudnn=has_recurrent_layer)
