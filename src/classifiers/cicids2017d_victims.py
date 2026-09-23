"""Load CICIDS2017-DistriNet category checkpoints as attack victims.

The returned module consumes the saved RobustScaler-space 79-feature vectors and
returns five category logits (Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4). Model
parameters are frozen, but gradients still propagate from logits to the input so
latent attacks and Stage-B residual training can optimize adversarial samples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from datasets.cicids2017 import CICIDS2017Adapter
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
    adapter: CICIDS2017Adapter,
    expected_model_type: str,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load one category victim and fail on dataset/schema/class/model mismatch."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"missing CICIDS2017 victim checkpoint: {path}")
    manifest = adapter.feature_manifest()
    expected_classes = ("Benign", "DoS", "DDoS", "Recon", "BruteForce")
    if tuple(adapter.class_mapping().names) != expected_classes:
        raise ValueError("unexpected CICIDS2017 class order")
    preprocessing_manifest = adapter._processed / "preprocessing_manifest.json"
    digest = hashlib.sha256(preprocessing_manifest.read_bytes()).hexdigest()
    run_manifest_path = path.parent.parent / "classifier_run_manifest.json"
    if not run_manifest_path.exists():
        raise FileNotFoundError(f"missing victim run manifest: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("preprocessing_manifest_sha256") != digest:
        raise ValueError(f"{path}: preprocessing manifest hash mismatch")
    if tuple(run_manifest.get("heads", {}).get("category", {}).get("classes", ())) != expected_classes:
        raise ValueError(f"{path}: category class order mismatch")
    if int(run_manifest.get("features", -1)) != manifest.n_features:
        raise ValueError(f"{path}: run-manifest feature count mismatch")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"state_dict", "model_type", "model_kwargs", "num_features", "num_classes"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"{path}: checkpoint missing keys {sorted(missing)}")
    if str(checkpoint["model_type"]) != expected_model_type:
        raise ValueError(
            f"{path}: expected model_type {expected_model_type!r}, "
            f"got {checkpoint['model_type']!r}"
        )
    if int(checkpoint["num_features"]) != manifest.n_features:
        raise ValueError(
            f"{path}: expected {manifest.n_features} features, "
            f"got {checkpoint['num_features']}"
        )
    if int(checkpoint["num_classes"]) != CICIDS2017_CATEGORY_CLASSES:
        raise ValueError(
            f"{path}: expected {CICIDS2017_CATEGORY_CLASSES} category classes, "
            f"got {checkpoint['num_classes']}"
        )

    model = get_model(
        str(checkpoint["model_type"]),
        num_features=manifest.n_features,
        num_classes=CICIDS2017_CATEGORY_CLASSES,
        **dict(checkpoint["model_kwargs"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    has_recurrent_layer = any(isinstance(module, nn.LSTM) for module in model.modules())
    return _GradientSafeVictim(model, disable_cudnn=has_recurrent_layer)
