"""Continuous mixed-range β-VAE for corrected CICIoT2023 semantics.

All 39 CSV features are reconstructed as continuous aggregates. The 22 averaged
flag/service/protocol indicators use sigmoid-bounded raw outputs in [0,1]. No
Bernoulli sampling, hard threshold, integer protocol embedding, or derived
one-hot reconstruction is applied.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from sklearn.preprocessing import RobustScaler

PROTOCOL_REFERENCE_BUFFERS: frozenset[str] = frozenset({"feature_center", "feature_scale"})


def _body(input_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    return nn.Sequential(*layers)


class MixedInputBetaVAE(nn.Module):
    """Backward-named API implementing continuous window-aggregate semantics."""

    def __init__(
        self,
        partition: dict,
        latent_dim: int = 16,
        protocol_embed_dim: int = 4,
        encoder_hidden: tuple = (128, 64),
        decoder_hidden: tuple = (64, 128),
        n_pseudo_binary: int = 0,
        use_structured_continuous_decoder: bool = False,
        use_structured_physics_decoder: bool = False,
        structured_continuous_mode: str = "full",
        structured_std_floor: float = 0.0,
        latent_logvar_bounds: tuple[float, float] = (-6.0, 6.0),
    ) -> None:
        super().__init__()
        self.partition = partition
        self.latent_dim = latent_dim
        self.protocol_embed_dim = protocol_embed_dim  # ignored; retained in checkpoint config
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.n_pseudo_binary = 0
        self.use_structured_continuous_decoder = use_structured_continuous_decoder
        self.use_structured_physics_decoder = use_structured_physics_decoder
        self.structured_continuous_mode = structured_continuous_mode
        self.structured_std_floor = structured_std_floor
        self.latent_logvar_bounds = latent_logvar_bounds
        self.n_continuous = len(partition["continuous_idx"])
        self.n_independent_binary = 0
        if self.n_continuous != 39:
            raise ValueError("corrected CICIoT2023 VAE requires all 39 continuous features")

        self.encoder_body = _body(39, tuple(encoder_hidden))
        self.encoder_out = nn.Linear(encoder_hidden[-1], 2 * latent_dim)
        self.decoder_body = _body(latent_dim, tuple(decoder_hidden))
        self.head_continuous_mu = nn.Linear(decoder_hidden[-1], 39)
        self.head_continuous_logvar = nn.Linear(decoder_hidden[-1], 39)
        self.register_buffer("feature_center", torch.zeros(39, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.ones(39, dtype=torch.float32))
        self._scaler_registered = False

    def register_protocol_references(self, scaler: "RobustScaler") -> None:
        """Register affine feature scaling; historical method name kept for callers."""

        center = np.asarray(scaler.center_, dtype=np.float32)
        scale = np.asarray(scaler.scale_, dtype=np.float32)
        if center.shape != (39,) or scale.shape != (39,):
            raise ValueError("scaler must match the full 39-feature schema")
        if np.any(scale == 0.0):
            raise ValueError("scaler contains a zero scale")
        self.feature_center = torch.tensor(center, dtype=torch.float32)
        self.feature_scale = torch.tensor(scale, dtype=torch.float32)
        self._scaler_registered = True

    def continuous_scaled_to_raw(self, values: torch.Tensor) -> torch.Tensor:
        indices = self.partition["continuous_idx"]
        center = self.feature_center.to(values.device)[indices]
        scale = self.feature_scale.to(values.device)[indices]
        return values * scale.unsqueeze(0) + center.unsqueeze(0)

    def continuous_raw_to_scaled(self, values: torch.Tensor) -> torch.Tensor:
        indices = self.partition["continuous_idx"]
        center = self.feature_center.to(values.device)[indices]
        scale = self.feature_scale.to(values.device)[indices]
        return (values - center.unsqueeze(0)) / scale.unsqueeze(0)

    def _structure_raw(self, raw: torch.Tensor) -> torch.Tensor:
        structured = raw.clone()
        bounded = self.partition["bounded_continuous_idx"]
        # Values in these columns are window means of packet indicators.
        structured[:, bounded] = torch.sigmoid(raw[:, bounded])
        if self.use_structured_continuous_decoder:
            unbounded = self.partition["unbounded_continuous_idx"]
            structured[:, unbounded] = structured[:, unbounded].clamp_min(0.0)
            ttl = 2
            structured[:, ttl] = torch.sigmoid(raw[:, ttl] / 32.0) * 255.0
            min_idx, max_idx, avg_idx = 31, 32, 33
            minimum = torch.nn.functional.softplus(raw[:, min_idx])
            average = minimum + torch.nn.functional.softplus(raw[:, avg_idx] - raw[:, min_idx])
            maximum = average + torch.nn.functional.softplus(raw[:, max_idx] - raw[:, avg_idx])
            structured[:, min_idx] = minimum
            structured[:, avg_idx] = average
            structured[:, max_idx] = maximum
            structured[:, 37] = torch.nn.functional.softplus(raw[:, 37])
            std = self.structured_std_floor + torch.nn.functional.softplus(raw[:, 34])
            structured[:, 34] = std
            structured[:, 38] = std.square()
            if self.use_structured_physics_decoder:
                structured[:, 35] = average
                structured[:, 30] = structured[:, 37] * average
        return structured

    def encode(self, x_39: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_39.ndim != 2 or x_39.shape[1] != 39:
            raise ValueError(f"expected (N,39), got {tuple(x_39.shape)}")
        out = self.encoder_out(self.encoder_body(x_39))
        return out[:, :self.latent_dim], out[:, self.latent_dim:].clamp(*self.latent_logvar_bounds)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode_internal(self, z: torch.Tensor) -> dict:
        hidden = self.decoder_body(z)
        raw_head = self.head_continuous_mu(hidden)
        raw_unstructured = self.continuous_scaled_to_raw(raw_head)
        raw = self._structure_raw(raw_unstructured)
        scaled = self.continuous_raw_to_scaled(raw)
        logvar = self.head_continuous_logvar(hidden).clamp(-7.0, 2.0)
        empty = scaled[:, :0]
        return {
            "continuous_mu": scaled,
            "continuous_mu_raw": raw,
            "continuous_mu_raw_unstructured": raw_unstructured,
            "continuous_logvar": logvar,
            "binary_logits": empty,
            "protocol_logits": empty,
            "pseudo_binary_sigmoid": None,
        }

    def decode_to_39(
        self,
        z: torch.Tensor,
        scaler: "RobustScaler | None" = None,
        mode: str = "soft",
    ) -> tuple[torch.Tensor, dict]:
        if mode not in {"soft", "hard"}:
            raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")
        decoded = self.decode_internal(z)
        # Soft/hard are identical: averaged indicators remain continuous.
        metadata = {
            "continuous_mu": decoded["continuous_mu"],
            "continuous_mu_raw": decoded["continuous_mu_raw"],
            "continuous_logvar": decoded["continuous_logvar"],
            "binary_logits": decoded["binary_logits"],
            "protocol_logits": decoded["protocol_logits"],
            "protocol_idx_batch": None,
        }
        return decoded["continuous_mu"], metadata

    def forward(self, x_39: torch.Tensor) -> dict:
        mu, logvar = self.encode(x_39)
        decoded = self.decode_internal(self.reparameterize(mu, logvar))
        return {"mu": mu, "logvar": logvar, **decoded}
