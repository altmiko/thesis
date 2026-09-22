"""Mixed-range β-VAE for flow-based NIDS features.

Two decoder paths, selected at construction:

* ``typed``  (default when a ``FeatureManifest`` is supplied) — the generic,
  dataset-agnostic head in :mod:`vae.decoder`. Feature dimensionality and per-feature
  activations come from the manifest; there are no hard-coded indices or a fixed
  feature count.
* ``legacy`` (default when only a ``partition`` dict is supplied) — the original
  CICIoT structured continuous decoder, kept intact for the A0 ablation and for
  callers that have not yet been migrated to the manifest.

Raw<->model-space scaling is provided by an injected transform (a
:class:`datasets.transforms.FeatureTransform` or an sklearn scaler). The decoder
REFUSES to run before that transform is registered — it never silently falls back
to center=0 / scale=1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover
    from sklearn.preprocessing import RobustScaler

    from datasets.feature_manifest import FeatureManifest
    from datasets.transforms import FeatureTransform

PROTOCOL_REFERENCE_BUFFERS: frozenset[str] = frozenset({"feature_center", "feature_scale"})


class TransformNotRegisteredError(RuntimeError):
    """Raised when decoding is attempted before a feature transform is registered."""


def _body(input_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    return nn.Sequential(*layers)


class MixedInputBetaVAE(nn.Module):
    """Per-class β-VAE with a shared trunk and a manifest-typed (or legacy) decoder."""

    def __init__(
        self,
        partition: dict | None = None,
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
        *,
        manifest: "FeatureManifest | None" = None,
        decoder_kind: str | None = None,
        continuous_logvar_bounds: tuple[float, float] = (-7.0, 2.0),
        encoder_input_transform: str = "none",
    ) -> None:
        super().__init__()
        if manifest is None and partition is None:
            raise ValueError("MixedInputBetaVAE requires either a manifest or a partition")

        self.manifest = manifest
        self.partition = partition
        self.latent_dim = latent_dim
        self.protocol_embed_dim = protocol_embed_dim  # ignored; retained for checkpoint config
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.n_pseudo_binary = 0
        self.use_structured_continuous_decoder = use_structured_continuous_decoder
        self.use_structured_physics_decoder = use_structured_physics_decoder
        self.structured_continuous_mode = structured_continuous_mode
        self.structured_std_floor = structured_std_floor
        self.latent_logvar_bounds = latent_logvar_bounds
        self.continuous_logvar_bounds = continuous_logvar_bounds
        if encoder_input_transform not in {"none", "asinh"}:
            raise ValueError(
                "encoder_input_transform must be 'none' or 'asinh', got "
                f"{encoder_input_transform!r}"
            )
        self.encoder_input_transform = encoder_input_transform

        # Dynamic feature count — no hard-coded 39.
        if manifest is not None:
            self.n_features = manifest.n_features
            self._cont_idx = list(range(self.n_features))
        else:
            self._cont_idx = list(partition["continuous_idx"])  # type: ignore[index]
            self.n_features = len(self._cont_idx)
        self.n_continuous = self.n_features  # backward-compat attribute
        self.n_independent_binary = 0

        self.decoder_kind = decoder_kind or ("typed" if manifest is not None else "legacy")
        if self.decoder_kind not in {"typed", "legacy"}:
            raise ValueError(f"decoder_kind must be 'typed' or 'legacy', got {self.decoder_kind!r}")
        if self.decoder_kind == "typed" and manifest is None:
            raise ValueError("decoder_kind='typed' requires a manifest")

        self.encoder_body = _body(self.n_features, tuple(encoder_hidden))
        self.encoder_out = nn.Linear(encoder_hidden[-1], 2 * latent_dim)
        self.decoder_body = _body(latent_dim, tuple(decoder_hidden))

        if self.decoder_kind == "typed":
            from vae.decoder import TypedDecoder

            self.typed_decoder = TypedDecoder(
                manifest, decoder_hidden[-1], logvar_bounds=continuous_logvar_bounds
            )
        else:
            self.head_continuous_mu = nn.Linear(decoder_hidden[-1], self.n_features)
            self.head_continuous_logvar = nn.Linear(decoder_hidden[-1], self.n_features)

        self.register_buffer("feature_center", torch.zeros(self.n_features, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.ones(self.n_features, dtype=torch.float32))
        self._scaler_registered = False

    # ------------------------------------------------------------------ #
    # Transform registration (raw <-> model space)
    # ------------------------------------------------------------------ #
    def register_feature_transform(self, transform: "FeatureTransform") -> None:
        """Register an affine transform from a FeatureTransform (preferred)."""
        center = np.asarray(transform.center, dtype=np.float32)
        scale = np.asarray(transform.scale, dtype=np.float32)
        self._set_affine(center, scale)

    def register_protocol_references(self, scaler: "RobustScaler") -> None:
        """Register affine scaling from an sklearn scaler (historical method name)."""
        center = np.asarray(scaler.center_, dtype=np.float32)
        scale = np.asarray(scaler.scale_, dtype=np.float32)
        self._set_affine(center, scale)

    def _set_affine(self, center: np.ndarray, scale: np.ndarray) -> None:
        if center.shape != (self.n_features,) or scale.shape != (self.n_features,):
            raise ValueError(
                f"transform must match {self.n_features} features, got "
                f"center{center.shape} scale{scale.shape}"
            )
        if np.any(scale == 0.0):
            raise ValueError("transform contains a zero scale")
        device = self.feature_center.device
        self.feature_center = torch.tensor(center, dtype=torch.float32, device=device)
        self.feature_scale = torch.tensor(scale, dtype=torch.float32, device=device)
        self._scaler_registered = True

    def _require_transform(self) -> None:
        if not self._scaler_registered:
            raise TransformNotRegisteredError(
                "decode() called before a feature transform was registered; refusing "
                "silent identity scaling. Call register_feature_transform(...) or "
                "register_protocol_references(scaler)."
            )

    def continuous_scaled_to_raw(self, values: torch.Tensor) -> torch.Tensor:
        self._require_transform()
        center = self.feature_center.to(values.device)
        scale = self.feature_scale.to(values.device)
        return values * scale.unsqueeze(0) + center.unsqueeze(0)

    def continuous_raw_to_scaled(self, values: torch.Tensor) -> torch.Tensor:
        self._require_transform()
        center = self.feature_center.to(values.device)
        scale = self.feature_scale.to(values.device)
        return (values - center.unsqueeze(0)) / scale.unsqueeze(0)

    # ------------------------------------------------------------------ #
    # Legacy CICIoT structured decoder (A0 path)
    # ------------------------------------------------------------------ #
    def _structure_raw(self, raw: torch.Tensor) -> torch.Tensor:
        structured = raw.clone()
        bounded = self.partition["bounded_continuous_idx"]
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

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected (N,{self.n_features}), got {tuple(x.shape)}")
        encoder_x = torch.asinh(x) if self.encoder_input_transform == "asinh" else x
        out = self.encoder_out(self.encoder_body(encoder_x))
        return out[:, : self.latent_dim], out[:, self.latent_dim :].clamp(*self.latent_logvar_bounds)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode_internal(self, z: torch.Tensor) -> dict:
        self._require_transform()
        hidden = self.decoder_body(z)
        if self.decoder_kind == "typed":
            out = self.typed_decoder(hidden)
            raw = out["x_typed"]
            raw_unstructured = out["pre_activation"]
            logvar = out["logvar"]
            scaled = self.continuous_raw_to_scaled(raw)
        else:
            raw_head = self.head_continuous_mu(hidden)
            raw_unstructured = self.continuous_scaled_to_raw(raw_head)
            raw = self._structure_raw(raw_unstructured)
            scaled = self.continuous_raw_to_scaled(raw)
            logvar = self.head_continuous_logvar(hidden).clamp(*self.continuous_logvar_bounds)
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

    def decode(self, z: torch.Tensor) -> dict:
        """Canonical decode returning the full output dict."""
        return self.decode_internal(z)

    def decode_to_39(
        self,
        z: torch.Tensor,
        scaler: "RobustScaler | None" = None,
        mode: str = "soft",
    ) -> tuple[torch.Tensor, dict]:
        """Backward-compatible alias. Name kept for existing attack/diagnostic callers.

        ``scaler`` is ignored (the registered transform is authoritative); ``mode`` is
        accepted for signature compatibility — soft/hard are identical because averaged
        aggregates stay continuous.
        """
        if mode not in {"soft", "hard"}:
            raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")
        decoded = self.decode_internal(z)
        metadata = {
            "continuous_mu": decoded["continuous_mu"],
            "continuous_mu_raw": decoded["continuous_mu_raw"],
            "continuous_logvar": decoded["continuous_logvar"],
            "binary_logits": decoded["binary_logits"],
            "protocol_logits": decoded["protocol_logits"],
            "protocol_idx_batch": None,
        }
        return decoded["continuous_mu"], metadata

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        decoded = self.decode_internal(self.reparameterize(mu, logvar))
        return {"mu": mu, "logvar": logvar, **decoded}
