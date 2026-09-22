"""Generic typed decoder head, driven entirely by a FeatureManifest.

The trunk (``decoder_body``) is shared; this module maps the trunk hidden state to
one output per feature using the feature's ``value_type`` — no dataset-specific
indices. Outputs are produced in RAW feature space:

    real                 x = a
    positive_continuous  x = softplus(a)
    integer_count        x = softplus(a)          (rounding is a separate eval-time
                                                    STE, not applied here, to keep
                                                    training/attack gradients intact)
    probability          x = sigmoid(a)           ([0,1] aggregate)
    bounded_continuous   x = l + (u - l) sigmoid(a)
    binary               x = sigmoid(a)           (STE threshold applied downstream)

``derived`` / ``categorical`` are intentionally rejected here: exact derived
features are recomputed by the constraint layer (Layer 0) from their parents, and
categorical decoding (logits / Gumbel) is a later-phase addition. Raising keeps the
contract honest rather than silently mistyping a feature.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.feature_manifest import FeatureManifest

_SUPPORTED = {"real", "positive_continuous", "integer_count", "probability", "bounded_continuous", "binary"}


class TypedDecoder(nn.Module):
    """Per-feature typed output head over a shared trunk hidden state."""

    def __init__(
        self,
        manifest: FeatureManifest,
        in_dim: int,
        *,
        logvar_bounds: tuple[float, float] = (-7.0, 2.0),
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.n_features = manifest.n_features
        self.logvar_bounds = logvar_bounds

        unsupported = sorted(
            {s.value_type for s in manifest.specs if s.value_type not in _SUPPORTED}
        )
        if unsupported:
            raise NotImplementedError(
                f"TypedDecoder cannot decode value_types {unsupported}; derived "
                "features are recomputed by Layer 0 and categorical decoding is a "
                "later phase. Type these features accordingly in the manifest."
            )

        self.pre = nn.Linear(in_dim, self.n_features)
        self.logvar_head = nn.Linear(in_dim, self.n_features)

        def _idx(vt: str) -> torch.Tensor:
            return torch.tensor(manifest.indices_of_value_type(vt), dtype=torch.long)

        self.register_buffer("idx_positive", _idx("positive_continuous"))
        self.register_buffer("idx_integer", _idx("integer_count"))
        self.register_buffer("idx_prob", _idx("probability"))
        self.register_buffer("idx_binary", _idx("binary"))
        self.register_buffer("idx_bounded", _idx("bounded_continuous"))
        # real features are the untouched complement (identity activation)

        bounded = [s for s in manifest.specs if s.value_type == "bounded_continuous"]
        self.register_buffer(
            "bounded_lower",
            torch.tensor([float(s.lower) for s in bounded], dtype=torch.float32),
        )
        self.register_buffer(
            "bounded_upper",
            torch.tensor([float(s.upper) for s in bounded], dtype=torch.float32),
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        a = self.pre(hidden)  # pre-activation, unstructured raw
        x = a.clone()
        if self.idx_positive.numel():
            x[:, self.idx_positive] = F.softplus(a[:, self.idx_positive])
        if self.idx_integer.numel():
            x[:, self.idx_integer] = F.softplus(a[:, self.idx_integer])
        if self.idx_prob.numel():
            x[:, self.idx_prob] = torch.sigmoid(a[:, self.idx_prob])
        if self.idx_binary.numel():
            x[:, self.idx_binary] = torch.sigmoid(a[:, self.idx_binary])
        if self.idx_bounded.numel():
            lo = self.bounded_lower.unsqueeze(0)
            hi = self.bounded_upper.unsqueeze(0)
            x[:, self.idx_bounded] = lo + (hi - lo) * torch.sigmoid(a[:, self.idx_bounded])
        logvar = self.logvar_head(hidden).clamp(*self.logvar_bounds)
        return {"pre_activation": a, "x_typed": x, "logvar": logvar}
