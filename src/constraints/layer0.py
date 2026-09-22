"""Layer 0 — hard, semantically inviolable projector (P0).

These follow from representation / datatype / measurement semantics and exact
mathematical identities, NOT from observed training ranges. The projector is
manifest-driven and differentiable (clamps have subgradients), so it can sit inside
gradient-based attacks.

Responsibilities:
* value-type domains: probability -> [0,1]; positive_continuous / integer_count ->
  [0, inf); bounded_continuous -> [lower, upper]; binary -> [0,1] (STE handled by
  caller); real -> untouched.
* immutable preservation: features flagged non-mutable are copied from the source
  sample (attack setting) rather than regenerated.
* exact derived recompute: features with an exact ``derivation`` are recomputed from
  their parents (hook; no exact derivations declared for CICIoT yet).

Layer 0 is structural, not a soft penalty. It should make Layer-0 validity ~100% by
construction.
"""
from __future__ import annotations

import torch

from datasets.feature_manifest import FeatureManifest


class Layer0Projector:
    def __init__(self, manifest: FeatureManifest) -> None:
        self.manifest = manifest
        n = manifest.n_features
        self.idx_prob = torch.tensor(manifest.indices_of_value_type("probability"), dtype=torch.long)
        self.idx_binary = torch.tensor(manifest.indices_of_value_type("binary"), dtype=torch.long)
        self.idx_pos = torch.tensor(
            manifest.indices_of_value_type("positive_continuous")
            + manifest.indices_of_value_type("integer_count"),
            dtype=torch.long,
        )
        bnd = [s for s in manifest.specs if s.value_type == "bounded_continuous"]
        self.idx_bounded = torch.tensor([s.model_index for s in bnd], dtype=torch.long)
        self.bounded_lower = torch.tensor([float(s.lower) for s in bnd], dtype=torch.float32)
        self.bounded_upper = torch.tensor([float(s.upper) for s in bnd], dtype=torch.float32)
        # Exact derived recompute hooks declared by the manifest. Each tuple is
        # (target index, generic operation tag, parent indices).
        self._derivations: list[tuple[int, str, tuple[int, ...]]] = []
        for spec in manifest.specs:
            if spec.primitive_or_derived != "derived":
                continue
            tag = spec.derivation
            if tag not in {"identity", "square", "product"}:
                raise ValueError(f"{spec.name}: unsupported derivation {tag!r}")
            parent_indices = tuple(manifest.index_by_name(name) for name in spec.parents)
            if tag in {"identity", "square"} and len(parent_indices) != 1:
                raise ValueError(f"{spec.name}: {tag} requires exactly one parent")
            if tag == "product" and not parent_indices:
                raise ValueError(f"{spec.name}: product requires at least one parent")
            self._derivations.append((spec.model_index, tag, parent_indices))
        self.n_features = n

    def project(
        self,
        x_raw: torch.Tensor,
        x_source: torch.Tensor | None = None,
        mutable_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project ``x_raw`` onto the Layer-0 feasible set.

        ``x_source`` + ``mutable_mask`` (1=mutable, 0=frozen) restore immutable
        features to their source values (attack setting).
        """
        x = x_raw.clone()
        if self.idx_prob.numel():
            x[:, self.idx_prob] = x[:, self.idx_prob].clamp(0.0, 1.0)
        if self.idx_binary.numel():
            x[:, self.idx_binary] = x[:, self.idx_binary].clamp(0.0, 1.0)
        if self.idx_pos.numel():
            x[:, self.idx_pos] = x[:, self.idx_pos].clamp_min(0.0)
        if self.idx_bounded.numel():
            lo = self.bounded_lower.to(x.device)
            hi = self.bounded_upper.to(x.device)
            x[:, self.idx_bounded] = torch.max(
                torch.min(x[:, self.idx_bounded], hi.unsqueeze(0)), lo.unsqueeze(0)
            )
        # immutable preservation
        if x_source is not None and mutable_mask is not None:
            m = mutable_mask.to(x.device).bool()
            if m.ndim == 1:
                m = m.unsqueeze(0)
            x = torch.where(m, x, x_source.to(x.device))
        # Exact derived recompute from post-projection primitive parents.
        for target_idx, tag, parent_indices in self._derivations:
            parents = x[:, list(parent_indices)]
            if tag == "identity":
                derived = parents[:, 0]
            elif tag == "square":
                derived = parents[:, 0].square()
            else:
                derived = parents.prod(dim=1)
            x[:, target_idx] = derived
        return x

    def validate(self, x_raw: torch.Tensor, tol: float = 1e-4) -> torch.Tensor:
        """Per-sample: all features within their Layer-0 domain."""
        x = x_raw
        n = x.shape[0]
        ok = torch.ones(n, dtype=torch.bool, device=x.device)
        if self.idx_prob.numel():
            sub = x[:, self.idx_prob]
            ok &= ((sub >= -tol) & (sub <= 1 + tol)).all(dim=1)
        if self.idx_binary.numel():
            sub = x[:, self.idx_binary]
            ok &= ((sub >= -tol) & (sub <= 1 + tol)).all(dim=1)
        if self.idx_pos.numel():
            ok &= (x[:, self.idx_pos] >= -tol).all(dim=1)
        if self.idx_bounded.numel():
            lo = self.bounded_lower.to(x.device).unsqueeze(0)
            hi = self.bounded_upper.to(x.device).unsqueeze(0)
            sub = x[:, self.idx_bounded]
            ok &= ((sub >= lo - tol) & (sub <= hi + tol)).all(dim=1)
        return ok

    def soft_penalty(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Differentiable pre-projection regularizer C0 (mean domain violation).

        Encourages the decoder to produce in-domain values *before* projection.
        Zero for values already within the Layer-0 feasible set.
        """
        pen = x_raw.new_tensor(0.0)
        terms = 0
        if self.idx_prob.numel():
            s = x_raw[:, self.idx_prob]
            pen = pen + (torch.relu(-s) + torch.relu(s - 1.0)).mean()
            terms += 1
        if self.idx_binary.numel():
            s = x_raw[:, self.idx_binary]
            pen = pen + (torch.relu(-s) + torch.relu(s - 1.0)).mean()
            terms += 1
        if self.idx_pos.numel():
            pen = pen + torch.relu(-x_raw[:, self.idx_pos]).mean()
            terms += 1
        if self.idx_bounded.numel():
            lo = self.bounded_lower.to(x_raw.device).unsqueeze(0)
            hi = self.bounded_upper.to(x_raw.device).unsqueeze(0)
            s = x_raw[:, self.idx_bounded]
            pen = pen + (torch.relu(lo - s) + torch.relu(s - hi)).mean()
            terms += 1
        return pen / max(terms, 1)
