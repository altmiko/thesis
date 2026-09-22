"""Dataset-specific perturbation masks with a differentiable dependency stage.

A :class:`DatasetMask` declares, by *feature name* (never by raw column index), the
attacker-facing role of every feature in a dataset:

* ``PERTURBABLE``    - attacker-controlled degrees of freedom. Only these receive a
  direct latent/residual update from the VAE attack.
* ``DERIVED_EXACT``  - not independently optimized. Recomputed deterministically from
  perturbable/frozen parents by an *exact*, extractor-consistent formula. Because the
  recompute is plain differentiable tensor algebra, classifier-loss gradients flow
  through a derived feature into its perturbable parents.
* ``FROZEN``         - copied verbatim from the original sample.

The mask is *resolved* against a :class:`~datasets.feature_manifest.FeatureManifest`
(the frozen feature-order contract). Resolution fails loudly on any missing / duplicated
feature, on a partition that does not cover the manifest exactly, or on an expected
1-indexed position that does not match the manifest - the mask can never silently
mis-index a checkpoint or scaler.

Nothing here is dataset-specific: the concrete perturbable list and the verified
formulas live in ``attack.masks.<dataset>``; this module only wires them to a manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

import torch

from constraints.layer0 import Layer0Projector
from datasets.feature_manifest import FeatureManifest, ManifestError

# A derivation formula maps (raw batch, name->column index) -> the derived column.
DerivationFn = Callable[[torch.Tensor, dict[str, int]], torch.Tensor]


class FeatureState(str, Enum):
    PERTURBABLE = "PERTURBABLE"
    DERIVED_EXACT = "DERIVED_EXACT"
    FROZEN = "FROZEN"


@dataclass(frozen=True)
class DerivedFeature:
    """One deterministic dependency: ``name = formula(parents)`` in raw units."""

    name: str
    parents: tuple[str, ...]
    formula: DerivationFn
    expression: str  # human-readable identity, e.g. "TLf / max(count_fwd, 1)"
    source: str  # verification provenance (code / audit / empirical check)


@dataclass(frozen=True)
class DatasetMask:
    """Declarative, name-based perturbation mask for one dataset.

    ``derived`` is stored in application (topological) order: a derived feature may
    read an earlier derived feature as a parent (e.g. a segment-size average that
    equals a just-recomputed mean).
    """

    dataset_name: str
    n_features: int
    perturbable: tuple[str, ...]
    derived: tuple[DerivedFeature, ...]
    expected_perturbable_index1: tuple[int, ...]
    notes: str = ""

    def resolve(self, manifest: FeatureManifest) -> "ResolvedMask":
        if manifest.dataset_name != self.dataset_name:
            raise ManifestError(
                f"mask dataset {self.dataset_name!r} != manifest dataset "
                f"{manifest.dataset_name!r}"
            )
        if manifest.n_features != self.n_features:
            raise ManifestError(
                f"mask expects {self.n_features} features but manifest has "
                f"{manifest.n_features} for {self.dataset_name!r}"
            )
        names = manifest.names
        name_to_idx = {n: i for i, n in enumerate(names)}

        derived_names = tuple(d.name for d in self.derived)
        # 1) every declared feature must exist and be unique.
        for group, label in ((self.perturbable, "PERTURBABLE"), (derived_names, "DERIVED_EXACT")):
            seen: set[str] = set()
            for nm in group:
                if nm not in name_to_idx:
                    raise ManifestError(f"{label} feature {nm!r} not in manifest {self.dataset_name!r}")
                if nm in seen:
                    raise ManifestError(f"duplicate {label} feature {nm!r}")
                seen.add(nm)
        overlap = set(self.perturbable) & set(derived_names)
        if overlap:
            raise ManifestError(f"features declared both PERTURBABLE and DERIVED_EXACT: {sorted(overlap)}")

        # 2) expected 1-indexed positions must match the manifest exactly.
        got_index1 = tuple(name_to_idx[n] + 1 for n in self.perturbable)
        if tuple(self.expected_perturbable_index1) != got_index1:
            raise ManifestError(
                f"{self.dataset_name}: PERTURBABLE 1-indexed positions "
                f"{got_index1} != expected {tuple(self.expected_perturbable_index1)}"
            )

        # 3) derived parents must resolve; a derived parent may only be perturbable,
        #    frozen, or an *earlier* derived feature (no forward / cyclic reference).
        available: set[str] = set(self.perturbable)
        frozen_and_perturbable = set(names) - set(derived_names)
        available |= frozen_and_perturbable
        emitted: set[str] = set()
        for d in self.derived:
            for parent in d.parents:
                if parent not in name_to_idx:
                    raise ManifestError(f"derived {d.name!r} parent {parent!r} not in manifest")
                if parent in derived_names and parent not in emitted:
                    raise ManifestError(
                        f"derived {d.name!r} references derived parent {parent!r} before it is "
                        "computed; reorder DatasetMask.derived topologically"
                    )
            emitted.add(d.name)

        perturbable_idx = tuple(name_to_idx[n] for n in self.perturbable)
        derived_idx = tuple(name_to_idx[d.name] for d in self.derived)
        frozen_idx = tuple(i for i in range(self.n_features) if names[i] not in set(self.perturbable) | set(derived_names))
        # 4) exact partition of the whole feature vector.
        if len(perturbable_idx) + len(derived_idx) + len(frozen_idx) != self.n_features:
            raise ManifestError("PERTURBABLE / DERIVED_EXACT / FROZEN do not partition the feature vector")

        return ResolvedMask(
            mask=self,
            manifest=manifest,
            name_to_idx=name_to_idx,
            perturbable_idx=perturbable_idx,
            derived_idx=derived_idx,
            frozen_idx=frozen_idx,
        )


class ResolvedMask:
    """A :class:`DatasetMask` bound to a concrete manifest (indices + tensors)."""

    def __init__(
        self,
        *,
        mask: DatasetMask,
        manifest: FeatureManifest,
        name_to_idx: dict[str, int],
        perturbable_idx: tuple[int, ...],
        derived_idx: tuple[int, ...],
        frozen_idx: tuple[int, ...],
    ) -> None:
        self.mask = mask
        self.manifest = manifest
        self.name_to_idx = name_to_idx
        self.n_features = manifest.n_features
        self.perturbable_idx = perturbable_idx
        self.derived_idx = derived_idx
        self.frozen_idx = frozen_idx
        self._perturbable_t = torch.as_tensor(perturbable_idx, dtype=torch.long)
        self._derived_t = torch.as_tensor(derived_idx, dtype=torch.long)
        self._frozen_t = torch.as_tensor(frozen_idx, dtype=torch.long)

    # -- masks -----------------------------------------------------------------
    def perturbable_mask(self) -> torch.Tensor:
        m = torch.zeros(self.n_features, dtype=torch.bool)
        if self._perturbable_t.numel():
            m[self._perturbable_t] = True
        return m

    def frozen_mask(self) -> torch.Tensor:
        m = torch.zeros(self.n_features, dtype=torch.bool)
        if self._frozen_t.numel():
            m[self._frozen_t] = True
        return m

    # -- generator wiring ------------------------------------------------------
    def generator_projector(self) -> Layer0Projector:
        """Layer-0 projector that clamps value-type domains and preserves immutables
        but performs **no** manifest-declared derived recompute.

        The manifest may declare *inverse* derivations (e.g. total-length = count x
        mean) that would clobber a feature this mask treats as directly perturbable.
        Stripping them and applying only this mask's verified derivations keeps the
        two views from fighting.
        """
        primitive_specs = [
            replace(s, primitive_or_derived="primitive", derivation=None, parents=())
            for s in self.manifest.specs
        ]
        primitive_manifest = FeatureManifest(
            primitive_specs,
            dataset_name=self.manifest.dataset_name,
            dataset_version=self.manifest.dataset_version,
            schema_version=self.manifest.schema_version,
        )
        return Layer0Projector(primitive_manifest)

    # -- differentiable dependency stage --------------------------------------
    def restore_frozen(self, raw: torch.Tensor, raw_original: torch.Tensor) -> torch.Tensor:
        if not self._frozen_t.numel():
            return raw
        out = raw.clone()
        idx = self._frozen_t.to(raw.device)
        out[:, idx] = raw_original[:, idx]
        return out

    def recompute(self, raw: torch.Tensor) -> torch.Tensor:
        """Overwrite every DERIVED_EXACT column with its formula, in declared order.

        Differentiable: each column is plain tensor algebra over parent columns, so a
        classifier gradient reaching a derived feature flows on to its perturbable
        parents.
        """
        out = raw
        for d in self.mask.derived:
            col = d.formula(out, self.name_to_idx)
            out = out.clone()
            out[:, self.name_to_idx[d.name]] = col
        return out

    def apply(self, raw_generated: torch.Tensor, raw_original: torch.Tensor) -> torch.Tensor:
        """Full dependency stage: restore frozen, then recompute exact dependencies."""
        return self.recompute(self.restore_frozen(raw_generated, raw_original))

    # -- sanity assertions -----------------------------------------------------
    def frozen_violation_mask(
        self, raw_adv: torch.Tensor, raw_original: torch.Tensor, *, atol: float = 1e-5, rtol: float = 1e-4
    ) -> torch.Tensor:
        """Per-sample bool: True where any FROZEN feature moved beyond tolerance."""
        if not self._frozen_t.numel():
            return torch.zeros(raw_adv.shape[0], dtype=torch.bool, device=raw_adv.device)
        idx = self._frozen_t.to(raw_adv.device)
        a = raw_adv[:, idx]
        b = raw_original[:, idx]
        moved = (a - b).abs() > (atol + rtol * b.abs())
        return moved.any(dim=1)

    def derived_consistency_mask(
        self,
        raw_adv: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        tol: float = 1e-3,
        atol: float = 1e-4,
        rtol: float = 1e-3,
    ) -> torch.Tensor:
        """Per-sample bool: True where a DERIVED feature disagrees with its formula.

        When a per-feature ``scale`` (the transform's robust scale) is given the
        comparison is done in model space (``|a-b| / scale <= tol``), which matches the
        precision the classifier and the stored float32 artifact actually operate in;
        otherwise a raw-space ``atol + rtol*|b|`` comparison is used.
        """
        if not self._derived_t.numel():
            return torch.zeros(raw_adv.shape[0], dtype=torch.bool, device=raw_adv.device)
        recomputed = self.recompute(raw_adv)
        idx = self._derived_t.to(raw_adv.device)
        a = raw_adv[:, idx]
        b = recomputed[:, idx]
        if scale is not None:
            bad = (a - b).abs() / scale.to(raw_adv.device)[idx] > tol
        else:
            bad = (a - b).abs() > (atol + rtol * b.abs())
        return bad.any(dim=1)

    def derived_consistency_counts(
        self, raw_adv: torch.Tensor, *, scale: torch.Tensor | None = None, tol: float = 1e-3,
        row_mask: torch.Tensor | None = None,
    ) -> dict[str, int]:
        """Per-derived-feature count of formula disagreements (model space if scaled)."""
        if not self._derived_t.numel():
            return {}
        recomputed = self.recompute(raw_adv)
        idx = self._derived_t.to(raw_adv.device)
        a = raw_adv[:, idx]
        b = recomputed[:, idx]
        if scale is not None:
            bad = (a - b).abs() / scale.to(raw_adv.device)[idx] > tol
        else:
            bad = (a - b).abs() > (1e-4 + 1e-3 * b.abs())
        if row_mask is not None:
            bad = bad[row_mask]
        counts = bad.long().sum(dim=0)
        return {self.mask.derived[j].name: int(counts[j]) for j in range(len(self._derived_t))}

    def per_feature_perturbation_frequency(
        self,
        raw_adv: torch.Tensor,
        raw_original: torch.Tensor,
        row_mask: torch.Tensor | None = None,
        *,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> dict[str, float]:
        """Fraction of (selected) rows in which each feature changed."""
        moved = (raw_adv - raw_original).abs() > (atol + rtol * raw_original.abs())
        if row_mask is not None:
            moved = moved[row_mask]
        denom = max(int(moved.shape[0]), 1)
        freqs = moved.float().sum(dim=0) / denom
        return {name: float(freqs[i]) for i, name in enumerate(self.manifest.names)}

    # -- diagnostics -----------------------------------------------------------
    def diagnostics(self) -> str:
        names = self.manifest.names
        lines = [
            f"Dataset: {self.manifest.dataset_name}",
            f"Features: {self.n_features}",
            "",
            f"PERTURBABLE ({len(self.perturbable_idx)}):",
        ]
        for i in self.perturbable_idx:
            lines.append(f"  {i:>2d}  {names[i]}")
        lines.append("")
        lines.append(f"DERIVED_EXACT ({len(self.derived_idx)}):")
        for d in self.mask.derived:
            i = self.name_to_idx[d.name]
            lines.append(f"  {i:>2d}  {d.name} = {d.expression}")
        lines.append("")
        lines.append(f"FROZEN: {len(self.frozen_idx)} features")
        return "\n".join(lines)
