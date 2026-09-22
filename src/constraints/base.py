"""Constraint layer interfaces.

Constraints operate in RAW feature space (post inverse-transform), the same space
the independent validators use, and resolve features via the manifest — never by
hard-coded indices. Two roles are kept distinct:

* generation-time: Layer 0 is a hard *projector* ``P0``; Layer 1/2 contribute soft
  *penalties* ``C1``/``C2`` used in losses / attack objectives.
* evaluation-time: every constraint also exposes ``validate`` returning a per-sample
  boolean mask, so validity can be checked *independently* of whether the generator
  projected — never by trusting the generator's own projection.

Layers:
    0 = hard, semantically inviolable (representation / datatype / exact identity)
    1 = generic soft constraints (universal algorithm, train-calibrated parameters)
    2 = dataset/extractor-specific or mined soft constraints
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from datasets.feature_manifest import FeatureManifest


@dataclass(frozen=True)
class ConstraintReport:
    name: str
    layer: int
    per_sample: torch.Tensor  # (N,) bool
    pass_rate: float


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.float32)


class Constraint(ABC):
    """A single named constraint bound to a manifest."""

    layer: int = 1

    def __init__(self, name: str, manifest: FeatureManifest) -> None:
        self.name = name
        self.manifest = manifest

    def col(self, name: str) -> int:
        return self.manifest.index_by_name(name)

    @abstractmethod
    def penalty(self, x_raw: torch.Tensor, ctx: dict | None = None) -> torch.Tensor:
        """Return a scalar soft-penalty (>=0), differentiable in ``x_raw``."""

    @abstractmethod
    def validate(self, x_raw: torch.Tensor, ctx: dict | None = None) -> torch.Tensor:
        """Return a per-sample boolean pass mask (N,)."""

    def report(self, x_raw: torch.Tensor, ctx: dict | None = None) -> ConstraintReport:
        mask = self.validate(x_raw, ctx)
        return ConstraintReport(
            name=self.name,
            layer=self.layer,
            per_sample=mask,
            pass_rate=float(mask.float().mean().item()) if mask.numel() else 1.0,
        )

    # subclasses that carry fitted parameters override these for (de)serialization
    def to_config(self) -> dict:
        return {"type": type(self).__name__, "name": self.name, "params": {}}
