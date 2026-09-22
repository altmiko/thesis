"""Dataset adapter interface.

A ``DatasetAdapter`` is the *only* place a concrete dataset (its feature schema,
label taxonomy, on-disk arrays, and optional mined constraints) is described.
Generic VAE / constraint / attack code depends on this interface, never on a
specific dataset module. Adding a new flow-based NIDS dataset means writing one
adapter + manifest; it must not require editing the model, attack, or validators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from datasets.feature_manifest import FeatureManifest
from datasets.transforms import FeatureTransform


@dataclass(frozen=True)
class ClassMapping:
    """Coarse (attack-category) label taxonomy used by the per-class VAEs."""

    names: tuple[str, ...]  # coarse class names, index == class id
    name_to_id: dict[str, int]
    id_to_name: dict[int, str]
    fine_to_coarse: dict[str, str] = field(default_factory=dict)  # optional 34->8 map

    @classmethod
    def from_names(
        cls, names: list[str], fine_to_coarse: dict[str, str] | None = None
    ) -> "ClassMapping":
        names_t = tuple(names)
        name_to_id = {n: i for i, n in enumerate(names_t)}
        id_to_name = {i: n for i, n in enumerate(names_t)}
        return cls(names_t, name_to_id, id_to_name, dict(fine_to_coarse or {}))

    @property
    def n_classes(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class Split:
    """One loaded data split in model (scaled) space with coarse labels."""

    name: str  # "train" | "val" | "test"
    x: np.ndarray  # (N, F) scaled features
    y: np.ndarray  # (N,) coarse class ids
    y_fine: np.ndarray | None = None  # optional fine labels


class DatasetAdapter(ABC):
    """Contract every dataset must implement."""

    #: stable dataset identifier, e.g. "ciciot2023"
    name: str = "unnamed"

    @abstractmethod
    def feature_manifest(self) -> FeatureManifest:
        """Return the validated feature manifest (feature order contract)."""

    @abstractmethod
    def class_mapping(self) -> ClassMapping:
        """Return the coarse class taxonomy used for per-class VAEs."""

    @abstractmethod
    def feature_transform(self) -> FeatureTransform:
        """Return the TRAIN-fit (or externally-wrapped) feature transform."""

    @abstractmethod
    def load_split(self, split: str) -> Split:
        """Load one of {"train","val","test"} as a :class:`Split`."""

    def dataset_constraints(self) -> dict[str, Any]:
        """Optional Layer-2 dataset/extractor-specific or mined constraints.

        Default: none. Adapters may return a serializable rule set consumed by the
        constraint engine (constraints package, later phase).
        """
        return {}
