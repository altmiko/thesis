"""Generic, dataset-agnostic feature/manifest/transform layer.

Public API used by the VAE, constraint, and attack code:
    FeatureSpec, FeatureManifest       -- feature semantics + ordering contract
    FeatureTransform                   -- train-only raw<->model scaling
    DatasetAdapter, ClassMapping, Split-- dataset interface
    get_adapter(name)                  -- adapter registry
"""
from __future__ import annotations

from datasets.base import ClassMapping, DatasetAdapter, Split
from datasets.feature_manifest import (
    MANIFEST_SCHEMA_VERSION,
    VALUE_TYPES,
    FeatureManifest,
    FeatureSpec,
    ManifestError,
)
from datasets.transforms import FeatureTransform, TransformNotFittedError

__all__ = [
    "ClassMapping",
    "DatasetAdapter",
    "Split",
    "FeatureManifest",
    "FeatureSpec",
    "ManifestError",
    "VALUE_TYPES",
    "MANIFEST_SCHEMA_VERSION",
    "FeatureTransform",
    "TransformNotFittedError",
    "get_adapter",
]


def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    """Return a dataset adapter by name (lazy import to avoid heavy deps)."""
    key = name.lower()
    if key in {"ciciot2023", "ciciot"}:
        from datasets.ciciot2023 import CICIoT2023Adapter

        return CICIoT2023Adapter(**kwargs)
    if key in {"cicids2017", "cicids2017_distrinet", "cicids"}:
        from datasets.cicids2017 import CICIDS2017Adapter

        return CICIDS2017Adapter(**kwargs)
    if key in {"cicids2018", "cicids2018_distrinet", "cse-cic-ids-2018"}:
        from datasets.cicids2018 import CICIDS2018Adapter

        return CICIDS2018Adapter(**kwargs)
    raise KeyError(f"unknown dataset adapter {name!r}")
