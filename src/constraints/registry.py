"""Serializable constraint registry.

Maps a constraint ``type`` string to its class so Layer-2 rule sets can be stored as
JSON/YAML and reconstructed against a manifest without importing dataset code into
the generic engine.
"""
from __future__ import annotations

from constraints.base import Constraint
from constraints.layer1 import (
    HalfRangeBound,
    MonotoneNondecreasing,
    ProductEquality,
    RobustTailBound,
)
from datasets.feature_manifest import FeatureManifest

CONSTRAINT_REGISTRY: dict[str, type] = {
    "RobustTailBound": RobustTailBound,
    "ProductEquality": ProductEquality,
    "MonotoneNondecreasing": MonotoneNondecreasing,
    "HalfRangeBound": HalfRangeBound,
}


def register_constraint(cls: type) -> type:
    """Class decorator / helper to add a constraint type to the registry."""
    CONSTRAINT_REGISTRY[cls.__name__] = cls
    return cls


def build_constraint(manifest: FeatureManifest, cfg: dict) -> Constraint:
    ctype = cfg.get("type")
    if ctype not in CONSTRAINT_REGISTRY:
        raise KeyError(f"unknown constraint type {ctype!r}; registered={sorted(CONSTRAINT_REGISTRY)}")
    cls = CONSTRAINT_REGISTRY[ctype]
    if not hasattr(cls, "from_config"):
        raise TypeError(f"{ctype} has no from_config; cannot deserialize")
    return cls.from_config(manifest, cfg)  # type: ignore[attr-defined]
