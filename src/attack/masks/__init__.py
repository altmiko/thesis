"""Dataset-aware perturbation masks (PERTURBABLE / DERIVED_EXACT / FROZEN).

Attack scripts request a mask by dataset name via :func:`get_dataset_mask` instead of
hard-coding column indices; the returned :class:`DatasetMask` is resolved against a
:class:`~datasets.feature_manifest.FeatureManifest` (fail-loud on any schema mismatch).
"""
from __future__ import annotations

from attack.masks.base import (
    DatasetMask,
    DerivedFeature,
    FeatureState,
    ResolvedMask,
)

_ALIASES = {
    "ciciot": "ciciot2023",
    "ciciot2023": "ciciot2023",
    "cicids": "cicids2017_distrinet",
    "cicids2017": "cicids2017_distrinet",
    "cicids2017_distrinet": "cicids2017_distrinet",
    "cicids2018": "cicids2018_distrinet",
    "cicids2018_distrinet": "cicids2018_distrinet",
}


def get_dataset_mask(name: str) -> DatasetMask:
    key = _ALIASES.get(name.lower())
    if key == "ciciot2023":
        from attack.masks.ciciot2023 import build_mask

        return build_mask()
    if key in ("cicids2017_distrinet", "cicids2018_distrinet"):
        from attack.masks.cicids2017_distrinet import build_mask

        return build_mask(key)
    raise KeyError(f"no perturbation mask registered for dataset {name!r}")


__all__ = [
    "DatasetMask",
    "DerivedFeature",
    "FeatureState",
    "ResolvedMask",
    "get_dataset_mask",
]
