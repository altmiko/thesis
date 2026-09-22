"""CICIoT2023 dataset adapter.

The manifest is built FROM ``preprocessing.schema`` (the existing single source of
truth) rather than being re-authored, so this adapter cannot drift from the frozen
39-feature contract. Perturbability (``mutable``) and exact derived relationships
are intentionally left un-set here: they are established by later train-mining /
constraint stages, not hand-listed.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from datasets.base import ClassMapping, DatasetAdapter, Split
from datasets.feature_manifest import FeatureManifest, FeatureSpec
from datasets.transforms import FeatureTransform

# preprocessing.schema is importable when repo ``src`` is on sys.path.
from preprocessing.schema import CATEGORY_MAP, FEATURE_SPECS

# CSV representation_type -> generic manifest value_type.
# Note: aggregated flag/service/protocol indicators are window MEANS in [0,1]
# (probability-like), NOT binary. Protocol Type and TTL are bounded continuous
# averaged codes in [0,255], NOT categorical.
_REPR_TO_VALUE_TYPE = {
    "bounded_aggregated_indicator": "probability",
    "bounded_continuous_aggregate": "bounded_continuous",
    "aggregated_code_like": "bounded_continuous",
    "nonnegative_continuous_aggregate": "positive_continuous",
}


def build_manifest() -> FeatureManifest:
    """Construct the CICIoT2023 manifest from the frozen schema specs."""
    specs: list[FeatureSpec] = []
    for idx, s in enumerate(FEATURE_SPECS):
        value_type = _REPR_TO_VALUE_TYPE.get(s.representation_type)
        if value_type is None:
            raise ValueError(
                f"{s.name}: no value_type mapping for representation "
                f"{s.representation_type!r}"
            )
        lower = 0.0 if s.expected_min is None else float(s.expected_min)
        upper = None if s.expected_max is None else float(s.expected_max)
        if value_type == "probability":
            lower, upper = 0.0, 1.0
        specs.append(
            FeatureSpec(
                name=s.name,
                model_index=idx,
                value_type=value_type,
                semantic_type=s.semantic_family,
                lower=lower,
                upper=upper,
                mutable=None,  # mined later (PerturbabilityScorer stage)
                primitive_or_derived="primitive",  # exact derivations set in Phase C
                parents=(),
                derivation=None,
                scaling="robust",  # matches the existing fitted RobustScaler
                aggregation=None,
                protocol_scope=None,
            )
        )
    return FeatureManifest(specs, dataset_name="ciciot2023", dataset_version="modA")


class CICIoT2023Adapter(DatasetAdapter):
    name = "ciciot2023"

    def __init__(self, repo_root: Path | str | None = None) -> None:
        self.repo_root = (
            Path(repo_root)
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self._processed = self.repo_root / "data" / "processed"
        self._manifest: FeatureManifest | None = None
        self._transform: FeatureTransform | None = None

    def feature_manifest(self) -> FeatureManifest:
        if self._manifest is None:
            self._manifest = build_manifest()
        return self._manifest

    def class_mapping(self) -> ClassMapping:
        # sklearn LabelEncoder over the coarse labels is alphabetical, which equals
        # sorted(set(CATEGORY_MAP.values())); verified against category_encoder.pkl.
        names = sorted(set(CATEGORY_MAP.values()))
        return ClassMapping.from_names(names, fine_to_coarse=dict(CATEGORY_MAP))

    def feature_transform(self) -> FeatureTransform:
        if self._transform is None:
            scaler_path = self._processed / "scaler.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"CICIoT2023 scaler not found at {scaler_path}; cannot build "
                    "FeatureTransform without train-fit scaling metadata"
                )
            with open(scaler_path, "rb") as fh:
                scaler = pickle.load(fh)
            self._transform = FeatureTransform.from_sklearn_scaler(
                scaler, self.feature_manifest()
            )
        return self._transform

    def load_split(self, split: str, *, mmap: bool = True) -> Split:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        mmap_mode = "r" if mmap else None
        x = np.load(self._processed / f"X_{split}.npy", mmap_mode=mmap_mode)
        y = np.load(self._processed / f"y_{split}_cat.npy")
        fine_path = self._processed / f"y_{split}.npy"
        y_fine = np.load(fine_path) if fine_path.exists() else None
        manifest = self.feature_manifest()
        manifest.assert_matches_array(x)
        return Split(name=split, x=x, y=y, y_fine=y_fine)
