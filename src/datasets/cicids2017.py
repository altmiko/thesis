"""CICIDS2017 (DistriNet) dataset adapter.

Processed artifacts live under ``data/processed/CICIDS_2017_Distrinet/`` (RobustScaler-
scaled ``X_{split}.npy`` + raw ``X_{split}_pristine.npy``, ``y_{split}_cat.npy``,
``label_encoders.json``, ``scaler.pkl``, ``preprocessing_manifest.json``). Supporting
CICIDS2017 requires writing ONLY this file: the generic VAE / constraint / attack /
validator code consumes the :class:`FeatureManifest` + :class:`FeatureTransform` and is
never edited.

Feature order is the frozen contract (``preprocessing_manifest.json:modelling_feature_names``,
79 columns). Each ``value_type`` is chosen from CICFlowMeter *extractor semantics* — not
the column name — with hard *semantic* bounds (not train min/max):

* ports / protocol / TCP initial-window bytes are finite-range codes -> ``bounded_continuous``;
* packet/flag/subflow tallies are non-negative integers -> ``integer_count``;
* byte totals, lengths, rates, IAT / active / idle statistics, ratios are non-negative
  reals with no finite upper bound -> ``positive_continuous``.

Perturbability remains train-mined (CFF). Six exact CICFlowMeter identities promoted
after zero train violations and 100% validation pass are declared as derived; Layer 0
recomputes their targets after perturbing primitive parents.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from datasets.base import ClassMapping, DatasetAdapter, Split
from datasets.feature_manifest import FeatureManifest, FeatureSpec, ManifestError
from datasets.transforms import FeatureTransform

# Finite hard ranges for the bounded codes (protocol number space / 16-bit fields).
_PORT_MAX = 65535.0
_PROTOCOL_MAX = 255.0
_WINDOW_MAX = 65535.0

# name -> (value_type, lower, upper, semantic_type). ``upper=None`` means no finite
# semantic upper bound. Chosen by extractor semantics; see module docstring.
_TYPING: dict[str, tuple[str, float, float | None, str]] = {
    "Src Port": ("bounded_continuous", 0.0, _PORT_MAX, "port"),
    "Dst Port": ("bounded_continuous", 0.0, _PORT_MAX, "port"),
    "Protocol": ("bounded_continuous", 0.0, _PROTOCOL_MAX, "protocol"),
    "Flow Duration": ("positive_continuous", 0.0, None, "duration"),
    "Total Fwd Packet": ("integer_count", 0.0, None, "packet_count"),
    "Total Bwd packets": ("integer_count", 0.0, None, "packet_count"),
    "Total Length of Fwd Packet": ("positive_continuous", 0.0, None, "byte_total"),
    "Total Length of Bwd Packet": ("positive_continuous", 0.0, None, "byte_total"),
    "Fwd Packet Length Max": ("positive_continuous", 0.0, None, "length_stat"),
    "Fwd Packet Length Min": ("positive_continuous", 0.0, None, "length_stat"),
    "Fwd Packet Length Mean": ("positive_continuous", 0.0, None, "length_stat"),
    "Fwd Packet Length Std": ("positive_continuous", 0.0, None, "length_stat"),
    "Bwd Packet Length Max": ("positive_continuous", 0.0, None, "length_stat"),
    "Bwd Packet Length Min": ("positive_continuous", 0.0, None, "length_stat"),
    "Bwd Packet Length Mean": ("positive_continuous", 0.0, None, "length_stat"),
    "Bwd Packet Length Std": ("positive_continuous", 0.0, None, "length_stat"),
    "Flow Bytes/s": ("positive_continuous", 0.0, None, "rate"),
    "Flow Packets/s": ("positive_continuous", 0.0, None, "rate"),
    "Flow IAT Mean": ("positive_continuous", 0.0, None, "iat_stat"),
    "Flow IAT Std": ("positive_continuous", 0.0, None, "iat_stat"),
    "Flow IAT Max": ("positive_continuous", 0.0, None, "iat_stat"),
    "Flow IAT Min": ("positive_continuous", 0.0, None, "iat_stat"),
    "Fwd IAT Total": ("positive_continuous", 0.0, None, "iat_stat"),
    "Fwd IAT Mean": ("positive_continuous", 0.0, None, "iat_stat"),
    "Fwd IAT Std": ("positive_continuous", 0.0, None, "iat_stat"),
    "Fwd IAT Max": ("positive_continuous", 0.0, None, "iat_stat"),
    "Fwd IAT Min": ("positive_continuous", 0.0, None, "iat_stat"),
    "Bwd IAT Total": ("positive_continuous", 0.0, None, "iat_stat"),
    "Bwd IAT Mean": ("positive_continuous", 0.0, None, "iat_stat"),
    "Bwd IAT Std": ("positive_continuous", 0.0, None, "iat_stat"),
    "Bwd IAT Max": ("positive_continuous", 0.0, None, "iat_stat"),
    "Bwd IAT Min": ("positive_continuous", 0.0, None, "iat_stat"),
    "Fwd PSH Flags": ("integer_count", 0.0, None, "flag_count"),
    "Bwd PSH Flags": ("integer_count", 0.0, None, "flag_count"),
    "Fwd URG Flags": ("integer_count", 0.0, None, "flag_count"),
    "Bwd URG Flags": ("integer_count", 0.0, None, "flag_count"),
    "Fwd Header Length": ("integer_count", 0.0, None, "header"),
    "Bwd Header Length": ("integer_count", 0.0, None, "header"),
    "Fwd Packets/s": ("positive_continuous", 0.0, None, "rate"),
    "Bwd Packets/s": ("positive_continuous", 0.0, None, "rate"),
    "Packet Length Min": ("positive_continuous", 0.0, None, "length_stat"),
    "Packet Length Max": ("positive_continuous", 0.0, None, "length_stat"),
    "Packet Length Mean": ("positive_continuous", 0.0, None, "length_stat"),
    "Packet Length Std": ("positive_continuous", 0.0, None, "length_stat"),
    "Packet Length Variance": ("positive_continuous", 0.0, None, "length_stat"),
    "FIN Flag Count": ("integer_count", 0.0, None, "flag_count"),
    "SYN Flag Count": ("integer_count", 0.0, None, "flag_count"),
    "RST Flag Count": ("integer_count", 0.0, None, "flag_count"),
    "PSH Flag Count": ("integer_count", 0.0, None, "flag_count"),
    "ACK Flag Count": ("integer_count", 0.0, None, "flag_count"),
    "URG Flag Count": ("integer_count", 0.0, None, "flag_count"),
    "CWR Flag Count": ("integer_count", 0.0, None, "flag_count"),
    "ECE Flag Count": ("integer_count", 0.0, None, "flag_count"),
    "Down/Up Ratio": ("positive_continuous", 0.0, None, "ratio"),
    "Average Packet Size": ("positive_continuous", 0.0, None, "length_stat"),
    "Fwd Segment Size Avg": ("positive_continuous", 0.0, None, "length_stat"),
    "Bwd Segment Size Avg": ("positive_continuous", 0.0, None, "length_stat"),
    "Fwd Bytes/Bulk Avg": ("positive_continuous", 0.0, None, "bulk"),
    "Fwd Packet/Bulk Avg": ("positive_continuous", 0.0, None, "bulk"),
    "Fwd Bulk Rate Avg": ("positive_continuous", 0.0, None, "bulk"),
    "Bwd Bytes/Bulk Avg": ("positive_continuous", 0.0, None, "bulk"),
    "Bwd Packet/Bulk Avg": ("positive_continuous", 0.0, None, "bulk"),
    "Bwd Bulk Rate Avg": ("positive_continuous", 0.0, None, "bulk"),
    "Subflow Fwd Packets": ("integer_count", 0.0, None, "subflow"),
    "Subflow Fwd Bytes": ("positive_continuous", 0.0, None, "subflow"),
    "Subflow Bwd Packets": ("integer_count", 0.0, None, "subflow"),
    "Subflow Bwd Bytes": ("positive_continuous", 0.0, None, "subflow"),
    "FWD Init Win Bytes": ("bounded_continuous", 0.0, _WINDOW_MAX, "window_bytes"),
    "Bwd Init Win Bytes": ("bounded_continuous", 0.0, _WINDOW_MAX, "window_bytes"),
    "Fwd Act Data Pkts": ("integer_count", 0.0, None, "packet_count"),
    "Fwd Seg Size Min": ("integer_count", 0.0, None, "header"),
    "Active Mean": ("positive_continuous", 0.0, None, "active_idle"),
    "Active Std": ("positive_continuous", 0.0, None, "active_idle"),
    "Active Max": ("positive_continuous", 0.0, None, "active_idle"),
    "Active Min": ("positive_continuous", 0.0, None, "active_idle"),
    "Idle Mean": ("positive_continuous", 0.0, None, "active_idle"),
    "Idle Std": ("positive_continuous", 0.0, None, "active_idle"),
    "Idle Max": ("positive_continuous", 0.0, None, "active_idle"),
    "Idle Min": ("positive_continuous", 0.0, None, "active_idle"),
}

# target -> (generic derivation tag, parents). These identities had zero violations
# across all 1,456,265 train rows and 100% pass across all 312,058 validation rows.
_DERIVATIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Packet Length Variance": ("square", ("Packet Length Std",)),
    "Average Packet Size": ("identity", ("Packet Length Mean",)),
    "Fwd Segment Size Avg": ("identity", ("Fwd Packet Length Mean",)),
    "Bwd Segment Size Avg": ("identity", ("Bwd Packet Length Mean",)),
    "Total Length of Fwd Packet": (
        "product",
        ("Total Fwd Packet", "Fwd Packet Length Mean"),
    ),
    "Total Length of Bwd Packet": (
        "product",
        ("Total Bwd packets", "Bwd Packet Length Mean"),
    ),
}


class CICIDS2017Adapter(DatasetAdapter):
    name = "cicids2017_distrinet"
    # Subclasses sharing the 79-feature CICFlowMeter layout override these three.
    processed_dirname = "CICIDS_2017_Distrinet"
    typing: dict[str, tuple[str, float, float | None, str]] = _TYPING
    derivations: dict[str, tuple[str, tuple[str, ...]]] = _DERIVATIONS

    def __init__(self, repo_root: Path | str | None = None) -> None:
        self.repo_root = (
            Path(repo_root)
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self._processed = self.repo_root / "data" / "processed" / self.processed_dirname
        self._manifest: FeatureManifest | None = None
        self._transform: FeatureTransform | None = None

    # -- feature order contract -------------------------------------------------
    def _feature_order(self) -> list[str]:
        man_path = self._processed / "preprocessing_manifest.json"
        if not man_path.exists():
            raise FileNotFoundError(f"missing {man_path}")
        payload = json.loads(man_path.read_text(encoding="utf-8"))
        names = payload.get("modelling_feature_names")
        if not names:
            raise ManifestError("preprocessing_manifest.json lacks modelling_feature_names")
        return list(names)

    def feature_manifest(self) -> FeatureManifest:
        if self._manifest is not None:
            return self._manifest
        names = self._feature_order()
        specs: list[FeatureSpec] = []
        for idx, name in enumerate(names):
            typing = self.typing.get(name)
            if typing is None:
                raise ManifestError(
                    f"no value_type declared for {self.name} feature {name!r}; "
                    f"extend {type(self).__module__}.typing"
                )
            value_type, lower, upper, semantic = typing
            derived = self.derivations.get(name)
            derivation, parents = derived if derived is not None else (None, ())
            specs.append(
                FeatureSpec(
                    name=name,
                    model_index=idx,
                    value_type=value_type,
                    semantic_type=semantic,
                    lower=lower,
                    upper=upper,
                    mutable=None,  # class-conditional CFF mask is loaded by the runner
                    primitive_or_derived="derived" if derived is not None else "primitive",
                    parents=parents,
                    derivation=derivation,
                    scaling="robust",  # matches the fitted RobustScaler (scaler.pkl)
                    aggregation=None,
                    protocol_scope=None,
                )
            )
        self._manifest = FeatureManifest(
            specs, dataset_name=self.name, dataset_version="distrinet"
        )
        return self._manifest

    def class_mapping(self) -> ClassMapping:
        enc_path = self._processed / "label_encoders.json"
        if not enc_path.exists():
            raise FileNotFoundError(f"missing {enc_path}")
        encoders = json.loads(enc_path.read_text(encoding="utf-8"))
        category = encoders.get("category")
        if not category:
            raise ManifestError("label_encoders.json lacks a 'category' encoder")
        # names ordered by encoder id: {Benign:0, DoS:1, DDoS:2, Recon:3, BruteForce:4}
        names = [name for name, _ in sorted(category.items(), key=lambda kv: kv[1])]
        return ClassMapping.from_names(names)

    def feature_transform(self) -> FeatureTransform:
        if self._transform is not None:
            return self._transform
        scaler_path = self._processed / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"{self.name} scaler not found at {scaler_path}; cannot build "
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
        manifest = self.feature_manifest()
        manifest.assert_matches_array(x)
        return Split(name=split, x=x, y=np.asarray(y, dtype=np.int64))
