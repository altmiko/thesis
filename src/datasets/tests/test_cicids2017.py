"""CICIDS2017-DistriNet adapter contract tests against real processed artifacts."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from datasets import get_adapter
from datasets.cicids2017 import CICIDS2017Adapter


def _require_artifacts(adapter: CICIDS2017Adapter) -> None:
    if not (adapter._processed / "preprocessing_manifest.json").exists():
        pytest.skip("CICIDS2017 processed artifacts unavailable")


def test_manifest_matches_preprocessing_order_and_semantics():
    adapter = CICIDS2017Adapter()
    _require_artifacts(adapter)
    manifest = adapter.feature_manifest()

    assert manifest.n_features == 79
    assert manifest.names == adapter._feature_order()
    assert manifest.mutable_mask() is None
    assert manifest["Src Port"].value_type == "bounded_continuous"
    assert manifest["Src Port"].upper == 65535.0
    assert manifest["Protocol"].value_type == "bounded_continuous"
    assert manifest["Protocol"].upper == 255.0
    assert manifest["Total Fwd Packet"].value_type == "integer_count"
    assert manifest["SYN Flag Count"].value_type == "integer_count"
    assert manifest["Flow Bytes/s"].value_type == "positive_continuous"
    assert manifest["Packet Length Variance"].lower == 0.0
    assert not manifest.indices_of_value_type("categorical")
    assert not manifest.indices_of_value_type("binary")
    derived = {manifest[index].name for index in manifest.derived_indices()}
    assert derived == {
        "Packet Length Variance",
        "Average Packet Size",
        "Fwd Segment Size Avg",
        "Bwd Segment Size Avg",
        "Total Length of Fwd Packet",
        "Total Length of Bwd Packet",
    }
    assert manifest["Packet Length Variance"].derivation == "square"
    assert manifest["Total Length of Fwd Packet"].derivation == "product"


def test_class_mapping_uses_category_encoder_ids():
    adapter = CICIDS2017Adapter()
    _require_artifacts(adapter)
    mapping = adapter.class_mapping()

    assert mapping.n_classes == 5
    assert list(mapping.names) == ["Benign", "DoS", "DDoS", "Recon", "BruteForce"]
    assert mapping.name_to_id["Benign"] == 0
    assert mapping.name_to_id["BruteForce"] == 4


def test_transform_matches_saved_scaler_and_scaled_array():
    adapter = CICIDS2017Adapter()
    _require_artifacts(adapter)
    scaler_path = adapter._processed / "scaler.pkl"
    if not scaler_path.exists():
        pytest.skip("CICIDS2017 scaler unavailable")

    transform = adapter.feature_transform()
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    pristine = np.asarray(
        np.load(adapter._processed / "X_train_pristine.npy", mmap_mode="r")[:256],
        dtype=np.float64,
    )
    saved_scaled = np.asarray(
        np.load(adapter._processed / "X_train.npy", mmap_mode="r")[:256],
        dtype=np.float64,
    )

    transformed = transform.transform(pristine)
    assert np.allclose(transformed, scaler.transform(pristine), rtol=1e-12, atol=1e-12)
    assert np.allclose(transformed, saved_scaled, rtol=1e-5, atol=1e-5)
    assert np.allclose(transform.inverse_transform(transformed), pristine, rtol=1e-6, atol=1e-4)


def test_split_width_labels_and_invalid_name():
    adapter = CICIDS2017Adapter()
    _require_artifacts(adapter)
    split = adapter.load_split("val")

    assert split.name == "val"
    assert split.x.shape == (312058, 79)
    assert split.y.shape == (312058,)
    assert split.y.dtype == np.int64
    labels = np.asarray(split.y)
    assert set(np.unique(labels).tolist()) == set(range(5))
    with pytest.raises(ValueError, match="train/val/test"):
        adapter.load_split("validation")


def test_registry_returns_cicids2017_adapter():
    assert isinstance(get_adapter("cicids2017"), CICIDS2017Adapter)
    assert isinstance(get_adapter("cicids2017_distrinet"), CICIDS2017Adapter)
