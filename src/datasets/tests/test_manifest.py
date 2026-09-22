"""Phase A verification: manifest, transform, and CICIoT2023 adapter.

Run (thesis env):
    PYTHONPATH=src python -m pytest src/datasets/tests/test_manifest.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from datasets import (
    FeatureManifest,
    FeatureSpec,
    FeatureTransform,
    ManifestError,
    TransformNotFittedError,
    get_adapter,
)
from datasets.ciciot2023 import CICIoT2023Adapter, build_manifest
from preprocessing.schema import FEATURE_NAMES


# --------------------------------------------------------------------------- #
# FeatureSpec invariants
# --------------------------------------------------------------------------- #
def test_bounded_continuous_requires_bounds():
    with pytest.raises(ManifestError):
        FeatureSpec("x", 0, "bounded_continuous", "sem", lower=0.0, upper=None)


def test_derived_requires_parents():
    with pytest.raises(ManifestError):
        FeatureSpec("x", 0, "positive_continuous", "sem", primitive_or_derived="derived")


def test_unknown_value_type_rejected():
    with pytest.raises(ManifestError):
        FeatureSpec("x", 0, "not_a_type", "sem")


def test_lower_gt_upper_rejected():
    with pytest.raises(ManifestError):
        FeatureSpec("x", 0, "bounded_continuous", "sem", lower=5.0, upper=1.0)


# --------------------------------------------------------------------------- #
# FeatureManifest ordering / lookups
# --------------------------------------------------------------------------- #
def _toy_manifest() -> FeatureManifest:
    specs = [
        FeatureSpec("a", 0, "positive_continuous", "count", lower=0.0),
        FeatureSpec("b", 1, "probability", "flag", lower=0.0, upper=1.0),
        FeatureSpec("c", 2, "bounded_continuous", "ttl", lower=0.0, upper=255.0),
    ]
    return FeatureManifest(specs, dataset_name="toy")


def test_order_contract_violation_detected():
    bad = [
        FeatureSpec("a", 0, "positive_continuous", "count"),
        FeatureSpec("b", 5, "probability", "flag", lower=0.0, upper=1.0),  # wrong index
    ]
    with pytest.raises(ManifestError):
        FeatureManifest(bad, dataset_name="toy")


def test_duplicate_names_detected():
    dup = [
        FeatureSpec("a", 0, "positive_continuous", "count"),
        FeatureSpec("a", 1, "probability", "flag", lower=0.0, upper=1.0),
    ]
    with pytest.raises(ManifestError):
        FeatureManifest(dup, dataset_name="toy")


def test_lookups():
    m = _toy_manifest()
    assert m.n_features == 3
    assert m.index_by_name("b") == 1
    assert m.index_by_semantic("ttl") == [2]
    assert m.indices_of_value_type("probability") == [1]
    with pytest.raises(KeyError):
        m.index_by_name("zzz")


def test_array_and_scaler_mismatch_fail_loud():
    m = _toy_manifest()
    with pytest.raises(ManifestError):
        m.assert_matches_array(np.zeros((4, 2)))
    m.assert_matches_array(np.zeros((4, 3)))  # ok

    class _S:
        center_ = np.zeros(2)
        scale_ = np.ones(2)

    with pytest.raises(ManifestError):
        m.assert_matches_scaler(_S())


def test_names_order_check():
    m = _toy_manifest()
    with pytest.raises(ManifestError):
        m.assert_names_match(["b", "a", "c"])
    m.assert_names_match(["a", "b", "c"])


def test_content_hash_excludes_mutability():
    m = _toy_manifest()
    h0 = m.content_hash
    m2 = m.with_mutability({"a": True, "b": False, "c": True})
    assert m2.content_hash == h0  # mining perturbability must not invalidate ckpts
    assert m.mutable_mask() is None  # un-mined -> None (no silent all-mutable)
    assert m2.mutable_mask() == [True, False, True]


def test_manifest_roundtrip(tmp_path):
    m = _toy_manifest()
    p = tmp_path / "m.json"
    m.save(p)
    m2 = FeatureManifest.load(p)
    assert m2.names == m.names
    assert m2.content_hash == m.content_hash


# --------------------------------------------------------------------------- #
# FeatureTransform
# --------------------------------------------------------------------------- #
def test_transform_requires_fit():
    m = _toy_manifest()
    t = FeatureTransform(m)
    with pytest.raises(TransformNotFittedError):
        t.transform(np.zeros((2, 3)))


def test_transform_train_only_roundtrip():
    m = _toy_manifest()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3)) * np.array([10.0, 0.3, 40.0]) + np.array([5.0, 0.5, 128.0])
    t = FeatureTransform(m).fit(x, provenance="train")
    xs = t.transform(x)
    xr = t.inverse_transform(xs)
    assert np.allclose(xr, x, atol=1e-6)


def test_transform_state_roundtrip_and_hash_guard(tmp_path):
    m = _toy_manifest()
    x = np.random.default_rng(1).normal(size=(100, 3))
    t = FeatureTransform(m).fit(x)
    p = tmp_path / "t.json"
    t.save(p)
    t2 = FeatureTransform.load(p, m)
    assert np.allclose(t2.center, t.center)
    # wrong manifest -> hash guard fires
    other = FeatureManifest(
        [FeatureSpec("a", 0, "real", "x"), FeatureSpec("b", 1, "real", "x"),
         FeatureSpec("c", 2, "real", "x")],
        dataset_name="toy",
    )
    with pytest.raises(ManifestError):
        FeatureTransform.load(p, other)


# --------------------------------------------------------------------------- #
# CICIoT2023 adapter (uses real artifacts under data/processed)
# --------------------------------------------------------------------------- #
def test_ciciot_manifest_matches_schema_order():
    m = build_manifest()
    assert m.n_features == 39
    assert m.names == list(FEATURE_NAMES)
    # value-type census from representation mapping
    assert len(m.indices_of_value_type("probability")) == 22
    assert len(m.indices_of_value_type("bounded_continuous")) == 2  # TTL + Protocol
    assert len(m.indices_of_value_type("positive_continuous")) == 15
    # flags/services are probability, NOT binary
    assert m["TCP"].value_type == "probability"
    assert m["syn_flag_number"].value_type == "probability"
    assert m["Time_To_Live"].upper == 255.0
    assert m.mutable_mask() is None


def test_ciciot_class_mapping():
    a = CICIoT2023Adapter()
    cm = a.class_mapping()
    assert cm.n_classes == 8
    assert list(cm.names) == ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
    assert cm.fine_to_coarse["DDOS-ICMP_FLOOD"] == "DDoS"


def test_ciciot_transform_wraps_existing_scaler():
    import pickle
    a = CICIoT2023Adapter()
    scaler_path = a._processed / "scaler.pkl"
    if not scaler_path.exists():
        pytest.skip("scaler.pkl not available")
    t = a.feature_transform()
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    x = np.random.default_rng(2).normal(size=(64, 39))
    # wrapped transform must reproduce sklearn scaler exactly
    assert np.allclose(t.transform(x), scaler.transform(x), atol=1e-9)
    assert np.allclose(t.inverse_transform(t.transform(x)), x, atol=1e-6)


def test_ciciot_load_split_contract():
    a = CICIoT2023Adapter()
    if not (a._processed / "X_val.npy").exists():
        pytest.skip("processed arrays not available")
    split = a.load_split("val")
    assert split.x.shape[1] == 39
    cm = a.class_mapping()
    ys = np.asarray(split.y[:10000])
    assert ys.min() >= 0 and ys.max() < cm.n_classes


def test_registry():
    assert isinstance(get_adapter("ciciot2023"), CICIoT2023Adapter)
    with pytest.raises(KeyError):
        get_adapter("nope")
