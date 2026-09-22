"""Phase C verification: Layer 0/1/2 constraints + engine.

Run (thesis env):
    PYTHONPATH=src python -m pytest src/constraints/tests/test_constraints.py -q -p no:faulthandler
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from constraints import (
    ConstraintEngine,
    HalfRangeBound,
    Layer0Projector,
    MonotoneNondecreasing,
    ProductEquality,
    RobustTailBound,
    build_constraint,
    dump_layer2,
    load_layer2,
)
from datasets import FeatureManifest, FeatureSpec
from datasets.ciciot2023 import CICIoT2023Adapter

_REPO = Path(__file__).resolve().parents[3]


def _manifest() -> FeatureManifest:
    specs = [
        FeatureSpec("pos", 0, "positive_continuous", "count", lower=0.0),
        FeatureSpec("prob", 1, "probability", "flag", lower=0.0, upper=1.0),
        FeatureSpec("ttl", 2, "bounded_continuous", "ttl", lower=0.0, upper=255.0),
        FeatureSpec("Min", 3, "positive_continuous", "size", lower=0.0),
        FeatureSpec("AVG", 4, "positive_continuous", "size", lower=0.0),
        FeatureSpec("Max", 5, "positive_continuous", "size", lower=0.0),
        FeatureSpec("Std", 6, "positive_continuous", "size", lower=0.0),
        FeatureSpec("Number", 7, "positive_continuous", "count", lower=0.0),
        FeatureSpec("Tot", 8, "positive_continuous", "size", lower=0.0),
    ]
    return FeatureManifest(specs, dataset_name="toy")


# --------------------------------------------------------------------------- #
# Layer 0
# --------------------------------------------------------------------------- #
def test_layer0_projects_into_domain():
    m = _manifest()
    p = Layer0Projector(m)
    x = torch.tensor([[-5.0, 2.0, 900.0, 1, 2, 3, 1, 4, 8]], dtype=torch.float32)
    xp = p.project(x)
    assert xp[0, 0] >= 0.0                 # positive clamped
    assert 0.0 <= xp[0, 1] <= 1.0          # probability clamped
    assert 0.0 <= xp[0, 2] <= 255.0        # bounded clamped
    assert bool(p.validate(xp).all())
    assert not bool(p.validate(x).all())   # original was out of domain


def test_layer0_immutable_preserved():
    m = _manifest()
    p = Layer0Projector(m)
    src = torch.zeros(1, 9)
    src[0, 3] = 42.0  # Min in source
    cand = torch.full((1, 9), 7.0)
    mutable = torch.ones(9, dtype=torch.bool)
    mutable[3] = False  # freeze Min
    out = p.project(cand, x_source=src, mutable_mask=mutable)
    assert out[0, 3].item() == 42.0
    assert out[0, 0].item() == 7.0


def test_layer0_recomputes_manifest_declared_derivations():
    manifest = FeatureManifest(
        [
            FeatureSpec("a", 0, "positive_continuous", "x", lower=0.0),
            FeatureSpec("b", 1, "positive_continuous", "x", lower=0.0),
            FeatureSpec(
                "a_square",
                2,
                "positive_continuous",
                "x",
                lower=0.0,
                primitive_or_derived="derived",
                parents=("a",),
                derivation="square",
            ),
            FeatureSpec(
                "ab",
                3,
                "positive_continuous",
                "x",
                lower=0.0,
                primitive_or_derived="derived",
                parents=("a", "b"),
                derivation="product",
            ),
        ],
        dataset_name="derived-toy",
    )
    projector = Layer0Projector(manifest)
    candidate = torch.tensor([[3.0, 4.0, 999.0, 999.0]])
    projected = projector.project(candidate)

    assert projected.tolist() == [[3.0, 4.0, 9.0, 12.0]]


# --------------------------------------------------------------------------- #
# Layer 1 generic constraints
# --------------------------------------------------------------------------- #
def test_robust_tail_fit_train_only():
    m = _manifest()
    rng = np.random.default_rng(0)
    x_train = np.abs(rng.normal(size=(4000, 9))) * np.arange(1, 10)
    c = RobustTailBound.fit(m, x_train, feature_names=["pos", "Max"], tau=4.0)
    inlier = torch.tensor(np.median(x_train, axis=0, keepdims=True), dtype=torch.float32)
    assert bool(c.validate(inlier).all())
    assert float(c.penalty(inlier)) == pytest.approx(0.0, abs=1e-5)
    outlier = inlier.clone()
    outlier[0, m.index_by_name("Max")] += 1e6
    assert not bool(c.validate(outlier).all())
    assert float(c.penalty(outlier)) > 0.0


def test_robust_tail_uses_std_fallback_for_sparse_zero_iqr_feature():
    m = _manifest()
    x_train = np.zeros((1000, 9), dtype=np.float64)
    x_train[:10, m.index_by_name("pos")] = 100.0
    c = RobustTailBound.fit(m, x_train, feature_names=["pos"], tau=4.0)
    sparse_inlier = torch.zeros((1, 9), dtype=torch.float32)
    sparse_inlier[0, m.index_by_name("pos")] = 20.0

    assert c.iqr.item() > 1.0
    assert bool(c.validate(sparse_inlier).all())



def test_robust_tail_adaptive_tau_targets_joint_train_coverage():
    m = _manifest()
    x_train = np.abs(np.random.default_rng(7).normal(size=(4000, 9)))
    c = RobustTailBound.fit(m, x_train, tau=None, coverage=0.99)
    passed = c.validate(torch.tensor(x_train, dtype=torch.float32))

    assert float(passed.float().mean()) >= 0.99

def test_product_equality():
    m = _manifest()
    c = ProductEquality(m, "Tot", ["Number", "AVG"], rtol=0.05)
    x = torch.zeros(2, 9)
    x[0, 7], x[0, 4], x[0, 8] = 3.0, 4.0, 12.0   # 3*4=12 -> pass
    x[1, 7], x[1, 4], x[1, 8] = 3.0, 4.0, 50.0   # mismatch -> fail
    mask = c.validate(x)
    assert bool(mask[0]) and not bool(mask[1])
    assert float(c.penalty(x)) > 0.0


def test_monotone_and_halfrange():
    m = _manifest()
    mono = MonotoneNondecreasing(m, ["Min", "AVG", "Max"])
    hr = HalfRangeBound(m, "Std", "Min", "Max", rtol=0.05)
    x = torch.zeros(2, 9)
    x[0, 3], x[0, 4], x[0, 5], x[0, 6] = 1.0, 2.0, 5.0, 1.0   # ordered, std<=2 -> pass
    x[1, 3], x[1, 4], x[1, 5], x[1, 6] = 5.0, 2.0, 1.0, 9.0   # unordered, huge std
    assert bool(mono.validate(x)[0]) and not bool(mono.validate(x)[1])
    assert bool(hr.validate(x)[0]) and not bool(hr.validate(x)[1])


# --------------------------------------------------------------------------- #
# Layer 2 + registry + engine
# --------------------------------------------------------------------------- #
def test_registry_roundtrip():
    m = _manifest()
    c = ProductEquality(m, "Tot", ["Number", "AVG"], rtol=0.1, name="p")
    cfg = c.to_config()
    c2 = build_constraint(m, cfg)
    assert isinstance(c2, ProductEquality) and c2.rtol == 0.1


def test_layer2_load_and_engine_hierarchical():
    m = _manifest()
    rules = [
        {"type": "MonotoneNondecreasing", "name": "ord", "params": {"features": ["Min", "AVG", "Max"]}},
        {"type": "ProductEquality", "name": "tot", "params": {"target": "Tot", "factors": ["Number", "AVG"], "rtol": 0.05}},
    ]
    l2 = load_layer2({"dataset": "toy", "constraints": rules}, m)
    assert all(c.layer == 2 for c in l2)
    engine = ConstraintEngine(m, layer1=[], layer2=l2)
    x = torch.zeros(2, 9)
    x[0, 3], x[0, 4], x[0, 5], x[0, 7], x[0, 8] = 1.0, 2.0, 3.0, 2.0, 4.0  # ordered, 2*2=4 pass
    x[1, 3], x[1, 4], x[1, 5], x[1, 7], x[1, 8] = 3.0, 2.0, 1.0, 2.0, 99.0  # both fail
    v = engine.validate(x)
    assert bool(v["pass_l0_l1_l2"][0]) and not bool(v["pass_l0_l1_l2"][1])
    assert v["rate_l0"] == 1.0  # both in domain
    assert 0.0 <= v["rate_l0_l1_l2"] <= 1.0
    # dump roundtrip
    dumped = dump_layer2(l2, "toy")
    l2b = load_layer2(dumped, m)
    assert len(l2b) == len(l2)


def test_layer2_dataset_mismatch_rejected():
    m = _manifest()
    with pytest.raises(ValueError):
        load_layer2({"dataset": "other", "constraints": []}, m)


# --------------------------------------------------------------------------- #
# Real CICIoT data: Layer 0 should pass ~everywhere; Layer 2 rules load
# --------------------------------------------------------------------------- #
def test_ciciot_layer0_high_pass_and_layer2_loads():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    split = adapter.load_split("val")
    idx = np.arange(min(20000, split.x.shape[0]))
    x_raw = torch.tensor(transform.inverse_transform(np.asarray(split.x[idx])), dtype=torch.float32)

    l2 = load_layer2(_REPO / "constraints" / "ciciot2023" / "mined.json", manifest)
    engine = ConstraintEngine(manifest, layer1=[], layer2=l2)
    v = engine.validate(x_raw)
    # real clean data must be within Layer-0 semantic domain essentially always
    assert v["rate_l0"] > 0.99
    # Layer-2 rates are informative, not asserted high (calibration is a later step)
    assert 0.0 <= v["rate_l0_l1_l2"] <= 1.0
