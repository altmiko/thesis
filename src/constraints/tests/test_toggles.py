"""Layer on/off toggles: active_layers honored in project / penalty / validate.

Run (thesis env):
    PYTHONPATH=src python -m pytest src/constraints/tests/test_toggles.py -q -p no:faulthandler
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from constraints import (
    ConstraintEngine,
    Layer0Projector,
    ProductEquality,
    RobustTailBound,
    build_engine,
    parse_layers,
)
from datasets import FeatureManifest, FeatureSpec


def _manifest() -> FeatureManifest:
    specs = [
        FeatureSpec("prob", 0, "probability", "flag", lower=0.0, upper=1.0),
        FeatureSpec("Number", 1, "positive_continuous", "count", lower=0.0),
        FeatureSpec("AVG", 2, "positive_continuous", "size", lower=0.0),
        FeatureSpec("Tot", 3, "positive_continuous", "size", lower=0.0),
    ]
    return FeatureManifest(specs, dataset_name="toy")


def _full_engine(m):
    x_train = np.abs(np.random.default_rng(0).normal(2, 1, size=(2000, 4)))
    l1 = [RobustTailBound.fit(m, x_train, feature_names=["AVG"], tau=4.0)]
    l2 = [ProductEquality(m, "Tot", ["Number", "AVG"], rtol=0.05, name="tot")]
    return ConstraintEngine(m, layer1=l1, layer2=l2)


def test_parse_layers():
    assert parse_layers("012", {0}) == {0, 1, 2}
    assert parse_layers("0,2", {0}) == {0, 2}
    assert parse_layers([1], {0}) == {1}
    assert parse_layers(None, {0, 1}) == {0, 1}
    with pytest.raises(ValueError):
        parse_layers("3", {0})


def test_default_active_is_all_present():
    m = _manifest()
    eng = _full_engine(m)
    assert eng.active_layers == {0, 1, 2}


def test_penalty_respects_active_layers():
    m = _manifest()
    eng = _full_engine(m)
    # sample with a Layer-2 product violation (Tot != Number*AVG)
    x = torch.tensor([[0.5, 3.0, 4.0, 99.0]])
    eng.set_active_layers("0")
    p0 = eng.penalty(x)
    assert float(p0["c1"]) == 0.0 and float(p0["c2"]) == 0.0
    eng.set_active_layers("012")
    p2 = eng.penalty(x)
    assert float(p2["c2"]) > 0.0  # layer-2 now contributes


def test_project_toggle():
    m = _manifest()
    eng = _full_engine(m)
    x = torch.tensor([[5.0, -3.0, 1.0, 1.0]])  # prob=5 (>1), Number=-3 (<0): out of domain
    eng.set_active_layers("12")  # layer 0 OFF
    assert torch.allclose(eng.project(x), x)  # no projection
    eng.set_active_layers("012")  # layer 0 ON
    xp = eng.project(x)
    assert 0.0 <= float(xp[0, 0]) <= 1.0 and float(xp[0, 1]) >= 0.0


def test_validate_rates_reflect_active_layers():
    m = _manifest()
    eng = _full_engine(m)
    x = torch.tensor([[0.5, 3.0, 4.0, 99.0]])  # domain-valid, but Tot!=N*AVG (L2 fail)
    eng.set_active_layers("01")
    v01 = eng.validate(x)
    assert v01["rate_l0_l1_l2"] == 1.0  # L2 inactive -> all-pass
    eng.set_active_layers("012")
    v012 = eng.validate(x)
    assert v012["rate_l0_l1_l2"] == 0.0  # L2 active -> fails
    assert v012["active_layers"] == [0, 1, 2]


def test_build_engine_builds_only_requested():
    m = _manifest()
    x_train = np.abs(np.random.default_rng(1).normal(2, 1, size=(1000, 4)))
    eng = build_engine(m, "01", layer1_fit_x_raw=x_train)
    assert eng.active_layers == {0, 1}
    assert len(eng.layer1) == 1 and eng.layer2 == []
    # requesting layer 2 without a source is refused
    with pytest.raises(ValueError):
        build_engine(m, "012", layer1_fit_x_raw=x_train)
    # requesting layer 1 without train data is refused
    with pytest.raises(ValueError):
        build_engine(m, "1")
