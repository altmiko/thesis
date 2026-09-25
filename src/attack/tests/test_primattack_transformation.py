"""Formal phi(x,p) contract, numerical safety, and victim-loss gradient checks."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from attack.primattack_budget import class_calibration, load_calibration
from attack.realizability.base import FeatureRole
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel
from datasets import get_adapter
from src.classifiers.cicids2017d_victims import load_category_victim


@pytest.fixture(scope="module")
def context():
    adapter = get_adapter("cicids2017")
    model = CICIDS2017PrimitiveModel(adapter.feature_manifest())
    raw_all = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    labels = np.load(adapter._processed / "y_test_cat.npy", mmap_mode="r")
    dos = adapter.class_mapping().name_to_id["DoS"]
    rows = np.flatnonzero(labels == dos)[:512]
    raw = torch.tensor(
        np.ascontiguousarray(np.asarray(raw_all[rows]), dtype=np.float32)
    )
    calibration = load_calibration(
        adapter.repo_root / "artifacts/primattack/budget_calibration.json"
    )
    class_cfg = class_calibration(calibration, "DoS", "maximum-evaluated")
    bounds = model.per_flow_bounds(raw, class_cfg.bounds_config())
    transform = adapter.feature_transform()
    center = torch.tensor(transform.center, dtype=torch.float32)
    scale = torch.tensor(transform.scale, dtype=torch.float32)
    victim = load_category_victim(
        adapter.repo_root / "outputs/cicids2017distrinet/models/mlp_category.pt",
        adapter=adapter,
        expected_model_type="mlp",
    )
    victim.eval()
    for parameter in victim.parameters():
        parameter.requires_grad_(False)
    return adapter, model, raw, bounds, center, scale, victim


def test_primitive_spec_is_complete_and_directional(context):
    _, model, _, _, _, _, _ = context
    specs = {spec.name: spec for spec in model.primitives()}
    assert set(specs) == {"p", "delay", "shape"}
    assert specs["p"].dtype == "discrete_integer"
    assert specs["p"].units == "bytes_per_forward_packet"
    assert specs["delay"].dtype == "discrete_integer"
    assert specs["delay"].units == "microseconds_total_forward_delay"
    assert specs["shape"].absolute_lower_bound == 0.0
    assert specs["shape"].absolute_upper_bound == 1.0
    assert all(spec.dependencies for spec in specs.values())
    assert "packet-level" in specs["p"].semantic_risk


def test_zero_perturbation_identity_is_exact(context):
    _, model, raw, _, _, _, _ = context
    controls = {
        "p": torch.zeros(len(raw)),
        "delay": torch.zeros(len(raw)),
        "shape": torch.zeros(len(raw)),
    }
    assert torch.equal(model.generate(raw, controls), raw)
    assert torch.equal(model.generate(raw, controls, quantize=True), raw)


def test_declared_dependency_completeness_per_primitive(context):
    _, model, raw, bounds, _, _, _ = context
    specs = {spec.name: spec for spec in model.primitives()}
    roles = model.roles()
    pad_active = model.active_mask(raw, "p")
    timing_active = model.active_mask(raw, "delay")
    for primitive, requested in (
        ("p", {
            "p": torch.where(pad_active, torch.full_like(bounds["p"], 2.0),
                             torch.zeros_like(bounds["p"])),
            "delay": torch.zeros_like(bounds["delay"]),
            "shape": torch.zeros_like(bounds["shape"]),
        }),
        ("delay", {
            "p": torch.zeros_like(bounds["p"]),
            "delay": torch.where(timing_active, torch.full_like(bounds["delay"], 10.0),
                                 torch.zeros_like(bounds["delay"])),
            "shape": torch.full_like(bounds["shape"], 0.5),
        }),
    ):
        adv = model.generate(raw, requested)
        delta = (adv - raw).abs()
        active = requested[primitive] > 0.0
        assert bool(active.any())
        changed_columns = {
            model.feature_names[column]
            for column in torch.nonzero(
                delta[active].gt(1e-6).any(0), as_tuple=False
            ).flatten().tolist()
        }
        assert changed_columns <= set(specs[primitive].dependencies)
        declared = set().union(*(set(spec.dependencies) for spec in specs.values()))
        assert changed_columns <= declared
        frozen = [model.i[name] for name, (role, _) in roles.items()
                  if role in {FeatureRole.FROZEN, FeatureRole.INVARIANT, FeatureRole.LEVEL_C}]
        assert torch.equal(adv[:, frozen], raw[:, frozen])


def test_timing_equations_and_shape_endpoints(context):
    _, model, raw, _, _, _, _ = context
    i = model.i
    active = model.active_mask(raw, "delay")
    delay = torch.where(
        active,
        torch.full((len(raw),), 100.0, dtype=raw.dtype),
        torch.zeros(len(raw), dtype=raw.dtype),
    )
    p = torch.zeros_like(delay)
    proportional = model.generate(
        raw, {"p": p, "delay": delay, "shape": torch.zeros_like(delay)}
    )
    uniform = model.generate(
        raw, {"p": p, "delay": delay, "shape": torch.ones_like(delay)}
    )
    assert bool(active.any())
    assert torch.allclose(
        proportional[active, i["Fwd IAT Total"]],
        raw[active, i["Fwd IAT Total"]] + delay[active],
        atol=1e-3,
    )
    gaps = (raw[active, i["Total Fwd Packet"]] - 1.0).clamp(min=1.0)
    assert torch.allclose(
        uniform[active, i["Fwd IAT Min"]],
        raw[active, i["Fwd IAT Min"]] + delay[active] / gaps,
        atol=1e-3,
    )
    assert torch.allclose(
        uniform[active, i["Fwd IAT Std"]],
        raw[active, i["Fwd IAT Std"]],
        atol=1e-3,
    )
    assert bool((uniform[:, i["Flow Duration"]] >= raw[:, i["Flow Duration"]]).all())
    assert bool((uniform[:, i["Flow IAT Mean"]] <= uniform[:, i["Flow IAT Max"]] + 1e-3).all())


def test_padding_size_and_derived_equations(context):
    _, model, raw, bounds, _, _, _ = context
    i = model.i
    p = torch.minimum(bounds["p"], torch.full_like(bounds["p"], 3.0))
    adv = model.generate(
        raw, {"p": p, "delay": torch.zeros_like(p), "shape": torch.zeros_like(p)}
    )
    active = p > 0
    assert bool(active.any())
    nf = raw[:, i["Total Fwd Packet"]]
    assert torch.allclose(
        adv[active, i["Total Length of Fwd Packet"]],
        raw[active, i["Total Length of Fwd Packet"]] + nf[active] * p[active],
        atol=1e-4,
    )
    assert bool((adv[:, i["Total Length of Fwd Packet"]] >= raw[:, i["Total Length of Fwd Packet"]]).all())
    assert torch.allclose(
        adv[:, i["Average Packet Size"]], adv[:, i["Packet Length Mean"]], atol=1e-5
    )
    assert torch.allclose(
        adv[:, i["Packet Length Variance"]],
        adv[:, i["Packet Length Std"]].square(),
        rtol=1e-4,
        atol=1.0,
    )


def test_projection_is_discrete_and_hard_budgeted(context):
    _, model, raw, bounds, _, _, _ = context
    requested = {
        "p": bounds["p"] + 1000.75,
        "delay": bounds["delay"] + 1000.75,
        "shape": bounds["shape"] + 10.0,
    }
    projected = model.project_controls(raw, requested, bounds)
    assert bool((projected["p"] <= torch.floor(bounds["p"])).all())
    assert bool((projected["delay"] <= torch.floor(bounds["delay"]) + 1e-7).all())
    assert bool((projected["shape"] <= bounds["shape"] + 1e-7).all())
    assert torch.equal(projected["p"], torch.round(projected["p"]))
    assert torch.equal(projected["delay"], torch.round(projected["delay"]))
    adv = model.generate(raw, projected, quantize=True)
    assert bool(torch.isfinite(adv).all())
    for name in model.integer_features():
        value = adv[:, model.i[name]]
        assert bool(torch.isclose(value, torch.round(value), atol=1e-3, rtol=0.0).all()), name


def _victim_loss(model, victim, raw, center, scale, p, delay, shape):
    adv = model.generate(raw, {"p": p, "delay": delay, "shape": shape})
    target = torch.zeros(len(raw), dtype=torch.long)
    return F.cross_entropy(victim((adv - center) / scale), target, reduction="sum")


def test_autograd_matches_finite_difference_through_victim(context):
    _, model, raw_all, _, center, scale, victim = context
    active = model.active_mask(raw_all, "p") & model.active_mask(raw_all, "delay")
    row = int(torch.nonzero(active, as_tuple=False)[0])
    raw = raw_all[row:row + 1]
    p = torch.tensor([2.25], dtype=torch.float32, requires_grad=True)
    delay = torch.tensor([100.0], dtype=torch.float32, requires_grad=True)
    shape = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)
    loss = _victim_loss(model, victim, raw, center, scale, p, delay, shape)
    grad_p, grad_delay, grad_shape = torch.autograd.grad(loss, (p, delay, shape))

    def finite_difference(name: str, step: float) -> float:
        values = {"p": p, "delay": delay, "shape": shape}
        with torch.no_grad():
            plus_args = {k: (v + step if k == name else v) for k, v in values.items()}
            minus_args = {k: (v - step if k == name else v) for k, v in values.items()}
            plus = _victim_loss(model, victim, raw, center, scale, **plus_args)
            minus = _victim_loss(model, victim, raw, center, scale, **minus_args)
        return float((plus - minus) / (2.0 * step))

    assert float(grad_p) == pytest.approx(finite_difference("p", 1e-2), rel=5e-2, abs=2e-4)
    assert float(grad_delay) == pytest.approx(
        finite_difference("delay", 1e-1), rel=5e-2, abs=2e-3
    )
    assert float(grad_shape) == pytest.approx(
        finite_difference("shape", 1e-3), rel=5e-2, abs=2e-3
    )


def test_numerical_safety_tiny_duration_boundaries_and_nonfinite_rejection(context):
    _, model, raw, _, _, _, _ = context
    i = model.i
    boundary = raw[:4].clone()
    boundary[:, i["Flow Duration"]] = torch.tensor([0.0, 1e-12, 1.0, 2.0])
    boundary[:, i["Fwd IAT Total"]] = 0.0
    identity = model.generate(
        boundary,
        {"p": torch.zeros(4), "delay": torch.zeros(4), "shape": torch.zeros(4)},
    )
    assert torch.equal(identity, boundary)
    padded = model.generate(
        boundary,
        {"p": torch.ones(4), "delay": torch.zeros(4), "shape": torch.zeros(4)},
    )
    assert bool(torch.isfinite(padded).all())
    bad = {
        "p": torch.full((4,), float("nan")),
        "delay": torch.zeros(4),
        "shape": torch.zeros(4),
    }
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.generate(boundary, bad)
