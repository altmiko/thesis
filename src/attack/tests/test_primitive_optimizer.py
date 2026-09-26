"""Invariants of the quantization-aware PrimAttack searches on a real victim."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from attack.primattack_budget import class_calibration, load_calibration
from attack.primitive_optimizer import (
    AttackObjective,
    RealizedSearch,
    hybrid_valid_gate,
    optimize_primitive_candidates,
    optimize_primitive_cw,
    optimize_primitive_pgd,
    targeted_margin,
)
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel
from datasets import get_adapter
from src.classifiers.cicids2017d_victims import load_category_victim

BUDGET = 64
SOURCE = get_adapter("cicids2017").class_mapping().name_to_id["DoS"]


@pytest.fixture(scope="module")
def setup():
    ad = get_adapter("cicids2017")
    model = CICIDS2017PrimitiveModel(ad.feature_manifest())
    transform = ad.feature_transform()
    center = torch.tensor(transform.center, dtype=torch.float32)
    scale = torch.tensor(transform.scale, dtype=torch.float32)
    calibration = load_calibration(ad.repo_root / "artifacts/primattack/budget_calibration.json")
    cfg = class_calibration(calibration, "DoS", "maximum-evaluated").bounds_config()
    test = ad.load_split("test")
    rows = np.flatnonzero(test.y == ad.class_mapping().name_to_id["DoS"])[:96]
    raw = torch.tensor(np.ascontiguousarray(np.asarray(
        np.load(ad._processed / "X_test_pristine.npy", mmap_mode="r")[rows]), dtype=np.float32))
    victim = load_category_victim(
        ad.repo_root / "outputs/cicids2017distrinet/models/cnn_category.pt",
        adapter=ad, expected_model_type="cnn",
    )
    caps = model.infer_capabilities(raw)
    bounds = model.per_flow_bounds(raw, cfg, capabilities=caps)
    gate = hybrid_valid_gate(ad.name)
    args = (model, victim, raw, center, scale, bounds, caps)
    results = {
        "hybrid": optimize_primitive_candidates(
            *args, steps=8, learning_rate=0.1, seed=42, restarts=2, validity_fn=gate),
        "hybrid_budget": optimize_primitive_candidates(
            *args, steps=8, learning_rate=0.1, seed=42, restarts=None, validity_fn=gate,
            eval_budget=BUDGET),
        "pgd": optimize_primitive_pgd(
            *args, steps=10, step_size=0.05, restarts=3, seed=42, validity_fn=gate,
            eval_budget=BUDGET),
        "cw": optimize_primitive_cw(
            *args, steps=10, learning_rate=0.05, stages=3, validity_fn=gate,
            eval_budget=BUDGET),
        "hybrid_untargeted": optimize_primitive_candidates(
            *args, steps=8, learning_rate=0.1, seed=42, restarts=None, validity_fn=gate,
            eval_budget=BUDGET, objective=AttackObjective("untargeted", SOURCE)),
        "pgd_untargeted": optimize_primitive_pgd(
            *args, steps=10, step_size=0.05, restarts=3, seed=42, validity_fn=gate,
            eval_budget=BUDGET, objective=AttackObjective("untargeted", SOURCE)),
    }
    return model, victim, raw, center, scale, bounds, caps, gate, results


METHODS = ("hybrid", "hybrid_budget", "pgd", "cw")
UNTARGETED = ("hybrid_untargeted", "pgd_untargeted")


@pytest.mark.parametrize("method", UNTARGETED)
def test_untargeted_success_is_leaving_the_source_class_on_the_realized_flow(setup, method):
    model, victim, raw, center, scale, bounds, caps, gate, results = setup
    result = results[method]
    adv = model.generate(raw, result.projected, quantize=True, capabilities=caps)
    assert torch.equal(adv, result.adversarial_raw)
    with torch.no_grad():
        logits = victim((adv - center) / scale)
    others = logits.clone()
    others[:, SOURCE] = -torch.inf
    expected_margin = logits[:, SOURCE] - others.amax(1)
    assert torch.allclose(expected_margin, result.objective_margin, atol=1e-5)
    assert torch.equal(result.success, (logits.argmax(1) != SOURCE) & gate(adv))
    assert bool((result.total_evaluations <= BUDGET).all())



@pytest.mark.parametrize("method", METHODS)
def test_result_is_the_realized_projection_of_its_request(setup, method):
    model, victim, raw, center, scale, bounds, caps, gate, results = setup
    result = results[method]
    projected = model.project_controls(raw, result.requested, bounds, capabilities=caps)
    for name in ("p", "delay", "shape"):
        assert torch.equal(projected[name], result.projected[name])
    adv = model.generate(raw, projected, quantize=True, capabilities=caps)
    assert torch.equal(adv, result.adversarial_raw)
    with torch.no_grad():
        logits = victim((adv - center) / scale)
    assert torch.allclose(logits, result.logits, atol=1e-5)
    assert torch.allclose(targeted_margin(logits), result.objective_margin, atol=1e-5)
    assert torch.equal(result.success, (logits.argmax(1) == 0) & gate(adv))


@pytest.mark.parametrize("method", METHODS)
def test_projected_controls_stay_in_the_hard_integer_box(setup, method):
    _, _, _, _, _, bounds, _, _, results = setup
    result = results[method]
    p, delay, shape = (result.projected[name] for name in ("p", "delay", "shape"))
    assert torch.equal(p, torch.round(p)) and torch.equal(delay, torch.round(delay))
    assert bool((p >= 0).all() and (p <= torch.floor(bounds["p"])).all())
    assert bool((delay >= 0).all() and (delay <= torch.floor(bounds["delay"])).all())
    assert bool((shape >= 0).all() and (shape <= bounds["shape"]).all())
    assert bool((shape[delay == 0] == 0).all())


@pytest.mark.parametrize("method", ("hybrid_budget", "pgd", "cw"))
def test_per_flow_evaluation_budget_is_never_exceeded(setup, method):
    result = setup[-1][method]
    assert bool((result.total_evaluations <= BUDGET).all())
    first = result.first_success_evaluation
    assert torch.equal(first > 0, result.success)
    assert bool((first[result.success] <= result.total_evaluations[result.success]).all())


def test_hybrid_never_worse_than_identity_or_any_exact_padding(setup):
    """Success-first selection dominates every exhaustively enumerated padding candidate."""
    model, victim, raw, center, scale, bounds, caps, gate, results = setup
    result = results["hybrid"]
    zeros = torch.zeros(len(raw))
    reference_success = torch.zeros(len(raw), dtype=torch.bool)
    best_failure_margin = torch.full((len(raw),), torch.inf)
    with torch.no_grad():
        for value in range(int(torch.floor(bounds["p"]).max()) + 1):
            controls = model.project_controls(
                raw, {"p": zeros + value, "delay": zeros, "shape": zeros}, bounds,
                capabilities=caps,
            )
            adv = model.generate(raw, controls, quantize=True, capabilities=caps)
            logits = victim((adv - center) / scale)
            success = (logits.argmax(1) == 0) & gate(adv)
            reference_success |= success
            best_failure_margin = torch.where(
                success, best_failure_margin,
                torch.minimum(best_failure_margin, targeted_margin(logits)),
            )
    assert bool(result.success[reference_success].all())
    failed = ~reference_success
    assert bool((result.objective_margin[failed] <= best_failure_margin[failed] + 1e-5).all())


def test_pinned_padding_control_cannot_poison_timing_steps():
    """CICIDS2018 Recon flows without padding capability and without backward packets have a
    non-finite partial derivative of phi w.r.t. padding at p = 0. The pinned coordinate must get
    an exactly-zero gradient; otherwise Prim-C&W crashes and PGD/Hybrid silently freeze the row
    (sign(NaN) == 0)."""
    ad = get_adapter("cicids2018")
    model = CICIDS2017PrimitiveModel(ad.feature_manifest())
    transform = ad.feature_transform()
    center = torch.tensor(transform.center, dtype=torch.float32)
    scale = torch.tensor(transform.scale, dtype=torch.float32)
    test = ad.load_split("test")
    names = list(ad.feature_manifest().names)
    raw_all = np.load(ad._processed / "X_test_pristine.npy", mmap_mode="r")
    recon = np.flatnonzero(test.y == ad.class_mapping().name_to_id["Recon"])
    block = np.asarray(raw_all[recon[:20000]], dtype=np.float32)
    keep = (block[:, names.index("Total Bwd packets")] == 0) & (
        block[:, names.index("Total Fwd Packet")] >= 2)
    raw = torch.tensor(block[keep][:64])
    caps = model.infer_capabilities(raw)
    calibration = load_calibration(ad.repo_root / "artifacts/primattack/budget_calibration_cicids2018.json")
    bounds = model.per_flow_bounds(
        raw, class_calibration(calibration, "Recon", "maximum-evaluated").bounds_config(),
        capabilities=caps)
    pinned_padding = (bounds["p"] < 1.0) & (bounds["delay"] >= 1.0)
    assert bool(pinned_padding.any()), "fixture must contain timing-only flows"
    victim = load_category_victim(
        ad.repo_root / "outputs/cicids2018distrinet/classifiers_multiseed/runs/seed_42/models/mlp_category.pt",
        adapter=ad, expected_model_type="mlp",
    )
    rows = torch.nonzero(pinned_padding, as_tuple=False).flatten()
    search = RealizedSearch(model, victim, raw, center, scale, bounds, caps, validity_fn=None)
    q = torch.full((rows.numel(), 3), 0.5).requires_grad_(True)
    grad = torch.autograd.grad(targeted_margin(search.surrogate_logits(rows, q)).sum(), q)[0]
    assert bool(torch.isfinite(grad).all())
    assert bool((grad[:, 0] == 0).all())
    assert bool((grad[:, 1] != 0).any())
