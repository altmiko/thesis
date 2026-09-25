"""Invariants of the quantization-aware PrimAttack candidate search on a real victim."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from attack.primattack_budget import class_calibration, load_calibration
from attack.primitive_optimizer import optimize_primitive_candidates, targeted_margin
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel
from datasets import get_adapter
from src.classifiers.cicids2017d_victims import load_category_victim


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
    result = optimize_primitive_candidates(
        model, victim, raw, center, scale, bounds, caps,
        steps=8, learning_rate=0.1, seed=42, restarts=2,
    )
    return model, victim, raw, center, scale, bounds, caps, result


def test_result_is_the_realized_projection_of_its_request(setup):
    model, victim, raw, center, scale, bounds, caps, result = setup
    projected = model.project_controls(raw, result.requested, bounds, capabilities=caps)
    for name in ("p", "delay", "shape"):
        assert torch.equal(projected[name], result.projected[name])
    adv = model.generate(raw, projected, quantize=True, capabilities=caps)
    assert torch.equal(adv, result.adversarial_raw)
    with torch.no_grad():
        logits = victim((adv - center) / scale)
    assert torch.allclose(logits, result.logits, atol=1e-5)
    assert torch.allclose(targeted_margin(logits), result.target_margin, atol=1e-5)


def test_projected_controls_stay_in_the_hard_integer_box(setup):
    _, _, _, _, _, bounds, _, result = setup
    p, delay, shape = (result.projected[name] for name in ("p", "delay", "shape"))
    assert torch.equal(p, torch.round(p)) and torch.equal(delay, torch.round(delay))
    assert bool((p >= 0).all() and (p <= torch.floor(bounds["p"])).all())
    assert bool((delay >= 0).all() and (delay <= torch.floor(bounds["delay"])).all())
    assert bool((shape >= 0).all() and (shape <= bounds["shape"]).all())
    assert bool((shape[delay == 0] == 0).all())


def test_never_worse_than_identity_or_any_exact_padding(setup):
    """Success-first selection dominates every exhaustively enumerated padding candidate."""
    model, victim, raw, center, scale, bounds, caps, result = setup
    zeros = torch.zeros(len(raw))
    best_margin = torch.full((len(raw),), torch.inf)
    with torch.no_grad():
        for value in range(int(torch.floor(bounds["p"]).max()) + 1):
            controls = model.project_controls(
                raw, {"p": zeros + value, "delay": zeros, "shape": zeros}, bounds,
                capabilities=caps,
            )
            adv = model.generate(raw, controls, quantize=True, capabilities=caps)
            best_margin = torch.minimum(
                best_margin, targeted_margin(victim((adv - center) / scale))
            )
    reference_success = best_margin < 0
    success = result.logits.argmax(1) == 0
    assert bool(success[reference_success].all())
    failed = ~reference_success
    assert bool((result.target_margin[failed] <= best_margin[failed] + 1e-5).all())
