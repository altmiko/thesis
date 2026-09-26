"""Invariants of the quantization-aware PrimAttack searches on a real victim."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from attack.primattack_budget import class_calibration, load_calibration
from attack.primitive_optimizer import (
    CANDIDATE_EXACT_PADDING,
    ROW_MODE_JOINT,
    ROW_MODE_PADDING_ONLY,
    ROW_MODE_TIMING_ONLY,
    AttackObjective,
    RealizedSearch,
    hybrid_valid_gate,
    optimize_primitive_candidates,
    optimize_primitive_cw,
    optimize_primitive_pgd,
    row_primitive_modes,
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
    raw_all = np.load(ad._processed / "X_test_pristine.npy", mmap_mode="r")
    dos = np.flatnonzero(test.y == ad.class_mapping().name_to_id["DoS"])
    fmin = np.asarray(raw_all[dos, list(ad.feature_manifest().names).index("Fwd Packet Length Min")])
    # 48 padding-capable flows (no empty forward packet) + 48 flows with an empty forward
    # packet that have timing headroom (timing-only rows)
    empty = dos[fmin == 0][:2000]
    empty_raw = torch.tensor(np.asarray(raw_all[empty]), dtype=torch.float32)
    headroom = model.per_flow_bounds(empty_raw, cfg)["delay"].numpy() >= 1.0
    rows = np.sort(np.concatenate([dos[fmin > 0][:48], empty[headroom][:48]]))
    raw = torch.tensor(np.ascontiguousarray(np.asarray(raw_all[rows]), dtype=np.float32))
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
    assert torch.equal(result.success, (logits.argmax(1) != SOURCE) & gate(adv, raw))
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
    assert torch.equal(result.success, (logits.argmax(1) == 0) & gate(adv, raw))


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


ALL_METHODS = METHODS + UNTARGETED


@pytest.mark.parametrize("method", ALL_METHODS)
def test_padding_ineligible_rows_never_pad_or_fill_an_empty_packet(setup, method):
    """Flows with an empty forward packet have no padding capability: every optimizer returns
    p == 0 exactly and leaves Fwd Packet Length Min at 0 (no filled empty packet)."""
    model, _, raw, _, _, bounds, caps, _, results = setup
    result = results[method]
    empty = raw[:, model.i["Fwd Packet Length Min"]] == 0
    assert bool(empty.any()) and not bool(caps.pad_allowed[empty].any())
    assert bool((bounds["p"][empty] == 0).all())
    assert bool((result.projected["p"][empty] == 0).all())
    assert bool((result.adversarial_raw[empty, model.i["Fwd Packet Length Min"]] == 0).all())
    modes = np.asarray(row_primitive_modes(bounds))
    timing = empty.numpy() & (bounds["delay"] >= 1.0).numpy()
    assert set(modes[timing]) == {ROW_MODE_TIMING_ONLY}
    assert not set(modes[empty.numpy()]) & {ROW_MODE_JOINT, ROW_MODE_PADDING_ONLY}


def test_timing_only_rows_spend_the_whole_budget_on_timing(setup):
    """Hybrid: timing-only rows skip the padding enumeration at zero cost, so an unsuccessful
    row spends its full evaluation budget on timing refinement, and timing is actually moved."""
    model, _, raw, _, _, bounds, caps, _, results = setup
    result = results["hybrid_budget"]
    modes = np.asarray(row_primitive_modes(bounds))
    rows = torch.as_tensor(modes == ROW_MODE_TIMING_ONLY)
    assert bool(rows.any())
    assert not bool((result.candidate_source[rows] == CANDIDATE_EXACT_PADDING).any())
    failed = rows & ~result.success
    assert bool(failed.any())
    # identity (1) + 2 evaluations per gradient step; at most one evaluation can be left over
    assert bool((result.total_evaluations[failed] >= BUDGET - 1).all())
    assert bool((result.surrogate_evaluations[failed] > 0).all())
    assert bool((result.projected["delay"][failed] > 0).any())


class _DurationVictim(torch.nn.Module):
    """Benign iff the scaled Fwd IAT Total exceeds a per-flow threshold (2 logits). The flow
    is identified by its Src Port (never written by the primitives), so the victim is correct
    on any subset / order / repetition of rows the search scores."""

    def __init__(self, key_col: int, keys: torch.Tensor, col: int, threshold: torch.Tensor,
                 slope: torch.Tensor) -> None:
        super().__init__()
        order = torch.argsort(keys)
        self.key_col, self.keys, self.col = key_col, keys[order], col
        self.threshold, self.slope = threshold[order], slope[order]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.searchsorted(self.keys, x[:, self.key_col].contiguous())
        benign = (x[:, self.col] - self.threshold[pos]) * self.slope[pos]
        return torch.stack((benign, torch.zeros_like(benign)), 1)


@pytest.mark.parametrize("method", ("hybrid", "pgd", "cw"))
def test_timing_only_rows_can_return_a_timing_only_success(setup, method):
    """A victim that only reacts to added forward delay: padding-ineligible rows succeed with
    p == 0 and delay > 0 (the attack is a real timing-only search, not a dead padding axis)."""
    model, _, raw, center, scale, bounds, caps, _, _ = setup
    modes = np.asarray(row_primitive_modes(bounds))
    port = raw[:, model.i["Src Port"]].numpy()
    _, first = np.unique(port, return_index=True)
    unique_port = np.zeros(len(port), dtype=bool)
    unique_port[first] = True
    keep = torch.as_tensor((modes == ROW_MODE_TIMING_ONLY) & unique_port)
    sub = raw[keep]
    assert sub.shape[0] >= 8
    b = {k: v[keep] for k, v in bounds.items()}
    c = model.infer_capabilities(sub)
    col, key = model.i["Fwd IAT Total"], model.i["Src Port"]
    reach = b["delay"] / scale[col]
    thr = (sub[:, col] - center[col]) / scale[col] + 0.2 * reach
    # logits in units of 10% of the reachable delay, so C&W's cost term cannot dominate
    victim = _DurationVictim(key, (sub[:, key] - center[key]) / scale[key], col, thr,
                             10.0 / reach)
    args = (model, victim, sub, center, scale, b, c)
    if method == "hybrid":
        res = optimize_primitive_candidates(*args, steps=8, learning_rate=0.1, seed=42,
                                            restarts=None, validity_fn=None, eval_budget=BUDGET)
    elif method == "pgd":
        res = optimize_primitive_pgd(*args, steps=10, step_size=0.05, restarts=3, seed=42,
                                     validity_fn=None, eval_budget=BUDGET)
    else:
        res = optimize_primitive_cw(*args, steps=10, learning_rate=0.05, stages=3,
                                    validity_fn=None, eval_budget=BUDGET)
    assert bool(res.success.all())
    assert bool((res.projected["p"] == 0).all()) and bool((res.projected["delay"] > 0).all())


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
            success = (logits.argmax(1) == 0) & gate(adv, raw)
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
