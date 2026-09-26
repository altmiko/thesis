from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from joblib import parallel_backend
import pandas  # Load pyarrow/pandas before torch on Windows to avoid DLL teardown faults.
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _path in (str(REPO_ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from comparisons.capgd_cicids2017 import (  # noqa: E402
    RawCICIDSVictim,
    build_capgd_prim_support_resources,
    build_capgd_resources,
    evaluate_capgd_output,
    finalize_capgd_output,
    fit_train_minmax,
    make_capgd,
)
from comparisons.cpgd_prim_support import (  # noqa: E402
    CPGDConfig,
    CPGDPrimSupportAttack,
)
from datasets import get_adapter  # noqa: E402


@pytest.fixture(scope="module")
def resources():
    return build_capgd_resources(REPO_ROOT)

@pytest.fixture(scope="module")
def prim_support_resources(resources):
    manifest = get_adapter("cicids2017").feature_manifest()
    return build_capgd_prim_support_resources(resources, manifest)


class _RawBoxVictim(nn.Module):
    def __init__(self, low: np.ndarray, high: np.ndarray, feature_index: int) -> None:
        super().__init__()
        self.register_buffer("low", torch.as_tensor(low))
        self.register_buffer("span", torch.as_tensor(np.maximum(high - low, 1e-12)))
        self.feature_index = feature_index

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        score = ((raw - self.low) / self.span)[:, self.feature_index] - 0.5
        floor = torch.full_like(score, -4.0)
        return torch.stack((4.0 * score, -4.0 * score, floor, floor, floor), dim=1)


def test_fit_train_minmax_uses_every_row_without_loading_holdouts(tmp_path: Path) -> None:
    values = np.asarray(
        [[4.0, -2.0] + [0.0] * 77, [1.0, 8.0] + [0.0] * 77, [3.0, 5.0] + [0.0] * 77],
        dtype=np.float32,
    )
    path = tmp_path / "X_train_pristine.npy"
    np.save(path, values)
    low, high = fit_train_minmax(path, chunk_rows=1)
    assert low[:2].tolist() == [1.0, -2.0]
    assert high[:2].tolist() == [4.0, 8.0]


def test_raw_victim_applies_existing_robust_transform() -> None:
    victim = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        victim.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]))
    wrapper = RawCICIDSVictim(
        victim,
        center=np.asarray([10.0, 20.0, 30.0], np.float32),
        scale=np.asarray([2.0, 4.0, 5.0], np.float32),
    )
    raw = torch.tensor([[12.0, 24.0, 35.0]])
    expected = victim(torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.allclose(wrapper(raw), expected)


def test_final_projection_restores_frozen_and_recomputes_dependencies(resources) -> None:
    processed = REPO_ROOT / "data/processed/CICIDS_2017_Distrinet"
    clean_np = np.asarray(np.load(processed / "X_test_pristine.npy", mmap_mode="r")[:2]).copy()
    clean = torch.as_tensor(clean_np)
    candidate = clean.clone()
    frozen_index = resources.resolved_mask.frozen_idx[0]
    candidate[:, frozen_index] += 123.0
    direct_index = resources.resolved_mask.perturbable_idx[1]
    candidate[:, direct_index] += 10.0

    projected = finalize_capgd_output(resources, clean, candidate)
    assert torch.equal(projected[:, frozen_index], clean[:, frozen_index])
    assert not bool(resources.resolved_mask.derived_consistency_mask(projected).any())
    assert not bool(resources.resolved_mask.frozen_violation_mask(projected, clean).any())


def test_clean_rows_pass_capgd_and_validator_constraints(resources) -> None:
    processed = REPO_ROOT / "data/processed/CICIDS_2017_Distrinet"
    clean = np.asarray(np.load(processed / "X_test_pristine.npy", mmap_mode="r")[:8]).copy()
    result = evaluate_capgd_output(resources, clean, clean, norm="L2", eps=0.5)
    assert np.asarray(result["internal_constraint_valid"], bool).all()
    assert np.asarray(result["distance_ok"], bool).all()
    assert np.asarray(result["hybrid_valid"], bool).all()
    assert np.allclose(result["distance"], 0.0)


def test_prim_support_finalizer_changes_only_allowed_coordinates(
    resources, prim_support_resources,
) -> None:
    processed = REPO_ROOT / "data/processed/CICIDS_2017_Distrinet"
    clean = torch.as_tensor(
        np.asarray(np.load(processed / "X_test_pristine.npy", mmap_mode="r")[:2]).copy()
    )
    candidate = clean + 1.0
    projected = finalize_capgd_output(prim_support_resources, clean, candidate)
    support = torch.as_tensor(prim_support_resources.support_mask)

    assert torch.equal(projected[:, ~support], clean[:, ~support])
    assert torch.equal(projected[:, support], candidate[:, support])
    assert bool((projected[:, support] != clean[:, support]).any())
    # Deriving matched support must not mutate the pre-existing native configuration.
    assert resources.configuration == "capgd_native"
    assert resources.resolved_mask is not None
    assert prim_support_resources.configuration == "capgd_prim_support"
    assert prim_support_resources.resolved_mask is None


def test_capgd_prim_support_has_gradients_and_optimizes_inside_mask(
    prim_support_resources,
) -> None:
    processed = REPO_ROOT / "data/processed/CICIDS_2017_Distrinet"
    clean = torch.as_tensor(
        np.asarray(np.load(processed / "X_test_pristine.npy", mmap_mode="r")[:2]).copy()
    )
    support_idx = int(np.flatnonzero(prim_support_resources.support_mask)[0])
    victim = _RawBoxVictim(
        prim_support_resources.train_min,
        prim_support_resources.train_max,
        support_idx,
    )
    labels = victim(clean).argmax(1)
    attack = make_capgd(
        prim_support_resources,
        victim,
        device="cpu",
        seed=42,
        norm="L2",
        eps=0.5,
        steps=3,
    )

    normalized = prim_support_resources.scaler.transform(clean).detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(attack.get_logits(normalized), labels)
    (gradient,) = torch.autograd.grad(loss, normalized)
    support = torch.as_tensor(prim_support_resources.support_mask)
    assert bool((gradient[:, support].abs() > 0).any())
    assert torch.equal(attack.mutable_mask.bool().cpu(), support)

    with parallel_backend("threading"):
        candidate = attack(clean, labels)
    adversarial = finalize_capgd_output(prim_support_resources, clean, candidate)
    assert torch.equal(adversarial[:, ~support], clean[:, ~support])
    assert bool((adversarial[:, support] != clean[:, support]).any())
    evaluated = evaluate_capgd_output(
        prim_support_resources,
        clean.numpy(),
        adversarial.detach().numpy(),
        norm="L2",
        eps=0.5,
    )
    assert np.asarray(evaluated["distance_ok"], bool).all()
    assert np.asarray(evaluated["hybrid_valid"]).shape == (len(clean),)


@pytest.mark.parametrize("norm", ["L2", "Linf"])
def test_cpgd_gradients_projection_and_evaluator_compatibility(
    prim_support_resources, norm: str,
) -> None:
    processed = REPO_ROOT / "data/processed/CICIDS_2017_Distrinet"
    clean = torch.as_tensor(
        np.asarray(np.load(processed / "X_test_pristine.npy", mmap_mode="r")[:4]).copy()
    )
    support = torch.as_tensor(prim_support_resources.support_mask)
    support_idx = int(torch.nonzero(support, as_tuple=False)[0])
    victim = _RawBoxVictim(
        prim_support_resources.train_min,
        prim_support_resources.train_max,
        support_idx,
    )
    labels = victim(clean).argmax(1)
    config = CPGDConfig(
        epsilon=0.2,
        norm=norm,
        step_size=0.05,
        iterations=3,
        constraint_penalty_weight=1e-3,
        random_start=False,
    )
    attack = CPGDPrimSupportAttack(
        prim_support_resources, victim, config=config, seed=42, device="cpu"
    )

    normalized = prim_support_resources.scaler.transform(clean).detach().requires_grad_(True)
    score, attack_loss, constraint_loss = attack.objective(normalized, labels)
    (score_gradient,) = torch.autograd.grad(score, normalized, retain_graph=True)
    (victim_gradient,) = torch.autograd.grad(attack_loss, normalized, retain_graph=True)
    assert bool((score_gradient[:, support].abs() > 0).any())
    assert bool((victim_gradient[:, support].abs() > 0).any())

    violating = clean.clone()
    mean_idx = get_adapter("cicids2017").feature_manifest().index_by_name(
        "Fwd Packet Length Mean"
    )
    violating[:, mean_idx] += 100.0
    violating.requires_grad_(True)
    penalty = attack.constraint_violation(violating).sum()
    (penalty_gradient,) = torch.autograd.grad(penalty, violating)
    assert constraint_loss.requires_grad
    assert bool((penalty_gradient[:, support].abs() > 0).any())

    result = attack.run(clean, labels)
    adversarial = result.adversarial_raw
    assert torch.equal(adversarial[:, ~support], clean[:, ~support])
    assert bool((adversarial[:, support] != clean[:, support]).any())
    clean_normalized = prim_support_resources.scaler.transform(clean)
    adversarial_normalized = prim_support_resources.scaler.transform(adversarial)
    delta = adversarial_normalized - clean_normalized
    distance = delta.norm(p=2, dim=1) if norm == "L2" else delta.abs().amax(dim=1)
    assert bool((distance <= config.epsilon + 1e-5).all())

    evaluated = evaluate_capgd_output(
        prim_support_resources,
        clean.numpy(),
        adversarial.numpy(),
        norm=config.norm,
        eps=config.epsilon,
    )
    for key in ("hybrid_valid", "hard_structural_valid", "distance", "distance_ok"):
        assert np.asarray(evaluated[key]).shape == (len(clean),)
