"""Sanity checks for the VAE-Latent-Raw and VAE-Latent-Masked decoder-movement attacks.

Covers the checklist for both variants (z_adv is the only leaf; detaching the decoder kills
the attack gradient; z_adv=z0 gives no adversarial change; same seed reproduces results; the
reported prediction is after projection; identical clean-correct denominator across variants)
plus the masked-specific closure checks (no frozen feature changes; no DERIVED_EXACT feature is
independently optimized; every changed feature lies in the perturbation/dependency closure).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from attack.masks import get_dataset_mask
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel, SCALER_ATOL, _INTEGER_FEATURES
from attack.vae_latent_primitive import LatentAttackConfig
from attack.vae_latent_variants import LatentMaskedAttack, LatentRawAttack
from datasets import get_adapter
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import load_stage_a


@pytest.fixture(scope="module")
def setup():
    ad = get_adapter("cicids2017")
    man = ad.feature_manifest()
    model = CICIDS2017PrimitiveModel(man)
    rmask = get_dataset_mask(ad.name).resolve(man)
    projector = rmask.generator_projector()
    transform = ad.feature_transform()
    center = torch.tensor(transform.center, dtype=torch.float32)
    scale = torch.tensor(transform.scale, dtype=torch.float32)
    test = ad.load_split("test")
    rows = np.flatnonzero(test.y == ad.class_mapping().name_to_id["DoS"])[:32]
    raw = torch.tensor(np.ascontiguousarray(np.asarray(
        np.load(ad._processed / "X_test_pristine.npy", mmap_mode="r")[rows]), dtype=np.float32))
    repo = ad.repo_root
    vae, _ = load_stage_a(ad, repo / "outputs/cicids2017_vae_attacks/stage_a/vae_DoS.pt")
    victim = load_category_victim(repo / "outputs/cicids2017distrinet/models/mlp_category.pt")
    int_idx = [model.i[n] for n in rmask.mask.perturbable if n in _INTEGER_FEATURES]
    cfg = LatentAttackConfig(steps=8, learning_rate=0.08, epsilon_z=10.0)
    return dict(model=model, rmask=rmask, projector=projector, center=center, scale=scale,
                raw=raw, vae=vae, victim=victim, int_idx=int_idx, cfg=cfg)


def _make(setup, variant):
    if variant == "raw":
        return LatentRawAttack(setup["projector"], setup["cfg"])
    return LatentMaskedAttack(setup["rmask"], setup["projector"], setup["cfg"],
                              integer_perturbable_idx=setup["int_idx"])


@pytest.mark.parametrize("variant", ["raw", "masked"])
def test_optimizer_leaf_is_only_z_adv(setup, variant):
    atk = _make(setup, variant)
    raw, center, scale = setup["raw"], setup["center"], setup["scale"]
    with torch.no_grad():
        z0, _ = setup["vae"].encode((raw - center) / scale)
        base = setup["vae"].decode(z0)["continuous_mu_raw"]
    z = (z0 + 0.1 * torch.randn_like(z0)).requires_grad_(True)
    decoded = setup["vae"].decode(z)["continuous_mu_raw"]
    x = atk._realize_continuous(raw, decoded, base)
    assert atk.optimizer_parameters(z) == [z]
    assert z.is_leaf and z.requires_grad
    assert not x.is_leaf  # feature vector is a non-leaf function of z, never optimized directly


@pytest.mark.parametrize("variant", ["raw", "masked"])
def test_classifier_gradient_flows_to_z_and_detach_kills_it(setup, variant):
    atk = _make(setup, variant)
    raw, center, scale, victim = setup["raw"], setup["center"], setup["scale"], setup["victim"]
    with torch.no_grad():
        z0, _ = setup["vae"].encode((raw - center) / scale)
        base = setup["vae"].decode(z0)["continuous_mu_raw"]
    z = (z0 + 0.3 * torch.randn_like(z0)).requires_grad_(True)
    decoded = setup["vae"].decode(z)["continuous_mu_raw"]
    x = atk._realize_continuous(raw, decoded, base)
    victim((x - center) / scale).sum().backward()
    assert z.grad is not None and float(z.grad.norm()) > 0.0

    # Detaching the decoder output severs every path to z_adv.
    z2 = (z0 + 0.3 * torch.randn_like(z0)).requires_grad_(True)
    decoded2 = setup["vae"].decode(z2)["continuous_mu_raw"].detach()
    x2 = atk._realize_continuous(raw, decoded2, base)
    logits2 = victim((x2 - center) / scale)
    assert not logits2.requires_grad


@pytest.mark.parametrize("variant", ["raw", "masked"])
def test_z_adv_equals_z0_gives_no_adversarial_change(setup, variant):
    atk = _make(setup, variant)
    raw, center, scale, victim = setup["raw"], setup["center"], setup["scale"], setup["victim"]
    with torch.no_grad():
        z0, _ = setup["vae"].encode((raw - center) / scale)
        base = setup["vae"].decode(z0)["continuous_mu_raw"]
        x = atk._realize_continuous(raw, base, base)  # decoded_adv == decoded_base -> zero movement
        x_final = atk._finalize(raw, x)
        clean_pred = victim((raw - center) / scale).argmax(1)
        adv_pred = victim((x_final - center) / scale).argmax(1)
    # movement is exactly zero -> realized vector matches pristine (within scaler/recompute tol)
    assert torch.allclose(x_final, raw, atol=1e-3, rtol=1e-3)
    assert torch.equal(clean_pred, adv_pred)


@pytest.mark.parametrize("variant", ["raw", "masked"])
def test_same_seed_reproduces_result(setup, variant):
    raw, center, scale, victim = setup["raw"], setup["center"], setup["scale"], setup["victim"]
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        atk = _make(setup, variant)
        res = atk.attack(setup["vae"], victim, raw, center, scale, target_class=0)
        outs.append(res.x_adv_realized_raw)
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("variant", ["raw", "masked"])
def test_reported_prediction_is_after_projection(setup, variant):
    raw, center, scale, victim = setup["raw"], setup["center"], setup["scale"], setup["victim"]
    torch.manual_seed(0)
    atk = _make(setup, variant)
    res = atk.attack(setup["vae"], victim, raw, center, scale, target_class=0)
    recomputed = victim((res.x_adv_realized_raw - center) / scale)
    assert torch.allclose(res.realized_logits, recomputed, atol=1e-5)


def test_clean_correct_denominator_identical_across_variants(setup):
    raw, center, scale, victim = setup["raw"], setup["center"], setup["scale"], setup["victim"]
    clean = victim((raw - center) / scale).argmax(1)
    # clean-correct depends only on (raw, victim), never on the attack variant.
    for variant in ("raw", "masked"):
        torch.manual_seed(0)
        atk = _make(setup, variant)
        res = atk.attack(setup["vae"], victim, raw, center, scale, target_class=0)
        assert torch.equal(res.clean_logits.argmax(1), clean)


def test_masked_freezes_frozen_and_only_perturbs_closure(setup):
    raw, center, scale, victim = setup["raw"], setup["center"], setup["scale"], setup["victim"]
    rmask = setup["rmask"]
    torch.manual_seed(0)
    atk = _make(setup, "masked")
    res = atk.attack(setup["vae"], victim, raw, center, scale, target_class=0)
    adv = res.x_adv_realized_raw

    # 1) no FROZEN feature changes.
    fidx = rmask._frozen_t
    assert torch.allclose(adv[:, fidx], raw[:, fidx], atol=SCALER_ATOL, rtol=1e-4)

    # 2) DERIVED_EXACT are recomputed from parents, never independently optimized.
    assert not rmask.derived_consistency_mask(adv, scale=scale, tol=1e-3).any()

    # 3) every changed feature lies in the perturbation/dependency closure.
    changed = (adv - raw).abs() > (1e-4 + 1e-3 * raw.abs())
    closure = torch.zeros(raw.shape[1], dtype=torch.bool)
    closure[list(rmask.perturbable_idx)] = True
    closure[list(rmask.derived_idx)] = True
    assert bool((changed.any(0) & ~closure).sum() == 0)


def test_raw_variant_moves_features_outside_the_mask(setup):
    """The raw diagnostic is *not* constrained to the mask -- it must move frozen features too,
    which is why it is not a valid attack (frozen/derived consistency fails by design)."""
    raw, center, scale, victim = setup["raw"], setup["center"], setup["scale"], setup["victim"]
    rmask = setup["rmask"]
    torch.manual_seed(0)
    atk = _make(setup, "raw")
    res = atk.attack(setup["vae"], victim, raw, center, scale, target_class=0)
    adv = res.x_adv_realized_raw
    frozen_moved = rmask.frozen_violation_mask(adv, raw, atol=SCALER_ATOL, rtol=1e-4)
    assert bool(frozen_moved.any())
