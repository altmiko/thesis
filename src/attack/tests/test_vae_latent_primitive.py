"""Proof that the proposed attack genuinely depends on the VAE latent/decoder path."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel
from attack.run_cicids2017_primitive_attack import train_envelope
from attack.vae_latent_primitive import LatentAttackConfig, LatentPrimitiveAttack
from datasets import get_adapter
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import load_stage_a


@pytest.fixture(scope="module")
def setup():
    ad = get_adapter("cicids2017")
    man = ad.feature_manifest(); model = CICIDS2017PrimitiveModel(man)
    transform = ad.feature_transform()
    center = torch.tensor(transform.center, dtype=torch.float32)
    scale = torch.tensor(transform.scale, dtype=torch.float32)
    train_raw = np.load(ad._processed / "X_train_pristine.npy", mmap_mode="r")
    env = train_envelope(train_raw, model.i)
    cfg = {"p_max": 1460.0, "alpha_max": 100.0, "mtu_cap": 0.0,
           **{f"env_{k}": v for k, v in env.items()}}
    test = ad.load_split("test")
    rows = np.flatnonzero(test.y == ad.class_mapping().name_to_id["DoS"])[:32]
    raw = torch.tensor(np.ascontiguousarray(np.asarray(
        np.load(ad._processed / "X_test_pristine.npy", mmap_mode="r")[rows]), dtype=np.float32))
    repo = ad.repo_root
    vae, _ = load_stage_a(
        ad, repo / "outputs/cicids2017_vae_stage_a/vae_DoS.pt",
        expected_class_name="DoS",
    )
    victim = load_category_victim(
        repo / "outputs/cicids2017distrinet/models/mlp_category.pt",
        adapter=ad, expected_model_type="mlp",
    )
    bounds = model.per_flow_bounds(raw, cfg)
    return model, vae, victim, raw, center, scale, bounds


def _graph(setup):
    model, vae, victim, raw, center, scale, bounds = setup
    x = (raw - center) / scale
    with torch.no_grad():
        z0, _ = vae.encode(x)
        decoded0 = vae.decode(z0)["continuous_mu_raw"].detach()
    # Realistic operating point: the attack uses a random latent start (off the relu(0) dead
    # point), so evaluate the decoder gradient at z0 + small delta, as the optimizer does.
    torch.manual_seed(0)
    z = (z0.detach() + 0.1 * torch.randn_like(z0)).requires_grad_(True)
    decoded = vae.decode(z)["continuous_mu_raw"]
    controls = model.infer_primitives_from_decoded(raw, decoded, decoded0, bounds)
    adv = model.generate(raw, controls)
    logits = victim((adv - center) / scale)
    return z, decoded, controls, adv, logits


def test_classifier_gradient_flows_through_decoder_to_z(setup):
    z, _, _, _, logits = _graph(setup)
    target = torch.zeros(logits.shape[0], dtype=torch.long)
    torch.nn.functional.cross_entropy(logits, target).backward()
    assert z.grad is not None
    assert bool(torch.isfinite(z.grad).all())
    assert float(z.grad.norm()) > 0.0


def test_detaching_decoder_breaks_attack_gradient(setup):
    model, vae, victim, raw, center, scale, bounds = setup
    x = (raw - center) / scale
    with torch.no_grad():
        z0, _ = vae.encode(x); decoded0 = vae.decode(z0)["continuous_mu_raw"].detach()
    z = z0.detach().clone().requires_grad_(True)
    decoded_detached = vae.decode(z)["continuous_mu_raw"].detach()
    controls = model.infer_primitives_from_decoded(raw, decoded_detached, decoded0, bounds)
    adv = model.generate(raw, controls)
    logits = victim((adv - center) / scale)
    # With decoder output detached and victim params frozen, classifier loss has no z path.
    assert not logits.requires_grad


def test_decoder_movement_changes_controls_and_adversarial_vector(setup):
    model, vae, _, raw, center, scale, bounds = setup
    with torch.no_grad():
        z0, _ = vae.encode((raw - center) / scale)
        d0 = vae.decode(z0)["continuous_mu_raw"]
        z1 = z0 + 0.25 * torch.ones_like(z0)
        d1 = vae.decode(z1)["continuous_mu_raw"]
        c0 = model.infer_primitives_from_decoded(raw, d0, d0, bounds)
        c1 = model.infer_primitives_from_decoded(raw, d1, d0, bounds)
        a0 = model.generate(raw, c0); a1 = model.generate(raw, c1)
    assert not torch.allclose(d0, d1)
    assert (not torch.allclose(c0["p"], c1["p"])) or (not torch.allclose(c0["alpha"], c1["alpha"]))
    assert not torch.allclose(a0, a1)


def test_optimizer_parameter_list_contains_only_z_adv(setup):
    z, _, controls, _, _ = _graph(setup)
    attack = LatentPrimitiveAttack(setup[0], LatentAttackConfig(steps=1))
    params = attack.optimizer_parameters(z)
    assert params == [z]
    assert z.is_leaf and z.requires_grad
    assert not controls["p"].is_leaf and not controls["alpha"].is_leaf


def test_single_packet_timing_disabled_in_latent_graph(setup):
    model, vae, _, _, center, scale, _ = setup
    ad = get_adapter("cicids2017")
    all_raw = np.asarray(np.load(ad._processed / "X_test_pristine.npy", mmap_mode="r")[:10000])
    idx = np.flatnonzero(all_raw[:, model.i["Total Fwd Packet"]] < 2)[:16]
    raw = torch.tensor(np.ascontiguousarray(all_raw[idx], dtype=np.float32))
    train_raw = np.load(ad._processed / "X_train_pristine.npy", mmap_mode="r")
    env = train_envelope(train_raw, model.i)
    bounds = model.per_flow_bounds(raw, {"p_max":1460.,"alpha_max":100.,"mtu_cap":0.,
        **{f"env_{k}":v for k,v in env.items()}})
    with torch.no_grad():
        z0, _ = vae.encode((raw - center) / scale)
        d0 = vae.decode(z0)["continuous_mu_raw"]
    z = (z0 + torch.ones_like(z0)).detach().requires_grad_(True)
    d1 = vae.decode(z)["continuous_mu_raw"]
    controls = model.infer_primitives_from_decoded(raw, d1, d0, bounds)
    alpha_grad = torch.autograd.grad(controls["alpha"].sum(), z, retain_graph=True)[0]
    adv = model.generate(raw, controls)
    timing_names = (
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
        "Fwd IAT Min", "Flow Duration", "Flow IAT Mean", "Flow IAT Max",
    )
    timing_idx = [model.i[name] for name in timing_names]
    assert torch.equal(controls["alpha"], torch.ones_like(controls["alpha"]))
    assert torch.equal(alpha_grad, torch.zeros_like(alpha_grad))
    assert torch.allclose(adv[:, timing_idx], raw[:, timing_idx], atol=1e-3, rtol=1e-4)


def test_final_result_reclassified_after_projection(setup):
    model, vae, victim, raw, center, scale, bounds = setup
    attack = LatentPrimitiveAttack(model, LatentAttackConfig(steps=2, learning_rate=0.02))
    result = attack.attack(vae, victim, raw[:8], center, scale,
                           {k:v[:8] for k,v in bounds.items()})
    with torch.no_grad():
        expected = victim((result.x_adv_realized_raw-center)/scale)
    assert torch.allclose(result.realized_logits, expected)
    assert torch.allclose(result.controls_realized["p"], torch.round(result.controls_realized["p"]))


@pytest.mark.parametrize(
    ("steps", "epsilon_z", "init_noise"),
    [(0, 10.0, 0.0), (2, 0.0, 0.3)],
    ids=("zero-step", "zero-radius"),
)
def test_no_movement_controls_are_identity(setup, steps, epsilon_z, init_noise):
    model, vae, victim, raw, center, scale, bounds = setup
    attack = LatentPrimitiveAttack(
        model,
        LatentAttackConfig(
            steps=steps,
            learning_rate=0.02,
            epsilon_z=epsilon_z,
            init_noise=init_noise,
        ),
    )
    torch.manual_seed(123)
    result = attack.attack(
        vae,
        victim,
        raw[:8],
        center,
        scale,
        {key: value[:8] for key, value in bounds.items()},
    )
    assert torch.equal(result.z_adv, result.z0)
    assert torch.equal(
        result.controls_realized["p"],
        torch.zeros_like(result.controls_realized["p"]),
    )
    assert torch.equal(
        result.controls_realized["alpha"],
        torch.ones_like(result.controls_realized["alpha"]),
    )
    assert torch.allclose(result.x_adv_realized_raw, raw[:8], atol=1e-3, rtol=1e-4)
    assert torch.equal(result.realized_logits.argmax(1), result.clean_logits.argmax(1))


def test_attack_is_invariant_to_unrelated_batch_rows(setup):
    model, vae, victim, raw, center, scale, bounds = setup
    attack = LatentPrimitiveAttack(
        model, LatentAttackConfig(steps=6, learning_rate=0.03, epsilon_z=10.0)
    )
    torch.manual_seed(321)
    full = attack.attack(
        vae, victim, raw[:16], center, scale,
        {key: value[:16] for key, value in bounds.items()},
    )
    torch.manual_seed(321)
    subset = attack.attack(
        vae, victim, raw[:8], center, scale,
        {key: value[:8] for key, value in bounds.items()},
    )
    assert torch.allclose(full.z_adv[:8], subset.z_adv, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        full.x_adv_realized_raw[:8], subset.x_adv_realized_raw, atol=1e-5, rtol=1e-6
    )
    assert torch.equal(full.realized_logits[:8].argmax(1), subset.realized_logits.argmax(1))
