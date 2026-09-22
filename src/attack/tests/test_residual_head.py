"""Phase E verification: masked residual head, derived recompute, attack integration.

Run (thesis env):
    PYTHONPATH=src python -m pytest src/attack/tests/test_residual_head.py -q -p no:faulthandler
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from attack.residual_head import ResidualAttackGenerator, ResidualHead
from constraints.layer0 import Layer0Projector
from datasets import FeatureManifest, FeatureSpec, FeatureTransform
from datasets.ciciot2023 import CICIoT2023Adapter
from vae.model import MixedInputBetaVAE


def _ciciot_generator(mode="both"):
    adapter = CICIoT2023Adapter()
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    torch.manual_seed(0)
    vae = MixedInputBetaVAE(manifest=manifest, latent_dim=16)
    vae.register_feature_transform(transform)
    projector = Layer0Projector(manifest)
    mutable = torch.ones(manifest.n_features, dtype=torch.bool)
    frozen_names = ["Protocol Type", "Time_To_Live"]
    frozen_idx = [manifest.index_by_name(n) for n in frozen_names]
    for i in frozen_idx:
        mutable[i] = False
    head = ResidualHead(16, manifest.n_features) if mode != "latent" else None
    gen = ResidualAttackGenerator(vae, projector, mutable, head=head, mode=mode)
    return adapter, manifest, transform, vae, projector, gen, frozen_idx


def _load_ddos(adapter, n=256):
    split = adapter.load_split("val")
    cm = adapter.class_mapping()
    idx = np.where(np.asarray(split.y) == cm.name_to_id["DDoS"])[0][:n]
    return torch.tensor(np.asarray(split.x[idx]), dtype=torch.float32)


@pytest.mark.parametrize("mode", ["latent", "residual", "both"])
def test_generate_respects_mask_and_layer0(mode):
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    _a, manifest, _t, vae, projector, gen, frozen_idx = _ciciot_generator(mode)
    x = _load_ddos(adapter)
    # perturb the latent so latent/both modes yield a non-trivial displacement
    z = vae.encode(x)[0] + 0.5 * torch.randn(x.shape[0], 16)
    x_adv, meta = gen.generate(x, z=z)
    for i in frozen_idx:
        assert torch.allclose(x_adv[:, i], x[:, i], atol=1e-5)
    # Layer-0 domain satisfied by construction
    assert bool(projector.validate(meta["raw_adv"]).all())
    # something actually changed on mutable features (not a no-op)
    assert not torch.allclose(x_adv, x, atol=1e-6)


def test_derived_feature_recompute_exact():
    # synthetic manifest with Std and Variance; Variance is recomputed = Std^2
    specs = [
        FeatureSpec("prim", 0, "positive_continuous", "x", lower=0.0),
        FeatureSpec("Std", 1, "positive_continuous", "size", lower=0.0),
        FeatureSpec("Variance", 2, "positive_continuous", "size", lower=0.0),
    ]
    manifest = FeatureManifest(specs, dataset_name="toy")
    rng = np.random.default_rng(0)
    x_raw = np.abs(rng.normal(2, 1, size=(1000, 3)))
    x_raw[:, 2] = x_raw[:, 1] ** 2
    transform = FeatureTransform(manifest).fit(x_raw)
    torch.manual_seed(0)
    vae = MixedInputBetaVAE(manifest=manifest, latent_dim=4, encoder_hidden=(16,), decoder_hidden=(16,))
    vae.register_feature_transform(transform)
    projector = Layer0Projector(manifest)
    mutable = torch.ones(3, dtype=torch.bool)
    head = ResidualHead(4, 3)
    var_idx, std_idx = manifest.index_by_name("Variance"), manifest.index_by_name("Std")
    gen = ResidualAttackGenerator(
        vae, projector, mutable, head=head, mode="both",
        derivations=[(var_idx, lambda raw: raw[:, std_idx] ** 2)],
    )
    x = torch.tensor(transform.transform(x_raw[:64]), dtype=torch.float32)
    _x_adv, meta = gen.generate(x)
    raw = meta["raw_adv"]
    assert torch.allclose(raw[:, var_idx], raw[:, std_idx] ** 2, atol=1e-4)


def test_residual_attack_step_is_differentiable_and_reduces_target_loss():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    _a, manifest, _t, vae, _p, gen, _f = _ciciot_generator(mode="both")
    x = _load_ddos(adapter, n=256)

    # tiny frozen victim: benign target class 0
    torch.manual_seed(1)
    victim = torch.nn.Linear(manifest.n_features, 8)
    for p in victim.parameters():
        p.requires_grad_(False)
    target = torch.zeros(x.shape[0], dtype=torch.long)  # "benign"

    z = vae.encode(x)[0].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z, *gen.head.parameters()], lr=1e-2)
    first = last = None
    for step in range(40):
        opt.zero_grad()
        x_adv, _ = gen.generate(x, z=z)
        loss = F.cross_entropy(victim(x_adv), target)
        loss.backward()
        assert z.grad is not None and any(p.grad is not None for p in gen.head.parameters())
        opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    assert last < first  # victim-target loss decreased via the residual path
