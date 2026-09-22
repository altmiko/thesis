"""Phase B verification: manifest-driven typed VAE, dynamic dims, fail-loud transform.

Run (thesis env):
    PYTHONPATH=src python -m pytest src/vae/tests/test_typed_vae.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from datasets import FeatureManifest, FeatureSpec, FeatureTransform
from datasets.ciciot2023 import CICIoT2023Adapter
from vae.model import MixedInputBetaVAE, TransformNotRegisteredError
from vae.schema import get_partition


def _synthetic_manifest(n_extra_real: int = 3) -> FeatureManifest:
    specs = [
        FeatureSpec("pc", 0, "positive_continuous", "count", lower=0.0),
        FeatureSpec("pr", 1, "probability", "flag", lower=0.0, upper=1.0),
        FeatureSpec("bc", 2, "bounded_continuous", "ttl", lower=0.0, upper=255.0),
    ]
    for i in range(n_extra_real):
        specs.append(FeatureSpec(f"r{i}", 3 + i, "real", "misc"))
    return FeatureManifest(specs, dataset_name="synthetic")


# --------------------------------------------------------------------------- #
# Typed path on CICIoT (real manifest + transform)
# --------------------------------------------------------------------------- #
def test_typed_vae_ciciot_ranges():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "scaler.pkl").exists():
        pytest.skip("scaler.pkl unavailable")
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    torch.manual_seed(0)
    model = MixedInputBetaVAE(manifest=manifest, latent_dim=16)
    assert model.decoder_kind == "typed"
    assert model.n_features == 39
    model.register_feature_transform(transform)
    model.eval()

    z = torch.randn(64, 16)
    out = model.decode(z)
    raw = out["continuous_mu_raw"].detach().numpy()
    prob_idx = manifest.indices_of_value_type("probability")
    pos_idx = manifest.indices_of_value_type("positive_continuous")
    bnd_idx = manifest.indices_of_value_type("bounded_continuous")
    assert np.all(raw[:, prob_idx] >= -1e-6) and np.all(raw[:, prob_idx] <= 1 + 1e-6)
    assert np.all(raw[:, pos_idx] >= -1e-6)
    for j in bnd_idx:
        lo, hi = manifest[j].lower, manifest[j].upper
        assert np.all(raw[:, j] >= lo - 1e-4) and np.all(raw[:, j] <= hi + 1e-4)
    assert np.isfinite(raw).all()


def test_fail_loud_without_transform():
    manifest = _synthetic_manifest()
    model = MixedInputBetaVAE(manifest=manifest, latent_dim=4)
    with pytest.raises(TransformNotRegisteredError):
        model.decode(torch.randn(2, 4))


def test_asinh_encoder_transform_stabilizes_extreme_scaled_inputs():
    manifest = _synthetic_manifest()
    torch.manual_seed(42)
    plain = MixedInputBetaVAE(manifest=manifest, latent_dim=4)
    stable = MixedInputBetaVAE(
        manifest=manifest, latent_dim=4, encoder_input_transform="asinh"
    )
    stable.load_state_dict(plain.state_dict())
    extreme = torch.full((4, manifest.n_features), 1e8)

    mu_plain, _ = plain.encode(extreme)
    mu_stable, _ = stable.encode(extreme)

    assert torch.isfinite(mu_stable).all()
    assert mu_stable.abs().max() < mu_plain.abs().max() * 1e-4


def test_unknown_encoder_input_transform_rejected():
    with pytest.raises(ValueError, match="encoder_input_transform"):
        MixedInputBetaVAE(
            manifest=_synthetic_manifest(),
            latent_dim=4,
            encoder_input_transform="log1p",
        )


def test_decode_to_39_alias_contract():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "scaler.pkl").exists():
        pytest.skip("scaler.pkl unavailable")
    model = MixedInputBetaVAE(manifest=adapter.feature_manifest(), latent_dim=16)
    model.register_feature_transform(adapter.feature_transform())
    model.eval()
    x_scaled, meta = model.decode_to_39(torch.randn(8, 16), mode="hard")
    assert x_scaled.shape == (8, 39)
    assert set(meta) >= {"continuous_mu", "continuous_mu_raw", "continuous_logvar", "protocol_idx_batch"}


# --------------------------------------------------------------------------- #
# Dynamic dimensionality (no 39 assumption) + trains
# --------------------------------------------------------------------------- #
def test_dynamic_dims_forward_and_learns():
    manifest = _synthetic_manifest(n_extra_real=3)  # 6 features
    n = manifest.n_features
    rng = np.random.default_rng(0)
    x_raw = np.column_stack(
        [
            np.abs(rng.normal(5, 2, size=2000)),          # positive
            rng.uniform(0, 1, size=2000),                 # probability
            rng.uniform(0, 255, size=2000),               # bounded
            rng.normal(0, 1, size=2000),
            rng.normal(3, 1, size=2000),
            rng.normal(-2, 1, size=2000),
        ]
    )
    transform = FeatureTransform(manifest).fit(x_raw, provenance="train")
    x_scaled = torch.tensor(transform.transform(x_raw), dtype=torch.float32)

    torch.manual_seed(0)
    model = MixedInputBetaVAE(manifest=manifest, latent_dim=4, encoder_hidden=(16,), decoder_hidden=(16,))
    assert model.n_features == n
    model.register_feature_transform(transform)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        out = model(x_scaled)
        recon = torch.mean((out["continuous_mu"] - x_scaled) ** 2)
        kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"] ** 2 - out["logvar"].exp())
        loss = recon + 0.01 * kl
        loss.backward()
        opt.step()
        losses.append(float(recon))
    assert losses[-1] < losses[0] * 0.7  # reconstruction meaningfully improved


# --------------------------------------------------------------------------- #
# Legacy path unchanged (A0) + real-data reconstruction smoke
# --------------------------------------------------------------------------- #
def test_legacy_partition_path_still_works():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "scaler.pkl").exists():
        pytest.skip("scaler.pkl unavailable")
    import pickle
    with open(adapter._processed / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    partition = get_partition()
    model = MixedInputBetaVAE(
        partition=partition,
        latent_dim=16,
        use_structured_continuous_decoder=True,
        structured_std_floor=0.01,
    )
    assert model.decoder_kind == "legacy"
    assert model.n_features == 39
    model.register_protocol_references(scaler)
    model.eval()
    out = model.decode(torch.randn(16, 16))
    raw = out["continuous_mu_raw"].detach().numpy()
    bounded = partition["bounded_continuous_idx"]
    assert np.all(raw[:, bounded] >= -1e-6) and np.all(raw[:, bounded] <= 1 + 1e-6)


def test_typed_vae_reconstructs_real_class_data():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    split = adapter.load_split("val")
    cm = adapter.class_mapping()
    ddos = cm.name_to_id["DDoS"]
    idx = np.where(np.asarray(split.y) == ddos)[0][:4096]
    x = torch.tensor(np.asarray(split.x[idx]), dtype=torch.float32)

    torch.manual_seed(0)
    model = MixedInputBetaVAE(manifest=manifest, latent_dim=16)
    model.register_feature_transform(transform)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(60):
        opt.zero_grad()
        out = model(x)
        recon = torch.mean((out["continuous_mu"] - x) ** 2)
        loss = recon + 0.5 * (-0.5 * torch.mean(1 + out["logvar"] - out["mu"] ** 2 - out["logvar"].exp()))
        loss.backward()
        opt.step()
        if step == 0:
            first = float(recon)
        last = float(recon)
    assert last < first  # the typed VAE learns to reconstruct real DDoS flows
