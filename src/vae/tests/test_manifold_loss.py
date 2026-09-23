"""Phase D verification: manifold ELBO (recon/KL/free-bits/constraints) + diagnostics no-crash.

Run (thesis env):
    PYTHONPATH=src python -m pytest src/vae/tests/test_manifold_loss.py -q -p no:faulthandler
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from constraints import ConstraintEngine, RobustTailBound, load_layer2
from datasets.ciciot2023 import CICIoT2023Adapter
from vae.losses import compute_manifold_elbo
from vae.model import MixedInputBetaVAE

_REPO = Path(__file__).resolve().parents[3]


def _engine_and_model():
    adapter = CICIoT2023Adapter()
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    split = adapter.load_split("val")
    cm = adapter.class_mapping()
    idx = np.where(np.asarray(split.y) == cm.name_to_id["DDoS"])[0][:1024]
    x_scaled = torch.tensor(np.asarray(split.x[idx]), dtype=torch.float32)
    x_raw = transform.inverse_transform(np.asarray(split.x[idx]))

    l1 = [RobustTailBound.fit(manifest, x_raw, feature_names=["Header_Length", "Rate", "IAT"], tau=6.0)]
    l2 = load_layer2(_REPO / "old_constraints" / "ciciot2023" / "mined.json", manifest)
    engine = ConstraintEngine(manifest, layer1=l1, layer2=l2)

    torch.manual_seed(0)
    model = MixedInputBetaVAE(manifest=manifest, latent_dim=16)
    model.register_feature_transform(transform)
    return adapter, manifest, transform, engine, model, x_scaled


def test_manifold_elbo_trains_and_components_finite():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    _a, _m, _t, engine, model, x = _engine_and_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(60):
        opt.zero_grad()
        out = model(x)
        elbo = compute_manifold_elbo(
            x, out, engine, beta=0.5,
            free_bits_lambda=0.1,
            constraint_l1_weight=0.1, constraint_l2_weight=0.1,
        )
        elbo["loss"].backward()
        opt.step()
        for k in ("loss", "recon_continuous", "kl", "constraint_l1", "constraint_l2"):
            assert torch.isfinite(elbo[k]).all(), k
        assert float(elbo["kl"]) >= 0.0
        assert float(elbo["constraint_l1"]) >= 0.0 and float(elbo["constraint_l2"]) >= 0.0
        if step == 0:
            first = float(elbo["recon_continuous"])
        last = float(elbo["recon_continuous"])
    assert last < first  # learns to reconstruct


def test_free_bits_floor_on_kl():
    # KL with a free-bits floor must be >= lambda * latent_dim on a (near) collapsed
    # posterior (mu~0, logvar~0 -> per-dim KL ~ 0, clamped up to lambda).
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    _a, _m, _t, engine, model, x = _engine_and_model()
    out = model(x)
    # force a collapsed-ish posterior
    out = dict(out)
    out["mu"] = torch.zeros_like(out["mu"])
    out["logvar"] = torch.zeros_like(out["logvar"])
    lam = 0.1
    elbo = compute_manifold_elbo(x, out, engine, beta=1.0, free_bits_lambda=lam)
    assert float(elbo["kl"]) >= lam * model.latent_dim - 1e-4


def test_pre_projection_penalty_zero_for_typed_output():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    _a, _m, _t, engine, model, x = _engine_and_model()
    out = model(x)
    # typed decoder output is already Layer-0 valid -> pre-projection C0 ~ 0
    pen = engine.layer0.soft_penalty(out["continuous_mu_raw"])
    assert float(pen) == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Diagnostics no longer crash under the continuous schema
# --------------------------------------------------------------------------- #
def test_diagnostics_no_crash_continuous_schema():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    import pickle
    from vae.dataset import PerClassDataset
    from vae.diagnostics import _diag_per_feature_recon, _diag_unconditional_validity
    from vae.schema import get_partition

    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    with open(adapter._processed / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    split = adapter.load_split("val")
    cm = adapter.class_mapping()
    ddos = cm.name_to_id["DDoS"]
    partition = get_partition()

    model = MixedInputBetaVAE(manifest=manifest, latent_dim=16)
    model.register_feature_transform(transform)
    model.eval()

    # small per-class dataset slice
    sel = np.where(np.asarray(split.y) == ddos)[0][:256]
    X = np.asarray(split.x[sel], dtype=np.float32)
    y = np.full(X.shape[0], ddos, dtype=np.int64)
    val_ds = PerClassDataset(X, y, ddos, scaler, partition)

    pfr = _diag_per_feature_recon(model, val_ds, partition, device="cpu")
    assert pfr["protocol_top1_accuracy"] is None  # no protocol head anymore
    assert len(pfr["recon_nll_continuous"]) == 39

    uncond = _diag_unconditional_validity(
        model, val_ds, scaler, partition, device="cpu",
        n_samples=128, validate_batch_fn=None, seed=1234,
    )
    assert uncond["protocol_binary_consistency"] == 1.0  # vacuous, no derived binaries
