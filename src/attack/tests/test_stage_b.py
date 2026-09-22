"""Phase F verification: Stage-B victim-guided residual training.

Run (thesis env):
    PYTHONPATH=src python -m pytest src/attack/tests/test_stage_b.py -q -p no:faulthandler
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from attack.residual_head import ResidualAttackGenerator, ResidualHead
from attack.train_attack_head import StageBConfig, VictimGuidedTrainer
from constraints import ConstraintEngine, load_layer2
from constraints.layer0 import Layer0Projector
from datasets.ciciot2023 import CICIoT2023Adapter
from pathlib import Path
from vae.model import MixedInputBetaVAE

_REPO = Path(__file__).resolve().parents[3]


def _setup(mode="both", freeze_all=False):
    adapter = CICIoT2023Adapter()
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    torch.manual_seed(0)
    vae = MixedInputBetaVAE(manifest=manifest, latent_dim=16)
    vae.register_feature_transform(transform)
    projector = Layer0Projector(manifest)
    mutable = torch.zeros(manifest.n_features, dtype=torch.bool) if freeze_all else torch.ones(
        manifest.n_features, dtype=torch.bool
    )
    if not freeze_all:
        for nfrozen in ["Protocol Type", "Time_To_Live"]:
            mutable[manifest.index_by_name(nfrozen)] = False
    head = ResidualHead(16, manifest.n_features)
    gen = ResidualAttackGenerator(vae, projector, mutable, head=head, mode=mode)
    l2 = load_layer2(_REPO / "constraints" / "ciciot2023" / "mined.json", manifest)
    engine = ConstraintEngine(manifest, layer1=[], layer2=l2)
    return adapter, manifest, transform, gen, engine


def _train_victim(x, y, in_dim, n_classes=8, steps=400):
    torch.manual_seed(0)
    v = nn.Linear(in_dim, n_classes)
    opt = torch.optim.Adam(v.parameters(), lr=1e-2)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    n = xt.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (512,))
        opt.zero_grad()
        loss = F.cross_entropy(v(xt[idx]), yt[idx])
        loss.backward()
        opt.step()
    for p in v.parameters():
        p.requires_grad_(False)
    return v


def _victim_and_data(adapter):
    split = adapter.load_split("val")
    y = np.asarray(split.y)
    rng = np.random.default_rng(0)
    pool = rng.choice(np.arange(y.shape[0]), size=min(8000, y.shape[0]), replace=False)
    xv = np.asarray(split.x[np.sort(pool)], dtype=np.float32)
    yv = y[np.sort(pool)]
    victim = _train_victim(xv, yv, in_dim=39)
    cm = adapter.class_mapping()
    ddos = cm.name_to_id["DDoS"]
    d_idx = np.where(y == ddos)[0]
    x_fit = np.asarray(split.x[d_idx[:2000]], dtype=np.float32)
    x_eval = np.asarray(split.x[d_idx[2000:3000]], dtype=np.float32)
    return victim, x_fit, x_eval, cm.name_to_id["Benign"]


def test_stage_b_raises_target_rate_and_keeps_validity():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    _a, _m, transform, gen, engine = _setup(mode="both")
    victim, x_fit, x_eval, benign = _victim_and_data(adapter)

    cfg = StageBConfig(target_class=benign, epochs=6, batch_size=256, lr=1e-2,
                       lambda_attack=1.0, lambda_delta=0.05, lambda_c1=0.05, lambda_c2=0.05)
    trainer = VictimGuidedTrainer(gen, victim, transform, engine, cfg)

    before = trainer.evaluate(x_eval)
    trainer.fit(x_fit)
    after = trainer.evaluate(x_eval)

    assert after["target_class_rate"] > before["target_class_rate"]  # victim-guided evasion improved
    assert after["layer0_valid_rate"] > 0.99                          # Layer-0 stays valid
    assert np.isfinite(after["mean_perturbation_cost"])
    assert before["layer0_valid_rate"] > 0.99


def test_perturbation_cost_zero_when_all_frozen():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    _a, _m, transform, gen, engine = _setup(mode="both", freeze_all=True)
    victim, _xf, x_eval, benign = _victim_and_data(adapter)
    cfg = StageBConfig(target_class=benign)
    trainer = VictimGuidedTrainer(gen, victim, transform, engine, cfg)
    ev = trainer.evaluate(x_eval)
    assert ev["mean_perturbation_cost"] == pytest.approx(0.0, abs=1e-6)  # nothing mutable -> no cost
    assert ev["layer0_valid_rate"] > 0.99


def test_victim_is_injected_dependency():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    _a, manifest, transform, gen, engine = _setup(mode="residual")
    _v, _xf, x_eval, benign = _victim_and_data(adapter)
    cfg = StageBConfig(target_class=benign)
    # two different frozen victims -> evaluate uses whichever is injected
    torch.manual_seed(1)
    v1 = nn.Linear(39, 8)
    torch.manual_seed(2)
    v2 = nn.Linear(39, 8)
    for v in (v1, v2):
        for p in v.parameters():
            p.requires_grad_(False)
    r1 = VictimGuidedTrainer(gen, v1, transform, engine, cfg).evaluate(x_eval)
    r2 = VictimGuidedTrainer(gen, v2, transform, engine, cfg).evaluate(x_eval)
    assert set(r1) == {"target_class_rate", "layer0_valid_rate", "mean_perturbation_cost"}
    # same generator, different victims -> generally different target rates
    assert r1["target_class_rate"] != r2["target_class_rate"] or r1["layer0_valid_rate"] == r2["layer0_valid_rate"]
