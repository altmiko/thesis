"""Deliverable verification: every ablation A0-A6 builds and runs from config.

Run (thesis env):
    PYTHONPATH=src python -m pytest src/experiments/tests/test_ablations.py -q -p no:faulthandler
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from datasets.ciciot2023 import CICIoT2023Adapter
from experiments.ablations import PRESETS, build_ablation


def test_all_presets_build_and_run():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    split = adapter.load_split("val")
    x_scaled = torch.tensor(np.asarray(split.x[:512]), dtype=torch.float32)
    x_raw = transform.inverse_transform(np.asarray(split.x[:512]))

    for name in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
        bundle = build_ablation(name, adapter, layer1_fit_x_raw=x_raw)
        assert bundle.config.name == name
        # base decode works for every preset
        out = bundle.vae.decode(torch.randn(8, 16))
        assert out["continuous_mu"].shape == (8, 39)
        # constraint layers present per flags
        assert (len(bundle.engine.layer1) > 0) == bundle.config.use_layer1
        assert (len(bundle.engine.layer2) > 0) == bundle.config.use_layer2
        # residual presets produce Layer-0-valid adversarial samples
        if bundle.config.use_residual_head:
            x_adv, meta = bundle.generator.generate(x_scaled)
            assert x_adv.shape == x_scaled.shape
            assert bool(bundle.engine.layer0.validate(meta["raw_adv"]).all())


def test_layer1_requires_train_sample():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    # A3 uses Layer 1 -> must refuse to build without a train sample (leakage guard)
    with pytest.raises(ValueError):
        build_ablation("A3", adapter, layer1_fit_x_raw=None)


def test_preset_ladder_monotonic_flags():
    order = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    seen_typed = False
    for name in order:
        c = PRESETS[name]
        if c.decoder_kind == "typed":
            seen_typed = True
        assert seen_typed == (name != "A0")  # only A0 is legacy
    assert PRESETS["A6"].victim_guided_stage_b and not PRESETS["A5"].victim_guided_stage_b


def test_active_layers_override():
    adapter = CICIoT2023Adapter()
    if not (adapter._processed / "X_val.npy").exists():
        pytest.skip("processed arrays unavailable")
    transform = adapter.feature_transform()
    split = adapter.load_split("val")
    x_raw = transform.inverse_transform(np.asarray(split.x[:512]))

    # request only layers 0 and 1 for an otherwise-full preset
    b = build_ablation("A6", adapter, active_layers="01", layer1_fit_x_raw=x_raw)
    assert b.engine.active_layers == {0, 1}
    assert len(b.engine.layer1) == 1 and b.engine.layer2 == []
    assert b.generator.apply_layer0 is True

    # only layer 0
    b0 = build_ablation("A6", adapter, active_layers="0", layer1_fit_x_raw=x_raw)
    assert b0.engine.active_layers == {0}
    assert b0.engine.layer1 == [] and b0.engine.layer2 == []

    # 0,1,2
    b2 = build_ablation("A6", adapter, active_layers="012", layer1_fit_x_raw=x_raw)
    assert b2.engine.active_layers == {0, 1, 2}
    assert len(b2.engine.layer2) > 0

    # layer 0 disabled -> generator skips projection
    bno0 = build_ablation("A6", adapter, active_layers="12", layer1_fit_x_raw=x_raw)
    assert bno0.generator.apply_layer0 is False
