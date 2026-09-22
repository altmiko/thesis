"""Realizability contract for the CICIDS2017 primitive-control attack.

Run (thesis env):
    PYTHONPATH=".;src" python -m pytest src/attack/tests/test_primitive_controls.py -q -p no:faulthandler
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from attack.realizability.base import FeatureRole
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel
from attack.realizability.validator import RealizabilityValidator
from datasets import get_adapter


@pytest.fixture(scope="module")
def setup():
    ad = get_adapter("cicids2017")
    man = ad.feature_manifest()
    model = CICIDS2017PrimitiveModel(man)
    val = RealizabilityValidator(model)
    raw = torch.tensor(
        np.asarray(np.load(ad._processed / "X_test_pristine.npy", mmap_mode="r")[:4000], np.float64)
    )
    return model, val, raw, model.i


def _ctl(n, p, a):
    return {"p": torch.full((n,), float(p), dtype=torch.float64),
            "alpha": torch.full((n,), float(a), dtype=torch.float64)}


def test_identity_is_realizable(setup):
    model, val, raw, _ = setup
    adv = model.generate(raw, _ctl(raw.shape[0], 0.0, 1.0))
    assert all(int(v.sum()) == 0 for v in val.validate(adv, raw).categories.values())
    assert torch.allclose(adv, raw, atol=1e-9)


@pytest.mark.parametrize("p_val,alpha_val", [(0.0, 1.0), (50.0, 2.0), (250.0, 3.0), (800.0, 20.0)])
def test_padding_and_dilation_stay_realizable(setup, p_val, alpha_val):
    model, val, raw, _ = setup
    adv = model.generate(raw, _ctl(raw.shape[0], p_val, alpha_val))
    cats = val.validate(adv, raw).categories
    assert all(int(v.sum()) == 0 for v in cats.values()), {k: int(v.sum()) for k, v in cats.items()}


def test_uniform_padding_shift_identities(setup):
    """new_total = old_total + n_fwd*p; new_fwd_mean = old_mean + p; fwd_std invariant."""
    model, _, raw, i = setup
    p = 37.0
    adv = model.generate(raw, _ctl(raw.shape[0], p, 1.0))
    nf = raw[:, i["Total Fwd Packet"]]
    active = nf >= 1
    tl0, tl1 = raw[:, i["Total Length of Fwd Packet"]], adv[:, i["Total Length of Fwd Packet"]]
    assert torch.allclose(tl1[active], tl0[active] + nf[active] * p, atol=1e-6)
    m0, m1 = raw[:, i["Fwd Packet Length Mean"]], adv[:, i["Fwd Packet Length Mean"]]
    assert torch.allclose(m1[active], m0[active] + p, atol=1e-4)
    assert torch.allclose(adv[:, i["Fwd Packet Length Std"]], raw[:, i["Fwd Packet Length Std"]])
    # Fwd Segment Size Avg tracks the mean exactly.
    assert torch.allclose(adv[:, i["Fwd Segment Size Avg"]], m1, atol=1e-4)


def test_combined_length_stats_recomputed(setup):
    """Previously-frozen combined packet-length stats must move & stay consistent under padding."""
    model, _, raw, i = setup
    adv = model.generate(raw, _ctl(raw.shape[0], 40.0, 1.0))
    # Packet Length Mean must change on at least some flows (was a bug when frozen).
    changed = (adv[:, i["Packet Length Mean"]] - raw[:, i["Packet Length Mean"]]).abs() > 1e-3
    assert bool(changed.any())
    # internal consistency: variance == std^2, mean in [min, max].
    var, std = adv[:, i["Packet Length Variance"]], adv[:, i["Packet Length Std"]]
    assert torch.allclose(var, std ** 2, atol=1.0, rtol=1e-3)
    assert bool((adv[:, i["Packet Length Min"]] <= adv[:, i["Packet Length Mean"]] + 1e-4).all())
    assert bool((adv[:, i["Packet Length Mean"]] <= adv[:, i["Packet Length Max"]] + 1e-4).all())


def test_rates_nonneg_and_correct(setup):
    model, _, raw, i = setup
    adv = model.generate(raw, _ctl(raw.shape[0], 60.0, 4.0))
    dur_s = (adv[:, i["Flow Duration"]] / 1e6).clamp(min=1e-12)
    nf = raw[:, i["Total Fwd Packet"]]
    for r in ("Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"):
        assert bool((adv[:, i[r]] >= 0).all())
    assert torch.allclose(adv[:, i["Fwd Packets/s"]], nf / dur_s, atol=1e-2, rtol=1e-3)


def test_timing_ordering(setup):
    model, _, raw, i = setup
    adv = model.generate(raw, _ctl(raw.shape[0], 0.0, 6.0))
    assert bool((adv[:, i["Flow Duration"]] > 0).all())
    assert bool((adv[:, i["Fwd IAT Max"]] <= adv[:, i["Fwd IAT Total"]] + 1e-3).all())
    assert bool((adv[:, i["Fwd IAT Total"]] <= adv[:, i["Flow Duration"]] + 1e-3).all())
    assert bool((adv[:, i["Flow IAT Mean"]] <= adv[:, i["Flow IAT Max"]] + 1e-3).all())


def test_single_forward_packet_disables_timing(setup):
    """n_fwd < 2 => alpha has no effect (no forward IAT sequence)."""
    model, _, raw, i = setup
    nf = raw[:, i["Total Fwd Packet"]]
    single = nf < 2
    if not bool(single.any()):
        pytest.skip("no single-forward-packet flows in sample")
    adv = model.generate(raw, _ctl(raw.shape[0], 0.0, 50.0))
    for f in ("Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Min", "Flow Duration"):
        assert torch.allclose(adv[single, i[f]], raw[single, i[f]], atol=1e-6)


def test_projection_makes_integer_fields_integral(setup):
    model, val, raw, _ = setup
    ctl = _ctl(raw.shape[0], 42.7, 3.3)
    proj = model.project_controls(raw, ctl)
    assert torch.allclose(proj["p"], torch.round(proj["p"]))  # p is integer bytes
    adv = model.generate(raw, proj, quantize=True)
    for name in model.integer_features():
        v = adv[:, model.i[name]]
        assert bool((v - torch.round(v)).abs().le(1e-3).all()), name
    assert int(val.validate(adv, raw).categories["discreteness_fail"].sum()) == 0


def test_frozen_features_exactly_preserved(setup):
    model, val, raw, _ = setup
    adv = model.generate(raw, _ctl(raw.shape[0], 300.0, 8.0), quantize=True)
    fidx = torch.tensor([model.i[n] for n in val.frozen_names], dtype=torch.long)
    assert torch.equal(adv[:, fidx], raw[:, fidx])
    # Level-C held-constant (Fᶜ) and proven-invariant (I) features are preserved & labelled.
    roles = model.roles()
    assert roles["Fwd Act Data Pkts"][0] is FeatureRole.LEVEL_C
    assert roles["Subflow Fwd Bytes"][0] is FeatureRole.LEVEL_C
    assert roles["Active Mean"][0] is FeatureRole.LEVEL_C  # held, NOT claimed invariant
    assert roles["Fwd Packet Length Std"][0] is FeatureRole.INVARIANT  # proven invariant


def test_gradient_flows_into_primitives(setup):
    model, _, raw, i = setup
    n = raw.shape[0]
    p = torch.full((n,), 40.0, dtype=torch.float64, requires_grad=True)
    alpha = torch.full((n,), 2.0, dtype=torch.float64, requires_grad=True)
    adv = model.generate(raw, {"p": p, "alpha": alpha})
    (adv[:, i["Flow Bytes/s"]].sum() + adv[:, i["Fwd IAT Mean"]].sum()
     + adv[:, i["Fwd Packet Length Mean"]].sum() + adv[:, i["Packet Length Std"]].sum()).backward()
    assert torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0
    assert torch.isfinite(alpha.grad).all() and float(alpha.grad.abs().sum()) > 0
