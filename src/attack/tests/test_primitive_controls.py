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
    active = model.active_mask(raw, "p")  # semantic: fwd packets AND fwd payload present
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


def test_no_forward_payload_disables_padding(setup):
    """A flow with no forward payload (Total Length of Fwd Packet == 0) is not paddable:
    pad_allowed=False, p_hi=0, and p is forced to identity even if requested (Recon fix)."""
    model, _, raw, i = setup
    no_payload = raw[:, i["Total Length of Fwd Packet"]] <= 0
    if not bool(no_payload.any()):
        pytest.skip("no zero-forward-payload flows in sample")
    caps = model.infer_capabilities(raw)
    assert not bool(caps.pad_allowed[no_payload].any())  # never admissible without payload
    # requesting a large p on these flows must not change the forward length block
    adv = model.generate(raw, _ctl(raw.shape[0], 28.0, 1.0))
    for f in ("Total Length of Fwd Packet", "Fwd Packet Length Max",
              "Fwd Packet Length Min", "Fwd Packet Length Mean"):
        assert torch.allclose(adv[no_payload, i[f]], raw[no_payload, i[f]], atol=1e-6)


def test_bounds_gated_by_capabilities(setup):
    """per_flow_bounds forces the identity cap where a primitive is inadmissible, while the
    numeric (envelope) cap is retained separately for provenance."""
    model, _, raw, i = setup
    env_feats = ("Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
                 "Total Length of Fwd Packet", "Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Std",
                 "Fwd IAT Mean", "Flow Duration")
    cfg = {"p_max": 1460.0, "max_relative_duration_change": 10.0,
           **{f"env_{n}": float(raw[:, i[n]].max()) + 1e6 for n in env_feats}}
    caps = model.infer_capabilities(raw)
    b = model.per_flow_bounds(raw, cfg, capabilities=caps)
    # inadmissible padding => p_hi == 0; inadmissible timing => alpha_hi == 1 (identity)
    assert bool((b["p"][~caps.pad_allowed] == 0).all())
    assert torch.allclose(b["alpha"][~caps.timing_allowed],
                          torch.ones_like(b["alpha"][~caps.timing_allowed]))
    # semantic cap never exceeds the numeric cap; equals it where admissible
    assert bool((b["p"] <= b["p_numeric"] + 1e-6).all())
    assert torch.allclose(b["p"][caps.pad_allowed], b["p_numeric"][caps.pad_allowed], atol=1e-6)


def test_capability_reasons_are_consistent(setup):
    """Every disabled primitive carries a machine-readable reason; enabled ones say *_ALLOWED."""
    from attack.realizability.base import (PAD_ALLOWED, TIMING_ALLOWED,
                                           NO_FORWARD_PAYLOAD, SINGLE_FWD_PACKET,
                                           ZERO_TIMING_HEADROOM)
    model, _, raw, _ = setup
    caps = model.infer_capabilities(raw)
    pad = caps.pad_allowed.cpu().numpy()
    tim = caps.timing_allowed.cpu().numpy()
    for k, ok in enumerate(pad):
        assert (caps.pad_reason[k] == PAD_ALLOWED) == bool(ok)
        if not ok:
            assert caps.pad_reason[k] == NO_FORWARD_PAYLOAD or caps.pad_reason[k] != PAD_ALLOWED
    for k, ok in enumerate(tim):
        assert (caps.timing_reason[k] == TIMING_ALLOWED) == bool(ok)
        if not ok:
            assert caps.timing_reason[k] in (SINGLE_FWD_PACKET, ZERO_TIMING_HEADROOM)


def test_projection_makes_integer_fields_integral(setup):
    model, val, raw, _ = setup
    ctl = _ctl(raw.shape[0], 42.7, 3.3)
    bounds = {
        "p": torch.full((raw.shape[0],), 100.0, dtype=raw.dtype),
        "alpha": torch.full((raw.shape[0],), 5.0, dtype=raw.dtype),
    }
    proj = model.project_controls(raw, ctl, bounds)
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
