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


def _ctl(n, p, delay, shape=0.0):
    return {
        "p": torch.full((n,), float(p), dtype=torch.float64),
        "delay": torch.full((n,), float(delay), dtype=torch.float64),
        "shape": torch.full((n,), float(shape), dtype=torch.float64),
    }


def test_identity_is_realizable(setup):
    model, val, raw, _ = setup
    adv = model.generate(raw, _ctl(raw.shape[0], 0.0, 0.0))
    assert all(int(v.sum()) == 0 for v in val.validate(adv, raw).categories.values())
    assert torch.allclose(adv, raw, atol=1e-9)


@pytest.mark.parametrize(
    "p_val,delay_val,shape_val",
    [(0.0, 0.0, 0.0), (50.0, 100.0, 0.0), (250.0, 1000.0, 0.5), (800.0, 5000.0, 1.0)],
)
def test_padding_and_delay_stay_realizable(setup, p_val, delay_val, shape_val):
    model, val, raw, _ = setup
    adv = model.generate(raw, _ctl(raw.shape[0], p_val, delay_val, shape_val), quantize=True)
    cats = val.validate(adv, raw).categories
    assert all(int(v.sum()) == 0 for v in cats.values()), {k: int(v.sum()) for k, v in cats.items()}


def test_uniform_padding_shift_identities(setup):
    """new_total = old_total + n_fwd*p; new_fwd_mean = old_mean + p; fwd_std invariant."""
    model, _, raw, i = setup
    p = 37.0
    adv = model.generate(raw, _ctl(raw.shape[0], p, 0.0))
    nf = raw[:, i["Total Fwd Packet"]]
    active = model.active_mask(raw, "p")  # semantic: fwd payload AND no empty fwd packet
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
    adv = model.generate(raw, _ctl(raw.shape[0], 40.0, 0.0))
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
    adv = model.generate(raw, _ctl(raw.shape[0], 60.0, 4000.0, 0.5))
    dur_s = (adv[:, i["Flow Duration"]] / 1e6).clamp(min=1e-12)
    nf = raw[:, i["Total Fwd Packet"]]
    for r in ("Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"):
        assert bool((adv[:, i[r]] >= 0).all())
    assert torch.allclose(adv[:, i["Fwd Packets/s"]], nf / dur_s, atol=1e-2, rtol=1e-3)


def test_timing_ordering(setup):
    model, _, raw, i = setup
    adv = model.generate(raw, _ctl(raw.shape[0], 0.0, 5000.0, 1.0))
    assert bool((adv[:, i["Flow Duration"]] > 0).all())
    assert bool((adv[:, i["Fwd IAT Max"]] <= adv[:, i["Fwd IAT Total"]] + 1e-3).all())
    assert bool((adv[:, i["Fwd IAT Total"]] <= adv[:, i["Flow Duration"]] + 1e-3).all())
    assert bool((adv[:, i["Flow IAT Mean"]] <= adv[:, i["Flow IAT Max"]] + 1e-3).all())


def test_single_forward_packet_disables_timing(setup):
    """n_fwd < 2 => delay has no effect (no forward IAT sequence)."""
    model, _, raw, i = setup
    nf = raw[:, i["Total Fwd Packet"]]
    single = nf < 2
    if not bool(single.any()):
        pytest.skip("no single-forward-packet flows in sample")
    adv = model.generate(raw, _ctl(raw.shape[0], 0.0, 5000.0, 1.0))
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
    adv = model.generate(raw, _ctl(raw.shape[0], 28.0, 0.0))
    for f in ("Total Length of Fwd Packet", "Fwd Packet Length Max",
              "Fwd Packet Length Min", "Fwd Packet Length Mean"):
        assert torch.allclose(adv[no_payload, i[f]], raw[no_payload, i[f]], atol=1e-6)


def _paddable_timing_rows(model, raw, i):
    """Flows that satisfy every padding and timing condition (payload, no empty packet, IATs)."""
    rows = ((raw[:, i["Total Length of Fwd Packet"]] > 0)
            & (raw[:, i["Fwd Packet Length Mean"]] > 0)
            & (raw[:, i["Fwd Packet Length Min"]] > 0)
            & (raw[:, i["Total Fwd Packet"]] >= 2) & (raw[:, i["Fwd IAT Total"]] > 0))
    assert bool(rows.any()), "sample needs paddable, timing-capable flows"
    return raw[rows]


def test_empty_forward_packet_disables_padding_but_not_timing(setup):
    """Same flows, only Fwd Packet Length Min set to 0 (one empty forward packet): padding is no
    longer admissible (reason EMPTY_FWD_PACKET), timing capability is unchanged."""
    from attack.realizability.base import EMPTY_FWD_PACKET
    model, _, raw, i = setup
    src = _paddable_timing_rows(model, raw, i)
    empty = src.clone()
    empty[:, i["Fwd Packet Length Min"]] = 0.0
    base, caps = model.infer_capabilities(src), model.infer_capabilities(empty)
    assert bool(base.pad_allowed.all())
    assert not bool(caps.pad_allowed.any())
    assert set(caps.pad_reason) == {EMPTY_FWD_PACKET}
    assert torch.equal(caps.timing_allowed, base.timing_allowed)
    assert bool(caps.timing_allowed.all())


def test_real_flows_with_an_empty_forward_packet_are_not_paddable(setup):
    from attack.realizability.base import EMPTY_FWD_PACKET
    model, _, raw, i = setup
    rows = (raw[:, i["Fwd Packet Length Min"]] == 0) & (raw[:, i["Total Length of Fwd Packet"]] > 0)
    assert bool(rows.any())
    caps = model.infer_capabilities(raw)
    assert not bool(caps.pad_allowed[rows].any())
    assert {caps.pad_reason[k] for k in torch.nonzero(rows).flatten().tolist()} == {EMPTY_FWD_PACKET}


def test_padding_stays_available_without_empty_forward_packets(setup):
    """Payload present and Fwd Packet Length Min > 0: padding admissible, box = numeric cap."""
    from attack.realizability.base import PAD_ALLOWED
    model, _, raw, i = setup
    src = _paddable_timing_rows(model, raw, i)
    caps = model.infer_capabilities(src)
    assert bool(caps.pad_allowed.all()) and set(caps.pad_reason) == {PAD_ALLOWED}
    env = ("Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
           "Total Length of Fwd Packet", "Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Std",
           "Fwd IAT Mean", "Flow Duration")
    cfg = {"p_max": 1460.0, "max_relative_duration_change": 10.0,
           **{f"env_{n}": float(src[:, i[n]].max()) + 1e6 for n in env}}
    b = model.per_flow_bounds(src, cfg, capabilities=caps)
    assert bool((b["p"] >= 1.0).all()) and torch.equal(b["p"], b["p_numeric"])
    adv = model.generate(src, _ctl(src.shape[0], 11.0, 0.0), quantize=True, capabilities=caps)
    assert torch.allclose(adv[:, i["Fwd Packet Length Min"]], src[:, i["Fwd Packet Length Min"]] + 11.0)


def test_empty_forward_packet_is_never_filled_by_any_request(setup):
    """Recompute invariant: whatever (p, delay, shape) is requested, a source with an empty
    forward packet leaves the primitive map with Fwd Packet Length Min == 0 and an unchanged
    forward-length block; bounds and projection pin p to 0."""
    model, _, raw, i = setup
    rows = raw[:, i["Fwd Packet Length Min"]] == 0
    src = raw[rows]
    caps = model.infer_capabilities(src)
    cfg = {"p_max": 1460.0, "max_relative_duration_change": 10.0,
           **{f"env_{n}": float(raw[:, i[n]].max()) + 1e6 for n in (
               "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
               "Total Length of Fwd Packet", "Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Std",
               "Fwd IAT Mean", "Flow Duration")}}
    b = model.per_flow_bounds(src, cfg, capabilities=caps)
    assert bool((b["p"] == 0).all())
    for p, delay, shape in ((1.0, 0.0, 0.0), (900.0, 0.0, 0.0), (500.0, 20000.0, 0.5)):
        ctl = _ctl(src.shape[0], p, delay, shape)
        proj = model.project_controls(src, ctl, b, capabilities=caps)
        assert bool((proj["p"] == 0).all())
        for adv in (model.generate(src, ctl, quantize=True, capabilities=caps),
                    model.generate(src, proj, quantize=True, capabilities=caps)):
            assert bool((adv[:, i["Fwd Packet Length Min"]] == 0).all())
            for f in ("Total Length of Fwd Packet", "Fwd Packet Length Max",
                      "Fwd Packet Length Mean", "Packet Length Min"):
                assert torch.equal(adv[:, i[f]], src[:, i[f]]), f


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
    # inadmissible padding/timing are forced to identity
    assert bool((b["p"][~caps.pad_allowed] == 0).all())
    assert torch.equal(
        b["delay"][~caps.timing_allowed],
        torch.zeros_like(b["delay"][~caps.timing_allowed]),
    )
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
    ctl = _ctl(raw.shape[0], 42.7, 3300.4, 0.7)
    bounds = {
        "p": torch.full((raw.shape[0],), 100.0, dtype=raw.dtype),
        "delay": torch.full((raw.shape[0],), 5000.0, dtype=raw.dtype),
        "shape": torch.ones((raw.shape[0],), dtype=raw.dtype),
    }
    proj = model.project_controls(raw, ctl, bounds)
    assert torch.allclose(proj["p"], torch.round(proj["p"]))
    assert torch.allclose(proj["delay"], torch.round(proj["delay"]))
    adv = model.generate(raw, proj, quantize=True)
    for name in model.integer_features():
        v = adv[:, model.i[name]]
        assert bool((v - torch.round(v)).abs().le(1e-3).all()), name
    assert int(val.validate(adv, raw).categories["discreteness_fail"].sum()) == 0


def test_frozen_features_exactly_preserved(setup):
    model, val, raw, _ = setup
    adv = model.generate(raw, _ctl(raw.shape[0], 300.0, 8000.0, 0.5), quantize=True)
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
    delay = torch.full((n,), 2000.0, dtype=torch.float64, requires_grad=True)
    shape = torch.full((n,), 0.5, dtype=torch.float64, requires_grad=True)
    adv = model.generate(raw, {"p": p, "delay": delay, "shape": shape})
    (adv[:, i["Flow Bytes/s"]].sum() + adv[:, i["Fwd IAT Mean"]].sum()
     + adv[:, i["Fwd IAT Std"]].sum() + adv[:, i["Fwd Packet Length Mean"]].sum()
     + adv[:, i["Packet Length Std"]].sum()).backward()
    assert torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0
    assert torch.isfinite(delay.grad).all() and float(delay.grad.abs().sum()) > 0
    assert torch.isfinite(shape.grad).all() and float(shape.grad.abs().sum()) > 0
