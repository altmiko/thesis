"""The source-conditioned PROTOCOL rule "an empty forward packet stays empty" (PROTO_0080).

A perturbed flow whose source has ``Fwd Packet Length Min == 0`` must keep it at 0: a positive
minimum would mean the empty packet received bytes. Checked on real genuine flows of both
datasets so every other rule is satisfied and only the transition can fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from datasets import get_adapter
from validation import load_validator
from validation.attack_interface import structural_masks

RULE = "PROTO_0080"
FEATURE = "Fwd Packet Length Min"


@pytest.fixture(scope="module", params=[("cicids2017", "cicids2017_distrinet"),
                                        ("cicids2018", "cicids2018_distrinet")])
def genuine(request):
    cli, dataset = request.param
    ad = get_adapter(cli)
    v = load_validator(dataset)
    X_all = np.load(ad._processed / "X_test_pristine.npy", mmap_mode="r")
    X = np.asarray(X_all[np.linspace(0, len(X_all) - 1, 40000).astype(np.int64)], np.float64)
    col = list(ad.feature_manifest().names).index(FEATURE)
    ok = v.validate_batch(X, X).hybrid_valid
    empty = X[ok & (X[:, col] == 0)][:200]
    nonempty = X[ok & (X[:, col] > 0)][:200]
    assert len(empty) and len(nonempty), "fixture needs valid flows with and without empty packets"
    return dataset, v, col, empty, nonempty


def test_rule_is_a_protocol_transition_rule_in_both_datasets(genuine):
    dataset, v, _, _, _ = genuine
    rule = {r.id: r for r in v.rules}[RULE]
    assert rule.source_type == "PROTOCOL" and rule.rule_type == "zero_preserved"
    assert rule.params == {"feature": FEATURE}


def test_filling_an_empty_forward_packet_is_rejected(genuine):
    dataset, v, col, empty, _ = genuine
    adv = empty.copy()
    adv[:, col] = 7.0  # only the minimum moves: the empty packet now carries 7 bytes
    b = v.validate_batch(adv, empty)
    assert b.violation(RULE).all()
    assert not b.protocol_valid.any() and not b.hybrid_valid.any()
    masks = structural_masks(adv, dataset=dataset, source_raw=empty)
    assert not masks["hybrid_valid"].any() and not masks["hard_structural_valid"].any()


def test_empty_packet_left_empty_is_not_a_violation(genuine):
    _, v, _, empty, _ = genuine
    b = v.validate_batch(empty.copy(), empty)
    assert b.eligible[RULE].all() and not b.violation(RULE).any()
    assert b.hybrid_valid.all()


def test_rule_does_not_apply_when_the_source_has_no_empty_forward_packet(genuine):
    _, v, col, _, nonempty = genuine
    adv = nonempty.copy()
    adv[:, col] += 5.0
    b = v.validate_batch(adv, nonempty)
    assert not b.eligible[RULE].any() and not b.violation(RULE).any()


def test_unperturbed_flows_never_violate_the_transition(genuine):
    """Without a source the rows ARE the source (genuine flows): the rule is ineligible."""
    _, v, col, empty, _ = genuine
    adv = empty.copy()
    adv[:, col] = 7.0
    b = v.validate_batch(adv)
    assert not b.eligible[RULE].any()


def test_source_shape_must_match(genuine):
    _, v, _, empty, _ = genuine
    with pytest.raises(ValueError):
        v.validate_batch(empty, empty[:-1])
