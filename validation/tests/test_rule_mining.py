"""Mining unit tests (self-contained)."""
from __future__ import annotations

import numpy as np

from validation.mining.candidate_templates import generate_candidates
from validation.mining.evaluate_candidates import residual_tightness, derive_tolerance
from validation.mining.mine_unary import classify_feature
from validation.mining.prune_rules import prune
from validation.validator.rule import Rule
from validation.validator.tolerance import suggest_tolerance, ABS_CAP, REL_CAP


def _cat(reg):
    return lambda f: reg.get(f, "x")


def test_candidate_grammar_detects_monotone_chain():
    feats = ["G Min", "G Mean", "G Max", "other"]
    reg = {f: "grp" for f in feats}
    fams = generate_candidates(feats, _cat(reg))
    chains = [r for r in fams["arithmetic"] if r.rule_type == "monotone_chain"]
    assert any(r.params["features"] == ["G Min", "G Mean", "G Max"] for r in chains)


def test_no_bare_le_template_generated():
    feats = ["a", "b", "c"]
    reg = {f: "grp" for f in feats}
    fams = generate_candidates(feats, _cat(reg))
    all_rules = [r for fam in fams.values() for r in fam]
    assert not any(r.rule_type == "le" for r in all_rules)


def test_residual_tightness_separates_true_and_false_relations():
    n = 5000
    rng = np.random.default_rng(0)
    b = rng.uniform(1, 100, n)
    X = np.stack([b, b, b + rng.uniform(50, 100, n)], axis=1)  # col0==col1 exact; col2 far
    idx = {"a": 0, "b": 1, "c": 2}
    true_rule = Rule("t", "MINED", "equality", {"lhs": "a", "rhs": "b"}, ["a", "b"])
    false_rule = Rule("f", "MINED", "equality", {"lhs": "a", "rhs": "c"}, ["a", "c"])
    assert residual_tightness(true_rule, X, idx)[0] < 1e-6
    assert residual_tightness(false_rule, X, idx)[0] > 1e-2


def test_suggest_tolerance_floors_and_caps():
    tol = suggest_tolerance(np.zeros(100), np.ones(100))
    assert tol.absolute >= 1e-6 and tol.relative >= 1e-6
    tol2 = suggest_tolerance(np.full(100, 1e12), np.ones(100))
    assert tol2.absolute <= ABS_CAP and tol2.relative <= REL_CAP


def test_prune_extractor_domination():
    mined = [Rule("m", "MINED", "equality", {"lhs": "a", "rhs": "b"}, ["a", "b"])]
    ext = [Rule("EXT_1", "EXTRACTOR", "equality", {"lhs": "a", "rhs": "b"}, ["a", "b"])]
    kept, pruned = prune(mined, ext)
    assert kept == [] and pruned and "EXTRACTOR" in pruned[0][1]


def test_prune_duplicate_expression():
    r1 = Rule("m1", "MINED", "square_relation", {"lhs": "v", "base": "s"}, ["v", "s"])
    r2 = Rule("m2", "MINED", "square_relation", {"lhs": "v", "base": "s"}, ["v", "s"])
    kept, pruned = prune([r1, r2], [])
    assert len(kept) == 1 and len(pruned) == 1


def test_classify_feature_types():
    assert classify_feature(np.array([0, 1, 0, 1.0]))["value_type"] == "binary"
    assert classify_feature(np.full(100, 5.0))["value_type"] == "constant"
    assert classify_feature(np.arange(100.0))["value_type"] == "integer"
    assert classify_feature(np.linspace(0, 1, 100))["value_type"] == "numeric"
