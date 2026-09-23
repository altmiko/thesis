"""Engine/rule/result unit tests (self-contained; no dataset needed)."""
from __future__ import annotations

import numpy as np

from validation.validator.engine import Validator
from validation.validator.rule import Rule
from validation.validator.tolerance import Tolerance, is_close


def _toy_validator():
    feats = ["a", "b", "c"]
    rules = [
        Rule("SCH_1", "SCHEMA", "integer", {"feature": "a"}, ["a"]),
        Rule("PROTO_1", "PROTOCOL", "nonnegative", {"feature": "a"}, ["a"]),
        Rule("EXT_1", "EXTRACTOR", "equality", {"lhs": "c", "rhs": "b"}, ["c", "b"],
             tolerance=Tolerance(1e-6, 1e-6)),
        Rule("MINED_1", "MINED", "monotone_chain", {"features": ["a", "b", "c"]}, ["a", "b", "c"],
             tolerance=Tolerance(1e-6, 0.0)),
    ]
    return Validator("toy", feats, rules)


def test_all_valid_row():
    v = _toy_validator()
    r = v.validate([1.0, 2.0, 2.0])
    assert r.schema_valid and r.protocol_valid and r.extractor_valid and r.mined_valid
    assert r.hard_structural_valid and r.hybrid_valid


def test_schema_integer_failure_is_structural():
    v = _toy_validator()
    r = v.validate([1.5, 2.0, 2.0])
    assert not r.schema_valid
    assert not r.hard_structural_valid and not r.hybrid_valid
    assert any(f["id"] == "SCH_1" for f in r.failed_rules)


def test_protocol_nonnegative_failure():
    v = _toy_validator()
    r = v.validate([-1.0, 2.0, 2.0])
    assert not r.protocol_valid and not r.hard_structural_valid


def test_mined_failure_keeps_hard_valid_but_not_hybrid():
    # chain a<=b<=c violated (3<=2), but schema/extractor/protocol still hold.
    v = _toy_validator()
    r = v.validate([3.0, 2.0, 2.0])
    assert r.hard_structural_valid is True          # SCHEMA+EXTRACTOR+PROTOCOL pass
    assert r.mined_valid is False
    assert r.hybrid_valid is False                  # separation of hard vs hybrid
    assert any(f["id"] == "MINED_1" for f in r.failed_rules)


def test_extractor_failure_reports_observed_expected():
    v = _toy_validator()
    r = v.validate([1.0, 2.0, 5.0])                 # c != b
    fr = [f for f in r.failed_rules if f["id"] == "EXT_1"][0]
    assert fr["observed"] == 5.0 and fr["expected"] == 2.0
    assert fr["absolute_error"] == 3.0


def test_plausibility_never_affects_structural():
    v = _toy_validator()                            # no plausibility profile
    r = v.validate([1.0, 2.0, 2.0])
    assert r.in_distribution is True               # default all-pass
    assert r.structurally_valid is True


def test_result_exports():
    v = _toy_validator()
    r = v.validate([3.0, 2.0, 2.0])
    d = r.to_dict()
    assert set(["hard_structural_valid", "hybrid_valid", "failed_rules"]) <= set(d)
    assert "FAILED RULES" in r.to_markdown()
    assert isinstance(r.to_json(), str)


def test_is_close_semantics():
    assert bool(is_close(1.0, 1.0 + 1e-9, 1e-6, 0.0))
    assert not bool(is_close(1.0, 2.0, 1e-6, 1e-6))


def test_ratio_eligibility_zero_denominator_is_not_violation():
    feats = ["x", "y", "z"]
    rules = [Rule("MINED_R", "MINED", "ratio_equality",
                  {"lhs": "x", "numerator": "y", "denominator": "z"}, ["x", "y", "z"],
                  tolerance=Tolerance(1e-6, 1e-6))]
    v = Validator("t", feats, rules)
    # denominator 0 -> rule ineligible -> not a violation
    r = v.validate([999.0, 5.0, 0.0])
    assert r.mined_valid is True
