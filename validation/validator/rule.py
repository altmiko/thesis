"""The :class:`Rule` object: a validation rule represented as *data*, not as an
arbitrary Python ``if`` buried in a validator.

Every rule declares:

* ``id``          -- stable identifier (``SCH_0007``, ``MINED_0187``, ``EXT_0042``, ``PROTO_0003``)
* ``source_type`` -- exactly one primary provenance: SCHEMA | MINED | EXTRACTOR | PROTOCOL
* ``rule_type``   -- the template it instantiates (see RULE_TYPES)
* ``params``      -- feature references / constants for that template
* ``hardness``    -- HARD (structural/definitional) | EMPIRICAL (mined) | PROTOCOL
* ``tolerance``   -- absolute + relative closeness for approximate relations
* ``description`` -- plain-English explanation
* ``provenance``  -- origin / external_reference / automatically_mined
* ``evidence``    -- train/validation support + rows tested

A rule is *evaluated* against a raw feature matrix and returns two per-sample
masks: ``satisfied`` and ``eligible``. A sample **violates** a rule iff it is
eligible and not satisfied; an ineligible sample (e.g. a conditional rule whose
antecedent does not hold, or a ratio whose denominator is ~0) is never a
violation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from validation.validator.tolerance import Tolerance, is_close, ABS_FLOOR

SOURCE_TYPES = ("SCHEMA", "MINED", "EXTRACTOR", "PROTOCOL")

# rule_type -> whether it uses a numeric tolerance (approximate) or is exact/logical
APPROXIMATE_TYPES = {
    "equality", "scaled_equality", "square_relation", "sqrt_relation",
    "sum_equality", "difference_equality", "product_equality", "ratio_equality",
    "constant",
}
RULE_TYPES = APPROXIMATE_TYPES | {
    "finite", "integer", "binary", "nonnegative", "nonpositive", "categorical",
    "le", "ge", "monotone_chain", "implication_zero", "implication_pos",
}


@dataclass
class Rule:
    id: str
    source_type: str
    rule_type: str
    params: dict[str, Any]
    features: list[str]
    description: str = ""
    hardness: str = "HARD"
    tolerance: Tolerance = field(default_factory=Tolerance)
    provenance: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"{self.id}: source_type {self.source_type!r} not in {SOURCE_TYPES}")
        if self.rule_type not in RULE_TYPES:
            raise ValueError(f"{self.id}: unknown rule_type {self.rule_type!r}")
        if isinstance(self.tolerance, dict):
            self.tolerance = Tolerance.from_dict(self.tolerance)

    # ---- evaluation -------------------------------------------------------
    def evaluate(self, X: np.ndarray, idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(satisfied, eligible)`` per-sample boolean masks.

        ``X`` is ``(N, F)`` raw features; ``idx`` maps feature name -> column.
        Missing features make the rule inapplicable (all-eligible-False).
        """
        n = X.shape[0]
        all_true = np.ones(n, dtype=bool)
        try:
            g = lambda f: X[:, idx[f]].astype(np.float64)  # noqa: E731
        except KeyError:
            return all_true, np.zeros(n, dtype=bool)

        at, rt = self.tolerance.absolute, self.tolerance.relative
        t = self.rule_type
        p = self.params

        if t == "finite":
            return np.isfinite(g(p["feature"])), all_true
        if t == "integer":
            v = g(p["feature"])
            return is_close(v, np.round(v), at, 0.0), all_true
        if t == "binary":
            v = g(p["feature"])
            return (is_close(v, 0.0, at, 0.0) | is_close(v, 1.0, at, 0.0)), all_true
        if t == "nonnegative":
            return g(p["feature"]) >= -at, all_true
        if t == "nonpositive":
            return g(p["feature"]) <= at, all_true
        if t == "constant":
            return is_close(g(p["feature"]), float(p["value"]), at, rt), all_true
        if t == "categorical":
            v = g(p["feature"])
            dom = np.asarray(p["domain"], np.float64)
            sat = np.zeros(n, dtype=bool)
            for d in dom:
                sat |= is_close(v, d, at, 0.0)
            return sat, all_true
        if t == "le":
            return g(p["lhs"]) <= g(p["rhs"]) + at, all_true
        if t == "ge":
            return g(p["lhs"]) >= g(p["rhs"]) - at, all_true
        if t == "equality":
            return is_close(g(p["lhs"]), g(p["rhs"]), at, rt), all_true
        if t == "scaled_equality":
            return is_close(g(p["lhs"]), float(p["k"]) * g(p["rhs"]), at, rt), all_true
        if t == "square_relation":
            return is_close(g(p["lhs"]), g(p["base"]) ** 2, at, rt), all_true
        if t == "sqrt_relation":
            base = g(p["base"])
            elig = base >= -at
            return is_close(g(p["lhs"]), np.sqrt(np.clip(base, 0.0, None)), at, rt), elig
        if t == "sum_equality":
            exp = np.zeros(n, np.float64)
            for f in p["addends"]:
                exp = exp + g(f)
            return is_close(g(p["lhs"]), exp, at, rt), all_true
        if t == "difference_equality":
            return is_close(g(p["lhs"]), g(p["minuend"]) - g(p["subtrahend"]), at, rt), all_true
        if t == "product_equality":
            exp = np.full(n, float(p.get("k", 1.0)), np.float64)
            for f in p["factors"]:
                exp = exp * g(f)
            return is_close(g(p["lhs"]), exp, at, rt), all_true
        if t == "ratio_equality":
            denom = g(p["denominator"])
            elig = np.abs(denom) > max(at, ABS_FLOOR)
            with np.errstate(divide="ignore", invalid="ignore"):
                exp = np.where(elig, g(p["numerator"]) / denom, 0.0)
            return is_close(g(p["lhs"]), exp, at, rt), elig
        if t == "monotone_chain":
            feats = p["features"]
            sat = all_true.copy()
            for a, b in zip(feats[:-1], feats[1:]):
                sat &= g(a) <= g(b) + at
            return sat, all_true
        if t == "implication_zero":
            ante = is_close(g(p["antecedent"]), 0.0, at, 0.0)
            cons = is_close(g(p["consequent"]), 0.0, at, 0.0)
            return (~ante) | cons, ante  # eligible only where antecedent holds
        if t == "implication_pos":
            ante = g(p["antecedent"]) > at
            cons = g(p["consequent"]) >= -at
            return (~ante) | cons, ante
        raise ValueError(f"{self.id}: unhandled rule_type {t!r}")

    # ---- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "rule_type": self.rule_type,
            "hardness": self.hardness,
            "params": self.params,
            "features": self.features,
            "tolerance": self.tolerance.to_dict(),
            "description": self.description,
            "provenance": self.provenance,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            source_type=d["source_type"],
            rule_type=d["rule_type"],
            hardness=d.get("hardness", "HARD"),
            params=d.get("params", {}),
            features=d.get("features", []),
            tolerance=Tolerance.from_dict(d.get("tolerance")),
            description=d.get("description", ""),
            provenance=d.get("provenance", {}),
            evidence=d.get("evidence", {}),
        )

    def expression(self) -> str:
        """Compact human-readable expression string."""
        p = self.params
        t = self.rule_type
        if t in ("finite", "integer", "nonnegative", "nonpositive", "binary"):
            sym = {"finite": "finite", "integer": "integer", "nonnegative": ">= 0",
                   "nonpositive": "<= 0", "binary": "in {0,1}"}[t]
            return f"{p['feature']} {sym}"
        if t == "constant":
            return f"{p['feature']} == {p['value']}"
        if t == "categorical":
            return f"{p['feature']} in {sorted(p['domain'])}"
        if t == "le":
            return f"{p['lhs']} <= {p['rhs']}"
        if t == "ge":
            return f"{p['lhs']} >= {p['rhs']}"
        if t == "equality":
            return f"{p['lhs']} ~= {p['rhs']}"
        if t == "scaled_equality":
            return f"{p['lhs']} ~= {p['k']} * {p['rhs']}"
        if t == "square_relation":
            return f"{p['lhs']} ~= {p['base']}^2"
        if t == "sqrt_relation":
            return f"{p['lhs']} ~= sqrt({p['base']})"
        if t == "sum_equality":
            return f"{p['lhs']} ~= " + " + ".join(p["addends"])
        if t == "difference_equality":
            return f"{p['lhs']} ~= {p['minuend']} - {p['subtrahend']}"
        if t == "product_equality":
            k = p.get("k", 1.0)
            prefix = "" if k == 1.0 else f"{k} * "
            return f"{p['lhs']} ~= {prefix}" + " * ".join(p["factors"])
        if t == "ratio_equality":
            return f"{p['lhs']} ~= {p['numerator']} / {p['denominator']}"
        if t == "monotone_chain":
            return " <= ".join(p["features"])
        if t == "implication_zero":
            return f"{p['antecedent']} == 0  =>  {p['consequent']} == 0"
        if t == "implication_pos":
            return f"{p['antecedent']} > 0  =>  {p['consequent']} >= 0"
        return f"<{t}>"
