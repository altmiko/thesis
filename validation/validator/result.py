"""Structured validation results.

The validator never returns a bare ``True``/``False``. A :class:`BatchResult`
holds per-rule satisfied/eligible masks for a whole matrix; a :class:`SampleResult`
is a per-sample view that can explain exactly which rules failed and why.

Validity concepts are kept explicitly separate:

* ``schema_valid`` / ``mined_valid`` / ``extractor_valid`` / ``protocol_valid``
  -- per-provenance conjunctions.
* ``hard_structural_valid = SCHEMA & EXTRACTOR & PROTOCOL``  (definitional only)
* ``hybrid_valid          = hard_structural & MINED``        (+ empirical invariants)
* ``structurally_valid``  -- configurable alias (default == hybrid_valid).
* ``in_distribution`` / ``plausibility_score`` -- distributional, NEVER folded into
  structural validity.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from validation.validator.rule import Rule
from validation.validator.tolerance import abs_error, rel_error

_ALL = ("SCHEMA", "MINED", "EXTRACTOR", "PROTOCOL")


@dataclass
class BatchResult:
    rules: list[Rule]
    satisfied: dict[str, np.ndarray]   # rule.id -> (N,) bool
    eligible: dict[str, np.ndarray]    # rule.id -> (N,) bool
    n: int
    include_mined_in_structural: bool = True
    plausibility: dict[str, Any] | None = None  # {"in_distribution":(N,),"score":(N,),"violations":[..]}

    # ---- per-rule ---------------------------------------------------------
    def violation(self, rid: str) -> np.ndarray:
        return self.eligible[rid] & ~self.satisfied[rid]

    def _group_valid(self, source_type: str) -> np.ndarray:
        out = np.ones(self.n, dtype=bool)
        for r in self.rules:
            if r.source_type == source_type:
                out &= ~self.violation(r.id)
        return out

    @property
    def schema_valid(self) -> np.ndarray:
        return self._group_valid("SCHEMA")

    @property
    def mined_valid(self) -> np.ndarray:
        return self._group_valid("MINED")

    @property
    def extractor_valid(self) -> np.ndarray:
        return self._group_valid("EXTRACTOR")

    @property
    def protocol_valid(self) -> np.ndarray:
        return self._group_valid("PROTOCOL")

    @property
    def hard_structural_valid(self) -> np.ndarray:
        return self.schema_valid & self.extractor_valid & self.protocol_valid

    @property
    def hybrid_valid(self) -> np.ndarray:
        return self.hard_structural_valid & self.mined_valid

    @property
    def structurally_valid(self) -> np.ndarray:
        return self.hybrid_valid if self.include_mined_in_structural else self.hard_structural_valid

    @property
    def in_distribution(self) -> np.ndarray:
        if self.plausibility is None:
            return np.ones(self.n, dtype=bool)
        return self.plausibility["in_distribution"]

    @property
    def plausibility_score(self) -> np.ndarray:
        if self.plausibility is None:
            return np.ones(self.n, dtype=float)
        return self.plausibility["score"]

    # ---- aggregate metrics ------------------------------------------------
    def rates(self) -> dict[str, float]:
        return {
            "schema_valid": float(self.schema_valid.mean()),
            "mined_valid": float(self.mined_valid.mean()),
            "extractor_valid": float(self.extractor_valid.mean()),
            "protocol_valid": float(self.protocol_valid.mean()),
            "hard_structural_valid": float(self.hard_structural_valid.mean()),
            "hybrid_valid": float(self.hybrid_valid.mean()),
            "in_distribution": float(self.in_distribution.mean()),
        }

    def per_rule_violation_rate(self) -> dict[str, float]:
        out = {}
        for r in self.rules:
            elig = self.eligible[r.id]
            n_elig = int(elig.sum())
            out[r.id] = float(self.violation(r.id).sum() / n_elig) if n_elig else 0.0
        return out

    def rules_rejecting_any(self) -> dict[str, int]:
        """rule.id -> number of samples it (eligibly) rejects, only for rules that reject > 0."""
        out = {}
        for r in self.rules:
            c = int(self.violation(r.id).sum())
            if c:
                out[r.id] = c
        return out

    # ---- per-sample view --------------------------------------------------
    def result(self, i: int, X: np.ndarray | None = None, idx: dict[str, int] | None = None) -> "SampleResult":
        return SampleResult(self, i, X, idx)


class SampleResult:
    """Per-sample validity view with explanations."""

    def __init__(self, batch: BatchResult, i: int, X: np.ndarray | None = None,
                 idx: dict[str, int] | None = None):
        self._b = batch
        self._i = i
        self._X = X
        self._idx = idx

    def _flag(self, name: str) -> bool:
        return bool(getattr(self._b, name)[self._i])

    hard_structural_valid = property(lambda s: s._flag("hard_structural_valid"))
    hybrid_valid = property(lambda s: s._flag("hybrid_valid"))
    structurally_valid = property(lambda s: s._flag("structurally_valid"))
    schema_valid = property(lambda s: s._flag("schema_valid"))
    mined_valid = property(lambda s: s._flag("mined_valid"))
    extractor_valid = property(lambda s: s._flag("extractor_valid"))
    protocol_valid = property(lambda s: s._flag("protocol_valid"))
    in_distribution = property(lambda s: s._flag("in_distribution"))

    @property
    def plausibility_score(self) -> float:
        return float(self._b.plausibility_score[self._i])

    @property
    def failed_rules(self) -> list[dict]:
        out = []
        for r in self._b.rules:
            if self._b.violation(r.id)[self._i]:
                out.append(self._explain(r))
        return out

    @property
    def passed_rules(self) -> list[str]:
        return [r.id for r in self._b.rules
                if self._b.eligible[r.id][self._i] and self._b.satisfied[r.id][self._i]]

    def counts_by_source(self) -> dict[str, dict[str, int]]:
        out = {s: {"passed": 0, "failed": 0} for s in _ALL}
        for r in self._b.rules:
            if not self._b.eligible[r.id][self._i]:
                continue
            key = "failed" if self._b.violation(r.id)[self._i] else "passed"
            out[r.source_type][key] += 1
        return out

    def _explain(self, r: Rule) -> dict:
        d = {
            "id": r.id,
            "source_type": r.source_type,
            "rule_type": r.rule_type,
            "expression": r.expression(),
            "description": r.description,
            "features": {},
            "tolerance": r.tolerance.to_dict(),
        }
        if self._X is not None and self._idx is not None:
            row = self._X[self._i]
            for f in r.features:
                if f in self._idx:
                    d["features"][f] = float(row[self._idx[f]])
            obs, exp = self._observed_expected(r, row)
            if obs is not None:
                d["observed"] = obs
                d["expected"] = exp
                d["absolute_error"] = float(abs(obs - exp))
                d["relative_error"] = float(abs(obs - exp) / (abs(exp) + 1e-6))
        return d

    def _observed_expected(self, r: Rule, row: np.ndarray):
        idx = self._idx
        p = r.params
        g = lambda f: float(row[idx[f]])  # noqa: E731
        try:
            t = r.rule_type
            if t in ("equality", "scaled_equality", "square_relation", "sqrt_relation",
                     "sum_equality", "difference_equality", "product_equality", "ratio_equality"):
                obs = g(p["lhs"])
                if t == "equality":
                    exp = g(p["rhs"])
                elif t == "scaled_equality":
                    exp = float(p["k"]) * g(p["rhs"])
                elif t == "square_relation":
                    exp = g(p["base"]) ** 2
                elif t == "sqrt_relation":
                    exp = g(p["base"]) ** 0.5
                elif t == "sum_equality":
                    exp = sum(g(f) for f in p["addends"])
                elif t == "difference_equality":
                    exp = g(p["minuend"]) - g(p["subtrahend"])
                elif t == "product_equality":
                    exp = float(p.get("k", 1.0))
                    for f in p["factors"]:
                        exp *= g(f)
                else:  # ratio_equality
                    exp = g(p["numerator"]) / g(p["denominator"])
                return obs, exp
        except (KeyError, ZeroDivisionError):
            return None, None
        return None, None

    def to_dict(self) -> dict:
        return {
            "hard_structural_valid": self.hard_structural_valid,
            "hybrid_valid": self.hybrid_valid,
            "structurally_valid": self.structurally_valid,
            "schema_valid": self.schema_valid,
            "mined_valid": self.mined_valid,
            "extractor_valid": self.extractor_valid,
            "protocol_valid": self.protocol_valid,
            "in_distribution": self.in_distribution,
            "plausibility_score": self.plausibility_score,
            "counts_by_source": self.counts_by_source(),
            "failed_rules": self.failed_rules,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), indent=2, **kw)

    def to_markdown(self) -> str:
        from validation.validator.report import render_sample_result
        return render_sample_result(self)
