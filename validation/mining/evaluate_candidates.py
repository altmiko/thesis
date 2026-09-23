"""Support metrics + train-derived tolerance for candidate rules.

For every candidate we compute (§7): eligible_rows, satisfied_rows,
violated_rows, support, and (for approximate relations) the absolute/relative
error distribution. For implications we additionally compute antecedent_count /
antecedent_rate / conditional_support.

Tolerances for approximate mined rules are derived from the TRAIN residual
distribution (high percentile, floored + capped) via
``tolerance.suggest_tolerance``; logical rules (<=, chains, implications) use a
small fixed numerical floor.
"""
from __future__ import annotations

import numpy as np

from validation.validator.rule import Rule
from validation.validator.tolerance import Tolerance, suggest_tolerance, ABS_FLOOR

_APPROX = {"equality", "scaled_equality", "square_relation", "sqrt_relation",
           "sum_equality", "difference_equality", "product_equality", "ratio_equality"}


def observed_expected(rule: Rule, X: np.ndarray, idx: dict[str, int]):
    """Return (observed, expected, eligible) for an approximate rule, else None."""
    t = rule.rule_type
    if t not in _APPROX:
        return None
    p = rule.params
    g = lambda f: X[:, idx[f]].astype(np.float64)  # noqa: E731
    n = X.shape[0]
    elig = np.ones(n, dtype=bool)
    obs = g(p["lhs"])
    if t == "equality":
        exp = g(p["rhs"])
    elif t == "scaled_equality":
        exp = float(p["k"]) * g(p["rhs"])
    elif t == "square_relation":
        exp = g(p["base"]) ** 2
    elif t == "sqrt_relation":
        base = g(p["base"]); elig = base >= 0; exp = np.sqrt(np.clip(base, 0, None))
    elif t == "sum_equality":
        exp = np.zeros(n); [exp := exp + g(f) for f in p["addends"]]
    elif t == "difference_equality":
        exp = g(p["minuend"]) - g(p["subtrahend"])
    elif t == "product_equality":
        exp = np.full(n, float(p.get("k", 1.0))); [exp := exp * g(f) for f in p["factors"]]
    else:  # ratio_equality
        denom = g(p["denominator"]); elig = np.abs(denom) > ABS_FLOOR
        exp = np.where(elig, g(p["numerator"]) / np.where(elig, denom, 1.0), 0.0)
    return obs, exp, elig


def derive_tolerance(rule: Rule, X: np.ndarray, idx: dict[str, int],
                     percentile: float = 99.9) -> Tolerance:
    """Train-residual-derived tolerance for approximate rules; fixed floor otherwise."""
    oe = observed_expected(rule, X, idx)
    if oe is None:
        return Tolerance(ABS_FLOOR, 0.0)
    obs, exp, elig = oe
    if not elig.any():
        return Tolerance(ABS_FLOOR, 0.0)
    resid = np.abs(obs[elig] - exp[elig])
    return suggest_tolerance(resid, exp[elig], percentile=percentile)


def evaluate_rule(rule: Rule, X: np.ndarray, idx: dict[str, int]) -> dict:
    """Compute support metrics for ``rule`` on matrix ``X`` at its current tolerance."""
    sat, elig = rule.evaluate(X, idx)
    n = X.shape[0]
    n_elig = int(elig.sum())
    n_sat = int((sat & elig).sum())
    n_viol = int((~sat & elig).sum())
    m = {
        "rows_tested": n,
        "eligible_rows": n_elig,
        "satisfied_rows": n_sat,
        "violated_rows": n_viol,
        "support": (n_sat / n_elig) if n_elig else 0.0,
    }
    oe = observed_expected(rule, X, idx)
    if oe is not None:
        obs, exp, e = oe
        if e.any():
            ae = np.abs(obs[e] - exp[e])
            re = ae / (np.abs(exp[e]) + ABS_FLOOR)
            m["abs_error"] = {"p50": float(np.percentile(ae, 50)),
                              "p99": float(np.percentile(ae, 99)),
                              "max": float(ae.max())}
            m["rel_error"] = {"p50": float(np.percentile(re, 50)),
                              "p99": float(np.percentile(re, 99)),
                              "max": float(re.max())}
    if rule.rule_type in ("implication_zero", "implication_pos"):
        m["antecedent_count"] = n_elig
        m["antecedent_rate"] = (n_elig / n) if n else 0.0
        m["conditional_support"] = m["support"]
    return m


def residual_tightness(rule: Rule, X: np.ndarray, idx: dict[str, int]):
    """Scale-free tightness of an approximate relation.

    Returns ``(p999_relative, p999_absolute, eligible_rows)`` where the relative
    residual is ``|obs - exp| / (|obs| + |exp| + floor)`` in [0, 1]. A genuine
    identity has p999_relative ~ 1e-6..1e-4; a spurious relation has it ~ O(1).
    This is used as the ACCEPTANCE gate for approximate mined rules so that a
    self-derived tolerance can never make a false relation vacuously pass.
    Returns ``None`` for non-approximate (logical) rule types.
    """
    oe = observed_expected(rule, X, idx)
    if oe is None:
        return None
    obs, exp, e = oe
    if not e.any():
        return None
    o, x = obs[e], exp[e]
    ae = np.abs(o - x)
    rr = ae / (np.abs(o) + np.abs(x) + ABS_FLOOR)
    return float(np.percentile(rr, 99.9)), float(np.percentile(ae, 99.9)), int(e.sum())
