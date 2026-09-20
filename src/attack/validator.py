"""Structural validator for CICIoT2023 window-aggregated CSV features.

Structural checks are separate from empirical support/density and mutability.
No percentile, training min/max, integer protocol, or Bernoulli constraint is a
structural validity rule for this representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.preprocessing.schema import BOUNDED_AGGREGATED_FEATURES, FEATURE_METADATA

# Historical packet-protocol values retained for artifact readers, not validation.
VALID_PROTOCOLS = {0, 1, 2, 6, 17, 47}
FLOAT_TOL = 0.01
VAR_REL_TOL = 0.05
VAR_ABS_TOL = FLOAT_TOL


@dataclass
class ValidationResult:
    n_samples: int
    violations_per_rule: Dict[str, np.ndarray]

    @property
    def overall_valid(self) -> np.ndarray:
        if not self.violations_per_rule:
            return np.ones(self.n_samples, dtype=bool)
        return ~np.stack(list(self.violations_per_rule.values()), axis=1).any(axis=1)

    @property
    def validity_rate(self) -> float:
        return float(self.overall_valid.mean())

    def per_rule_violation_rate(self) -> Dict[str, float]:
        return {key: float(value.mean()) for key, value in self.violations_per_rule.items()}

    def summary(self) -> str:
        lines = [f"Validity rate: {self.validity_rate:.4%} ({int(self.overall_valid.sum())}/{self.n_samples})"]
        for key, rate in sorted(self.per_rule_violation_rate().items(), key=lambda item: -item[1]):
            if rate > 0:
                lines.append(f"  {key}: {rate:.4%}")
        return "\n".join(lines)


def validate_batch(X: np.ndarray, feature_names: List[str]) -> ValidationResult:
    frame = pd.DataFrame(X, columns=feature_names)
    violations: dict[str, np.ndarray] = {}

    for feature in feature_names:
        spec = FEATURE_METADATA.get(feature)
        if spec is None:
            continue
        values = frame[feature].to_numpy()
        if spec.expected_min is not None:
            violations[f"R_structural_min_{feature}"] = values < spec.expected_min - FLOAT_TOL
        if spec.expected_max is not None:
            violations[f"R_structural_max_{feature}"] = values > spec.expected_max + FLOAT_TOL

    if {"Min", "Max"}.issubset(frame.columns):
        violations["R_min_leq_max"] = frame["Min"].to_numpy() > frame["Max"].to_numpy() + FLOAT_TOL
    if {"Min", "AVG", "Max"}.issubset(frame.columns):
        average = frame["AVG"].to_numpy()
        violations["R_avg_in_range"] = (average < frame["Min"].to_numpy() - FLOAT_TOL) | (average > frame["Max"].to_numpy() + FLOAT_TOL)
    if {"Std", "Variance"}.issubset(frame.columns):
        expected = frame["Std"].to_numpy().square()
        absolute = np.abs(frame["Variance"].to_numpy() - expected)
        relative = absolute / (np.abs(expected) + 1e-8)
        violations["R_var_eq_std_sq"] = (absolute > VAR_ABS_TOL) & (relative > VAR_REL_TOL)

    return ValidationResult(len(frame), violations)
