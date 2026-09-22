"""Independent PAVE-style validity checks in original feature units.

The validator never projects or repairs a sample.  It combines hard domains from
an existing feature manifest with train-only min/max fallbacks, then checks the
untouched raw sample for finite values, ranges, and discrete types.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class FeatureConstraint:
    """One independently auditable feature-domain rule."""

    name: str
    kind: str
    lower: float | None = None
    upper: float | None = None
    integer: bool = False
    binary: bool = False
    source: str = ""
    uncertain: bool = False


# Exact normalized names only.  These are fallbacks for callers without a local
# manifest; repository adapters take precedence.  Deliberately no substring tests.
_EXACT_UNIVERSAL: dict[str, dict[str, Any]] = {
    "ttl": {"kind": "protocol_integer", "lower": 0.0, "upper": 255.0, "integer": True},
    "time_to_live": {"kind": "protocol_integer", "lower": 0.0, "upper": 255.0, "integer": True},
    "src_port": {"kind": "protocol_integer", "lower": 0.0, "upper": 65535.0, "integer": True},
    "source_port": {"kind": "protocol_integer", "lower": 0.0, "upper": 65535.0, "integer": True},
    "dst_port": {"kind": "protocol_integer", "lower": 0.0, "upper": 65535.0, "integer": True},
    "destination_port": {"kind": "protocol_integer", "lower": 0.0, "upper": 65535.0, "integer": True},
    "protocol": {"kind": "protocol_integer", "lower": 0.0, "upper": 255.0, "integer": True},
    "binary_flag": {"kind": "binary", "lower": 0.0, "upper": 1.0, "integer": True, "binary": True},
    "packet_count": {"kind": "statistical_integer", "lower": 0.0, "integer": True},
    "byte_count": {"kind": "continuous", "lower": 0.0},
    "duration": {"kind": "continuous", "lower": 0.0},
    "rate": {"kind": "continuous", "lower": 0.0},
}

# CICIDS fields are integer-valued protocol/code fields even though the generic
# manifest uses bounded_continuous to express their finite decoder domains.
_DATASET_INTEGER_OVERRIDES: dict[str, frozenset[str]] = {
    "cicids2017_distrinet": frozenset(
        {
            "src_port",
            "dst_port",
            "protocol",
            "fwd_init_win_bytes",
            "bwd_init_win_bytes",
        }
    )
}


def _normalize_name(name: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in str(name).strip()]
    return "_".join(part for part in "".join(chars).split("_") if part)


def _rate(mask: np.ndarray) -> float:
    return float(np.mean(mask))


def _to_numpy_bool(value: Any, expected: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    mask = np.asarray(value, dtype=np.bool_).reshape(-1)
    if mask.shape != (expected,):
        raise ValueError(f"mined checker returned shape {mask.shape}, expected {(expected,)}")
    return mask


def evaluate_mined_constraints(x_raw: np.ndarray, checker: Any) -> tuple[np.ndarray, dict[str, int]]:
    """Adapt an existing checker to a per-sample mask without coupling its rules.

    ``checker`` may be a callable or an object exposing ``validate``.  The local
    :class:`constraints.ConstraintEngine` is supported directly; only its result is
    consumed and no rules are copied into this module.
    """

    x = np.asarray(x_raw)
    try:
        import torch

        checker_input: Any = torch.as_tensor(x, dtype=torch.float32)
    except ImportError:  # pragma: no cover - this repository depends on torch
        checker_input = x

    result = checker.validate(checker_input) if hasattr(checker, "validate") else checker(checker_input)
    counts: dict[str, int] = {}
    if isinstance(result, Mapping):
        for key in ("pass_l0_l1_l2", "strict_valid", "valid", "layer2"):
            if key in result:
                mask = _to_numpy_bool(result[key], len(x))
                break
        else:
            raise ValueError("mined checker mapping has no recognized validity mask")
        per_constraint = result.get("per_constraint", {})
        if isinstance(per_constraint, Mapping):
            counts = {
                str(name): int((~_to_numpy_bool(value, len(x))).sum())
                for name, value in per_constraint.items()
            }
    else:
        mask = _to_numpy_bool(result, len(x))
    return mask, counts


class PAVEStyleValidator:
    """PAVE-inspired raw-space range and type validator.

    Fit only on raw training data.  Validation is observational: no projection,
    clipping, rounding, or mutation is implemented.
    """

    def __init__(self, *, integer_tolerance: float = 1e-6, range_tolerance: float = 1e-6) -> None:
        if integer_tolerance < 0 or range_tolerance < 0:
            raise ValueError("tolerances must be non-negative")
        self.integer_tolerance = float(integer_tolerance)
        self.range_tolerance = float(range_tolerance)
        self.feature_names: list[str] = []
        self.constraints: list[FeatureConstraint] = []
        self.dataset_name = "unspecified"
        self.is_fitted = False

    def fit(
        self,
        X_train_raw: np.ndarray,
        feature_names: list[str] | tuple[str, ...],
        schema: Any = None,
    ) -> "PAVEStyleValidator":
        """Fit train min/max fallbacks and resolve local feature semantics."""

        x = np.asarray(X_train_raw)
        names = list(feature_names)
        if x.ndim != 2 or x.shape[1] != len(names):
            raise ValueError(f"X_train_raw shape {x.shape} does not match {len(names)} feature names")
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")

        specs_by_name: dict[str, Any] = {}
        mapping_schema: Mapping[str, Any] | None = None
        if schema is not None and hasattr(schema, "specs"):
            schema.assert_names_match(names)
            specs_by_name = {spec.name: spec for spec in schema.specs}
            self.dataset_name = str(getattr(schema, "dataset_name", "unspecified"))
        elif isinstance(schema, Mapping):
            mapping_schema = schema
            self.dataset_name = str(schema.get("dataset_name", "custom"))
        elif schema is not None:
            raise TypeError("schema must be a FeatureManifest-like object, mapping, or None")

        constraints: list[FeatureConstraint] = []
        for index, name in enumerate(names):
            col = np.asarray(x[:, index], dtype=np.float64)
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                raise ValueError(f"training feature {name!r} has no finite values")
            train_lower = float(finite.min())
            train_upper = float(finite.max())

            if name in specs_by_name:
                constraint = self._from_manifest_spec(
                    specs_by_name[name], train_lower=train_lower, train_upper=train_upper
                )
            else:
                supplied = None
                if mapping_schema is not None:
                    supplied = mapping_schema.get(name, mapping_schema.get(_normalize_name(name)))
                constraint = self._from_mapping_or_name(
                    name,
                    supplied,
                    train_lower=train_lower,
                    train_upper=train_upper,
                )
            constraints.append(constraint)

        self.feature_names = names
        self.constraints = constraints
        self.is_fitted = True
        return self

    def _from_manifest_spec(
        self, spec: Any, *, train_lower: float, train_upper: float
    ) -> FeatureConstraint:
        value_type = str(spec.value_type)
        kind_by_type = {
            "binary": "binary",
            "integer_count": "statistical_integer",
            "categorical": "protocol_integer",
            "probability": "probability",
            "bounded_continuous": "continuous",
            "positive_continuous": "continuous",
            "derived": "continuous",
            "real": "continuous",
        }
        kind = kind_by_type.get(value_type, "continuous")
        integer = value_type in {"binary", "integer_count", "categorical"}
        binary = value_type == "binary"
        lower = None if spec.lower is None else float(spec.lower)
        upper = None if spec.upper is None else float(spec.upper)
        semantic_bound = lower is not None or upper is not None

        normalized = _normalize_name(spec.name)
        if normalized in _DATASET_INTEGER_OVERRIDES.get(self.dataset_name, frozenset()):
            integer = True
            kind = "protocol_integer"
            semantic_bound = True

        used_training = False
        if lower is None:
            lower = train_lower
            used_training = True
        if upper is None:
            upper = train_upper
            used_training = True
        if semantic_bound and used_training:
            source = "dataset_schema+training_range"
        elif semantic_bound:
            source = "dataset_schema"
        else:
            source = "training_range"
        return FeatureConstraint(
            name=str(spec.name),
            kind=kind,
            lower=lower,
            upper=upper,
            integer=integer,
            binary=binary,
            source=source,
            uncertain=not semantic_bound,
        )

    def _from_mapping_or_name(
        self,
        name: str,
        supplied: Any,
        *,
        train_lower: float,
        train_upper: float,
    ) -> FeatureConstraint:
        if isinstance(supplied, FeatureConstraint):
            data = asdict(supplied)
        elif isinstance(supplied, Mapping):
            data = dict(supplied)
        else:
            data = dict(_EXACT_UNIVERSAL.get(_normalize_name(name), {}))

        recognized = bool(data)
        lower = data.get("lower")
        upper = data.get("upper")
        used_training = False
        if lower is None:
            lower = train_lower
            used_training = True
        if upper is None:
            upper = train_upper
            used_training = True
        explicit_source = str(data.get("source", "universal" if recognized else "training_range"))
        source = (
            f"{explicit_source}+training_range"
            if recognized and used_training and "training_range" not in explicit_source
            else explicit_source
        )
        return FeatureConstraint(
            name=name,
            kind=str(data.get("kind", "continuous")),
            lower=float(lower),
            upper=float(upper),
            integer=bool(data.get("integer", False)),
            binary=bool(data.get("binary", False)),
            source=source,
            uncertain=bool(data.get("uncertain", not recognized)),
        )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("call fit() or load() before validation")

    def validate_sample(self, x: np.ndarray, mined_checker: Any = None) -> dict[str, Any]:
        """Validate one untouched raw-space sample."""

        result = self.validate_batch(np.asarray(x).reshape(1, -1), mined_checker=mined_checker)
        sample = dict(result["per_sample"][0])
        if "mined_valid_mask" in result:
            sample["mined_valid"] = bool(result["mined_valid_mask"][0])
            sample["strict_valid"] = bool(result["strict_valid_mask"][0])
        return sample

    def validate_batch(self, X: np.ndarray, mined_checker: Any = None) -> dict[str, Any]:
        """Validate untouched raw-space samples and return masks, rates, and failures."""

        self._require_fitted()
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.constraints):
            raise ValueError(f"X shape {x.shape} does not match {len(self.constraints)} constraints")
        if x.shape[0] == 0:
            raise ValueError("cannot validate an empty batch")

        n = x.shape[0]
        finite = np.isfinite(x)
        range_cells = finite.copy()
        type_cells = finite.copy()
        violations: list[list[dict[str, Any]]] = [[] for _ in range(n)]
        feature_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()

        for index, constraint in enumerate(self.constraints):
            values = x[:, index]
            finite_col = finite[:, index]
            if constraint.lower is not None:
                range_cells[:, index] &= values >= constraint.lower - self.range_tolerance
            if constraint.upper is not None:
                range_cells[:, index] &= values <= constraint.upper + self.range_tolerance
            if constraint.binary:
                type_cells[:, index] &= (
                    np.isclose(values, 0.0, atol=self.integer_tolerance, rtol=0.0)
                    | np.isclose(values, 1.0, atol=self.integer_tolerance, rtol=0.0)
                )
            elif constraint.integer:
                type_cells[:, index] &= np.abs(values - np.rint(values)) <= self.integer_tolerance

            for row in np.flatnonzero(~finite_col):
                self._record_violation(
                    violations, feature_counts, reason_counts, int(row), constraint.name,
                    values[row], "non-finite value"
                )
            below = (
                finite_col & (values < constraint.lower - self.range_tolerance)
                if constraint.lower is not None
                else np.zeros(n, dtype=np.bool_)
            )
            above = (
                finite_col & (values > constraint.upper + self.range_tolerance)
                if constraint.upper is not None
                else np.zeros(n, dtype=np.bool_)
            )
            for row in np.flatnonzero(below):
                self._record_violation(
                    violations, feature_counts, reason_counts, int(row), constraint.name,
                    values[row], f"below lower bound {constraint.lower:g}"
                )
            for row in np.flatnonzero(above):
                self._record_violation(
                    violations, feature_counts, reason_counts, int(row), constraint.name,
                    values[row], f"above upper bound {constraint.upper:g}"
                )
            if constraint.binary:
                bad_type = finite_col & ~type_cells[:, index]
                reason = "expected binary value"
            elif constraint.integer:
                bad_type = finite_col & ~type_cells[:, index]
                reason = "expected integer value"
            else:
                bad_type = np.zeros(n, dtype=np.bool_)
                reason = ""
            for row in np.flatnonzero(bad_type):
                self._record_violation(
                    violations, feature_counts, reason_counts, int(row), constraint.name,
                    values[row], reason
                )

        range_mask = range_cells.all(axis=1)
        type_mask = type_cells.all(axis=1)
        valid_mask = range_mask & type_mask
        result: dict[str, Any] = {
            "per_sample": [
                {
                    "valid": bool(valid_mask[row]),
                    "range_valid": bool(range_mask[row]),
                    "type_valid": bool(type_mask[row]),
                    "violations": violations[row],
                }
                for row in range(n)
            ],
            "total_samples": n,
            "valid_count": int(valid_mask.sum()),
            "validity_rate": _rate(valid_mask),
            "range_validity_rate": _rate(range_mask),
            "type_validity_rate": _rate(type_mask),
            "combined_validity_rate": _rate(valid_mask),
            "valid_mask": valid_mask,
            "range_valid_mask": range_mask,
            "type_valid_mask": type_mask,
            "violation_counts_by_feature": dict(feature_counts.most_common()),
            "violation_counts_by_reason": dict(reason_counts.most_common()),
        }
        if mined_checker is not None:
            mined_mask, mined_counts = evaluate_mined_constraints(x, mined_checker)
            strict_mask = valid_mask & mined_mask
            result.update(
                {
                    "mined_valid_count": int(mined_mask.sum()),
                    "mined_constraint_validity_rate": _rate(mined_mask),
                    "strict_valid_count": int(strict_mask.sum()),
                    "strict_validity_rate": _rate(strict_mask),
                    "mined_valid_mask": mined_mask,
                    "strict_valid_mask": strict_mask,
                    "mined_violation_counts_by_constraint": mined_counts,
                }
            )
        return result

    @staticmethod
    def _record_violation(
        violations: list[list[dict[str, Any]]],
        feature_counts: Counter[str],
        reason_counts: Counter[str],
        row: int,
        feature: str,
        value: float,
        reason: str,
    ) -> None:
        violations[row].append({"feature": feature, "value": float(value), "reason": reason})
        feature_counts[feature] += 1
        reason_counts[reason] += 1

    def validate_scaled_batch(self, X_scaled: np.ndarray, transform: Any, mined_checker: Any = None) -> dict[str, Any]:
        """Inverse-transform with an existing train-fit transform, then validate raw values."""

        x_raw = transform.inverse_transform(np.asarray(X_scaled))
        return self.validate_batch(x_raw, mined_checker=mined_checker)

    def summary(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "dataset": self.dataset_name,
            "integer_tolerance": self.integer_tolerance,
            "range_tolerance": self.range_tolerance,
            "constraints": [asdict(constraint) for constraint in self.constraints],
            "uncertain_features": [constraint.name for constraint in self.constraints if constraint.uncertain],
        }

    def format_audit(self) -> str:
        """Return a human-readable feature audit."""

        summary = self.summary()
        title = f"{str(summary['dataset']).upper()} PAVE-STYLE VALIDITY SCHEMA"
        lines = [title, "", f"{'Feature':32} {'Type':22} {'Lower':>12} {'Upper':>12}  Source", "-" * 104]
        for constraint in self.constraints:
            lower = "-" if constraint.lower is None else f"{constraint.lower:.6g}"
            upper = "-" if constraint.upper is None else f"{constraint.upper:.6g}"
            lines.append(
                f"{constraint.name[:32]:32} {constraint.kind[:22]:22} {lower:>12} {upper:>12}  {constraint.source}"
            )
        uncertain = summary["uncertain_features"]
        lines.extend(["", "Unrecognized / uncertain features:", "  " + (", ".join(uncertain) if uncertain else "none")])
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Persist the exact fitted registry for reuse without refitting."""

        self._require_fitted()
        Path(path).write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PAVEStyleValidator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        validator = cls(
            integer_tolerance=float(payload["integer_tolerance"]),
            range_tolerance=float(payload["range_tolerance"]),
        )
        validator.dataset_name = str(payload.get("dataset", "unspecified"))
        validator.constraints = [FeatureConstraint(**item) for item in payload["constraints"]]
        validator.feature_names = [constraint.name for constraint in validator.constraints]
        validator.is_fitted = True
        return validator
