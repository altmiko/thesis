"""Flow-level attack-semantic preservation proxy for CICIDS2017 PrimAttack.

This validator is intentionally separate from network/domain validity. It uses only saved flow
features, immutable metadata, projected primitive controls, and train-fitted thresholds. It does
not claim packet-level or real-world functionality preservation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np
import torch

from attack.primattack_budget import BudgetLevel
from attack.realizability.base import PrimitiveSpec
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel

_EPS = 1e-12


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_TESTABLE = "NOT_TESTABLE"


class SemanticStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_FULLY_TESTABLE = "NOT_FULLY_TESTABLE"


@dataclass(frozen=True)
class SemanticCheck:
    name: str
    required: bool
    status: np.ndarray
    reason_code: str
    explanation: str


@dataclass(frozen=True)
class PrimitiveCosts:
    original_duration: np.ndarray
    adversarial_duration: np.ndarray
    delta_duration: np.ndarray
    relative_duration_change: np.ndarray
    original_byte_quantity: np.ndarray
    adversarial_byte_quantity: np.ndarray
    added_byte_quantity: np.ndarray
    relative_byte_change: np.ndarray
    original_rate: np.ndarray
    adversarial_rate: np.ndarray
    rate_retention: np.ndarray
    padding_percent_of_forward_mean: np.ndarray
    normalized_padding_magnitude: np.ndarray
    normalized_timing_magnitude: np.ndarray


@dataclass(frozen=True)
class FlowSemanticReport:
    semantic_status: np.ndarray
    primitive_feasible: np.ndarray
    primitive_violation_reasons: list[list[str]]
    required_tests_passed: np.ndarray
    required_tests_failed: np.ndarray
    tests_not_testable: np.ndarray
    failure_reasons: list[list[str]]
    not_testable_reasons: list[list[str]]
    checks: tuple[SemanticCheck, ...]
    costs: PrimitiveCosts
    features_changed: list[list[str]]
    number_features_changed: np.ndarray

    @property
    def semantic_pass(self) -> np.ndarray:
        return self.semantic_status == SemanticStatus.PASS.value

    @property
    def semantic_testable(self) -> np.ndarray:
        return self.semantic_status != SemanticStatus.NOT_FULLY_TESTABLE.value


class SemanticRule:
    """Small class strategy interface; concrete rules remain transparent equations."""

    labels: frozenset[str] = frozenset()

    def supports(self, label: str) -> bool:
        return label in self.labels

    def evaluate(
        self,
        original: np.ndarray,
        adversarial: np.ndarray,
        i: Mapping[str, int],
        semantic_thresholds: Mapping[str, object],
    ) -> list[SemanticCheck]:
        raise NotImplementedError


class RateRetentionRule(SemanticRule):
    labels = frozenset({"DoS", "DDoS"})

    def evaluate(self, original, adversarial, i, semantic_thresholds):
        threshold = float(semantic_thresholds["flow_packets_per_second_lower"])
        rate = adversarial[:, i["Flow Packets/s"]]
        return [SemanticCheck(
            name="class_attack_like_flow_rate",
            required=True,
            status=np.where(rate >= threshold, CheckStatus.PASS.value, CheckStatus.FAIL.value),
            reason_code="RATE_BELOW_TRAIN_P05",
            explanation=(
                f"adversarial Flow Packets/s must remain >= class training p05 ({threshold:.12g})"
            ),
        )]


class ReconRule(SemanticRule):
    labels = frozenset({"Recon"})

    def evaluate(self, original, adversarial, i, semantic_thresholds):
        checks = []
        for property_name in semantic_thresholds.get("critical_not_testable", []):
            checks.append(SemanticCheck(
                name=property_name,
                required=True,
                status=np.full(original.shape[0], CheckStatus.NOT_TESTABLE.value, dtype="U16"),
                reason_code="NOT_TESTABLE_FROM_FLOW_DATA",
                explanation=(
                    "a single aggregate CICFlowMeter row does not contain the complete port set, "
                    "scan order, or distinct connection-attempt sequence"
                ),
            ))
        return checks


class BruteForceRule(SemanticRule):
    labels = frozenset({"BruteForce"})

    def evaluate(self, original, adversarial, i, semantic_thresholds):
        checks = []
        for property_name in semantic_thresholds.get("critical_not_testable", []):
            checks.append(SemanticCheck(
                name=property_name,
                required=True,
                status=np.full(original.shape[0], CheckStatus.NOT_TESTABLE.value, dtype="U16"),
                reason_code="NOT_TESTABLE_FROM_FLOW_DATA",
                explanation=(
                    "aggregate flow features do not expose authentication attempts, credentials, "
                    "payload semantics, or application outcomes"
                ),
            ))
        return checks


def primitive_costs(
    model: CICIDS2017PrimitiveModel,
    original: torch.Tensor,
    adversarial: torch.Tensor,
    projected: Mapping[str, torch.Tensor],
    bounds: Mapping[str, torch.Tensor],
) -> PrimitiveCosts:
    """Compute only quantities directly represented by the flow/primitive model."""
    i = model.i
    o = original.detach().cpu().numpy().astype(np.float64, copy=False)
    a = adversarial.detach().cpu().numpy().astype(np.float64, copy=False)
    p = projected["p"].detach().cpu().numpy().astype(np.float64, copy=False)
    delay = projected["delay"].detach().cpu().numpy().astype(np.float64, copy=False)
    p_hi = bounds["p"].detach().cpu().numpy().astype(np.float64, copy=False)
    delay_hi = bounds["delay"].detach().cpu().numpy().astype(np.float64, copy=False)

    d0 = o[:, i["Flow Duration"]]
    d1 = a[:, i["Flow Duration"]]
    b0 = o[:, i["Total Length of Fwd Packet"]] + o[:, i["Total Length of Bwd Packet"]]
    b1 = a[:, i["Total Length of Fwd Packet"]] + a[:, i["Total Length of Bwd Packet"]]
    r0 = o[:, i["Flow Packets/s"]]
    r1 = a[:, i["Flow Packets/s"]]
    fwd_mean = o[:, i["Fwd Packet Length Mean"]]
    return PrimitiveCosts(
        original_duration=d0,
        adversarial_duration=d1,
        delta_duration=d1 - d0,
        relative_duration_change=(d1 - d0) / np.maximum(d0, _EPS),
        original_byte_quantity=b0,
        adversarial_byte_quantity=b1,
        added_byte_quantity=b1 - b0,
        relative_byte_change=(b1 - b0) / np.maximum(b0, _EPS),
        original_rate=r0,
        adversarial_rate=r1,
        rate_retention=r1 / np.maximum(r0, _EPS),
        padding_percent_of_forward_mean=100.0 * p / np.maximum(fwd_mean, _EPS),
        normalized_padding_magnitude=np.divide(
            p, p_hi, out=np.zeros_like(p), where=p_hi > 0
        ),
        normalized_timing_magnitude=np.divide(
            delay,
            delay_hi,
            out=np.zeros_like(delay),
            where=delay_hi > 0,
        ),
    )


class FlowSemanticValidator:
    """Evaluate generic invariants plus one retained-class semantic strategy."""

    def __init__(
        self,
        model: CICIDS2017PrimitiveModel,
        calibration: Mapping[str, object],
        *,
        atol: float = 1e-6,
    ) -> None:
        if calibration.get("fit_split") != "train":
            raise ValueError("semantic thresholds must come from the training split")
        self.model = model
        self.calibration = calibration
        self.atol = float(atol)
        self.rules: tuple[SemanticRule, ...] = (
            RateRetentionRule(),
            ReconRule(),
            BruteForceRule(),
        )
        self._specs: dict[str, PrimitiveSpec] = {
            spec.name: spec for spec in model.primitives()
        }

    @staticmethod
    def _status(mask: np.ndarray) -> np.ndarray:
        return np.where(mask, CheckStatus.PASS.value, CheckStatus.FAIL.value)

    def _unchanged_check(
        self, name: str, original: np.ndarray, adversarial: np.ndarray, columns: Sequence[int],
        *, reason_code: str, explanation: str,
    ) -> SemanticCheck:
        if not columns:
            return SemanticCheck(
                name=name,
                required=True,
                status=np.full(original.shape[0], CheckStatus.NOT_TESTABLE.value, dtype="U16"),
                reason_code="NOT_TESTABLE_FROM_FLOW_DATA",
                explanation=explanation,
            )
        same = np.all(
            np.isclose(original[:, columns], adversarial[:, columns], atol=self.atol, rtol=0.0),
            axis=1,
        )
        return SemanticCheck(name, True, self._status(same), reason_code, explanation)

    def evaluate(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
        requested: Mapping[str, torch.Tensor],
        projected: Mapping[str, torch.Tensor],
        bounds: Mapping[str, torch.Tensor],
        *,
        class_name: str,
        budget: BudgetLevel,
        original_labels: np.ndarray | None = None,
        adversarial_labels: np.ndarray | None = None,
        original_metadata: Mapping[str, np.ndarray] | None = None,
        adversarial_metadata: Mapping[str, np.ndarray] | None = None,
    ) -> FlowSemanticReport:
        if class_name not in self.calibration["classes"]:
            raise ValueError(f"class {class_name!r} is absent from calibration")
        if original.shape != adversarial.shape:
            raise ValueError("original/adversarial shape mismatch")
        n = original.shape[0]
        i = self.model.i
        o = original.detach().cpu().numpy().astype(np.float64, copy=False)
        a = adversarial.detach().cpu().numpy().astype(np.float64, copy=False)
        req = {k: v.detach().cpu().numpy().astype(np.float64, copy=False) for k, v in requested.items()}
        proj = {k: v.detach().cpu().numpy().astype(np.float64, copy=False) for k, v in projected.items()}
        cap = {k: v.detach().cpu().numpy().astype(np.float64, copy=False) for k, v in bounds.items()}
        costs = primitive_costs(self.model, original, adversarial, projected, bounds)

        checks: list[SemanticCheck] = []
        if original_labels is None or adversarial_labels is None:
            label_status = np.full(n, CheckStatus.NOT_TESTABLE.value, dtype="U16")
        else:
            label_status = self._status(np.asarray(original_labels) == np.asarray(adversarial_labels))
        checks.append(SemanticCheck(
            "attack_label_metadata_unchanged", True, label_status,
            "ATTACK_LABEL_CHANGED", "source attack-label metadata must be copied unchanged",
        ))

        checks.append(self._unchanged_check(
            "protocol_unchanged", o, a, [i["Protocol"]],
            reason_code="PROTOCOL_CHANGED", explanation="IP protocol must remain unchanged",
        ))
        checks.append(self._unchanged_check(
            "service_ports_unchanged", o, a, [i["Src Port"], i["Dst Port"]],
            reason_code="SERVICE_ENDPOINT_CHANGED",
            explanation="source/destination service-port fields must remain unchanged",
        ))
        packet_count_names = (
            "Total Fwd Packet", "Total Bwd packets", "Fwd Act Data Pkts",
            "Subflow Fwd Packets", "Subflow Bwd Packets",
        )
        checks.append(self._unchanged_check(
            "packet_counts_unchanged", o, a, [i[name] for name in packet_count_names],
            reason_code="PACKET_COUNT_CHANGED",
            explanation="packet-count injection is outside the retained primitive contract",
        ))
        flag_names = tuple(name for name in self.model.feature_names if "Flag" in name)
        checks.append(self._unchanged_check(
            "control_flags_unchanged", o, a, [i[name] for name in flag_names],
            reason_code="CONTROL_FLAG_CHANGED",
            explanation="TCP control/flag aggregates are immutable under PrimAttack",
        ))

        if original_metadata is None or adversarial_metadata is None:
            endpoint_status = np.full(n, CheckStatus.NOT_TESTABLE.value, dtype="U16")
        else:
            available = [
                name for name in ("Src IP", "Dst IP")
                if name in original_metadata and name in adversarial_metadata
            ]
            if not available:
                endpoint_status = np.full(n, CheckStatus.NOT_TESTABLE.value, dtype="U16")
            else:
                endpoint_status = self._status(np.logical_and.reduce([
                    np.asarray(original_metadata[name]) == np.asarray(adversarial_metadata[name])
                    for name in available
                ]))
        checks.append(SemanticCheck(
            "flow_endpoints_and_direction_unchanged", True, endpoint_status,
            "FLOW_ENDPOINT_OR_DIRECTION_CHANGED",
            "source/destination IP metadata jointly preserve endpoints and flow direction",
        ))

        finite = np.isfinite(a).all(axis=1)
        checks.append(SemanticCheck(
            "finite_generated_features", True, self._status(finite),
            "NONFINITE_GENERATED_FEATURE", "generated artifacts must contain no NaN or Inf",
        ))
        checks.append(SemanticCheck(
            "traffic_volume_not_decreased", True,
            self._status(costs.added_byte_quantity >= -self.atol),
            "TRAFFIC_VOLUME_DECREASED",
            "padding/size primitives may not reduce represented traffic volume",
        ))

        delta = np.abs(a - o)
        features_changed: list[list[str]] = []
        allowed_rows = np.zeros_like(delta, dtype=bool)
        for row in range(n):
            allowed: set[str] = set()
            if proj["p"][row] > 0:
                allowed.update(self._specs["p"].dependencies)
            if proj["delay"][row] > 0:
                allowed.update(self._specs["delay"].dependencies)
            changed = [
                name for name, column in i.items() if delta[row, column] > self.atol
            ]
            features_changed.append(changed)
            for name in allowed:
                allowed_rows[row, i[name]] = True
        undeclared = np.any((delta > self.atol) & ~allowed_rows, axis=1)
        checks.append(SemanticCheck(
            "only_declared_primitive_dependencies_changed", True,
            self._status(~undeclared), "UNDECLARED_FEATURE_CHANGED",
            "phi may write only dependencies declared by the active primitive specifications",
        ))

        p_integer = np.isclose(proj["p"], np.rint(proj["p"]), atol=self.atol, rtol=0.0)
        delay_integer = np.isclose(
            proj["delay"], np.rint(proj["delay"]), atol=self.atol, rtol=0.0
        )
        primitive_ok = (
            np.isfinite(proj["p"]) & np.isfinite(proj["delay"]) & np.isfinite(proj["shape"])
            & (proj["p"] >= -self.atol)
            & (proj["p"] <= np.floor(cap["p"]) + self.atol)
            & p_integer
            & (proj["delay"] >= -self.atol)
            & (proj["delay"] <= np.floor(cap["delay"]) + self.atol)
            & delay_integer
            & (proj["shape"] >= -self.atol)
            & (proj["shape"] <= cap["shape"] + self.atol)
            & (costs.relative_duration_change <= budget.max_relative_duration_change + self.atol)
            & (costs.added_byte_quantity >= -self.atol)
        )
        checks.append(SemanticCheck(
            "primitive_budget_compliance", True, self._status(primitive_ok),
            "PRIMITIVE_BUDGET_VIOLATION",
            "projected controls and realized costs must remain inside the named hard budget",
        ))

        class_entry = self.calibration["classes"][class_name]
        semantic_thresholds = class_entry["semantic_thresholds"]
        rule = next((rule for rule in self.rules if rule.supports(class_name)), None)
        if rule is not None:
            checks.extend(rule.evaluate(o, a, i, semantic_thresholds))

        required_passed = np.zeros(n, dtype=np.int64)
        required_failed = np.zeros(n, dtype=np.int64)
        not_testable = np.zeros(n, dtype=np.int64)
        failure_reasons: list[list[str]] = [[] for _ in range(n)]
        not_testable_reasons: list[list[str]] = [[] for _ in range(n)]
        primitive_reasons: list[list[str]] = [[] for _ in range(n)]
        for row in range(n):
            if not primitive_ok[row]:
                primitive_reasons[row].append("PRIMITIVE_BUDGET_VIOLATION")
        for check in checks:
            if not check.required:
                continue
            pass_mask = check.status == CheckStatus.PASS.value
            fail_mask = check.status == CheckStatus.FAIL.value
            nt_mask = check.status == CheckStatus.NOT_TESTABLE.value
            required_passed += pass_mask
            required_failed += fail_mask
            not_testable += nt_mask
            for row in np.flatnonzero(fail_mask):
                failure_reasons[int(row)].append(check.reason_code)
            for row in np.flatnonzero(nt_mask):
                not_testable_reasons[int(row)].append(check.reason_code)

        status = np.full(n, SemanticStatus.PASS.value, dtype="U24")
        status[not_testable > 0] = SemanticStatus.NOT_FULLY_TESTABLE.value
        status[required_failed > 0] = SemanticStatus.FAIL.value
        return FlowSemanticReport(
            semantic_status=status,
            primitive_feasible=primitive_ok,
            primitive_violation_reasons=primitive_reasons,
            required_tests_passed=required_passed,
            required_tests_failed=required_failed,
            tests_not_testable=not_testable,
            failure_reasons=failure_reasons,
            not_testable_reasons=not_testable_reasons,
            checks=tuple(checks),
            costs=costs,
            features_changed=features_changed,
            number_features_changed=np.asarray([len(names) for names in features_changed]),
        )
