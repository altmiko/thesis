"""Train-only calibration and hard budget loading for PrimAttack.

Numerical budgets are derived from the training split of the selected CICFlowMeter DistriNet
dataset (CICIDS2017 or CSE-CIC-IDS-2018; same 79-feature layout). The module never loads
validation/test data and never observes victim predictions or attack success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from datasets import DatasetAdapter, get_adapter
from datasets.cicids2017 import CICIDS2017Adapter

SCHEMA_VERSION = 1
BUDGET_NAMES = ("restricted", "intermediate", "maximum-evaluated")
_LEVEL_QUANTILES = dict(zip(BUDGET_NAMES, (0.25, 0.50, 0.75)))
_STAT_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
_ATTACK_CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")
_QUANTITIES = (
    "Flow Duration",
    "Flow Packets/s",
    "Flow Bytes/s",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Fwd Packet Length Min",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Packet Length Mean",
    "Packet Length Std",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "Fwd Act Data Pkts",
)
_ENVELOPE_FEATURES = (
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Total Length of Fwd Packet",
    "Fwd IAT Total",
    "Fwd IAT Max",
    "Fwd IAT Std",
    "Fwd IAT Mean",
    "Flow Duration",
)


@dataclass(frozen=True)
class BudgetLevel:
    """One frozen class-conditional hard primitive budget."""

    name: str
    padding_bytes_per_forward_packet: float
    max_relative_duration_change: float

    def __post_init__(self) -> None:
        if self.name not in BUDGET_NAMES and self.name != "unbounded":
            raise ValueError(f"unknown budget name {self.name!r}")
        if self.padding_bytes_per_forward_packet < 0:
            raise ValueError("padding budget must be non-negative")
        if self.max_relative_duration_change < 0:
            raise ValueError("timing budget must be non-negative")


@dataclass(frozen=True)
class ClassCalibration:
    class_name: str
    budget: BudgetLevel
    envelope_upper: dict[str, float]
    min_flow_packets_per_second: float | None

    def bounds_config(self) -> dict[str, float]:
        config = {
            "p_max": self.budget.padding_bytes_per_forward_packet,
            "max_relative_duration_change": self.budget.max_relative_duration_change,
            **{f"env_{name}": value for name, value in self.envelope_upper.items()},
        }
        if self.min_flow_packets_per_second is not None:
            config["min_flow_packets_per_second"] = self.min_flow_packets_per_second
        return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result = result[np.isfinite(result)]
    if result.size == 0:
        raise ValueError("cannot calibrate an empty/non-finite quantity")
    return result


def robust_summary(values: np.ndarray) -> dict[str, Any]:
    """Transparent robust summary used verbatim in the calibration artifact."""
    v = _finite(values)
    q = np.quantile(v, _STAT_QUANTILES)
    median = float(q[4])
    return {
        "n": int(v.size),
        "median": median,
        "q1": float(q[3]),
        "q3": float(q[5]),
        "iqr": float(q[5] - q[3]),
        "mad": float(np.median(np.abs(v - median))),
        "percentiles": {
            f"p{int(prob * 100):02d}": float(value)
            for prob, value in zip(_STAT_QUANTILES, q)
        },
    }


def _padding_population(rows: np.ndarray, i: Mapping[str, int]) -> np.ndarray:
    eligible = (
        (rows[:, i["Total Fwd Packet"]] >= 1)
        & (rows[:, i["Total Length of Fwd Packet"]] > 0)
        & (rows[:, i["Fwd Packet Length Mean"]] > 0)
    )
    return _finite(rows[eligible, i["Fwd Packet Length Mean"]])


def _relative_duration_variation(rows: np.ndarray, i: Mapping[str, int]) -> np.ndarray:
    duration = _finite(rows[:, i["Flow Duration"]])
    median = max(float(np.median(duration)), 1.0)
    return np.abs(duration - median) / median


def _rounded_empirical_budget(values: np.ndarray, probability: float) -> float:
    # p is discrete. Round the observed empirical quantile to its nearest legal byte rather
    # than importing a packet/MTU constant or selecting a value from attack success.
    return float(max(0, int(np.rint(np.quantile(values, probability)))))


def calibrate(adapter: DatasetAdapter | None = None) -> dict[str, Any]:
    """Fit the complete artifact from ``X_train_pristine`` and ``y_train_cat`` only."""
    adapter = adapter or CICIDS2017Adapter()
    processed = adapter._processed
    x_path = processed / "X_train_pristine.npy"
    y_path = processed / "y_train_cat.npy"
    raw = np.load(x_path, mmap_mode="r")
    labels = np.load(y_path, mmap_mode="r")
    manifest = adapter.feature_manifest()
    mapping = adapter.class_mapping()
    i = {name: manifest.index_by_name(name) for name in manifest.names}
    # A common train-only plausibility envelope keeps budgets comparable across classes and
    # avoids treating a class-local structural zero as a physical upper bound.
    global_envelope_upper = {
        name: float(np.quantile(np.asarray(raw[:, i[name]], dtype=np.float64), 0.99))
        for name in _ENVELOPE_FEATURES
    }

    classes: dict[str, Any] = {}
    for class_name in _ATTACK_CLASSES:
        class_id = mapping.name_to_id[class_name]
        rows = np.asarray(raw[np.asarray(labels) == class_id], dtype=np.float64)
        if rows.size == 0:
            raise ValueError(f"training split has no rows for {class_name}")

        quantities = {
            name: robust_summary(rows[:, i[name]])
            for name in _QUANTITIES
        }
        padding_population = _padding_population(rows, i)
        duration_variation = _relative_duration_variation(rows, i)
        budgets: dict[str, Any] = {}
        for budget_name, probability in _LEVEL_QUANTILES.items():
            budgets[budget_name] = {
                "selection_quantile": probability,
                "padding_bytes_per_forward_packet": _rounded_empirical_budget(
                    padding_population, probability
                ),
                "max_relative_duration_change": float(
                    np.quantile(duration_variation, probability)
                ),
            }

        flow_rate = _finite(rows[:, i["Flow Packets/s"]])
        byte_rate = _finite(rows[:, i["Flow Bytes/s"]])
        duration = _finite(rows[:, i["Flow Duration"]])
        rate_required = class_name in {"DoS", "DDoS"}
        critical_not_testable = {
            "Recon": [
                "complete_scanned_port_set",
                "scan_sequence",
                "distinct_connection_attempt_count",
            ],
            "BruteForce": [
                "authentication_attempt_count",
                "credential_or_payload_semantics",
                "server_authentication_outcome",
            ],
        }.get(class_name, [])
        classes[class_name] = {
            "class_id": class_id,
            "n": int(rows.shape[0]),
            "quantities": quantities,
            "protocol_counts": {
                str(int(value)): int(count)
                for value, count in zip(*np.unique(rows[:, i["Protocol"]], return_counts=True))
            },
            "calibrated_feasible_envelope": {
                "method": (
                    "per-flow algebraic/global-train-p99 plausibility headroom intersected "
                    "with the class-conditional empirical primitive budget and semantic "
                    "rate-retention cap"
                ),
                "feature_upper_percentile": 0.99,
                "feature_upper_population": "complete pristine training split",
                "envelope_upper": global_envelope_upper,
                "padding_reference": {
                    "quantity": "positive Fwd Packet Length Mean",
                    "n": int(padding_population.size),
                    "summary": robust_summary(padding_population),
                },
                "timing_reference": {
                    "quantity": (
                        "absolute relative deviation of Flow Duration from its class median"
                    ),
                    "summary": robust_summary(duration_variation),
                },
            },
            "budgets": budgets,
            "semantic_thresholds": {
                "flow_packets_per_second_lower": float(np.quantile(flow_rate, 0.05)),
                "flow_bytes_per_second_lower": float(np.quantile(byte_rate, 0.05)),
                "flow_duration_upper": float(np.quantile(duration, 0.99)),
                "rate_retention_required": rate_required,
                "method": (
                    "class-conditional training p05 lower rate and p99 upper duration; "
                    "fixed before attack evaluation"
                ),
                "critical_not_testable": critical_not_testable,
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": adapter.name,
        "fit_split": "train",
        "selection_prohibited_inputs": [
            "validation_features",
            "test_features",
            "victim_predictions",
            "adversarial_success",
        ],
        "source": {
            "features": str(x_path),
            "features_sha256": _sha256(x_path),
            "labels": str(y_path),
            "labels_sha256": _sha256(y_path),
            "preprocessing_manifest": str(processed / "preprocessing_manifest.json"),
            "preprocessing_manifest_sha256": _sha256(
                processed / "preprocessing_manifest.json"
            ),
        },
        "budget_level_method": {
            "restricted": "training empirical p25",
            "intermediate": "training empirical p50",
            "maximum-evaluated": "training empirical p75",
            "padding_population": "positive class-conditional Fwd Packet Length Mean",
            "timing_population": (
                "class-conditional absolute relative Flow Duration deviation from class median"
            ),
            "note": (
                "These are evaluated flow-level envelopes, not universal physical maxima."
            ),
        },
        "classes": classes,
    }


def write_calibration(output: Path, adapter: DatasetAdapter | None = None) -> dict[str, Any]:
    payload = calibrate(adapter)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported PrimAttack calibration schema in {path}")
    if payload.get("fit_split") != "train":
        raise ValueError("PrimAttack calibration must be fitted on train only")
    return payload


def class_calibration(
    payload: Mapping[str, Any], class_name: str, budget_name: str
) -> ClassCalibration:
    if budget_name not in BUDGET_NAMES:
        raise ValueError(f"budget_name must be one of {BUDGET_NAMES}")
    entry = payload["classes"][class_name]
    budget_data = entry["budgets"][budget_name]
    semantic = entry["semantic_thresholds"]
    return ClassCalibration(
        class_name=class_name,
        budget=BudgetLevel(
            name=budget_name,
            padding_bytes_per_forward_packet=float(
                budget_data["padding_bytes_per_forward_packet"]
            ),
            max_relative_duration_change=float(
                budget_data["max_relative_duration_change"]
            ),
        ),
        envelope_upper={
            name: float(value)
            for name, value in entry["calibrated_feasible_envelope"][
                "envelope_upper"
            ].items()
        },
        min_flow_packets_per_second=(
            float(semantic["flow_packets_per_second_lower"])
            if semantic["rate_retention_required"]
            else None
        ),
    )


def unbounded_calibration(payload: Mapping[str, Any], class_name: str) -> ClassCalibration:
    """Envelope-only PrimAttack budget: p_max=+inf, max_relative_duration_change=+inf.

    Removes the empirical class budget cap entirely while keeping the SAME train-fit p99
    physical feasibility envelope and (DoS/DDoS) semantic min-rate floor as the calibrated
    budgets. `per_flow_bounds` then clamps the primitives to the p99 envelope headroom only,
    so feasibility is gated by physical plausibility + realizability + semantic rules -- not by
    the p25/p50/p75 quantile budgets. All statistics remain train-only (reused from the
    calibration artifact); nothing is fit on val/test.
    """
    base = class_calibration(payload, class_name, "maximum-evaluated")
    return ClassCalibration(
        class_name=class_name,
        budget=BudgetLevel(
            name="unbounded",
            padding_bytes_per_forward_packet=float("inf"),
            max_relative_duration_change=float("inf"),
        ),
        envelope_upper=dict(base.envelope_upper),
        min_flow_packets_per_second=base.min_flow_packets_per_second,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cicids2017",
                        help="adapter name for datasets.get_adapter (cicids2017 | cicids2018)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/primattack/budget_calibration.json"),
    )
    args = parser.parse_args()
    payload = write_calibration(args.output, get_adapter(args.dataset))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fit_split": payload["fit_split"],
                "classes": list(payload["classes"]),
            }
        )
    )


if __name__ == "__main__":
    main()
