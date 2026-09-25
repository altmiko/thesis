"""FlowSemanticValidator tri-state, hard-budget, and invariant regression tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from attack.flow_semantics import FlowSemanticValidator, SemanticStatus
from attack.primattack_budget import class_calibration, load_calibration
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel
from datasets import get_adapter


def _context(class_name: str, n: int = 32):
    adapter = get_adapter("cicids2017")
    model = CICIDS2017PrimitiveModel(adapter.feature_manifest())
    calibration = load_calibration(
        adapter.repo_root / "artifacts/primattack/budget_calibration.json"
    )
    class_id = adapter.class_mapping().name_to_id[class_name]
    labels = np.load(adapter._processed / "y_test_cat.npy", mmap_mode="r")
    raw_all = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    rows = np.flatnonzero(labels == class_id)[:n]
    raw = torch.tensor(
        np.ascontiguousarray(np.asarray(raw_all[rows]), dtype=np.float32)
    )
    cfg = class_calibration(calibration, class_name, "maximum-evaluated")
    bounds = model.per_flow_bounds(raw, cfg.bounds_config())
    validator = FlowSemanticValidator(model, calibration)
    metadata = {
        "Src IP": np.asarray([f"source-{value}" for value in rows]),
        "Dst IP": np.asarray([f"target-{value}" for value in rows]),
    }
    y = np.full(len(rows), class_id, dtype=np.int64)
    return model, validator, cfg, raw, bounds, metadata, y


def _identity(raw):
    zeros = torch.zeros(len(raw))
    return {"p": zeros.clone(), "delay": zeros.clone(), "shape": zeros.clone()}


def test_dos_identity_passes_all_flow_level_proxy_checks():
    model, validator, cfg, raw, bounds, metadata, labels = _context("DoS", 64)
    threshold = validator.calibration["classes"]["DoS"]["semantic_thresholds"][
        "flow_packets_per_second_lower"
    ]
    keep = raw[:, model.i["Flow Packets/s"]] >= float(threshold)
    raw = raw[keep]
    bounds = {name: value[keep] for name, value in bounds.items()}
    metadata = {name: value[keep.numpy()] for name, value in metadata.items()}
    labels = labels[keep.numpy()]
    controls = _identity(raw)
    report = validator.evaluate(
        raw, raw.clone(), controls, controls, bounds,
        class_name="DoS", budget=cfg.budget,
        original_labels=labels, adversarial_labels=labels.copy(),
        original_metadata=metadata, adversarial_metadata=metadata,
    )
    assert np.all(report.primitive_feasible)
    assert np.all(report.semantic_status == SemanticStatus.PASS.value)
    assert np.all(report.required_tests_failed == 0)


def test_each_generic_invariant_violation_is_detected():
    model, validator, cfg, raw, bounds, metadata, labels = _context("DoS", 8)
    controls = _identity(raw)
    cases = {
        "PROTOCOL_CHANGED": ("Protocol", 1.0),
        "PACKET_COUNT_CHANGED": ("Total Fwd Packet", 1.0),
        "CONTROL_FLAG_CHANGED": ("SYN Flag Count", 1.0),
        "TRAFFIC_VOLUME_DECREASED": ("Total Length of Bwd Packet", -1.0),
    }
    for reason, (feature, delta) in cases.items():
        adversarial = raw.clone()
        adversarial[:, model.i[feature]] += delta
        report = validator.evaluate(
            raw, adversarial, controls, controls, bounds,
            class_name="DoS", budget=cfg.budget,
            original_labels=labels, adversarial_labels=labels.copy(),
            original_metadata=metadata, adversarial_metadata=metadata,
        )
        assert np.all(report.semantic_status == SemanticStatus.FAIL.value)
        assert all(reason in reasons for reasons in report.failure_reasons)


@pytest.mark.parametrize("primitive", ["p", "delay"])
def test_intentional_primitive_budget_violation_fails_closed(primitive):
    model, validator, cfg, raw, bounds, metadata, labels = _context("DoS", 8)
    violating = _identity(raw)
    violating[primitive] = bounds[primitive] + (100.0 if primitive == "p" else 1000.0)
    adversarial = model.generate(raw, violating)
    report = validator.evaluate(
        raw, adversarial, violating, violating, bounds,
        class_name="DoS", budget=cfg.budget,
        original_labels=labels, adversarial_labels=labels.copy(),
        original_metadata=metadata, adversarial_metadata=metadata,
    )
    assert not np.any(report.primitive_feasible)
    assert np.all(report.semantic_status == SemanticStatus.FAIL.value)
    assert all("PRIMITIVE_BUDGET_VIOLATION" in reasons
               for reasons in report.primitive_violation_reasons)


def test_recon_critical_properties_are_not_testable_not_pass():
    _, validator, cfg, raw, bounds, metadata, labels = _context("Recon", 16)
    controls = _identity(raw)
    report = validator.evaluate(
        raw, raw.clone(), controls, controls, bounds,
        class_name="Recon", budget=cfg.budget,
        original_labels=labels, adversarial_labels=labels.copy(),
        original_metadata=metadata, adversarial_metadata=metadata,
    )
    assert np.all(report.semantic_status == SemanticStatus.NOT_FULLY_TESTABLE.value)
    assert np.all(report.tests_not_testable >= 3)
    assert not np.any(report.semantic_pass)
    assert all("NOT_TESTABLE_FROM_FLOW_DATA" in reasons
               for reasons in report.not_testable_reasons)


def test_bruteforce_application_semantics_are_not_testable_not_pass():
    _, validator, cfg, raw, bounds, metadata, labels = _context("BruteForce", 16)
    controls = _identity(raw)
    report = validator.evaluate(
        raw, raw.clone(), controls, controls, bounds,
        class_name="BruteForce", budget=cfg.budget,
        original_labels=labels, adversarial_labels=labels.copy(),
        original_metadata=metadata, adversarial_metadata=metadata,
    )
    assert np.all(report.semantic_status == SemanticStatus.NOT_FULLY_TESTABLE.value)
    assert not np.any(report.semantic_pass)


def test_missing_metadata_never_silently_becomes_pass():
    _, validator, cfg, raw, bounds, _, labels = _context("DoS", 8)
    controls = _identity(raw)
    report = validator.evaluate(
        raw, raw.clone(), controls, controls, bounds,
        class_name="DoS", budget=cfg.budget,
        original_labels=labels, adversarial_labels=labels.copy(),
        original_metadata=None, adversarial_metadata=None,
    )
    assert np.all(report.tests_not_testable >= 1)
    assert not np.any(report.semantic_pass)
