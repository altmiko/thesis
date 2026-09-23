"""Train-only PrimAttack calibration and frozen budget contract."""
from __future__ import annotations

import json

from attack.primattack_budget import BUDGET_NAMES, calibrate, load_calibration
from datasets import get_adapter


def test_frozen_artifact_is_train_only_and_success_independent():
    adapter = get_adapter("cicids2017")
    path = adapter.repo_root / "artifacts/primattack/budget_calibration.json"
    payload = load_calibration(path)
    assert payload["fit_split"] == "train"
    assert payload["source"]["features"].endswith("X_train_pristine.npy")
    assert payload["source"]["labels"].endswith("y_train_cat.npy")
    prohibited = set(payload["selection_prohibited_inputs"])
    assert {"test_features", "victim_predictions", "adversarial_success"} <= prohibited
    serialized = json.dumps(payload).lower()
    assert "x_test" not in serialized
    assert "attack_results" not in serialized


def test_calibration_reproduction_matches_frozen_budget_values():
    adapter = get_adapter("cicids2017")
    frozen = load_calibration(
        adapter.repo_root / "artifacts/primattack/budget_calibration.json"
    )
    reproduced = calibrate(adapter)
    for class_name in ("DoS", "DDoS", "Recon", "BruteForce"):
        assert reproduced["classes"][class_name]["n"] == frozen["classes"][class_name]["n"]
        assert reproduced["classes"][class_name]["budgets"] == frozen["classes"][class_name]["budgets"]
        assert reproduced["classes"][class_name]["semantic_thresholds"] == frozen["classes"][class_name]["semantic_thresholds"]


def test_named_budgets_are_empirical_and_monotone():
    adapter = get_adapter("cicids2017")
    payload = load_calibration(
        adapter.repo_root / "artifacts/primattack/budget_calibration.json"
    )
    assert BUDGET_NAMES == ("restricted", "intermediate", "maximum-evaluated")
    expected_probabilities = [0.25, 0.50, 0.75]
    for entry in payload["classes"].values():
        levels = [entry["budgets"][name] for name in BUDGET_NAMES]
        assert [level["selection_quantile"] for level in levels] == expected_probabilities
        assert [level["padding_bytes_per_forward_packet"] for level in levels] == sorted(
            level["padding_bytes_per_forward_packet"] for level in levels
        )
        assert [level["max_relative_duration_change"] for level in levels] == sorted(
            level["max_relative_duration_change"] for level in levels
        )
        assert entry["calibrated_feasible_envelope"]["feature_upper_population"] == (
            "complete pristine training split"
        )
