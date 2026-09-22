"""Focused checks for the independent PAVE-style validity baseline."""
from __future__ import annotations

import numpy as np
import pytest

from datasets import FeatureManifest, FeatureSpec, FeatureTransform
from datasets.ciciot2023 import build_manifest
from evaluation.pave_style_validator import PAVEStyleValidator
from evaluation.run_pave_validity import evaluate_attack_arrays


FEATURES = ["Time_To_Live", "binary_flag", "packet_count", "measurement"]
TRAIN = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [64.0, 1.0, 10.0, 2.0],
        [128.0, 0.0, 20.0, 3.0],
    ],
    dtype=np.float64,
)


@pytest.fixture()
def validator() -> PAVEStyleValidator:
    return PAVEStyleValidator().fit(TRAIN, FEATURES)


def _sample(**updates: float) -> np.ndarray:
    values = {
        "Time_To_Live": 64.0,
        "binary_flag": 0.0,
        "packet_count": 10.0,
        "measurement": 2.0,
    }
    values.update(updates)
    return np.array([values[name] for name in FEATURES], dtype=np.float64)


@pytest.mark.parametrize("ttl", [0.0, 255.0])
def test_ttl_universal_endpoints_are_valid(validator: PAVEStyleValidator, ttl: float):
    assert validator.validate_sample(_sample(Time_To_Live=ttl))["valid"]


@pytest.mark.parametrize("ttl", [-1.0, 256.0])
def test_ttl_outside_universal_domain_is_invalid(validator: PAVEStyleValidator, ttl: float):
    result = validator.validate_sample(_sample(Time_To_Live=ttl))
    assert not result["range_valid"]
    assert any(item["feature"] == "Time_To_Live" for item in result["violations"])


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_binary_endpoints_are_valid(validator: PAVEStyleValidator, value: float):
    assert validator.validate_sample(_sample(binary_flag=value))["valid"]


def test_fractional_binary_is_type_invalid_but_in_range(validator: PAVEStyleValidator):
    result = validator.validate_sample(_sample(binary_flag=0.5))
    assert result["range_valid"]
    assert not result["type_valid"]
    assert result["violations"][0]["reason"] == "expected binary value"


def test_packet_count_requires_nonnegative_integer(validator: PAVEStyleValidator):
    assert validator.validate_sample(_sample(packet_count=10.0))["valid"]
    fractional = validator.validate_sample(_sample(packet_count=10.4))
    negative = validator.validate_sample(_sample(packet_count=-1.0))
    assert fractional["range_valid"] and not fractional["type_valid"]
    assert not negative["range_valid"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_value_is_invalid(validator: PAVEStyleValidator, value: float):
    result = validator.validate_sample(_sample(measurement=value))
    assert not result["valid"]
    assert result["violations"][0]["reason"] == "non-finite value"


def test_unknown_feature_uses_training_minmax(validator: PAVEStyleValidator):
    assert validator.validate_sample(_sample(measurement=1.5))["valid"]
    result = validator.validate_sample(_sample(measurement=3.1))
    assert not result["range_valid"]
    summary = validator.summary()
    assert "measurement" in summary["uncertain_features"]


def test_local_ciciot_aggregates_are_not_misclassified_as_binary_or_integer():
    manifest = build_manifest()
    train = np.zeros((2, manifest.n_features), dtype=np.float64)
    train[1, manifest.index_by_name("Time_To_Live")] = 255.0
    train[1, manifest.index_by_name("syn_flag_number")] = 1.0
    fitted = PAVEStyleValidator().fit(train, manifest.names, schema=manifest)
    candidate = train[0].copy()
    candidate[manifest.index_by_name("Time_To_Live")] = 10.5
    candidate[manifest.index_by_name("syn_flag_number")] = 0.5

    result = fitted.validate_sample(candidate)
    constraints = {item.name: item for item in fitted.constraints}
    assert result["valid"]
    assert not constraints["Time_To_Live"].integer
    assert not constraints["syn_flag_number"].binary


def test_inverse_transform_path_uses_existing_transform_without_mutation():
    manifest = FeatureManifest(
        [
            FeatureSpec("packet_count", 0, "integer_count", "packet_count", lower=0.0),
            FeatureSpec("value", 1, "positive_continuous", "measurement", lower=0.0),
        ],
        dataset_name="toy",
    )
    raw_train = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]])
    transform = FeatureTransform(manifest).fit(raw_train, provenance="train")
    fitted = PAVEStyleValidator().fit(raw_train, manifest.names, schema=manifest)
    raw_candidate = np.array([[10.0, 25.0], [10.4, 25.0]])
    scaled = transform.transform(raw_candidate)
    unchanged = scaled.copy()

    result = fitted.validate_scaled_batch(scaled, transform)

    assert result["valid_mask"].tolist() == [True, False]
    assert np.array_equal(scaled, unchanged)


def test_existing_checker_remains_separate_and_combines_only_at_reporting(validator: PAVEStyleValidator):
    class ExistingChecker:
        def validate(self, x):
            mined = x[:, 3] <= 2.0
            return {"layer2": mined, "per_constraint": {"existing_rule": mined}}

    batch = np.stack([_sample(measurement=2.0), _sample(measurement=3.0)])
    result = validator.validate_batch(batch, mined_checker=ExistingChecker())

    assert result["valid_mask"].tolist() == [True, True]
    assert result["mined_valid_mask"].tolist() == [True, False]
    assert result["strict_valid_mask"].tolist() == [True, False]
    assert result["mined_violation_counts_by_constraint"] == {"existing_rule": 1}


def test_attack_metrics_use_originally_correct_denominator(validator: PAVEStyleValidator):
    batch = np.stack([_sample(), _sample(measurement=3.1), _sample()])
    validity = validator.validate_batch(batch)
    metrics = evaluate_attack_arrays(
        attack="pgd",
        validation=validity,
        y_true=np.array([1, 1, 1]),
        y_pred_clean=np.array([1, 1, 0]),
        y_pred_adv=np.array([0, 0, 0]),
    )

    assert metrics["originally_correct_samples"] == 2
    assert metrics["raw_asr"] == 1.0
    assert metrics["valid_successful_attacks"] == 1
    assert metrics["valid_asr"] == 0.5


def test_fitted_registry_round_trip(tmp_path, validator: PAVEStyleValidator):
    path = tmp_path / "validator.json"
    validator.save(path)
    loaded = PAVEStyleValidator.load(path)
    assert loaded.summary() == validator.summary()
    assert loaded.validate_sample(_sample()) == validator.validate_sample(_sample())
