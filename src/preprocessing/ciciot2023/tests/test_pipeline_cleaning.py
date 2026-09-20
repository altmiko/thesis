"""Regression tests for window-aggregate feature semantics and train-only fitting."""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.ciciot2023.pipeline import (
    CLUSTER_FEATURES,
    clean_features,
    clip_round,
    fit_scaler,
    sample_train,
)
from src.preprocessing.schema import (
    BOUNDED_AGGREGATED_FEATURES,
    FEATURE_NAMES,
    FEATURE_SPECS,
)


def _column(name: str) -> int:
    return FEATURE_NAMES.index(name)


def _matrix(rows: int = 3) -> np.ndarray:
    return np.zeros((rows, len(FEATURE_NAMES)), dtype=np.float32)


def test_fractional_positive_indicators_remain_fractional() -> None:
    x = _matrix()
    x[:, _column("ARP")] = [0.0, 0.2, 1.0]
    x[:, _column("syn_flag_number")] = [0.0, 0.06, 1.0]
    cleaned, upper = clean_features(x.copy(), np.array([True, True, False]), "none")
    assert upper is None
    np.testing.assert_allclose(cleaned[:, _column("ARP")], [0.0, 0.2, 1.0])
    np.testing.assert_allclose(cleaned[:, _column("syn_flag_number")], [0.0, 0.06, 1.0])


def test_fractional_count_values_remain_fractional() -> None:
    x = _matrix()
    x[:, _column("syn_count")] = [0.0, 0.06, 12.87]
    x[:, _column("Number")] = [9.5, 10.0, 15.0]
    cleaned, _ = clean_features(x.copy(), np.array([True, True, False]), "none")
    np.testing.assert_allclose(cleaned[:, _column("syn_count")], [0.0, 0.06, 12.87])
    np.testing.assert_allclose(cleaned[:, _column("Number")], [9.5, 10.0, 15.0])


def test_percentile_clipping_uses_training_only_and_does_not_round() -> None:
    x = _matrix(4)
    x[:, _column("ARP")] = [0.0, 0.2, 0.4, 1.0]
    x[:, _column("Rate")] = [1.0, 2.0, 3.0, 1000.0]
    train = np.array([True, True, True, False])
    cleaned, upper = clean_features(x.copy(), train, "train_percentile", 100.0)
    assert upper is not None
    assert cleaned[1, _column("ARP")] == pytest.approx(0.2)
    assert cleaned[3, _column("ARP")] == pytest.approx(0.4)
    assert cleaned[3, _column("Rate")] == pytest.approx(3.0)


def test_clip_round_name_has_no_rounding_semantics() -> None:
    x = _matrix(1)
    x[0, _column("ack_count")] = 0.49
    x[0, _column("HTTP")] = 0.2
    upper = np.full(len(FEATURE_NAMES), 10.0, dtype=np.float32)
    upper[_column("HTTP")] = 1.0
    result = clip_round(x, upper)
    assert result[0, _column("ack_count")] == pytest.approx(0.49)
    assert result[0, _column("HTTP")] == pytest.approx(0.2)


def test_scaler_fit_uses_natural_train_only() -> None:
    x = _matrix(5)
    x[:, _column("Rate")] = [0.0, 1.0, 2.0, 10_000.0, 20_000.0]
    train = np.array([True, True, True, False, False])
    scaler = fit_scaler(x, train)
    assert scaler.center_[_column("Rate")] == pytest.approx(1.0)
    assert scaler.center_[_column("Rate")] != pytest.approx(np.median(x[:, _column("Rate")]))


def test_sampler_never_returns_validation_or_test_indices() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(12, len(FEATURE_NAMES))).astype(np.float32)
    split = np.array([0] * 6 + [1] * 3 + [2] * 3, dtype=np.int8)
    category = np.array(["Web"] * 12, dtype=object)
    scaler = fit_scaler(x, split == 0)
    kept, _ = sample_train(x, scaler, split, category)
    assert np.array_equal(kept, np.arange(6))
    assert np.all(split[kept] == 0)


def test_feature_order_and_metadata_are_frozen() -> None:
    assert len(FEATURE_NAMES) == 39
    assert tuple(spec.name for spec in FEATURE_SPECS) == tuple(FEATURE_NAMES)
    assert FEATURE_NAMES[:4] == ["Header_Length", "Protocol Type", "Time_To_Live", "Rate"]
    assert FEATURE_NAMES[-3:] == ["IAT", "Number", "Variance"]
    assert CLUSTER_FEATURES == FEATURE_NAMES


def test_bounded_aggregate_metadata_has_structural_unit_interval() -> None:
    by_name = {spec.name: spec for spec in FEATURE_SPECS}
    for name in BOUNDED_AGGREGATED_FEATURES:
        assert by_name[name].representation_type == "bounded_aggregated_indicator"
        assert by_name[name].expected_min == 0.0
        assert by_name[name].expected_max == 1.0


def test_nonfinite_source_is_rejected() -> None:
    x = _matrix(2)
    x[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        clean_features(x, np.array([True, False]), "none")
