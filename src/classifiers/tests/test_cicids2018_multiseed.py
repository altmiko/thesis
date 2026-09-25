"""Behavioral tests for multi-seed classifier metric aggregation."""
from __future__ import annotations

import numpy as np

from scripts.train_cicids2018_classifiers_multiseed import (
    aggregate_class_rows,
    aggregate_rows,
    mean_std,
)


def test_mean_std_uses_sample_standard_deviation():
    mean, std = mean_std([0.8, 0.9, 1.0])
    assert mean == 0.9
    assert np.isclose(std, 0.1)


def test_aggregate_keeps_model_task_split_separate_and_averages_seeds():
    rows = []
    for seed, value in ((42, 0.8), (123, 0.9), (2024, 1.0)):
        rows.append(
            {
                "seed": seed,
                "model": "mlp",
                "display_name": "SimpleMLP",
                "task": "category",
                "split": "test",
                "n": 100,
                "training_seconds": float(seed),
                **{metric: value for metric in (
                    "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "weighted_precision", "weighted_recall", "weighted_f1",
                )},
            }
        )
    aggregate = aggregate_rows(rows)
    assert len(aggregate) == 1
    assert aggregate[0]["n_seeds"] == 3
    assert aggregate[0]["n_rows_per_seed"] == 100
    assert np.isclose(aggregate[0]["macro_f1_mean"], 0.9)
    assert np.isclose(aggregate[0]["macro_f1_sample_std"], 0.1)


def test_per_class_aggregation_reports_precision_recall_f1_variability():
    rows = [
        {
            "seed": seed,
            "model": "cnn",
            "display_name": "CNNOnly",
            "task": "binary",
            "split": "test",
            "class": "Attack",
            "support": 70,
            "precision": value,
            "recall": value - 0.1,
            "f1": value - 0.05,
        }
        for seed, value in ((42, 0.8), (123, 0.9), (2024, 1.0))
    ]
    aggregate = aggregate_class_rows(rows)
    assert len(aggregate) == 1
    assert aggregate[0]["support_per_seed"] == 70
    assert np.isclose(aggregate[0]["precision_mean"], 0.9)
    assert np.isclose(aggregate[0]["recall_mean"], 0.8)
    assert np.isclose(aggregate[0]["f1_sample_std"], 0.1)
