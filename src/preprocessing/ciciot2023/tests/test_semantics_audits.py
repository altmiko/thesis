"""Synthetic contracts for duplicate, feature-selection, and corrected VAE semantics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from src.preprocessing.ciciot2023.feature_analysis import _selection
from src.preprocessing.ciciot2023.feature_audit import run_duplicate_audit
from src.preprocessing.schema import BOUNDED_AGGREGATED_IDX, FEATURE_NAMES
from src.vae.dataset import PerClassDataset
from src.vae.model import MixedInputBetaVAE
from src.vae.schema import get_partition


def test_duplicate_audit_identifies_overlap_and_cross_label(tmp_path) -> None:
    x = np.zeros((7, len(FEATURE_NAMES)), dtype=np.float32)
    x[0, 0] = 1.0  # train A
    x[1, 0] = 1.0  # train duplicate A
    x[2, 0] = 2.0  # train B
    x[3, 0] = 1.0  # val same vector/same label
    x[4, 0] = 3.0  # val novel
    x[5, 0] = 2.0  # test vector seen in train, different label
    x[6, 0] = 4.0  # test novel
    split = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int8)
    labels = np.array([0, 0, 1, 0, 2, 2, 3], dtype=np.int32)

    report = run_duplicate_audit(x, split, labels, tmp_path, chunk_size=2)

    assert report["splits"]["train"]["duplicate_rows"] == 1
    assert report["intersections"]["train_val"]["same_label_overlap_rows"] == 1
    assert report["intersections"]["train_test"]["cross_label_overlap_rows"] == 1
    np.testing.assert_array_equal(np.load(tmp_path / "audits" / "novel_val_indices.npy"), [1])
    np.testing.assert_array_equal(np.load(tmp_path / "audits" / "novel_test_indices.npy"), [1])


def test_feature_selection_is_deterministic_and_preserves_schema_order() -> None:
    n = len(FEATURE_NAMES)
    spearman = np.eye(n)
    spearman[0, 30] = spearman[30, 0] = 0.97
    variability = pd.DataFrame({
        "category": ["ALL"] * n,
        "feature": FEATURE_NAMES,
        "near_constant": [False] * n,
    })
    importance = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mutual_information_coarse_8class": np.linspace(1.0, 0.0, n),
    })

    first, dropped_first = _selection(spearman, 0.95, variability, importance)
    second, dropped_second = _selection(spearman, 0.95, variability, importance)

    assert first == second
    assert dropped_first == dropped_second
    assert first == [feature for feature in FEATURE_NAMES if feature != "Tot sum"]
    assert dropped_first[0]["full_domain_schema_status"] == "retained"


def test_vae_partition_has_no_binary_or_categorical_protocol_targets() -> None:
    partition = get_partition()
    assert partition["continuous_idx"] == list(range(39))
    assert partition["independent_binary_idx"] == []
    assert partition["derived_binary_idx"] == []
    assert partition["protocol_idx"] == [1]


def test_vae_dataset_preserves_fractional_aggregates() -> None:
    raw = np.zeros((2, 39), dtype=np.float32)
    raw[:, 0] = [1.0, 2.0]
    raw[0, FEATURE_NAMES.index("HTTP")] = 0.2
    raw[0, FEATURE_NAMES.index("syn_count")] = 0.06
    scaler = RobustScaler().fit(raw)
    scaled = scaler.transform(raw).astype(np.float32)
    dataset = PerClassDataset(scaled, np.array([0, 0]), 0, scaler, get_partition())
    restored = scaler.inverse_transform(dataset.x_scaled.numpy())
    assert restored[0, FEATURE_NAMES.index("HTTP")] == 0.2
    assert restored[0, FEATURE_NAMES.index("syn_count")] == 0.06
    assert dataset.target_independent_binary.shape == (2, 0)


def test_vae_decoder_keeps_aggregate_indicators_continuous() -> None:
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(100, 39)).astype(np.float32)
    raw[:, list(BOUNDED_AGGREGATED_IDX)] = rng.uniform(0.0, 1.0, size=(100, len(BOUNDED_AGGREGATED_IDX)))
    scaler = RobustScaler().fit(raw)
    model = MixedInputBetaVAE(get_partition(), latent_dim=4, encoder_hidden=(16, 8), decoder_hidden=(8, 16))
    model.register_protocol_references(scaler)
    model.eval()
    decoded_scaled, _ = model.decode_to_39(torch.zeros((3, 4)), mode="hard")
    decoded_raw = scaler.inverse_transform(decoded_scaled.detach().numpy())
    bounded = decoded_raw[:, list(BOUNDED_AGGREGATED_IDX)]
    assert np.all((bounded >= 0.0) & (bounded <= 1.0))
    assert np.any((bounded > 0.0) & (bounded < 1.0))
