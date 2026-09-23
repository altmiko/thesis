"""Fail-loud identity checks for artifacts used by the CICIDS2017 attack experiments."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from datasets.cicids2017 import CICIDS2017Adapter
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import load_stage_a


ATTACK_CLASSES = {"DoS": 1, "DDoS": 2, "Recon": 3, "BruteForce": 4}
VICTIMS = ("mlp", "cnn", "lstm", "serial")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_schema_scaler_and_split_widths_match_exactly():
    adapter = CICIDS2017Adapter(REPO)
    manifest = adapter.feature_manifest()
    assert manifest.n_features == 79
    assert len(manifest.names) == len(set(manifest.names))

    with (adapter._processed / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    assert int(scaler.n_features_in_) == manifest.n_features
    assert len(scaler.center_) == manifest.n_features
    assert len(scaler.scale_) == manifest.n_features
    for split in ("train", "val", "test"):
        scaled = np.load(adapter._processed / f"X_{split}.npy", mmap_mode="r")
        pristine = np.load(adapter._processed / f"X_{split}_pristine.npy", mmap_mode="r")
        assert scaled.shape == pristine.shape
        assert scaled.shape[1] == manifest.n_features


def test_classifier_run_manifest_matches_current_preprocessing_manifest():
    run = json.loads(
        (REPO / "outputs/cicids2017distrinet/classifier_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    preprocessing = REPO / "data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json"
    assert run["preprocessing_manifest_sha256"] == _sha256(preprocessing)
    assert run["features"] == 79
    assert run["heads"]["category"]["classes"] == [
        "Benign", "DoS", "DDoS", "Recon", "BruteForce"
    ]


def test_all_victim_checkpoint_dimensions_match_experiment():
    model_dir = REPO / "outputs/cicids2017distrinet/models"
    for victim in VICTIMS:
        checkpoint = torch.load(
            model_dir / f"{victim}_category.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["model_type"] == victim
        assert int(checkpoint["num_features"]) == 79
        assert int(checkpoint["num_classes"]) == 5
        assert "state_dict" in checkpoint


def test_all_vae_checkpoint_classes_and_schema_match_experiment():
    adapter = CICIDS2017Adapter(REPO)
    manifest_hash = adapter.feature_manifest().content_hash
    checkpoint_dir = REPO / "outputs/cicids2017_vae_stage_a"
    for class_name, class_id in ATTACK_CLASSES.items():
        checkpoint = torch.load(
            checkpoint_dir / f"vae_{class_name}.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["class_name"] == class_name
        assert int(checkpoint["class_id"]) == class_id
        assert checkpoint["manifest_hash"] == manifest_hash
        assert int(checkpoint["model_config"]["latent_dim"]) == 16
        assert float(checkpoint["training_config"]["beta_target"]) == 0.5


def test_loaders_reject_wrong_model_and_class_identity():
    adapter = CICIDS2017Adapter(REPO)
    with pytest.raises(ValueError, match="expected model_type"):
        load_category_victim(
            REPO / "outputs/cicids2017distrinet/models/mlp_category.pt",
            adapter=adapter,
            expected_model_type="cnn",
        )
    with pytest.raises(ValueError, match="expected class"):
        load_stage_a(
            adapter,
            REPO / "outputs/cicids2017_vae_stage_a/vae_DoS.pt",
            expected_class_name="DDoS",
        )
