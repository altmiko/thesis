"""Dataset access for mining -- enforces TRAIN/VAL/TEST separation in code.

The miner discovers rules on TRAIN, confirms on VAL, and MUST NEVER read TEST
during discovery/threshold/tolerance selection. This module exposes ``train``
and ``val`` freely but guards ``test`` behind an explicit
``allow_test_for_final_reporting=True`` flag so accidental leakage is impossible
to write by mistake.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CICIDS_DIR = REPO_ROOT / "data" / "processed" / "CICIDS_2017_Distrinet"


def feature_names(data_dir: Path = CICIDS_DIR) -> list[str]:
    man = json.loads((data_dir / "preprocessing_manifest.json").read_text())
    return list(man["modelling_feature_names"])


def dataset_name(data_dir: Path = CICIDS_DIR) -> str:
    # Profile key for validator_v2 (schema/rules/reports dirs). Distinct from the
    # legacy artifact path old_constraints/cicids2017_distrinet/mined.json.
    return "cicids2017_distrinet"


def load_split(split: str, data_dir: Path = CICIDS_DIR, mmap: bool = True,
               allow_test_for_final_reporting: bool = False) -> np.ndarray:
    """Load a raw (pristine, un-scaled) feature matrix for a split.

    ``split`` in {"train","val","test"}. Reading ``test`` requires the explicit
    override flag to prevent leakage during mining.
    """
    if split == "test" and not allow_test_for_final_reporting:
        raise PermissionError(
            "TEST split is off-limits during mining (rule discovery / threshold / "
            "tolerance selection). Pass allow_test_for_final_reporting=True only "
            "for final evaluation, never for mining.")
    fname = {"train": "X_train_pristine.npy", "val": "X_val_pristine.npy",
             "test": "X_test_pristine.npy"}[split]
    return np.load(data_dir / fname, mmap_mode="r" if mmap else None)


def load_labels(split: str, data_dir: Path = CICIDS_DIR,
                allow_test_for_final_reporting: bool = False) -> np.ndarray:
    if split == "test" and not allow_test_for_final_reporting:
        raise PermissionError("TEST labels off-limits during mining.")
    return np.load(data_dir / f"y_{split}_cat.npy")


def sample_rows(X: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Deterministic row subsample as a materialized float64 array."""
    total = X.shape[0]
    if n >= total:
        return np.asarray(X, np.float64)
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(total, size=n, replace=False))
    return np.asarray(X[sel], np.float64)
