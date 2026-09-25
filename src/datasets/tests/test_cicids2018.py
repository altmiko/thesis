"""CICIDS2018-DistriNet adapter contract tests against real processed artifacts."""
from __future__ import annotations

import numpy as np
import pytest

from datasets import get_adapter
from datasets.cicids2018 import CICIDS2018Adapter


def _adapter() -> CICIDS2018Adapter:
    adapter = CICIDS2018Adapter()
    if not (adapter._processed / "preprocessing_manifest.json").exists():
        pytest.skip("CICIDS2018 processed artifacts unavailable")
    return adapter


def test_training_values_respect_manifest_bounds():
    """Layer 0 clamps to [lower, upper]; a bound violated by real TRAIN data would corrupt it."""
    adapter = _adapter()
    manifest = adapter.feature_manifest()
    x = np.load(adapter._processed / "X_train_pristine.npy", mmap_mode="r")
    manifest.assert_matches_array(x)
    for spec in manifest.specs:
        column = np.asarray(x[:, spec.model_index])
        if spec.lower is not None:
            assert column.min() >= spec.lower, spec.name
        if spec.upper is not None:
            assert column.max() <= spec.upper, spec.name
    assert manifest["Fwd Header Length"].lower == -32768.0


def test_transform_reproduces_saved_scaled_split():
    adapter = _adapter()
    transform = adapter.feature_transform()
    pristine = np.load(adapter._processed / "X_val_pristine.npy", mmap_mode="r")[:50_000]
    scaled = np.load(adapter._processed / "X_val.npy", mmap_mode="r")[:50_000]
    assert np.allclose(transform.transform(np.asarray(pristine)), scaled, rtol=1e-5, atol=1e-4)
    split = adapter.load_split("test")
    assert split.x.shape[1] == 79 and len(split.y) == len(split.x)
    assert set(np.unique(split.y)) == {0, 1, 2, 3, 4}


def test_registry_returns_cicids2018_adapter():
    assert isinstance(get_adapter("cicids2018"), CICIDS2018Adapter)
    assert isinstance(get_adapter("cicids2018_distrinet"), CICIDS2018Adapter)
