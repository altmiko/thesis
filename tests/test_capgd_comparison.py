from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas  # Load pyarrow/pandas before torch on Windows to avoid DLL teardown faults.
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _path in (str(REPO_ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from comparisons.capgd_cicids2017 import (  # noqa: E402
    RawCICIDSVictim,
    build_capgd_resources,
    evaluate_capgd_output,
    finalize_capgd_output,
    fit_train_minmax,
)


@pytest.fixture(scope="module")
def resources():
    return build_capgd_resources(REPO_ROOT)


def test_fit_train_minmax_uses_every_row_without_loading_holdouts(tmp_path: Path) -> None:
    values = np.asarray(
        [[4.0, -2.0] + [0.0] * 77, [1.0, 8.0] + [0.0] * 77, [3.0, 5.0] + [0.0] * 77],
        dtype=np.float32,
    )
    path = tmp_path / "X_train_pristine.npy"
    np.save(path, values)
    low, high = fit_train_minmax(path, chunk_rows=1)
    assert low[:2].tolist() == [1.0, -2.0]
    assert high[:2].tolist() == [4.0, 8.0]


def test_raw_victim_applies_existing_robust_transform() -> None:
    victim = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        victim.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]))
    wrapper = RawCICIDSVictim(
        victim,
        center=np.asarray([10.0, 20.0, 30.0], np.float32),
        scale=np.asarray([2.0, 4.0, 5.0], np.float32),
    )
    raw = torch.tensor([[12.0, 24.0, 35.0]])
    expected = victim(torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.allclose(wrapper(raw), expected)


def test_final_projection_restores_frozen_and_recomputes_dependencies(resources) -> None:
    processed = REPO_ROOT / "data/processed/CICIDS_2017_Distrinet"
    clean_np = np.asarray(np.load(processed / "X_test_pristine.npy", mmap_mode="r")[:2]).copy()
    clean = torch.as_tensor(clean_np)
    candidate = clean.clone()
    frozen_index = resources.resolved_mask.frozen_idx[0]
    candidate[:, frozen_index] += 123.0
    direct_index = resources.resolved_mask.perturbable_idx[1]
    candidate[:, direct_index] += 10.0

    projected = finalize_capgd_output(resources, clean, candidate)
    assert torch.equal(projected[:, frozen_index], clean[:, frozen_index])
    assert not bool(resources.resolved_mask.derived_consistency_mask(projected).any())
    assert not bool(resources.resolved_mask.frozen_violation_mask(projected, clean).any())


def test_clean_rows_pass_capgd_and_validator_constraints(resources) -> None:
    processed = REPO_ROOT / "data/processed/CICIDS_2017_Distrinet"
    clean = np.asarray(np.load(processed / "X_test_pristine.npy", mmap_mode="r")[:8]).copy()
    result = evaluate_capgd_output(resources, clean, clean, norm="L2", eps=0.5)
    assert np.asarray(result["internal_constraint_valid"], bool).all()
    assert np.asarray(result["distance_ok"], bool).all()
    assert np.asarray(result["hybrid_valid"], bool).all()
    assert np.allclose(result["distance"], 0.0)
