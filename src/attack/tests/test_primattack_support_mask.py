from __future__ import annotations

import numpy as np
import pytest
import torch

from attack.realizability.cicids2017 import (
    CICIDS2017PrimitiveModel,
    primattack_feature_support,
    primattack_joint_feature_mask,
    primattack_padding_feature_mask,
    primattack_timing_feature_mask,
)
from datasets import get_adapter


EXPECTED_INDEX0 = (
    3, 6, 8, 9, 10, 16, 17, 18, 20, 22, 23, 24, 25, 26, 38, 39,
    40, 41, 42, 43, 44, 54, 55,
)


@pytest.mark.parametrize("dataset", ["cicids2017", "cicids2018"])
def test_primattack_joint_mask_resolves_frozen_feature_order(dataset: str) -> None:
    manifest = get_adapter(dataset).feature_manifest()
    support = primattack_feature_support(manifest)
    mask = primattack_joint_feature_mask(manifest)

    assert tuple(torch.nonzero(mask, as_tuple=False).flatten().tolist()) == EXPECTED_INDEX0
    assert tuple(entry.index for entry in support) == EXPECTED_INDEX0
    assert all(entry.name == manifest.names[entry.index] for entry in support)
    assert all(entry.recomputation for entry in support)
    assert int(primattack_padding_feature_mask(manifest).sum()) == 12
    assert int(primattack_timing_feature_mask(manifest).sum()) == 12
    assert int(mask.sum()) == 23


def test_representative_primitive_recomputation_equals_declared_support() -> None:
    adapter = get_adapter("cicids2017")
    manifest = adapter.feature_manifest()
    model = CICIDS2017PrimitiveModel(manifest)
    # Strided over the whole test split: padding now needs flows without an empty forward
    # packet, which the first rows of the split do not cover for every declared coordinate.
    x_all = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    raw_np = np.asarray(x_all[np.linspace(0, len(x_all) - 1, 40000).astype(np.int64)]).copy()
    raw = torch.as_tensor(raw_np)
    caps = model.infer_capabilities(raw)
    zero = torch.zeros(len(raw), dtype=raw.dtype)

    padding = torch.where(caps.pad_allowed, torch.full_like(zero, 17.0), zero)
    delayed = torch.where(caps.timing_allowed, torch.full_like(zero, 1009.0), zero)
    padding_adv = model.generate(
        raw,
        {"p": padding, "delay": zero, "shape": zero},
        quantize=True,
        capabilities=caps,
    )
    timing_adv = model.generate(
        raw,
        {"p": zero, "delay": delayed, "shape": torch.full_like(zero, 0.37)},
        quantize=True,
        capabilities=caps,
    )

    observed_padding = (padding_adv != raw).any(dim=0)
    observed_timing = (timing_adv != raw).any(dim=0)
    expected_padding = primattack_padding_feature_mask(manifest)
    expected_timing = primattack_timing_feature_mask(manifest)

    # Equality proves both directions: no write escapes the mask and every declared
    # coordinate is demonstrably sensitive on representative feasible flows.
    assert torch.equal(observed_padding, expected_padding)
    assert torch.equal(observed_timing, expected_timing)
    assert model.controlled_idx == list(EXPECTED_INDEX0)
    assert torch.equal(observed_padding | observed_timing, primattack_joint_feature_mask(manifest))
