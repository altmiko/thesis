from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.paired_validity_gap import (
    InputSpec,
    align_by_row_id,
    contingency,
    holm_adjust,
    load_inputs,
    mcnemar_test,
    newcombe_paired_ci,
)


def _rows(raw: list[bool], valid_mask: list[bool]) -> list[dict[str, bool]]:
    assert len(raw) == len(valid_mask)
    return [
        {
            "eligible": True,
            "raw_success": r,
            "valid_success": r and v,
            "untargeted_raw_success": r,
            "untargeted_valid_success": r and v,
        }
        for r, v in zip(raw, valid_mask)
    ]


def test_all_samples_valid_has_zero_gap() -> None:
    result = contingency(_rows([True, False, True, False], [True] * 4))
    assert result["delta_asr"] == 0.0
    assert result["b"] == 0
    assert result["c"] == 0
    assert result["raw_asr"] == result["valid_asr"] == 0.5


def test_all_successful_attacks_invalid() -> None:
    result = contingency(_rows([True, False, True, True], [False] * 4))
    assert result["valid_asr"] == 0.0
    assert result["b"] == 3
    assert result["raw_asr"] == 0.75


def test_valid_success_must_be_subset_of_raw_success() -> None:
    rows = _rows([False], [True])
    rows[0]["valid_success"] = True
    with pytest.raises(AssertionError, match="c=1"):
        contingency(rows)


def test_exact_mcnemar_small_manual_table() -> None:
    result = mcnemar_test(4, 0)
    assert result["test_variant"] == "exact binomial McNemar"
    assert result["test_statistic"] is None
    assert result["p_value"] == pytest.approx(0.125)


def test_holm_adjustment_matches_hand_calculation() -> None:
    # Sorted p-values: .01, .03, .04 -> raw multipliers .03, .06, .04;
    # monotonic step-down adjusted values: .03, .06, .06.
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_newcombe_paired_ci_matches_published_reference() -> None:
    # Lydersen (2022), Table 1: a=1, b=4, c=0, d=11.
    # The cited Newcombe interval is -0.015 to 0.494 after rounding.
    low, high = newcombe_paired_ci(1, 4, 0, 11)
    assert low == pytest.approx(-0.014684852759915712)
    assert high == pytest.approx(0.4935117573531926)


def test_row_order_permutation_is_alignment_invariant() -> None:
    left, right, ids = align_by_row_id(
        ["r1", "r2", "r3"], np.array([1, 0, 1]),
        ["r3", "r1", "r2"], np.array([1, 1, 0]),
    )
    assert ids == ["r1", "r2", "r3"]
    assert np.array_equal(left, right)


def test_mismatched_sample_ids_abort() -> None:
    with pytest.raises(ValueError, match="mismatched sample IDs"):
        align_by_row_id(
            ["r1", "r2"], np.array([1, 0]),
            ["r1", "r3"], np.array([1, 0]),
        )


def _write_minimal_artifact(path: Path) -> None:
    n = 2
    np.savez_compressed(
        path,
        X_clean_raw=np.zeros((n, 3), dtype=np.float32),
        X_adv_raw=np.ones((n, 3), dtype=np.float32),
        row_id=np.array(["r1", "r2"]),
        dataset=np.array("toy_dataset"),
        method=np.array("toy_method"),
        class_name=np.array("DoS"),
        victim=np.array("mlp"),
        seed=np.array(42),
        true_label=np.ones(n, dtype=np.int64),
        clean_prediction=np.ones(n, dtype=np.int64),
        final_adversarial_prediction=np.zeros(n, dtype=np.int64),
        clean_correct=np.ones(n, dtype=bool),
        benign=np.ones(n, dtype=bool),
        evasion=np.ones(n, dtype=bool),
        target_success_flag=np.ones(n, dtype=bool),
        pave_valid=np.ones(n, dtype=bool),
        mined_valid=np.ones(n, dtype=bool),
        dep_ok=np.ones(n, dtype=bool),
        packet_ok=np.ones(n, dtype=bool),
        timing_ok=np.ones(n, dtype=bool),
        rate_ok=np.ones(n, dtype=bool),
        disc_ok=np.ones(n, dtype=bool),
        frozen_ok=np.ones(n, dtype=bool),
        realizable=np.ones(n, dtype=bool),
        strict_valid=np.ones(n, dtype=bool),
    )


def test_duplicated_seed_sample_rows_are_detected(tmp_path: Path) -> None:
    attack_dir = tmp_path / "attack_artifacts"
    attack_dir.mkdir()
    artifact = attack_dir / "DoS_mlp_seed42.npz"
    _write_minimal_artifact(artifact)
    (tmp_path / "attack_results.json").write_text(
        json.dumps({
            "dataset": "toy_dataset",
            "method_id": "toy_method",
            "cells": [{"artifact": str(artifact)}, {"artifact": str(artifact)}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate seed/sample artifact row"):
        load_inputs([InputSpec("Toy", tmp_path)])
