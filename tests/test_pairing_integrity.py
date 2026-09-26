"""Runtime pairing-integrity guards for the full paired adversarial evaluation.

These fail if two attacks on the same (victim,class,seed) would use different row IDs, order,
labels, clean predictions, or counts, and check the McNemar helper on a known contingency.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_cell(art: Path, v, cl, atk, seed, sids, cid, clean_pred=None):
    n = len(sids)
    clean_pred = np.full(n, cid) if clean_pred is None else clean_pred
    np.savez_compressed(
        art / f"{v}__{cl}__{atk}__seed{seed}.npz",
        sample_id=np.asarray(sids, dtype="U128"),
        true_class=np.full(n, cid, dtype=np.int64),
        clean_pred=np.asarray(clean_pred, dtype=np.int64),
    )


def _fixture(tmp_path, sids_by_attack, cid=1, clean_pred_by_attack=None):
    out = tmp_path / "full_adv_eval"
    art = out / "artifacts"
    art.mkdir(parents=True)
    v, cl, seed = "mlp", "DoS", 42
    ref = list(sids_by_attack.values())[0]
    selection = {v: {cl: {"class_id": cid, "sample_ids": list(ref)}}}
    roster = [{"name": a} for a in sids_by_attack]
    for a, sids in sids_by_attack.items():
        cp = None if clean_pred_by_attack is None else clean_pred_by_attack.get(a)
        _write_cell(art, v, cl, a, seed, sids, cid, cp)
    return out, {v: {"attack_seeds": [seed]}}, [cl], roster, selection


def test_assert_pairing_passes_on_identical_rows(tmp_path):
    run = _load("rfae", "scripts/run_full_adversarial_eval.py")
    ids = [f"s{i}" for i in range(20)]
    args = _fixture(tmp_path, {"pgd_tb": ids, "prim_opt_joint_p75": ids})
    run.assert_pairing(*args)  # must not raise


def test_assert_pairing_detects_row_mismatch(tmp_path):
    run = _load("rfae", "scripts/run_full_adversarial_eval.py")
    ids = [f"s{i}" for i in range(20)]
    shuffled = ids[::-1]  # same set, different ORDER -> must fail
    args = _fixture(tmp_path, {"pgd_tb": ids, "prim_opt_joint_p75": shuffled})
    with pytest.raises(AssertionError):
        run.assert_pairing(*args)


def test_assert_pairing_detects_count_mismatch(tmp_path):
    run = _load("rfae", "scripts/run_full_adversarial_eval.py")
    ids = [f"s{i}" for i in range(20)]
    args = _fixture(tmp_path, {"pgd_tb": ids, "prim_opt_joint_p75": ids[:19]})
    with pytest.raises(AssertionError):
        run.assert_pairing(*args)


def test_assert_pairing_detects_clean_pred_desync(tmp_path):
    run = _load("rfae", "scripts/run_full_adversarial_eval.py")
    ids = [f"s{i}" for i in range(20)]
    cp_bad = np.full(20, 1); cp_bad[0] = 2  # not clean-correct -> must fail
    args = _fixture(tmp_path, {"pgd_tb": ids, "prim_opt_joint_p75": ids},
                    clean_pred_by_attack={"prim_opt_joint_p75": cp_bad})
    with pytest.raises(AssertionError):
        run.assert_pairing(*args)


def test_mcnemar_known_contingency():
    an = _load("afae", "scripts/analyze_full_adversarial_eval.py")
    a = np.array([1, 1, 1, 1, 0, 0], dtype=bool)  # A succeeds on 4
    b = np.array([1, 0, 0, 0, 1, 0], dtype=bool)  # B succeeds on 2
    r = an.mcnemar(a, b)
    assert r["n"] == 6
    assert r["n11"] == 1 and r["n10_A_not_B"] == 3 and r["n01_B_not_A"] == 1 and r["n00"] == 1
    assert abs(r["prop_diff_A_minus_B"] - (3 - 1) / 6) < 1e-9
    assert 0.0 <= r["p_value"] <= 1.0
