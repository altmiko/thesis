"""Clean-acceptance regression (requires mined artifacts + CICIDS2017 data)."""
from __future__ import annotations

import pytest

from validation.mining import data_access as da

pytestmark = pytest.mark.skipif(
    not (da.CICIDS_DIR / "X_test_pristine.npy").exists()
    or not (da.CICIDS_DIR / "preprocessing_manifest.json").exists(),
    reason="CICIDS2017 pristine data not available")


def test_hard_structural_accepts_genuine_data():
    from validation import load_validator
    v = load_validator("cicids2017_distrinet")  # requires schema/rules profiles present
    from validation.evaluation.clean_acceptance import run
    res = run(n=20000)
    # genuine held-out flows must not be routinely rejected by HARD rules
    assert res["car_hard"] >= 0.999
    assert res["car_hybrid"] >= 0.999


def test_plausibility_is_separate_from_validity():
    from validation import load_validator
    from validation.mining import data_access as da2
    v = load_validator("cicids2017_distrinet")
    X = da2.load_split("test", allow_test_for_final_reporting=True)
    Xs = da2.sample_rows(X, 5000, seed=1)
    b = v.validate_batch(Xs)
    # in_distribution can differ from structural validity without changing it
    assert b.hard_structural_valid.mean() >= 0.999
    # plausibility rate is allowed to be lower and must not force structural=False
    assert (b.in_distribution.mean() <= b.hard_structural_valid.mean() + 1e-9)
