"""Synthetic-violation detection regression (requires artifacts + data)."""
from __future__ import annotations

import pytest

from validation.mining import data_access as da

pytestmark = pytest.mark.skipif(
    not (da.CICIDS_DIR / "X_test_pristine.npy").exists(),
    reason="CICIDS2017 pristine data not available")


def test_known_invalid_detection_rate_high():
    from validation.evaluation.synthetic_violations import run
    res = run(n=10000)
    assert res["overall_kidr"] >= 0.99
    # gross corruptions must be caught essentially always
    for k in ("min_gt_max", "negative_count", "variance_neq_std_sq",
              "flow_pkts_inconsistent", "count_fractional"):
        assert res["per_corruption"][k] >= 0.999
