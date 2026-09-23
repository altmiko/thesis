"""Unary (single-feature) template tests.

These are the restricted unary grammar of §5: ``finite``, ``integer``,
``binary``, ``constant``, ``nonnegative``, and small ``categorical`` domains.
They power the SCHEMA inference stage (``infer_schema.py``) rather than emitting
empirical MINED rules -- a per-feature type fact is a schema/type constraint,
not a discovered cross-feature invariant.

IMPORTANT (§5, §18): an observed train min/max is NOT turned into a hard
validity bound here. Only exact type facts (integrality, a genuinely tiny value
set, a truly constant column) become schema constraints.
"""
from __future__ import annotations

import numpy as np

INT_SUPPORT = 0.99999      # fraction of rows that must be integral
BINARY_DOMAIN = {0.0, 1.0}
MAX_CATEGORICAL = 8        # a column with <= this many distinct values -> categorical


def frac_integral(col: np.ndarray, atol: float = 1e-6) -> float:
    return float(np.mean(np.abs(col - np.round(col)) <= atol))


def frac_nonnegative(col: np.ndarray, atol: float = 1e-9) -> float:
    return float(np.mean(col >= -atol))


def distinct_values(col: np.ndarray, cap: int = 64) -> np.ndarray:
    u = np.unique(col)
    return u[:cap]


def classify_feature(col: np.ndarray) -> dict:
    """Return the inferred type facts + evidence for one feature column."""
    col = np.asarray(col, np.float64)
    n = col.size
    finite = bool(np.all(np.isfinite(col)))
    u = np.unique(col[np.isfinite(col)])
    nuniq = int(u.size)
    fi = frac_integral(col)
    fnn = frac_nonnegative(col)

    ev = {"rows_tested": n, "n_distinct": nuniq, "frac_integral": fi,
          "frac_nonnegative": fnn, "observed_min": float(col.min()),
          "observed_max": float(col.max())}

    if nuniq == 1:
        value_type = "constant"
        constant_value = float(u[0])
        domain = None
    elif set(u.tolist()) <= BINARY_DOMAIN:
        value_type = "binary"
        constant_value = None
        domain = [0.0, 1.0]
    elif nuniq <= MAX_CATEGORICAL and fi >= INT_SUPPORT:
        value_type = "categorical"
        constant_value = None
        domain = [float(x) for x in u.tolist()]
    elif fi >= INT_SUPPORT:
        value_type = "integer"
        constant_value = None
        domain = None
    else:
        value_type = "numeric"
        constant_value = None
        domain = None

    return {"value_type": value_type, "finite": finite,
            "constant_value": constant_value, "domain": domain,
            "nonnegative": fnn >= INT_SUPPORT, "evidence": ev}
