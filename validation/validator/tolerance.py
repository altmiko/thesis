"""Centralized floating-point tolerance handling.

Every approximate comparison in the framework flows through :func:`is_close`.
Tolerances are never scattered as bare literals in rule code; each approximate
rule carries an explicit :class:`Tolerance` (absolute + relative) that is stored
in its serialized form and rendered in the human-readable reports.

Closeness convention (numpy-style, asymmetric on ``expected``)::

    |observed - expected| <= absolute + relative * |expected|
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Smallest numerical floor a mined tolerance may collapse to. Prevents a
# perfectly-clean training residual (all zeros) from producing an absurdly
# tight rule that rejects genuine data on trivial float noise.
ABS_FLOOR: float = 1e-6
REL_FLOOR: float = 1e-6

# Caps: a mined tolerance derived from a residual distribution is never allowed
# to grow beyond these, otherwise the rule becomes vacuous ("A within a mile of B").
ABS_CAP: float = 1e9
REL_CAP: float = 0.10


@dataclass(frozen=True)
class Tolerance:
    """Absolute + relative tolerance for an approximate relation."""

    absolute: float = ABS_FLOOR
    relative: float = REL_FLOOR

    def to_dict(self) -> dict:
        return {"absolute": float(self.absolute), "relative": float(self.relative)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Tolerance":
        if not d:
            return cls()
        return cls(absolute=float(d.get("absolute", ABS_FLOOR)),
                   relative=float(d.get("relative", REL_FLOOR)))


def is_close(observed, expected, abs_tol: float = ABS_FLOOR, rel_tol: float = REL_FLOOR) -> np.ndarray:
    """Element-wise closeness mask: ``|obs - exp| <= abs_tol + rel_tol*|exp|``."""
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return np.abs(observed - expected) <= (abs_tol + rel_tol * np.abs(expected))


def abs_error(observed, expected) -> np.ndarray:
    return np.abs(np.asarray(observed, np.float64) - np.asarray(expected, np.float64))


def rel_error(observed, expected) -> np.ndarray:
    observed = np.asarray(observed, np.float64)
    expected = np.asarray(expected, np.float64)
    return np.abs(observed - expected) / (np.abs(expected) + ABS_FLOOR)


def suggest_tolerance(residual: np.ndarray, expected: np.ndarray,
                      percentile: float = 99.9) -> Tolerance:
    """Derive an approximate-rule tolerance from a TRAIN residual distribution.

    ``candidate = max(numerical_floor, high_percentile(|residual|))`` then capped.
    Returns both an absolute and a relative component so downstream rules keep the
    numpy closeness semantics. The relative part is estimated from the residual
    normalized by ``|expected|`` (scale-free), the absolute part from raw residual.
    """
    residual = np.abs(np.asarray(residual, np.float64))
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return Tolerance(ABS_FLOOR, REL_FLOOR)
    abs_c = float(np.percentile(residual, percentile))
    abs_c = min(max(abs_c, ABS_FLOOR), ABS_CAP)

    # relative component: robust ratio of residual to expected magnitude (+1 guards
    # near-zero expected values so tiny expecteds don't blow the relative term up).
    exp_full = np.abs(np.asarray(expected, np.float64))[: residual.size]
    ratio = residual / (exp_full + 1.0)
    rel_c = float(np.percentile(ratio, percentile)) if ratio.size else REL_FLOOR
    rel_c = min(max(rel_c, REL_FLOOR), REL_CAP)
    return Tolerance(abs_c, rel_c)
