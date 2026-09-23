"""Thesis metrics (§21): validity-aware attack success and validator quality.

All ASR variants share ONE denominator -- the number of eligible clean-correct
malicious samples -- so they are directly comparable (the denominator is never
silently changed between metrics).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def clean_acceptance_rate(valid_mask: np.ndarray) -> float:
    """CAR: fraction of genuine clean samples the validator accepts."""
    v = np.asarray(valid_mask, bool)
    return float(v.mean()) if v.size else 0.0


def known_invalid_detection_rate(valid_mask_on_corrupted: np.ndarray) -> float:
    """KIDR: fraction of known-invalid samples the validator rejects."""
    v = np.asarray(valid_mask_on_corrupted, bool)
    return float((~v).mean()) if v.size else 0.0


@dataclass
class TargetedASR:
    eligible: int
    raw: float
    hard_valid: float
    hybrid_valid: float
    valid_in_distribution: float

    def to_dict(self) -> dict:
        return asdict(self)


def targeted_asr_suite(evasion: np.ndarray, hard_valid: np.ndarray,
                       hybrid_valid: np.ndarray, in_distribution: np.ndarray,
                       eligible: int | None = None) -> TargetedASR:
    """Raw / hard-valid / hybrid-valid / valid+in-distribution targeted ASR.

    ``evasion`` (target->Benign success), ``hard_valid``, ``hybrid_valid`` and
    ``in_distribution`` are per-sample boolean arrays over the eligible
    clean-correct malicious set. ``eligible`` overrides the denominator; by
    default it is the array length (i.e. the arrays already cover exactly the
    eligible set).
    """
    e = np.asarray(evasion, bool)
    hv = np.asarray(hard_valid, bool)
    yv = np.asarray(hybrid_valid, bool)
    idn = np.asarray(in_distribution, bool)
    denom = int(eligible) if eligible is not None else int(e.size)
    if denom == 0:
        return TargetedASR(0, 0.0, 0.0, 0.0, 0.0)
    return TargetedASR(
        eligible=denom,
        raw=float(e.sum() / denom),
        hard_valid=float((e & hv).sum() / denom),
        hybrid_valid=float((e & yv).sum() / denom),
        valid_in_distribution=float((e & yv & idn).sum() / denom),
    )
