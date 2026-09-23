"""Interpretable hybrid validity framework (validator_v2).

A dataset-agnostic engine + per-dataset profiles for validating (adversarial)
network-flow feature vectors. Structural validity (SCHEMA / MINED / EXTRACTOR /
PROTOCOL rules) is kept strictly separate from distributional plausibility.

Public entrypoints::

    from validation import load_validator
    v = load_validator("cicids2017_distrinet")
    result = v.validate(x)              # single (F,) raw feature vector
    results = v.validate_batch(X)       # (N, F) raw feature matrix

See ``docs/validator_v2_methodology.md``.
"""
from __future__ import annotations

from pathlib import Path

from validation.validator.engine import Validator, load_validator

__all__ = ["Validator", "load_validator", "PACKAGE_ROOT", "RULES_DIR", "SCHEMA_DIR", "REPORTS_DIR"]

PACKAGE_ROOT: Path = Path(__file__).resolve().parent
SCHEMA_DIR: Path = PACKAGE_ROOT / "schema"
RULES_DIR: Path = PACKAGE_ROOT / "rules"
REPORTS_DIR: Path = PACKAGE_ROOT / "reports"
