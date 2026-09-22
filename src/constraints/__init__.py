"""Layered constraint architecture (generation P0/C1/C2 + evaluation validators)."""
from __future__ import annotations

from constraints.base import Constraint, ConstraintReport
from constraints.engine import ConstraintEngine, build_engine, parse_layers
from constraints.layer0 import Layer0Projector
from constraints.layer1 import (
    HalfRangeBound,
    MonotoneNondecreasing,
    ProductEquality,
    RobustTailBound,
)
from constraints.layer2 import dump_layer2, load_layer2
from constraints.registry import CONSTRAINT_REGISTRY, build_constraint, register_constraint

__all__ = [
    "Constraint",
    "ConstraintReport",
    "ConstraintEngine",
    "build_engine",
    "parse_layers",
    "Layer0Projector",
    "RobustTailBound",
    "ProductEquality",
    "MonotoneNondecreasing",
    "HalfRangeBound",
    "load_layer2",
    "dump_layer2",
    "CONSTRAINT_REGISTRY",
    "build_constraint",
    "register_constraint",
]
