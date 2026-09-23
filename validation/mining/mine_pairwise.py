"""Two-feature candidate templates (restricted, interpretable grammar).

Generates candidate MINED rules of the forms (§5):

    A ~= B            (equality)
    A <= B            (order)
    A ~= k * B        (scaled equality, small constants only)
    A ~= B^2          (square relation)

Candidates are enumerated **within a feature category** (guided by the registry)
to keep the search interpretable and bounded -- we do not fit arbitrary
coefficients, only the small constant set below.
"""
from __future__ import annotations

from itertools import permutations

from validation.validator.rule import Rule

SCALED_CONSTANTS = (2.0, 0.5, 10.0, 100.0, 1000.0, 1.0e6)


def _mk(rule_type: str, params: dict, features: list[str], name: str) -> Rule:
    return Rule(id="MINED_PENDING", name=name, source_type="MINED",
                rule_type=rule_type, params=params, features=features,
                hardness="EMPIRICAL", description="")


def generate(feature_order, category_of) -> list[Rule]:
    cands: list[Rule] = []
    # group by category
    groups: dict[str, list[str]] = {}
    for f in feature_order:
        groups.setdefault(category_of(f), []).append(f)

    for cat, feats in groups.items():
        for a, b in permutations(feats, 2):
            # equality A ~= B (one direction only; a<b lexicographically avoids the dup)
            if a < b:
                cands.append(_mk("equality", {"lhs": a, "rhs": b}, [a, b], f"eq::{a}=={b}"))
            # scaled A ~= k*B
            for k in SCALED_CONSTANTS:
                cands.append(_mk("scaled_equality", {"lhs": a, "rhs": b, "k": k},
                                 [a, b], f"scaled::{a}=={k}*{b}"))
            # square A ~= B^2
            cands.append(_mk("square_relation", {"lhs": a, "base": b}, [a, b], f"sq::{a}=={b}^2"))
    return cands

# NOTE: the bare order template ``A <= B`` was intentionally removed. Arbitrary
# same-category inequalities that merely happen to hold on the data are
# correlation, not a defensible structural invariant (methodology §18). Ordering
# is captured only where it is definitional -- the Min<=Mean<=Max statistical
# chains in ``mine_arithmetic.statistical_groups``.
