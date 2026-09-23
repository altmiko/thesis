"""Implication candidate templates (§5).

    A == 0  =>  B == 0
    A > 0   =>  B >= 0   (rarely interesting once PROTOCOL nonnegativity exists;
                          generated for completeness, usually pruned)

Only zero-implications between features of related families (same category, or a
count->length / count->rate pairing) are generated, to keep the grammar small
and interpretable. Acceptance additionally requires a meaningful antecedent
coverage (§5, §7): a rule that holds only because ``A == 0`` almost never occurs
is not retained.
"""
from __future__ import annotations

from itertools import permutations

from validation.validator.rule import Rule


def _mk(rule_type: str, params: dict, features: list[str], name: str) -> Rule:
    return Rule(id="MINED_PENDING", name=name, source_type="MINED",
                rule_type=rule_type, params=params, features=features,
                hardness="EMPIRICAL", description="")


def generate(feature_order, category_of) -> list[Rule]:
    cands: list[Rule] = []
    # ONLY defensible zero-implications: a direction with zero packets can carry
    # no bytes / no rate / no inter-arrival time in that direction. We do NOT
    # emit within-category permutation implications -- most of those hold only
    # because the antecedent rarely occurs and are not structural (methodology
    # §5, §18). Acceptance still requires meaningful antecedent coverage.
    counts = [f for f in feature_order if category_of(f) == "packet_count"]
    targets = [f for f in feature_order
               if category_of(f) in ("byte_count", "rate", "iat")]
    for a in counts:
        d = "Fwd" if "Fwd" in a else ("Bwd" if "Bwd" in a else None)
        if d is None:
            continue
        for b in targets:
            if d.lower() in b.lower() and a != b:
                cands.append(_mk("implication_zero", {"antecedent": a, "consequent": b},
                                 [a, b], f"impl0::{a}==0=>{b}==0"))
    return cands
