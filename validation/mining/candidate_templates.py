"""Aggregate the restricted candidate grammar.

Combines the unary/pairwise/arithmetic/implication generators into a single
candidate list, and deduplicates identical expressions up front. This is the
whole hypothesis space the miner will ever test -- a small, human-readable set
of templates, NOT an open-ended symbolic-regression search (§5).
"""
from __future__ import annotations

from validation.mining import mine_arithmetic, mine_implications, mine_pairwise
from validation.validator.rule import Rule


def _dedup(rules: list[Rule]) -> list[Rule]:
    seen: dict[str, Rule] = {}
    for r in rules:
        key = f"{r.rule_type}|{r.expression()}"
        if key not in seen:
            seen[key] = r
    return list(seen.values())


def generate_candidates(feature_order: list[str], category_of, *,
                        include_implications: bool = True) -> dict[str, list[Rule]]:
    """Return candidates grouped by grammar family (for reporting counts)."""
    families = {
        "pairwise": mine_pairwise.generate(feature_order, category_of),
        "arithmetic": mine_arithmetic.generate(feature_order, category_of),
    }
    if include_implications:
        families["implication"] = mine_implications.generate(feature_order, category_of)
    return {k: _dedup(v) for k, v in families.items()}
