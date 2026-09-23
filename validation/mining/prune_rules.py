"""Deterministic rule pruning (§19).

Removes redundant/dominated candidates *after* acceptance, recording a
``why_pruned`` reason for every removal:

* an exact ``A ~= B`` equality dominates the corresponding ``A <= B`` / ``B <= A``;
* a ``Min <= Mean <= Max`` chain dominates its pairwise ``<=`` components;
* an EXTRACTOR-declared identity dominates an empirically-mined duplicate
  (the extractor version carries the stronger provenance);
* duplicate expressions collapse to one.
"""
from __future__ import annotations

from validation.validator.rule import Rule


def _sig(rule: Rule):
    """Canonical semantic signature for domination/equivalence comparison."""
    t, p = rule.rule_type, rule.params
    if t == "equality":
        return ("eq", frozenset((p["lhs"], p["rhs"])))
    if t == "scaled_equality":
        return ("scaled", p["lhs"], p["rhs"], float(p["k"]))
    if t == "square_relation":
        return ("sq", p["lhs"], p["base"])
    if t == "sum_equality":
        return ("sum", p["lhs"], frozenset(p["addends"]))
    if t == "difference_equality":
        return ("diff", p["lhs"], p["minuend"], p["subtrahend"])
    if t == "product_equality":
        return ("prod", p["lhs"], frozenset(p["factors"]))
    if t == "ratio_equality":
        return ("ratio", p["lhs"], p["numerator"], p["denominator"])
    if t == "monotone_chain":
        return ("chain", tuple(p["features"]))
    if t == "le":
        return ("le", p["lhs"], p["rhs"])
    if t == "ge":
        return ("ge", p["lhs"], p["rhs"])
    return (t, rule.expression())


def prune(accepted: list[Rule], extractor_rules: list[Rule] | None = None):
    """Return (kept, pruned) where pruned = list of (rule, why_pruned)."""
    extractor_rules = extractor_rules or []
    ext_sigs = {_sig(r) for r in extractor_rules}
    # extractor equality pairs (unordered) also dominate mined <=
    ext_eq_pairs = {s[1] for s in ext_sigs if s[0] == "eq"}

    kept: list[Rule] = []
    pruned: list[tuple[Rule, str]] = []
    seen_sigs: set = set()

    # equality/chain sets among accepted mined rules for domination of <=
    eq_pairs = {frozenset((r.params["lhs"], r.params["rhs"]))
                for r in accepted if r.rule_type == "equality"}
    chain_le_pairs: set[tuple[str, str]] = set()
    for r in accepted:
        if r.rule_type == "monotone_chain":
            fs = r.params["features"]
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    chain_le_pairs.add((fs[i], fs[j]))

    for r in accepted:
        sig = _sig(r)
        # 1. exact duplicate expression
        if sig in seen_sigs:
            pruned.append((r, "duplicate expression of an already-kept rule"))
            continue
        # 2. dominated by an EXTRACTOR identity
        if sig in ext_sigs:
            ext_id = next((e.id for e in extractor_rules if _sig(e) == sig), "EXTRACTOR")
            pruned.append((r, f"dominated by EXTRACTOR rule {ext_id} (same relation, stronger provenance)"))
            continue
        # 3. <= dominated by an equality (mined or extractor) on the same pair
        if r.rule_type == "le":
            pair = frozenset((r.params["lhs"], r.params["rhs"]))
            if pair in eq_pairs or pair in ext_eq_pairs:
                pruned.append((r, "dominated by an equality on the same feature pair"))
                continue
            if (r.params["lhs"], r.params["rhs"]) in chain_le_pairs:
                pruned.append((r, "dominated by a Min<=Mean<=Max monotone chain"))
                continue
        # 4. scaled equality dominated by a plain equality on the same pair
        if r.rule_type == "scaled_equality":
            pair = frozenset((r.params["lhs"], r.params["rhs"]))
            if pair in eq_pairs or pair in ext_eq_pairs:
                pruned.append((r, "dominated by a plain equality on the same feature pair"))
                continue
        seen_sigs.add(sig)
        kept.append(r)
    return kept, pruned
