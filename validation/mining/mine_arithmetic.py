"""Three-feature arithmetic candidate templates (§5).

    A ~= B + C     (sum)
    A ~= B - C     (difference)
    A ~= B * C     (product)
    A ~= B / C     (ratio, division-by-zero handled by eligibility)

Also emits the statistical-ordering group templates:

    Min <= Mean <= Max      (monotone_chain)
    Variance ~= Std^2       (square_relation)

Candidates are generated with category guidance and simple name-family matching
(Fwd/Bwd -> combined) so the search stays interpretable and bounded rather than
scanning every unordered triple of 79 features.
"""
from __future__ import annotations

from itertools import combinations, permutations

from validation.validator.rule import Rule

STAT_TOKENS = ("Min", "Mean", "Max", "Std", "Variance", "Total", "Avg", "Average")


def _mk(rule_type: str, params: dict, features: list[str], name: str) -> Rule:
    return Rule(id="MINED_PENDING", name=name, source_type="MINED",
                rule_type=rule_type, params=params, features=features,
                hardness="EMPIRICAL", description="")


def _strip_token(name: str, token: str) -> str | None:
    """Return base name if ``name`` ends with ``token`` (space-delimited)."""
    if name == token:
        return ""
    if name.endswith(" " + token):
        return name[: -(len(token) + 1)].strip()
    return None


def statistical_groups(feature_order) -> list[Rule]:
    """Auto-detect Min/Mean/Max groups and Variance/Std pairs."""
    by_base: dict[str, dict[str, str]] = {}
    for f in feature_order:
        for tok in ("Min", "Mean", "Max", "Std", "Variance"):
            base = _strip_token(f, tok)
            if base is not None:
                by_base.setdefault(base, {})[tok] = f
    out: list[Rule] = []
    for base, d in by_base.items():
        if {"Min", "Mean", "Max"} <= set(d):
            feats = [d["Min"], d["Mean"], d["Max"]]
            out.append(_mk("monotone_chain", {"features": feats}, feats,
                           f"order::{base} Min<=Mean<=Max"))
        if {"Variance", "Std"} <= set(d):
            out.append(_mk("square_relation", {"lhs": d["Variance"], "base": d["Std"]},
                           [d["Variance"], d["Std"]], f"sq::{base} Variance==Std^2"))
    return out


def _direction_base(name: str) -> tuple[str, str] | None:
    """Return (direction, stripped) if the name carries a Fwd/Bwd direction token."""
    for tok in ("Fwd", "FWD", "Forward"):
        if tok in name.split():
            return "fwd", name
    for tok in ("Bwd", "BWD", "Backward"):
        if tok in name.split():
            return "bwd", name
    return None


def _norm_dir(name: str) -> str:
    parts = [p for p in name.split() if p.lower() not in ("fwd", "bwd", "forward", "backward")]
    return " ".join(parts).lower()


def directional_sums(feature_order) -> list[Rule]:
    """A ~= Fwd_X + Bwd_X where a combined feature A shares X's direction-free name."""
    fwd, bwd, plain = {}, {}, {}
    for f in feature_order:
        d = _direction_base(f)
        key = _norm_dir(f)
        if d is None:
            plain.setdefault(key, []).append(f)
        elif d[0] == "fwd":
            fwd[key] = f
        else:
            bwd[key] = f
    out: list[Rule] = []
    for key in set(fwd) & set(bwd):
        addends = [fwd[key], bwd[key]]
        for target in plain.get(key, []):
            out.append(_mk("sum_equality", {"lhs": target, "addends": addends},
                           [target] + addends, f"sum::{target}==Fwd+Bwd"))
    return out


def guided_products(feature_order, category_of) -> list[Rule]:
    """Total-length ~= count * mean, matched by direction; plus variance=std*std."""
    out: list[Rule] = []
    byte_counts = [f for f in feature_order if category_of(f) == "byte_count"]
    packet_counts = [f for f in feature_order if category_of(f) == "packet_count"]
    means = [f for f in feature_order if f.endswith("Packet Length Mean")]
    for tgt in byte_counts:
        d = "Fwd" if "Fwd" in tgt else ("Bwd" if "Bwd" in tgt else None)
        if d is None:
            continue
        cnt = next((c for c in packet_counts if d.lower() in c.lower()), None)
        mean = next((m for m in means if d.lower() in m.lower()), None)
        if cnt and mean:
            out.append(_mk("product_equality", {"lhs": tgt, "factors": [cnt, mean], "k": 1.0},
                           [tgt, cnt, mean], f"prod::{tgt}==count*mean"))
    return out


def category_arithmetic(feature_order, category_of, max_per_category: int = 4000) -> list[Rule]:
    """Bounded sum/difference/product/ratio scan within each category (>=3 feats)."""
    groups: dict[str, list[str]] = {}
    for f in feature_order:
        groups.setdefault(category_of(f), []).append(f)
    out: list[Rule] = []
    for cat, feats in groups.items():
        if len(feats) < 3:
            continue
        cnt = 0
        for target in feats:
            others = [x for x in feats if x != target]
            for b, c in combinations(others, 2):
                out.append(_mk("sum_equality", {"lhs": target, "addends": [b, c]},
                               [target, b, c], f"sum::{target}=={b}+{c}"))
                cnt += 1
                if cnt >= max_per_category:
                    break
            if cnt >= max_per_category:
                break
    return out


def generate(feature_order, category_of) -> list[Rule]:
    cands = []
    cands += statistical_groups(feature_order)
    cands += directional_sums(feature_order)
    cands += guided_products(feature_order, category_of)
    cands += category_arithmetic(feature_order, category_of)
    return cands
