"""Legacy vs validator_v2 comparison (§15).

Runs the preserved legacy CICIDS2017 structural validator (the mined
ConstraintEngine semantics, `validation/legacy.py`) and validator_v2 on the same
untouched sample, cross-tabulates agreement, and lists the rules responsible for
disagreements. The goal is auditability, not making v2 reproduce legacy.

Run:  python -m validation.evaluation.legacy_vs_v2
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from validation import load_validator
from validation.legacy import LegacyValidator
from validation.mining import data_access as da

PKG = Path(__file__).resolve().parents[1]
N = 100000


def _corrupted_set(v, clean, per_n=4000):
    """Build a corrupted evaluation set from clean rows using the synthetic corruptions."""
    from validation.evaluation.synthetic_violations import _corruptions
    base = clean[:per_n]
    parts = []
    for c in _corruptions(v.idx):
        Xc = base.copy()
        app = c["fn"](Xc, v.idx)
        parts.append(Xc[app])
    return np.concatenate(parts, axis=0) if parts else base


def run(n: int = N) -> dict:
    v = load_validator("cicids2017_distrinet")
    leg = LegacyValidator(v.feature_order)
    X = da.load_split("test", allow_test_for_final_reporting=True)
    Xs = da.sample_rows(X, n, seed=3)

    def compare(Xe):
        b = v.validate_batch(Xe)
        v2_hard = b.hard_structural_valid
        v2_hybrid = b.hybrid_valid
        leg_valid = leg.validate_batch(Xe)
        return b, v2_hard, v2_hybrid, leg_valid

    def crosstab(a, c):
        return {"TT": int((a & c).sum()), "TF": int((a & ~c).sum()),
                "FT": int((~a & c).sum()), "FF": int((~a & ~c).sum())}

    # --- clean ---
    b, v2_hard, v2_hybrid, leg_valid = compare(Xs)
    ct_hard = crosstab(leg_valid, v2_hard)
    ct_hybrid = crosstab(leg_valid, v2_hybrid)

    # --- corrupted (where disagreements surface) ---
    clean_ok = Xs[v.validate_batch(Xs).structurally_valid]
    Xco = _corrupted_set(v, clean_ok)
    bc, cv2_hard, cv2_hybrid, cleg = compare(Xco)
    ct_corr = crosstab(cleg, cv2_hybrid)

    dis_v2_stricter = cleg & ~cv2_hybrid          # legacy passes, v2 rejects
    dis_leg_stricter = ~cleg & cv2_hybrid         # v2 passes, legacy rejects
    v2_rules = Counter()
    if dis_v2_stricter.any():
        for r in v.rules:
            fired = int((bc.violation(r.id) & dis_v2_stricter).sum())
            if fired:
                v2_rules[f"{r.source_type}: {r.expression()}"] += fired
    leg_rules = Counter()
    if dis_leg_stricter.any():
        per = leg.per_rule(Xco[dis_leg_stricter])
        for name, mask in per.items():
            if mask.any():
                leg_rules[name] += int(mask.sum())

    md = _report(n, ct_hard, ct_hybrid, ct_corr, int(Xco.shape[0]), v2_rules, leg_rules,
                 int(dis_v2_stricter.sum()), int(dis_leg_stricter.sum()))
    out = PKG / "reports" / "legacy_vs_v2.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return {"n": int(Xs.shape[0]), "crosstab_hard": ct_hard, "crosstab_hybrid": ct_hybrid,
            "corrupted_n": int(Xco.shape[0]), "crosstab_corrupted": ct_corr,
            "v2_stricter": int(dis_v2_stricter.sum()), "legacy_stricter": int(dis_leg_stricter.sum())}


def _ct_block(title, ct, n):
    return [f"### {title}", "",
            "| | v2 = True | v2 = False |", "|---|---|---|",
            f"| **legacy = True** | {ct['TT']} | {ct['TF']} |",
            f"| **legacy = False** | {ct['FT']} | {ct['FF']} |",
            f"", f"Agreement: {(ct['TT']+ct['FF'])/n:.6f}", ""]


def _report(n, ct_hard, ct_hybrid, ct_corr, n_corr, v2_rules, leg_rules, n_v2s, n_legs) -> str:
    L = ["# Legacy vs validator_v2", "",
         f"Legacy = mined ConstraintEngine semantics "
         "(`old_constraints/cicids2017_distrinet/mined.json`: 8 monotone + 6 product "
         "identities). v2 shown at `hard_structural_valid` (SCHEMA+EXTRACTOR+PROTOCOL) "
         "and `hybrid_valid` (+MINED).", "",
         f"## Clean data ({n:,} untouched `test` flows)", ""]
    L += _ct_block("Legacy vs v2 hard_structural_valid", ct_hard, n)
    L += _ct_block("Legacy vs v2 hybrid_valid", ct_hybrid, n)
    L += [f"## Corrupted data ({n_corr:,} rows: clean flows with one injected violation each)", "",
          "This is where the two validators diverge — legacy only checks 14 mined "
          "monotone/product rules, whereas v2 also enforces SCHEMA type facts, EXTRACTOR "
          "identities and PROTOCOL non-negativity.", ""]
    L += _ct_block("Legacy vs v2 hybrid_valid (corrupted)", ct_corr, n_corr)
    L += ["## Disagreements (corrupted set)", "",
          f"- legacy accepts but v2-hybrid rejects: **{n_v2s}**",
          f"- v2-hybrid accepts but legacy rejects: **{n_legs}**", ""]
    if v2_rules:
        L += ["v2 rules responsible (legacy=True, v2=False):", ""]
        for k, c in v2_rules.most_common(20):
            L.append(f"- `{k}` — {c}")
        L.append("")
    if leg_rules:
        L += ["legacy constraints responsible (legacy=False, v2=True):", ""]
        for k, c in leg_rules.most_common(20):
            L.append(f"- `{k}` — {c}")
        L.append("")
    if not v2_rules and not leg_rules:
        L.append("No disagreements: the two validators agree on every sample in this set.")
    L += ["", "> This comparison is for auditability. v2 is intentionally NOT tuned to "
          "reproduce legacy; where they differ, the responsible rules are listed above so "
          "the difference is explainable."]
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
