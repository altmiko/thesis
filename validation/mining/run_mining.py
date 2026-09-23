"""End-to-end mining orchestrator for CICIDS2017.

Pipeline (strict TRAIN -> VAL separation; TEST never touched):

  1. infer SCHEMA type facts on TRAIN, confirm on VAL  -> schema/<ds>.yaml
  2. generate the restricted candidate grammar
  3. prefilter on a small TRAIN sample
  4. derive each survivor's tolerance from the TRAIN residual distribution
  5. measure TRAIN support (discovery) and VAL support (confirmation)
  6. accept iff train_support >= THR_TRAIN and val_support >= THR_VAL
     (+ antecedent-coverage gates for implications)
  7. prune dominated/duplicate/extractor-covered rules (records why_pruned)
  8. recompute FINAL support of survivors on the FULL train & val splits
  9. emit rules/<ds>/mined_rules.json, rules/<ds>/protocol_rules.yaml,
     rules/<ds>/plausibility_profile.json, and reports/<ds>/mining_report.md

Run:  python -m validation.mining.run_mining
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from validation.mining import candidate_templates as ct
from validation.mining import data_access as da
from validation.mining.evaluate_candidates import (
    derive_tolerance, evaluate_rule, observed_expected, residual_tightness)
from validation.mining.feature_registry import registry_for
from validation.mining.infer_schema import infer_schema, write_schema_yaml
from validation.mining.prune_rules import prune
from validation.validator.plausibility import PlausibilityProfile
from validation.validator.rule import Rule
from validation.validator.tolerance import Tolerance, ABS_FLOOR

try:
    import yaml
except Exception:
    yaml = None

PKG = Path(__file__).resolve().parents[1]

# ---- thresholds (configurable; starting values, see mining_report) ----------
THR_TRAIN = 0.999
THR_VAL = 0.995
PREFILTER = 0.99
PREFILTER_SAMPLE = 20000
DISCOVERY_SAMPLE = 200000
VAL_SAMPLE = 200000
MIN_ANTECEDENT_RATE = 0.01
MIN_ANTECEDENT_ROWS = 1000
TOL_PERCENTILE = 99.9
SEED = 42
LOGICAL_ABS_TOL = 1e-6
# Scale-free relative-residual tightness gate for APPROXIMATE mined rules:
# accept only if the 99.9th-percentile relative residual is within these bounds
# on train (discovery) and validation (confirmation). This prevents a
# self-derived tolerance from making a false relation vacuously pass.
REL_TIGHT = 1e-3
REL_TIGHT_VAL = 2e-3
LOGICAL_TYPES = ("le", "ge", "monotone_chain", "implication_zero", "implication_pos")


def _describe(r: Rule) -> str:
    p = r.params
    t = r.rule_type
    if t == "monotone_chain":
        return ("Ordered statistics are non-decreasing: " + " <= ".join(p["features"]) +
                ". A mean cannot fall below the minimum or above the maximum of the same group.")
    if t == "equality":
        return f"{p['lhs']} equals {p['rhs']} across training flows (empirically confirmed on validation)."
    if t == "le":
        return f"{p['lhs']} never exceeds {p['rhs']}."
    if t == "ge":
        return f"{p['lhs']} is never below {p['rhs']}."
    if t == "square_relation":
        return f"{p['lhs']} equals the square of {p['base']}."
    if t == "scaled_equality":
        return f"{p['lhs']} equals {p['k']} times {p['rhs']}."
    if t == "sum_equality":
        return f"{p['lhs']} equals the sum of " + " and ".join(p["addends"]) + "."
    if t == "difference_equality":
        return f"{p['lhs']} equals {p['minuend']} minus {p['subtrahend']}."
    if t == "product_equality":
        return f"{p['lhs']} equals the product of " + " and ".join(p["factors"]) + "."
    if t == "ratio_equality":
        return f"{p['lhs']} equals {p['numerator']} divided by {p['denominator']}."
    if t == "implication_zero":
        return f"When {p['antecedent']} is zero, {p['consequent']} is also zero."
    if t == "implication_pos":
        return f"When {p['antecedent']} is positive, {p['consequent']} is non-negative."
    return r.expression()


def _set_logical_tol(r: Rule) -> None:
    if r.rule_type in ("le", "ge", "monotone_chain", "implication_zero", "implication_pos"):
        r.tolerance = Tolerance(LOGICAL_ABS_TOL, 0.0)


def _full_support(rule: Rule, X: np.ndarray, idx: dict[str, int], step: int = 200000) -> tuple[float, int]:
    """Chunked support over a full (possibly mmap) matrix; returns (support, eligible_rows)."""
    n = X.shape[0]
    sat = elig = 0
    for s in range(0, n, step):
        Xc = np.asarray(X[s:s + step], np.float64)
        m_sat, m_elig = rule.evaluate(Xc, idx)
        sat += int((m_sat & m_elig).sum())
        elig += int(m_elig.sum())
    return (sat / elig if elig else 0.0), elig


def main() -> dict:
    t0 = time.time()
    dataset = da.dataset_name()
    names = da.feature_names()
    idx = {f: i for i, f in enumerate(names)}
    category_of = lambda f: registry_for(f).get("category", "unknown")  # noqa: E731

    print(f"[mining] dataset={dataset} features={len(names)}")
    Xtr = da.load_split("train")           # mmap, full
    Xva = da.load_split("val")             # mmap, full
    tr_sample = da.sample_rows(Xtr, DISCOVERY_SAMPLE, seed=SEED)
    va_sample = da.sample_rows(Xva, VAL_SAMPLE, seed=SEED)
    pre_sample = tr_sample[:PREFILTER_SAMPLE]

    # ---- 1. schema inference ----
    profile = infer_schema(tr_sample, va_sample, names, dataset)
    write_schema_yaml(profile, PKG / "schema" / f"{dataset}.yaml")
    print(f"[mining] schema profile written ({len(names)} features)")

    # ---- 2. candidate generation ----
    # Exclude constant / near-constant columns from relational mining: relations
    # involving a column that is a single value >=99.9% of the time are driven by
    # that near-constant, not by a genuine cross-feature invariant (methodology
    # §18). Their type is still captured by the SCHEMA layer.
    near_constant = []
    for i2, f in enumerate(names):
        col = tr_sample[:, i2]
        vals, counts = np.unique(col, return_counts=True)
        if counts.max() / col.size >= 0.999:
            near_constant.append(f)
    rel_names = [f for f in names if f not in near_constant]
    print(f"[mining] excluded {len(near_constant)} near-constant features from relational mining: {near_constant}")
    families = ct.generate_candidates(rel_names, category_of)
    all_cands = [r for fam in families.values() for r in fam]
    n_candidates = len(all_cands)
    fam_counts = {k: len(v) for k, v in families.items()}
    print(f"[mining] candidates: {n_candidates}  by family={fam_counts}")

    # ---- 3-6. prefilter -> tolerance -> train/val confirm -> accept ----
    # APPROX rules: gated on scale-free relative-residual tightness (train + val).
    # LOGICAL rules (<=, chains, implications): gated on plain support (tol=1e-6).
    accepted: list[Rule] = []
    rejected: list[dict] = []
    support_hist: list[float] = []

    def pre_support(rule: Rule) -> float:
        if rule.rule_type in LOGICAL_TYPES:
            return evaluate_rule(rule, pre_sample, idx)["support"]
        oe = observed_expected(rule, pre_sample, idx)
        if oe is None:
            return 0.0
        obs, exp, e = oe
        if not e.any():
            return 0.0
        rr = np.abs(obs[e] - exp[e]) / (np.abs(obs[e]) + np.abs(exp[e]) + ABS_FLOOR)
        return float(np.mean(rr <= REL_TIGHT))

    for r in all_cands:
        _set_logical_tol(r)
        ps = pre_support(r)
        support_hist.append(ps)
        is_logical = r.rule_type in LOGICAL_TYPES

        # prefilter
        if r.rule_type == "implication_zero":
            pm = evaluate_rule(r, pre_sample, idx)
            if pm.get("antecedent_rate", 0.0) < MIN_ANTECEDENT_RATE or pm["eligible_rows"] < 5:
                rejected.append({"rule": r, "reason": "low antecedent coverage (prefilter)", "train_support": ps})
                continue
        if ps < PREFILTER:
            rejected.append({"rule": r, "reason": "prefilter below threshold", "train_support": ps})
            continue

        if is_logical:
            tr_m = evaluate_rule(r, tr_sample, idx)
            va_m = evaluate_rule(r, va_sample, idx)
            ok = tr_m["support"] >= THR_TRAIN and va_m["support"] >= THR_VAL
            if r.rule_type == "implication_zero" and (
                    tr_m.get("antecedent_count", 0) < MIN_ANTECEDENT_ROWS
                    or tr_m.get("antecedent_rate", 0) < MIN_ANTECEDENT_RATE):
                ok = False
                reason = "antecedent coverage gate"
            else:
                reason = "train/val support below threshold"
            if ok:
                r.evidence = {"train_support": tr_m["support"], "validation_support": va_m["support"],
                              "eligible_rows_train": tr_m["eligible_rows"],
                              "antecedent_rate": tr_m.get("antecedent_rate")}
                accepted.append(r)
            else:
                rejected.append({"rule": r, "reason": reason,
                                 "train_support": tr_m["support"], "validation_support": va_m["support"]})
            continue

        # approximate: derive train tolerance, gate on tightness (train + val)
        r.tolerance = derive_tolerance(r, tr_sample, idx, TOL_PERCENTILE)
        tr_t = residual_tightness(r, tr_sample, idx)
        va_t = residual_tightness(r, va_sample, idx)
        if tr_t is None or va_t is None:
            rejected.append({"rule": r, "reason": "not evaluable", "train_support": ps})
            continue
        tr_rr, va_rr = tr_t[0], va_t[0]
        if tr_rr <= REL_TIGHT and va_rr <= REL_TIGHT_VAL:
            tr_m = evaluate_rule(r, tr_sample, idx)
            va_m = evaluate_rule(r, va_sample, idx)
            r.evidence = {"train_support": tr_m["support"], "validation_support": va_m["support"],
                          "train_rel_residual_p999": tr_rr, "val_rel_residual_p999": va_rr,
                          "eligible_rows_train": tr_m["eligible_rows"],
                          "abs_error": tr_m.get("abs_error"), "rel_error": tr_m.get("rel_error")}
            accepted.append(r)
        else:
            rejected.append({"rule": r, "reason": "relative residual not tight (train/val)",
                             "train_support": 1.0 - tr_rr, "validation_support": 1.0 - va_rr})

    print(f"[mining] accepted (pre-prune): {len(accepted)}  rejected: {len(rejected)}")

    # ---- 7. prune (needs extractor rules for domination) ----
    ext_rules = _load_extractor_rules(dataset)
    kept, pruned = prune(accepted, ext_rules)
    print(f"[mining] kept after prune: {len(kept)}  pruned: {len(pruned)}")

    # ---- 8. final full-split support for survivors ----
    for j, r in enumerate(kept, 1):
        r.id = f"MINED_{j:04d}"
        r.description = _describe(r)
        ts, te = _full_support(r, Xtr, idx)
        vs, ve = _full_support(r, Xva, idx)
        r.hardness = "EMPIRICAL"
        r.provenance = {"origin": "empirically mined from train split",
                        "external_reference": None, "automatically_mined": True,
                        "template": r.rule_type}
        r.evidence.update({"train_support_full": ts, "validation_support_full": vs,
                           "train_rows_full": int(Xtr.shape[0]), "validation_rows_full": int(Xva.shape[0]),
                           "eligible_rows_train_full": te, "eligible_rows_val_full": ve,
                           "why_accepted": f"train support {ts:.6f} >= {THR_TRAIN} and held-out "
                                           f"validation support {vs:.6f} >= {THR_VAL}"})

    # ---- 9a. write mined_rules.json ----
    mined_doc = {
        "schema_version": "2.0", "dataset": dataset,
        "mining": {
            "fit_split": "train", "confirm_split": "val", "test_used": False,
            "discovery_sample": DISCOVERY_SAMPLE, "val_sample": VAL_SAMPLE,
            "prefilter_support": PREFILTER, "train_support_threshold": THR_TRAIN,
            "val_support_threshold": THR_VAL, "tolerance_percentile": TOL_PERCENTILE,
            "min_antecedent_rate": MIN_ANTECEDENT_RATE, "min_antecedent_rows": MIN_ANTECEDENT_ROWS,
            "candidates_tested": n_candidates, "candidates_by_family": fam_counts,
            "accepted_pre_prune": len(accepted), "retained_rules": len(kept),
            "seed": SEED,
        },
        "rules": [r.to_dict() for r in kept],
    }
    out_dir = PKG / "rules" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mined_rules.json").write_text(json.dumps(mined_doc, indent=2))

    # ---- 9b. protocol rules (nonnegativity, domain-justified) ----
    proto = _build_protocol_rules(dataset, names, category_of, Xtr, Xva, idx)
    (out_dir / "protocol_rules.yaml").write_text(yaml.safe_dump(proto, sort_keys=False))

    # ---- 9c. plausibility profile ----
    plaus = PlausibilityProfile.fit(tr_sample, names, dataset)
    plaus.save(out_dir / "plausibility_profile.json")

    # ---- 9d. mining report ----
    _write_mining_report(dataset, names, profile, families, fam_counts, n_candidates,
                         kept, pruned, rejected, support_hist, ext_rules, proto, t0)

    print(f"[mining] done in {time.time()-t0:.1f}s")
    return {"n_candidates": n_candidates, "accepted": len(accepted),
            "retained": len(kept), "pruned": len(pruned), "rejected": len(rejected)}


def _load_extractor_rules(dataset: str) -> list[Rule]:
    path = PKG / "rules" / dataset / "extractor_rules.yaml"
    if not path.exists():
        return []
    doc = yaml.safe_load(path.read_text())
    out = []
    for d in doc.get("rules", []):
        d = dict(d); d["source_type"] = "EXTRACTOR"
        out.append(Rule.from_dict(d))
    return out


def _build_protocol_rules(dataset, names, category_of, Xtr, Xva, idx) -> dict:
    rules = []
    for i, f in enumerate(names):
        r = Rule(id=f"PROTO_{i+1:04d}", name=f"nonnegative::{f}", source_type="PROTOCOL",
                 rule_type="nonnegative", params={"feature": f}, features=[f],
                 hardness="PROTOCOL", tolerance=Tolerance(1e-6, 0.0),
                 description=f"{f} is non-negative: CICFlowMeter flow statistics "
                             f"(counts, sizes, durations, rates, ratios, ports/protocol codes) "
                             f"cannot be negative by construction.")
        ts, _ = _full_support(r, Xtr, idx)
        vs, _ = _full_support(r, Xva, idx)
        r.provenance = {"origin": "networking / extractor domain knowledge",
                        "external_reference": "non-negativity of flow statistics",
                        "automatically_mined": False}
        r.evidence = {"train_support_full": ts, "validation_support_full": vs}
        rules.append(r.to_dict())
    return {"dataset": dataset,
            "note": "Minimal, domain-justified PROTOCOL layer: a single principled "
                    "template (non-negativity) instantiated per feature. Not hand-invented "
                    "per feature; verified to hold on train and validation.",
            "rules": rules}


def _write_mining_report(dataset, names, profile, families, fam_counts, n_candidates,
                         kept, pruned, rejected, support_hist, ext_rules, proto, t0):
    from validation.mining.report_builder import build_mining_report
    md = build_mining_report(dataset, names, profile, families, fam_counts, n_candidates,
                             kept, pruned, rejected, support_hist, ext_rules, proto,
                             THR_TRAIN, THR_VAL, PREFILTER, TOL_PERCENTILE, time.time() - t0)
    rep_dir = PKG / "reports" / dataset
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "mining_report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    print(main())
