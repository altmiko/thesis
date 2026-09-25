"""Paired + statistical analysis of the two-dataset adversarial campaign.

Consumes ``outputs/adv_campaign/<dataset>/{config,selection}.json`` and the per-row
``artifacts/*.npz`` written by ``scripts/run_full_adversarial_eval.py`` for CICIDS2017- and
CSE-CIC-IDS-2018-DistriNet, and writes

* ``outputs/adv_campaign/analysis.json``      -- every rate, CI, and test (machine-readable)
* ``outputs/adv_campaign/analysis_tables.md`` -- the generated tables used by the report

Statistical design (per CLAUDE.md: victims are never pooled):

* Unit = one clean-correct eligible flow of one victim instance. Rows of different attack
  classes are distinct flows, so classes are pooled WITHIN a victim; victims, victim
  training replicates and attack seeds are never pooled into one test.
* Paired tests use the reference attack seed 42 (every attack of a victim hit the SAME rows,
  asserted by the driver). Other seeds enter only as run-to-run variability (mean +- sd).
* Binary paired contrasts: exact/asymptotic McNemar, Newcombe square-and-add 95% CI for the
  paired risk difference, Haldane-Anscombe discordant odds ratio; k>2 related methods:
  Cochran's Q. Holm correction within each contrast family per (dataset, victim).
* Cross-dataset (independent samples, same architecture, seed 42, 2018 replicate s42):
  Newcombe hybrid-score CI for the unpaired difference + Fisher exact test, Holm over attacks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2, fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.paired_validity_gap import (  # noqa: E402
    holm_adjust, mcnemar_test, newcombe_paired_ci, wilson_score_interval,
)

ROOT = REPO_ROOT / "outputs" / "adv_campaign"  # overridable with --root
DATASETS = {"cicids2017_distrinet": "CICIDS2017", "cicids2018_distrinet": "CSE-CIC-IDS-2018"}
REF_SEED = 42
ARCHS = ("mlp", "cnn", "ft_transformer")
ARCH_LABEL = {"mlp": "MLP", "cnn": "CNN", "ft_transformer": "FT-Transformer"}

# Levels of the nested success ladder (goal success = targeted->Benign for targeted attacks,
# any misclassification for untargeted attacks). None = level undefined for that attack.
LEVELS = ("raw", "valid", "feasible", "sp")
HEADLINE_ATTACKS = (
    "pgd_untargeted", "cw_untargeted", "pgd_tb", "cw_tb",
    "prim_search_joint_p50", "prim_search_joint_p75", "prim_search_joint_unb",
    "prim_rand_joint_p75", "capgd_native", "capgd_prim_p75",
)
REFERENCE_PRIM = "prim_search_joint_p75"
C2_COMPARATORS = ("pgd_untargeted", "cw_untargeted", "capgd_native",
                  "capgd_prim_p75", "prim_rand_joint_p75", "prim_search_joint_unb")


# ----------------------------------------------------------------------------- loading
def _load_dataset(ds: str) -> dict:
    out = ROOT / ds
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    roster = cfg["attack_roster"]
    rows: dict = {}  # (victim, attack, seed) -> dict of concatenated class arrays
    for victim, vinfo in cfg["victims"].items():
        for atk in roster:
            for seed in vinfo["attack_seeds"]:
                parts = defaultdict(list)
                for cname in cfg["classes"]:
                    p = out / "artifacts" / f"{victim}__{cname}__{atk['name']}__seed{seed}.npz"
                    d = np.load(p, allow_pickle=True)
                    n = len(d["sample_id"])
                    parts["class"].append(np.full(n, cname))
                    for key in ("evasion", "targeted_success", "domain_valid", "realizable"):
                        parts[key].append(d[key].astype(bool))
                    for key in ("primitive_feasible", "semantic_pass"):
                        parts[key].append(d[key].astype(np.int8))
                    for key in ("cost_total", "l2_scaled"):
                        parts[key].append(d[key].astype(np.float64))
                    parts["elapsed"].append(np.asarray([float(d["elapsed_seconds"])]))
                rows[(victim, atk["name"], seed)] = {k: np.concatenate(v) for k, v in parts.items()}
    return {"config": cfg, "selection": sel, "roster": {a["name"]: a for a in roster},
            "rows": rows}


def outcome(r: dict, atk: dict, level: str, *, goal: str | None = None) -> np.ndarray | None:
    """Per-row success indicator at a ladder level. ``goal`` overrides the attack's own."""
    g = goal or atk["goal"]
    base = r["targeted_success"] if g == "targeted_benign" else r["evasion"]
    if level == "raw":
        return base
    valid = base & r["domain_valid"]
    if level == "valid":
        return valid
    if atk["family"] != "primattack" and atk["kind"] != "prim_capgd":
        return None
    feasible = valid & (r["primitive_feasible"] == 1)
    if level == "feasible":
        return feasible
    return feasible & (r["semantic_pass"] == 1)  # sp


# ----------------------------------------------------------------------------- statistics
def rate_ci(x: np.ndarray) -> dict:
    n, k = int(x.size), int(x.sum())
    lo, hi = wilson_score_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci": [lo, hi]}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Method A vs method B on the same rows (success indicators)."""
    both = int((a & b).sum()); a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum()); neither = int((~a & ~b).sum())
    n = both + a_only + b_only + neither
    lo, hi = newcombe_paired_ci(both, a_only, b_only, neither)
    test = mcnemar_test(a_only, b_only)
    return {"n": n, "rate_a": (both + a_only) / n, "rate_b": (both + b_only) / n,
            "rd": (a_only - b_only) / n, "ci": [lo, hi], "a_only": a_only, "b_only": b_only,
            "or_ha": (a_only + 0.5) / (b_only + 0.5), "p": test["p_value"],
            "test": test["test_variant"]}


def cochran_q(mat: np.ndarray) -> dict:
    """Cochran's Q for an (n_rows, k_methods) binary matrix of related outcomes."""
    x = mat.astype(np.int64)
    k = x.shape[1]
    col, row = x.sum(0), x.sum(1)
    total = int(row.sum())
    denom = k * total - int((row ** 2).sum())
    if denom == 0:
        return {"q": 0.0, "df": k - 1, "p": 1.0, "rates": (col / x.shape[0]).tolist()}
    q = (k - 1) * (k * int((col ** 2).sum()) - total ** 2) / denom
    return {"q": float(q), "df": k - 1, "p": float(chi2.sf(q, k - 1)),
            "rates": (col / x.shape[0]).tolist()}


def newcombe_unpaired(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe hybrid score interval (method 10) for p1 - p2, independent samples."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_score_interval(k1, n1)
    l2, u2 = wilson_score_interval(k2, n2)
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def _holm_family(tests: list[dict]) -> None:
    ps = [t["p"] for t in tests]
    for t, adj in zip(tests, holm_adjust(ps) if ps else []):
        t["p_holm"] = adj


# ----------------------------------------------------------------------------- analysis
def analyze_dataset(ds: str, data: dict) -> dict:
    cfg, rows, roster = data["config"], data["rows"], data["roster"]
    victims = list(cfg["victims"])
    res: dict = {"victims": cfg["victims"], "rates": {}, "variability": {}, "per_class": {},
                 "contrasts": {}, "selection": {}}
    for v in victims:
        res["selection"][v] = {c: {k: data["selection"][v][c][k] for k in
                                   ("n_class_test", "n_eligible_total", "n_used",
                                    "clean_hybrid_valid_rate", "source_label_counts")}
                               for c in cfg["classes"]}
    # ---- rates (seed 42, Wilson) + across-seed variability ----
    for v in victims:
        seeds = cfg["victims"][v]["attack_seeds"]
        ref = REF_SEED if REF_SEED in seeds else seeds[0]
        for name, atk in roster.items():
            r = rows[(v, name, ref)]
            entry = {}
            for lvl in LEVELS:
                o = outcome(r, atk, lvl)
                entry[lvl] = rate_ci(o) if o is not None else None
            entry["evasion_valid"] = rate_ci(outcome(r, atk, "valid", goal="untargeted"))
            entry["domain_valid"] = rate_ci(r["domain_valid"])
            entry["mean_l2_scaled"] = float(r["l2_scaled"].mean())
            entry["elapsed_s"] = float(r["elapsed"].sum())
            res["rates"].setdefault(v, {})[name] = entry
            var = {}
            for lvl in ("raw", "valid"):
                vals = [float(outcome(rows[(v, name, s)], atk, lvl).mean()) for s in seeds]
                var[lvl] = {"values": vals, "mean": float(np.mean(vals)),
                            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
            res["variability"].setdefault(v, {})[name] = var
            if name in HEADLINE_ATTACKS:
                pc = {}
                for cname in cfg["classes"]:
                    m = r["class"] == cname
                    pc[cname] = {lvl: (float(o[m].mean()) if (o := outcome(r, atk, lvl)) is not None
                                       else None) for lvl in LEVELS}
                res["per_class"].setdefault(v, {})[name] = pc

    # ---- paired contrasts per victim (seed 42) ----
    for v in victims:
        seeds = cfg["victims"][v]["attack_seeds"]
        ref = REF_SEED if REF_SEED in seeds else seeds[0]
        R = lambda n: rows[(v, n, ref)]  # noqa: E731
        C: dict = {}
        # C1 validity gap: raw vs valid (goal success) for every attack
        C["C1_validity_gap"] = [
            {"attack": n, **paired(outcome(R(n), a, "raw"), outcome(R(n), a, "valid"))}
            for n, a in roster.items()]
        # C2 PrimAttack reference vs other families (valid evasion, untargeted)
        C["C2_family_valid_evasion"] = [
            {"a": REFERENCE_PRIM, "b": n,
             **paired(outcome(R(REFERENCE_PRIM), roster[REFERENCE_PRIM], "valid", goal="untargeted"),
                      outcome(R(n), roster[n], "valid", goal="untargeted"))}
            for n in C2_COMPARATORS if n in roster]
        # C3 primitive mode (joint / timing / padding), search, per budget; valid targeted
        C["C3_mode"] = []
        for b in ("p50", "p75", "unb"):
            names = [f"prim_search_{m}_{b}" for m in ("joint", "timing", "padding")]
            mats = np.stack([outcome(R(n), roster[n], "valid") for n in names], 1)
            q = cochran_q(mats)
            pairs = [{"a": names[i], "b": names[j], **paired(mats[:, i], mats[:, j])}
                     for i, j in ((0, 1), (0, 2), (1, 2))]
            C["C3_mode"].append({"budget": b, "cochran": q, "pairs": pairs})
        # C4 budget (p50 / p75 / unbounded), search joint; valid targeted and SP
        C["C4_budget"] = []
        for lvl in ("valid", "sp"):
            names = [f"prim_search_joint_{b}" for b in ("p50", "p75", "unb")]
            mats = np.stack([outcome(R(n), roster[n], lvl) for n in names], 1)
            pairs = [{"a": names[i], "b": names[j], **paired(mats[:, i], mats[:, j])}
                     for i, j in ((1, 0), (2, 1))]
            C["C4_budget"].append({"level": lvl, "cochran": cochran_q(mats), "pairs": pairs})
        # C6 optimizer: search vs random-feasible, every budget x mode (valid targeted)
        C["C6_optimizer"] = [
            {"a": f"prim_search_{m}_{b}", "b": f"prim_rand_{m}_{b}",
             **paired(outcome(R(f"prim_search_{m}_{b}"), roster[f"prim_search_{m}_{b}"], "valid"),
                      outcome(R(f"prim_rand_{m}_{b}"), roster[f"prim_rand_{m}_{b}"], "valid"))}
            for b in ("p50", "p75", "unb") for m in ("joint", "timing", "padding")]
        # C7 same box, different optimizer: PrimAttack search vs CAPGD in the p75 box (evasion)
        C["C7_box_optimizer"] = [
            {"a": REFERENCE_PRIM, "b": "capgd_prim_p75", "level": lvl,
             **paired(outcome(R(REFERENCE_PRIM), roster[REFERENCE_PRIM], lvl, goal="untargeted"),
                      outcome(R("capgd_prim_p75"), roster["capgd_prim_p75"], lvl, goal="untargeted"))}
            for lvl in ("raw", "valid", "sp")]
        # Holm within each family (flatten nested pair lists)
        for key, fam in C.items():
            flat = []
            for t in fam:
                if "pairs" in t:
                    flat.extend(t["pairs"])
                else:
                    flat.append(t)
            _holm_family(flat)
            for t in fam:
                if "cochran" in t:
                    t["cochran"]["p_holm"] = min(1.0, t["cochran"]["p"] * len(fam))
        res["contrasts"][v] = C
    return res


def cross_dataset(results: dict, data: dict) -> list[dict]:
    """Same architecture, independent samples: 2017 victim vs 2018 replicate s42, seed 42."""
    out = []
    d17, d18 = data["cicids2017_distrinet"], data["cicids2018_distrinet"]
    for arch in ARCHS:
        v17, v18 = arch, f"{arch}-s42"
        if v17 not in d17["config"]["victims"] or v18 not in d18["config"]["victims"]:
            continue
        tests = []
        for name in HEADLINE_ATTACKS:
            atk = d17["roster"][name]
            a = outcome(d17["rows"][(v17, name, REF_SEED)], atk, "valid", goal="untargeted")
            b = outcome(d18["rows"][(v18, name, REF_SEED)], atk, "valid", goal="untargeted")
            k1, n1, k2, n2 = int(a.sum()), a.size, int(b.sum()), b.size
            lo, hi = newcombe_unpaired(k1, n1, k2, n2)
            p = float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
            tests.append({"arch": arch, "attack": name, "metric": "valid_evasion",
                          "rate_2017": k1 / n1, "rate_2018": k2 / n2, "n_2017": n1,
                          "n_2018": n2, "rd": k1 / n1 - k2 / n2, "ci": [lo, hi], "p": p})
        _holm_family(tests)
        out.extend(tests)
    return out


# ----------------------------------------------------------------------------- markdown
def _pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}"


def _ci(e, d=1):
    if e is None:
        return "—"
    return f"{100 * e['rate']:.{d}f} [{100 * e['ci'][0]:.{d}f}, {100 * e['ci'][1]:.{d}f}]"


def _p(p):
    return "<1e-300" if p == 0 else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")


def _vlabel(ds, v):
    return ARCH_LABEL.get(v, None) or (ARCH_LABEL[v.rpartition("-s")[0]] + " s" + v.rpartition("-s")[2])


def write_markdown(results: dict, cross: list[dict], data: dict) -> str:
    L: list[str] = []
    for ds, res in results.items():
        name = DATASETS[ds]
        cfg = data[ds]["config"]
        victims = list(cfg["victims"])
        L.append(f"## {name}\n")
        # selection
        L.append(f"### {name}: eligible rows (clean-correct, seeded random sample, cap {cfg['n_per_class_cap']}/class)\n")
        L.append("| victim | " + " | ".join(f"{c} used / eligible / test" for c in cfg["classes"])
                 + " | clean hybrid_valid |")
        L.append("|---|" + "---|" * (len(cfg["classes"]) + 1))
        for v in victims:
            s = res["selection"][v]
            hv = np.mean([s[c]["clean_hybrid_valid_rate"] for c in cfg["classes"]])
            L.append(f"| {_vlabel(ds, v)} | " + " | ".join(
                f"{s[c]['n_used']} / {s[c]['n_eligible_total']} / {s[c]['n_class_test']}"
                for c in cfg["classes"]) + f" | {100 * hv:.2f}% |")
        L.append("")
        # headline ladder
        L.append(f"### {name}: success ladder per victim (reference seed 42, classes pooled; % [Wilson 95% CI])\n")
        L.append("Goal success = targeted→Benign for targeted attacks, misclassification for untargeted "
                 "attacks. valid = ∧ validator_v2 hybrid_valid; feasible = ∧ primitive-feasible; "
                 "SP = ∧ flow-semantic PASS.\n")
        for v in victims:
            L.append(f"**{_vlabel(ds, v)}** (N = {res['rates'][v][HEADLINE_ATTACKS[0]]['raw']['n']})\n")
            L.append("| attack | goal | raw | valid | feasible | SP | hybrid_valid rate | mean L2 (scaled) |")
            L.append("|---|---|---|---|---|---|---|---|")
            for n, e in res["rates"][v].items():
                g = "T" if data[ds]["roster"][n]["goal"] == "targeted_benign" else "U"
                L.append(f"| `{n}` | {g} | {_ci(e['raw'])} | {_ci(e['valid'])} | {_ci(e['feasible'])} | "
                         f"{_ci(e['sp'])} | {_pct(e['domain_valid']['rate'])} | "
                         f"{e['mean_l2_scaled']:.3g} |")
            L.append("")
        # variability
        L.append(f"### {name}: run-to-run variability (valid goal-success %, mean ± sd over attack seeds)\n")
        arch_groups = defaultdict(list)
        for v in victims:
            arch_groups[cfg["victims"][v]["arch"]].append(v)
        L.append("| attack | " + " | ".join(_vlabel(ds, v) for v in victims)
                 + (" | " + " | ".join(f"{ARCH_LABEL[a]} (all replicates×seeds)" for a in arch_groups)
                    if ds == "cicids2018_distrinet" else "") + " |")
        L.append("|---|" + "---|" * (len(victims) + (len(arch_groups) if ds == "cicids2018_distrinet" else 0)))
        for n in HEADLINE_ATTACKS:
            cells = [f"{100 * res['variability'][v][n]['valid']['mean']:.2f} ± "
                     f"{100 * res['variability'][v][n]['valid']['sd']:.2f}" for v in victims]
            if ds == "cicids2018_distrinet":
                for a, vs in arch_groups.items():
                    vals = [x for v in vs for x in res["variability"][v][n]["valid"]["values"]]
                    cells.append(f"{100 * np.mean(vals):.2f} ± {100 * np.std(vals, ddof=1):.2f}")
            L.append(f"| `{n}` | " + " | ".join(cells) + " |")
        L.append("")
        # per class
        L.append(f"### {name}: per-class valid goal-success (%, seed 42)\n")
        key = ["prim_search_joint_p75", "prim_search_joint_unb", "capgd_native", "pgd_untargeted"]
        L.append("| victim | attack | " + " | ".join(cfg["classes"]) + " |")
        L.append("|---|---|" + "---|" * len(cfg["classes"]))
        for v in victims:
            for n in key:
                pc = res["per_class"][v][n]
                L.append(f"| {_vlabel(ds, v)} | `{n}` | " + " | ".join(
                    _pct(pc[c]["valid"]) for c in cfg["classes"]) + " |")
        L.append("")
        # contrasts
        L.append(f"### {name}: paired tests (seed 42, per victim; Holm within family)\n")
        L.append("RD = rate(A) − rate(B) in percentage points with Newcombe 95% CI; "
                 "A-only/B-only = discordant rows; OR = Haldane–Anscombe (A-only+½)/(B-only+½).\n")
        for fam, title in (("C1_validity_gap", "C1 validity gap: raw vs valid goal-success (A = raw, B = valid)"),
                           ("C2_family_valid_evasion", f"C2 `{REFERENCE_PRIM}` vs other families — valid evasion (untargeted)"),
                           ("C6_optimizer", "C6 search vs random-feasible (valid targeted)"),
                           ("C7_box_optimizer", "C7 PrimAttack search vs CAPGD in the SAME p75 primitive box (untargeted)")):
            L.append(f"#### {title}\n")
            L.append("| victim | A | B | N | rate A | rate B | RD pp [95% CI] | A-only | B-only | OR | p | p_Holm |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for v in victims:
                for t in res["contrasts"][v][fam]:
                    a = t.get("a", t.get("attack")) + (f" ({t['level']})" if "level" in t else "")
                    b = t.get("b", "valid" if fam == "C1_validity_gap" else "")
                    L.append(f"| {_vlabel(ds, v)} | `{a}` | `{b}` | {t['n']} | {_pct(t['rate_a'], 2)} | {_pct(t['rate_b'], 2)} | "
                             f"{100 * t['rd']:+.2f} [{100 * t['ci'][0]:+.2f}, {100 * t['ci'][1]:+.2f}] | "
                             f"{t['a_only']} | {t['b_only']} | {t['or_ha']:.3g} | {_p(t['p'])} | {_p(t['p_holm'])} |")
            L.append("")
        for fam, title, lab in (("C3_mode", "C3 primitive mode (search, valid targeted)", "budget"),
                                ("C4_budget", "C4 budget p50 → p75 → unbounded (search joint)", "level")):
            L.append(f"#### {title}\n")
            L.append(f"| victim | {lab} | Cochran Q (df) | p | p_Holm | rates % | pairwise RD pp [CI], p_Holm |")
            L.append("|---|---|---|---|---|---|---|")
            for v in victims:
                for t in res["contrasts"][v][fam]:
                    q = t["cochran"]
                    pw = "; ".join(f"{t2['a'].split('_')[-2] if fam == 'C3_mode' else t2['a']}−"
                                   f"{t2['b'].split('_')[-2] if fam == 'C3_mode' else t2['b']}: "
                                   f"{100 * t2['rd']:+.2f} [{100 * t2['ci'][0]:+.2f}, {100 * t2['ci'][1]:+.2f}], {_p(t2['p_holm'])}"
                                   for t2 in t["pairs"])
                    L.append(f"| {_vlabel(ds, v)} | {t[lab]} | {q['q']:.1f} ({q['df']}) | {_p(q['p'])} | {_p(q['p_holm'])} | "
                             + "/".join(f"{100 * r:.1f}" for r in q["rates"]) + f" | {pw} |")
            L.append("")
    # cross-dataset
    L.append("## Cross-dataset comparison (independent samples; 2017 victim vs 2018 replicate s42, seed 42)\n")
    L.append("| arch | attack | metric | 2017 % | 2018 % | RD pp [Newcombe 95% CI] | Fisher p | p_Holm |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t in cross:
        L.append(f"| {ARCH_LABEL[t['arch']]} | `{t['attack']}` | {t['metric']} | {_pct(t['rate_2017'], 2)} | "
                 f"{_pct(t['rate_2018'], 2)} | {100 * t['rd']:+.2f} [{100 * t['ci'][0]:+.2f}, {100 * t['ci'][1]:+.2f}] | "
                 f"{_p(t['p'])} | {_p(t['p_holm'])} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="directory holding <dataset>/ campaign outputs")
    ROOT = ap.parse_args().root
    data = {ds: _load_dataset(ds) for ds in DATASETS}
    results = {ds: analyze_dataset(ds, d) for ds, d in data.items()}
    cross = cross_dataset(results, data)
    (ROOT / "analysis.json").write_text(
        json.dumps({"datasets": results, "cross_dataset": cross}, indent=1, default=float),
        encoding="utf-8")
    (ROOT / "analysis_tables.md").write_text(write_markdown(results, cross, data), encoding="utf-8")
    print(f"wrote {ROOT / 'analysis.json'} and {ROOT / 'analysis_tables.md'}")


if __name__ == "__main__":
    main()
