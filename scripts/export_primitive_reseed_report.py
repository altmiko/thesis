"""Aggregate the reseeded primitive-direct CICIDS2017 attack and emit a root MD report.

Produces ONE markdown file in the repo root containing:
  1. Effectiveness averaged over the NEW seeds (mean±std), every rate in PERCENT.
  2. The best (smallest normalized-cost) successful targeted Attack->Benign strict-valid
     example per class, inverse-transformed to raw CICFlowMeter units, shown side-by-side
     with the original (base) feature values (all features).
  3. A deltas-only table (changed features) for each of those best examples.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("outputs/cicids2017_primitive_attack_reseed/attack_results.json")
ARTDIR = Path("outputs/cicids2017_primitive_attack_reseed/attack_artifacts")
OUT = Path("CICIDS2017_primitive_reseed_adversarial_examples.md")

ROLE_TAG = {
    "derived_p": "Dp", "derived_t": "Dt", "derived": "D", "conditional": "C",
    "rate": "R", "invariant": "I", "frozen": "F", "level_c": "Fᶜ",
}
ID_TO_NAME = {0: "Benign", 1: "DoS", 2: "DDoS", 3: "Recon", 4: "BruteForce"}


def fnum(v: float) -> str:
    v = float(v)
    a = abs(v)
    if v == 0:
        return "0"
    if a >= 1e6 or a < 1e-3:
        return f"{v:.4e}"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.4f}"


def _mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def _std(xs):
    xs = [x for x in xs if x == x]
    return float(np.std(xs)) if len(xs) > 1 else 0.0


def pct(m, s=None):
    if m != m:
        return "-"
    return f"{m*100:.1f}±{s*100:.1f}" if s is not None else f"{m*100:.1f}"


def main() -> None:
    res = json.loads(RESULTS.read_text())
    cells = res["cells"]
    classes = list(dict.fromkeys(c["class"] for c in cells))
    victims = list(dict.fromkeys(c["victim"] for c in cells))
    seeds = sorted({c["seed"] for c in cells})
    names = list(res["feature_roles"].keys())
    roles = res["feature_roles"]

    grp = defaultdict(list)
    for c in cells:
        grp[(c["class"], c["victim"])].append(c)

    L = []
    L.append("# CICIDS2017-DistriNet — reseeded primitive-direct adversarial attack\n")
    L.append("Direct primitive-domain attack (`method_id = primitive_direct`): the attacker "
             "optimizes two per-flow primitives — forward packet-length augmentation `p` "
             "(bytes/fwd packet) and forward timing dilation `α` — through a differentiable "
             "realizability map (NO VAE in the gradient path); the VAE is used only as a "
             "val-anchored Mahalanobis realism gate (IDR).\n")
    L.append(f"- **Re-run with new seeds:** `{seeds}` (evaluation rows fixed across seeds; "
             "only optimizer initialization varies). All effectiveness numbers below are the "
             "mean±std **over these seeds**.")
    L.append(f"- **Denominator:** {res['denominator']}.")
    L.append(f"- **Threat model:** {res['threat_model']}.")
    L.append("- **Strict valid:** PAVE (Level-A) ∧ mined-density ∧ internal primitive-realizability.")
    L.append("- **Class ids:** `Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4`.")
    L.append("- All ASR / validity / IDR figures are **percentages**.\n")
    L.append("---\n")

    # ---------------- Table 1: effectiveness per class x victim (mean+/-std over seeds) --
    L.append("## 1. Effectiveness averaged over new seeds (mean±std %, per class × victim)\n")
    L.append("| Class | Victim | N clean-correct | Untargeted ASR | Targeted-Benign ASR | "
             "Strict Validity | Targeted Strict-Valid ASR | IDR | True-IDSR |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    per_class = defaultdict(lambda: defaultdict(list))
    for cl in classes:
        for v in victims:
            g = grp[(cl, v)]
            if not g:
                continue
            ncc = int(np.mean([x["n_clean_correct"] for x in g]))
            def ms(key):
                return (_mean([x[key] for x in g]), _std([x[key] for x in g]))
            ua, tb, sv = ms("untargeted_asr"), ms("targeted_benign_asr"), ms("strict_validity")
            tsv, idr, tid = ms("targeted_strict_valid_asr"), ms("IDR"), ms("true_idsr")
            L.append(f"| {cl} | {v} | {ncc} | {pct(*ua)} | {pct(*tb)} | {pct(*sv)} | "
                     f"{pct(*tsv)} | {pct(*idr)} | {pct(*tid)} |")
            for k, m in (("untargeted_asr", ua[0]), ("targeted_benign_asr", tb[0]),
                         ("targeted_strict_valid_asr", tsv[0]), ("strict_validity", sv[0]),
                         ("IDR", idr[0]), ("true_idsr", tid[0])):
                per_class[cl][k].append(m)

    # ---------------- Table 2: per-class macro + overall -------------------------------
    L.append("\n## 2. Per-class macro (mean over victims) and overall, %\n")
    L.append("| Class | Untargeted ASR | Targeted-Benign ASR | Targeted Strict-Valid ASR | "
             "Strict Validity | IDR | True-IDSR |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    keys = ("untargeted_asr", "targeted_benign_asr", "targeted_strict_valid_asr",
            "strict_validity", "IDR", "true_idsr")
    for cl in classes:
        L.append(f"| {cl} | " + " | ".join(pct(_mean(per_class[cl][k])) for k in keys) + " |")
    L.append("| **OVERALL** | " +
             " | ".join(pct(_mean([_mean(per_class[cl][k]) for cl in classes])) for k in keys) + " |")
    L.append("\n*Macro = unweighted mean over victims (and over classes for OVERALL).*\n")
    L.append("---\n")

    # ---------------- Best targeted examples per class ---------------------------------
    L.append("## 3. Best targeted adversarial examples (inverse-transformed, base vs adversarial)\n")
    L.append("For each attack class: the smallest normalized-cost row that was clean-correct and "
             "evaded to **Benign** while passing strict validity, chosen across all new seeds and "
             "victims. Values are **inverse-transformed raw CICFlowMeter units**.\n")

    delta_blocks = []
    for cl in classes:
        best = None  # (cost, victim, seed, row, npz)
        for v in victims:
            for s in seeds:
                f = ARTDIR / f"{cl}_{v}_seed{s}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                strict = d["strict_valid"].astype(bool)
                ok = d["clean_correct"].astype(bool) & d["benign"].astype(bool) & strict
                idxs = np.flatnonzero(ok)
                if idxs.size == 0:
                    continue
                r = int(idxs[np.argmin(d["cost_total"][idxs])])
                cost = float(d["cost_total"][r])
                if best is None or cost < best[0]:
                    best = (cost, v, s, r, d)
        if best is None:
            L.append(f"### {cl} → Benign\n\n*No successful strict-valid → Benign example across "
                     "the new seeds.*\n\n---\n")
            continue
        cost, v, s, r, d = best
        o = d["X_clean_raw"][r].astype(float)
        adv = d["X_adv_raw"][r].astype(float)
        L.append(f"### {cl} → Benign  (victim = {v}, seed = {s}, row = {r})\n")
        L.append(f"- primitives: forward packet-length augmentation `p = {float(d['p_cont'][r]):.3f}` → "
                 f"`p_real = {int(d['p'][r])}` bytes;  timing dilation `α = {float(d['alpha_cont'][r]):.4f}` → "
                 f"`α_real = {float(d['alpha'][r]):.4f}`  |  normalized cost = {cost:.4f}")
        L.append(f"- **true class = {cl}  →  adversarial prediction = "
                 f"{ID_TO_NAME[int(d['y_pred_adv'][r])]}**\n")
        L.append("#### Base vs. adversarial — all features (raw units)\n")
        L.append("| idx | feature | role | base (raw) | adversarial (raw) |")
        L.append("|---:|---|:--:|---:|---:|")
        for j, nm in enumerate(names):
            L.append(f"| {j} | {nm} | {ROLE_TAG[roles[nm]['role']]} | {fnum(o[j])} | {fnum(adv[j])} |")
        L.append("")

        # collect delta block for the class
        db = [f"### {cl} → Benign  (victim = {v}, seed = {s}, row = {r})\n",
              "| idx | feature | role | base | → | adversarial | Δ (adv − base) |",
              "|---:|---|:--:|---:|:--:|---:|---:|"]
        any_delta = False
        for j, nm in enumerate(names):
            if abs(adv[j] - o[j]) > 1e-5 + 1e-4 * abs(o[j]):
                any_delta = True
                db.append(f"| {j} | {nm} | {ROLE_TAG[roles[nm]['role']]} | {fnum(o[j])} | → | "
                          f"{fnum(adv[j])} | {fnum(adv[j] - o[j])} |")
        if not any_delta:
            db.append("| — | *(no feature changed)* | | | | | |")
        db.append("")
        delta_blocks.append("\n".join(db))
        L.append("---\n")

    # ---------------- deltas-only section ----------------------------------------------
    L.append("## 4. Feature deltas only (changed features per best example)\n")
    L.extend(delta_blocks)

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines); seeds={seeds}")


if __name__ == "__main__":
    main()
