"""Aggregate + report the realizability-aware primitive-control CICIDS2017 attack.

Reads the per-cell results/artifacts written by ``attack.run_cicids2017_primitive_attack``
and emits the thesis tables (main effectiveness, primitive cost, validity, failure reasons,
old-vs-new), multi-seed mean+/-std, 95% bootstrap CIs, p/alpha distributions, and the
smallest-cost successful Target->Benign strict-valid audit case per class.

Usage (thesis env):
    PYTHONPATH=".;src" python scripts/analyze_primitive_attack.py \
        --results outputs/cicids2017_primitive_attack/attack_results.json \
        --old outputs/cicids2017_vae_attacks_masked/attack_results.json \
        --out outputs/cicids2017_primitive_attack/results_tables.md
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel
from attack.realizability.validator import RealizabilityValidator
from datasets import get_adapter

RNG = np.random.default_rng(12345)
HEADLINE = ("untargeted_asr", "targeted_benign_asr", "targeted_strict_valid_asr")


# ------------------------------------------------------------------ helpers
def _mean(xs):
    xs = [x for x in xs if x == x]  # drop nan
    return float(np.mean(xs)) if xs else float("nan")


def _std(xs):
    xs = [x for x in xs if x == x]
    return float(np.std(xs)) if len(xs) > 1 else 0.0


def load_cell_masks(artifact: str) -> dict:
    d = np.load(artifact)
    return {k: d[k] for k in d.files}


def bootstrap_ci(cc: np.ndarray, success: np.ndarray, b: int = 2000):
    """95% bootstrap CI of P(success | clean-correct) resampling clean-correct rows."""
    idx = np.flatnonzero(cc)
    if idx.size == 0:
        return (float("nan"), float("nan"))
    s = success[idx].astype(float)
    boot = [s[RNG.integers(0, idx.size, idx.size)].mean() for _ in range(b)]
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("outputs/cicids2017_primitive_attack/attack_results.json"))
    ap.add_argument("--old", type=Path, default=Path("outputs/cicids2017_vae_attacks_masked/attack_results.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs/cicids2017_primitive_attack/results_tables.md"))
    a = ap.parse_args()

    res = json.loads(a.results.read_text())
    cells = res["cells"]
    classes = list(dict.fromkeys(c["class"] for c in cells))
    victims = list(dict.fromkeys(c["victim"] for c in cells))
    seeds = sorted({c["seed"] for c in cells})
    names = list(res["feature_roles"].keys())
    roles = res["feature_roles"]
    L = []
    L.append(f"# Results — realizability-aware primitive-control attack (CICIDS2017-DistriNet)\n")
    L.append(f"- Denominator: **{res['denominator']}**.")
    L.append(f"- Threat model: **{res['threat_model']}**.")
    L.append(f"- Seeds: {seeds} (fixed eval rows; init-noise varies optimization). "
             f"Config: {json.dumps(res['config'])}\n")

    # group cells by (class,victim) over seeds
    grp = defaultdict(list)
    for c in cells:
        grp[(c["class"], c["victim"])].append(c)

    # ---------- Table 1: main effectiveness (mean+/-std over seeds) ----------
    L.append("## Main effectiveness (per class x victim, mean±std over seeds)\n")
    L.append("| Class | Victim | N clean-correct | Untargeted ASR | Targeted-Benign ASR | Strict Validity | Targeted Strict-Valid ASR |")
    L.append("|---|---|--:|--:|--:|--:|--:|")
    per_class = defaultdict(lambda: defaultdict(list))
    pooled = defaultdict(lambda: [0, 0, 0, 0, 0])  # denom, unt, tb, strictvalid, tb_strict
    for cl in classes:
        for v in victims:
            g = grp[(cl, v)]
            if not g:
                continue
            ncc = int(np.mean([x["n_clean_correct"] for x in g]))
            ua = (_mean([x["untargeted_asr"] for x in g]), _std([x["untargeted_asr"] for x in g]))
            tb = (_mean([x["targeted_benign_asr"] for x in g]), _std([x["targeted_benign_asr"] for x in g]))
            sv = (_mean([x["strict_validity"] for x in g]), _std([x["strict_validity"] for x in g]))
            tsv = (_mean([x["targeted_strict_valid_asr"] for x in g]), _std([x["targeted_strict_valid_asr"] for x in g]))
            L.append(f"| {cl} | {v} | {ncc} | {ua[0]*100:.1f}±{ua[1]*100:.1f} | {tb[0]*100:.1f}±{tb[1]*100:.1f} "
                     f"| {sv[0]*100:.1f}±{sv[1]*100:.1f} | {tsv[0]*100:.1f}±{tsv[1]*100:.1f} |")
            for k, val in (("untargeted_asr", ua[0]), ("targeted_benign_asr", tb[0]),
                           ("strict_validity", sv[0]), ("targeted_strict_valid_asr", tsv[0])):
                per_class[cl][k].append(val)
            # pooled from seed-averaged counts
            pooled[cl][0] += ncc
            pooled[cl][1] += int(np.mean([x["n_untargeted_success"] for x in g]))
            pooled[cl][2] += int(np.mean([x["n_targeted_benign_success"] for x in g]))
            pooled[cl][3] += int(np.mean([x["n_strict_valid"] for x in g]))
            pooled[cl][4] += int(np.mean([x["n_targeted_strict_valid"] for x in g]))
    # per-class macro + micro rows
    L.append("\n### Per-class macro (mean over victims) and pooled/micro\n")
    L.append("| Class | Macro Untargeted ASR | Macro Targeted-Benign ASR | Macro Targeted Strict-Valid ASR | Micro Targeted-Benign ASR | Micro Targeted Strict-Valid ASR |")
    L.append("|---|--:|--:|--:|--:|--:|")
    tot = [0, 0, 0, 0, 0]
    for cl in classes:
        d = pooled[cl]
        for j in range(5):
            tot[j] += d[j]
        micro_tb = d[2] / d[0] if d[0] else float("nan")
        micro_tsv = d[4] / d[0] if d[0] else float("nan")
        L.append(f"| {cl} | {_mean(per_class[cl]['untargeted_asr'])*100:.1f} "
                 f"| {_mean(per_class[cl]['targeted_benign_asr'])*100:.1f} "
                 f"| {_mean(per_class[cl]['targeted_strict_valid_asr'])*100:.1f} "
                 f"| {micro_tb*100:.1f} | {micro_tsv*100:.1f} |")
    overall_macro_tb = _mean([_mean(per_class[cl]['targeted_benign_asr']) for cl in classes])
    overall_macro_tsv = _mean([_mean(per_class[cl]['targeted_strict_valid_asr']) for cl in classes])
    L.append(f"| **OVERALL** | {_mean([_mean(per_class[cl]['untargeted_asr']) for cl in classes])*100:.1f} "
             f"| {overall_macro_tb*100:.1f} | {overall_macro_tsv*100:.1f} "
             f"| {tot[2]/tot[0]*100:.1f} | {tot[4]/tot[0]*100:.1f} |")
    L.append("\n*Macro = unweighted mean over victims/classes; micro = pooled successes / pooled clean-correct.*\n")

    # ---------- Table 2: primitive cost + distributions (seed 42) ----------
    L.append("## Primitive cost & perturbation distribution (seed 42)\n")
    L.append("| Class | Victim | median p | p95 p | %p@cap | median α | p95 α | %α@cap | median cost | p95 cost |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    dist_rows = {}
    for cl in classes:
        for v in victims:
            art = f"outputs/cicids2017_primitive_attack/attack_artifacts/{cl}_{v}_seed42.npz"
            if not Path(art).exists():
                continue
            d = load_cell_masks(art)
            cc = d["clean_correct"].astype(bool)
            p, al = d["p"][cc], d["alpha"][cc]
            phi, ahi = d["p_hi"][cc], d["alpha_hi"][cc]
            ct = d["cost_total"][cc]
            p_at = float(np.mean(p >= phi - 1e-3)) if p.size else float("nan")
            a_at = float(np.mean(al >= ahi - 1e-3)) if al.size else float("nan")
            L.append(f"| {cl} | {v} | {np.median(p):.0f} | {np.percentile(p,95):.0f} | {p_at*100:.1f} "
                     f"| {np.median(al):.3f} | {np.percentile(al,95):.3f} | {a_at*100:.1f} "
                     f"| {np.median(ct):.3f} | {np.percentile(ct,95):.3f} |")
            dist_rows[(cl, v)] = d

    # p/alpha on successful targeted-valid subset (section 12)
    L.append("\n### p/α on SUCCESSFUL targeted-strict-valid samples only (pooled per class, seed 42)\n")
    L.append("| Class | n success | p mean | p median | p p95 | p max | α mean | α median | α p95 | α max |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for cl in classes:
        P, A = [], []
        for v in victims:
            d = dist_rows.get((cl, v))
            if d is None:
                continue
            sub = d["clean_correct"].astype(bool) & d["benign"].astype(bool) & \
                  d["pave_valid"].astype(bool) & d["mined_valid"].astype(bool) & d["realizable"].astype(bool)
            P.append(d["p"][sub]); A.append(d["alpha"][sub])
        P = np.concatenate(P) if P else np.array([]); A = np.concatenate(A) if A else np.array([])
        if P.size == 0:
            L.append(f"| {cl} | 0 | - | - | - | - | - | - | - | - |"); continue
        L.append(f"| {cl} | {P.size} | {P.mean():.1f} | {np.median(P):.0f} | {np.percentile(P,95):.0f} | {P.max():.0f} "
                 f"| {A.mean():.3f} | {np.median(A):.3f} | {np.percentile(A,95):.3f} | {A.max():.3f} |")

    # ---------- Table 3: validity breakdown (seed 42, macro over victims) ----------
    L.append("\n## Validity breakdown (mean over victims, seed 42)\n")
    L.append("| Class | Feature (PAVE) | Dependency | Mined | Discrete | Realizability-aware | Strict |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for cl in classes:
        acc = defaultdict(list)
        for v in victims:
            g = [x for x in grp[(cl, v)] if x["seed"] == 42]
            if not g:
                continue
            x = g[0]
            acc["pave"].append(x["pave_validity"]); acc["dep"].append(x["dependency_validity"])
            acc["mined"].append(x["mined_validity"]); acc["disc"].append(x["discreteness_validity"])
            acc["real"].append(x["realizability_aware_validity"]); acc["strict"].append(x["strict_validity"])
        L.append(f"| {cl} | {_mean(acc['pave'])*100:.1f} | {_mean(acc['dep'])*100:.1f} | {_mean(acc['mined'])*100:.1f} "
                 f"| {_mean(acc['disc'])*100:.1f} | {_mean(acc['real'])*100:.1f} | {_mean(acc['strict'])*100:.1f} |")

    # ---------- Table 4: failure reasons (pooled seed 42) ----------
    L.append("\n## Failure reasons (pooled over class×victim, seed 42, among clean-correct)\n")
    L.append("| Rule category | Fail count | Fail % |")
    L.append("|---|--:|--:|")
    fc = defaultdict(int); denom_all = 0
    for (cl, v), d in dist_rows.items():
        cc = int(d["clean_correct"].sum()); denom_all += cc
        for k, cat in (("dependency", "dep_ok"), ("packet_summary", "packet_ok"),
                       ("timing", "timing_ok"), ("negative_rate", "rate_ok"),
                       ("discreteness", "disc_ok"), ("frozen", "frozen_ok")):
            fc[k] += int(((~d[cat].astype(bool)) & d["clean_correct"].astype(bool)).sum())
    for k, c in fc.items():
        L.append(f"| {k} | {c} | {c/denom_all*100:.3f} |")

    # ---------- Table 5: bootstrap CIs (pooled per class, seed 42) ----------
    L.append("\n## 95% bootstrap CIs for headline metrics (pooled clean-correct, seed 42)\n")
    L.append("| Class | Targeted-Benign ASR [95% CI] | Targeted Strict-Valid ASR [95% CI] |")
    L.append("|---|--:|--:|")
    for cl in classes:
        cc_all, tb_all, tsv_all = [], [], []
        for v in victims:
            d = dist_rows.get((cl, v))
            if d is None:
                continue
            cc = d["clean_correct"].astype(bool)
            strict = d["pave_valid"].astype(bool) & d["mined_valid"].astype(bool) & d["realizable"].astype(bool)
            cc_all.append(cc); tb_all.append(d["benign"].astype(bool)); tsv_all.append(d["benign"].astype(bool) & strict)
        cc_all = np.concatenate(cc_all); tb_all = np.concatenate(tb_all); tsv_all = np.concatenate(tsv_all)
        tb = tb_all[cc_all].mean(); tsv = tsv_all[cc_all].mean()
        ltb, htb = bootstrap_ci(cc_all, tb_all); ltsv, htsv = bootstrap_ci(cc_all, tsv_all)
        L.append(f"| {cl} | {tb*100:.1f} [{ltb*100:.1f}, {htb*100:.1f}] | {tsv*100:.1f} [{ltsv*100:.1f}, {htsv*100:.1f}] |")

    # ---------- Table 6: old vs new ----------
    if a.old.exists():
        L.append("\n## Old (aggregate 9-feature, ablation A4) vs New (primitive-control)\n")
        ad = get_adapter("cicids2017"); man = ad.feature_manifest(); tr = ad.feature_transform()
        model = CICIDS2017PrimitiveModel(man); val = RealizabilityValidator(model)
        sc = torch.tensor(np.asarray(tr.scale, np.float64)); ce = torch.tensor(np.asarray(tr.center, np.float64))
        leg = json.loads(a.old.read_text())
        lr = [vd["A4"] for cd in leg["classes"].values() for vd in cd["victims"].values()]
        old_tb = _mean([m.get("targeted_benign_rate", float("nan")) for m in lr])
        old_asr = _mean([m["ASR_raw"] for m in lr])
        old_validasr = _mean([m.get("ASR_L0_L1_L2", float("nan")) for m in lr])
        old_cost = _mean([m["mean_normalized_cost"] for m in lr])
        # old physical realizability (packet/timing/rate) from inverse-transformed scaled artifacts
        cats = defaultdict(int); N = 0
        for f in sorted(glob.glob("outputs/cicids2017_vae_attacks_masked/attack_artifacts/*_A4.npz")):
            arr = np.load(f)
            radv = torch.tensor(arr["X_adv"].astype(np.float64)) * sc + ce
            rcln = torch.tensor(arr["X_clean"].astype(np.float64)) * sc + ce
            rep = val.validate(radv, rcln); N += radv.shape[0]
            for cat in ("packet_summary_fail", "timing_fail", "negative_rate_fail"):
                cats[cat] += int(rep.categories[cat].sum())
        # new physical realizability
        new_cats = defaultdict(int); Nn = 0
        for d in dist_rows.values():
            Nn += d["clean_correct"].shape[0]
            new_cats["packet_summary_fail"] += int((~d["packet_ok"].astype(bool)).sum())
            new_cats["timing_fail"] += int((~d["timing_ok"].astype(bool)).sum())
            new_cats["negative_rate_fail"] += int((~d["rate_ok"].astype(bool)).sum())
        new_tb = tot[2] / tot[0]; new_tsv = tot[4] / tot[0]
        new_cost = _mean([c["cost_total_mean"] for c in cells if c["seed"] == 42])
        L.append("| Method | Targeted-Benign ASR | Targeted Strict-Valid ASR | Mean cost | Packet-summary fails | Timing fails | Neg-rate fails |")
        L.append("|---|--:|--:|--:|--:|--:|--:|")
        L.append(f"| old aggregate (A4) | {old_tb*100:.1f} | {old_validasr*100:.1f} | {old_cost:.2f} "
                 f"| {cats['packet_summary_fail']}/{N} | {cats['timing_fail']}/{N} | {cats['negative_rate_fail']}/{N} |")
        L.append(f"| new primitive | {new_tb*100:.1f} | {new_tsv*100:.1f} | {new_cost:.2f} "
                 f"| {new_cats['packet_summary_fail']}/{Nn} | {new_cats['timing_fail']}/{Nn} | {new_cats['negative_rate_fail']}/{Nn} |")
        L.append(f"\n*Old realizability recomputed with the SAME internal validator on inverse-transformed "
                 f"old adversarial samples (physical categories only; the aggregate attack has no frozen/dependency contract).*")

    # ---------- Audit cases: smallest-cost successful Target->Benign strict-valid per class ----------
    L.append("\n## Adversarial audit cases (smallest-cost successful Target→Benign strict-valid, seed 42)\n")
    idn = get_adapter("cicids2017").class_mapping().id_to_name
    for cl in classes:
        best = None
        for v in victims:
            d = dist_rows.get((cl, v))
            if d is None:
                continue
            strict = d["pave_valid"].astype(bool) & d["mined_valid"].astype(bool) & d["realizable"].astype(bool)
            ok = d["clean_correct"].astype(bool) & d["benign"].astype(bool) & strict
            idxs = np.flatnonzero(ok)
            if idxs.size == 0:
                continue
            r = idxs[np.argmin(d["cost_total"][idxs])]
            cost = float(d["cost_total"][r])
            if best is None or cost < best[0]:
                best = (cost, cl, v, r, d)
        if best is None:
            L.append(f"### {cl}: no successful targeted strict-valid sample\n"); continue
        cost, _, v, r, d = best
        o, adv = d["X_clean_raw"][r].astype(float), d["X_adv_raw"][r].astype(float)
        L.append(f"### {cl} → Benign (victim={v}, row={r})\n")
        L.append(f"- p_cont={d['p_cont'][r]:.3f} → p_real={d['p'][r]:.0f} bytes; "
                 f"α_cont={d['alpha_cont'][r]:.4f} → α_real={d['alpha'][r]:.4f}; normalized cost={cost:.4f}")
        L.append(f"- base pred={idn[int(d['y_pred_clean'][r])]} → final pred={idn[int(d['y_pred_adv'][r])]}\n")
        L.append("| feature | role | base | adversarial |")
        L.append("|---|:--:|--:|--:|")
        for j, nm in enumerate(names):
            if abs(adv[j] - o[j]) > 1e-5 + 1e-4 * abs(o[j]):
                rr = roles[nm]["role"]
                tag = {"primitive_controlled": "prim", "direct_derived": "d-der",
                       "conditional_derived": "c-der", "rate": "rate"}.get(rr, rr)
                L.append(f"| {nm} | {tag} | {o[j]:.4g} | {adv[j]:.4g} |")
        # impossible-case checks
        c = lambda n: adv[names.index(n)]
        checks = {
            "fwd mean<min": c("Fwd Packet Length Mean") < c("Fwd Packet Length Min") - 1e-6,
            "fwd max>total": c("Fwd Packet Length Max") > c("Total Length of Fwd Packet") + 1e-6,
            "pkt mean<min": c("Packet Length Mean") < c("Packet Length Min") - 1e-6,
            "fwd IAT max>total": c("Fwd IAT Max") > c("Fwd IAT Total") + 1e-3,
            "fwd IAT total>dur": c("Fwd IAT Total") > c("Flow Duration") + 1e-3,
            "flow IAT mean>max": c("Flow IAT Mean") > c("Flow IAT Max") + 1e-3,
            "any rate<0": min(c("Flow Bytes/s"), c("Flow Packets/s"), c("Fwd Packets/s"), c("Bwd Packets/s")) < 0,
            "duration<=0": c("Flow Duration") <= 0,
        }
        L.append(f"\n*Impossible-case checks (all must be False):* " +
                 ", ".join(f"{k}={bool(x)}" for k, x in checks.items()))
        L.append("")

    a.out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {a.out}  ({len(L)} lines)")
    print(f"OVERALL macro targeted-benign ASR={overall_macro_tb*100:.1f}%  "
          f"macro targeted-strict-valid ASR={overall_macro_tsv*100:.1f}%  "
          f"micro TB={tot[2]/tot[0]*100:.1f}%  micro TSV={tot[4]/tot[0]*100:.1f}%")


if __name__ == "__main__":
    main()
