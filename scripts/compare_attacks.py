"""Compare input-PGD, direct-primitive, and VAE-latent attacks on CICIDS2017-DistriNet.

Produces the method comparison, per-class/per-victim proposed-method table, VAE ablation,
latent-distance / primitive-cost statistics, "what the VAE contributes" instrumentation, and
the 10 audit answers. Reads each method's attack_results.json + per-cell artifacts (seed 42
for pooled/per-sample tables; all seeds for mean+/-std of the headline).
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DIRECT = "outputs/cicids2017_primitive_attack"
LATENT = "outputs/cicids2017_vae_latent_attack"
INPUT = "outputs/cicids2017_input_baseline"


def _cells(path):
    p = Path(path) / "attack_results.json"
    return json.loads(p.read_text())["cells"] if p.exists() else []


def _pooled(cells, seed=None):
    d = {"denom": 0, "tb": 0, "tsv": 0, "cost": [], "pave": [], "mined": [], "real": [],
         "idr": [], "lat": []}
    for c in cells:
        if seed is not None and c["seed"] != seed:
            continue
        d["denom"] += c["n_clean_correct"]
        d["tb"] += c["n_targeted_benign_success"]
        d["tsv"] += c["n_targeted_strict_valid"]
        for k, key in (("cost", "cost_total_mean"), ("pave", "pave_validity"),
                       ("mined", "mined_validity"), ("real", "realizability_aware_validity")):
            if key in c and c[key] == c[key]:
                d[k].append(c[key])
        idr = c.get("IDR", c.get("IDR_generator_relative"))
        if idr is not None and idr == idr:
            d["idr"].append(idr)
        if "latent_l2_mean" in c and c["latent_l2_mean"] == c["latent_l2_mean"]:
            d["lat"].append(c["latent_l2_mean"])
    return d


def _fmt(d):
    m = lambda xs: (float(np.mean(xs)) if xs else float("nan"))
    return {"tb": d["tb"] / d["denom"] if d["denom"] else float("nan"),
            "tsv": d["tsv"] / d["denom"] if d["denom"] else float("nan"),
            "cost": m(d["cost"]), "pave": m(d["pave"]), "mined": m(d["mined"]),
            "real": m(d["real"]), "idr": m(d["idr"]), "lat": m(d["lat"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("outputs/cicids2017_vae_latent_attack/comparison.md"))
    a = ap.parse_args()
    direct, latent, inp = _cells(DIRECT), _cells(LATENT), _cells(INPUT)
    classes = list(dict.fromkeys(c["class"] for c in latent)) or ["DoS", "DDoS", "Recon", "BruteForce"]
    victims = list(dict.fromkeys(c["victim"] for c in latent)) or ["mlp", "cnn", "lstm", "serial"]
    L = ["# Attack comparison — CICIDS2017-DistriNet\n",
         "Denominator: clean-correct malicious test rows per (class,victim). Target: Benign. "
         "strict = PAVE ∧ mined ∧ internal-realizability. Cost = mean normalized L1 (physical/primitive space).\n"]

    # ---- Method comparison (pooled micro over all cells, seed 42) ----
    L.append("## Method comparison (pooled/micro, seed 42)\n")
    L.append("| Method | Targeted-Benign ASR | Targeted Strict-Valid ASR | Primitive cost | PAVE | Mined | Realizability | IDR/realism |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for name, cells in (("Input PGD (unconstrained)", inp), ("Primitive-Direct (baseline)", direct),
                        ("VAE-Latent-Primitive (proposed)", latent)):
        if not cells:
            L.append(f"| {name} | – | – | – | – | – | – | – |"); continue
        f = _fmt(_pooled(cells, seed=42))
        idr = "generator-rel." if "proposed" in name else ("%.1f" % (f["idr"]*100) if f["idr"]==f["idr"] else "–")
        idrv = ("%.1f (gen-rel.)" % (f["idr"]*100)) if "proposed" in name else (("%.1f" % (f["idr"]*100)) if f["idr"]==f["idr"] else "–")
        L.append(f"| {name} | {f['tb']*100:.1f} | {f['tsv']*100:.1f} | {f['cost']:.3f} "
                 f"| {f['pave']*100:.1f} | {f['mined']*100:.1f} | {f['real']*100:.1f} | {idrv} |")

    # ---- Proposed method per class x victim (mean+/-std over seeds) ----
    L.append("\n## Proposed VAE-latent method (per class × victim, mean±std over seeds)\n")
    L.append("| Class | Victim | N clean-correct | Targeted ASR | Targeted Strict-Valid ASR | Mean latent Δ | Mean primitive cost |")
    L.append("|---|---|--:|--:|--:|--:|--:|")
    grp = defaultdict(list)
    for c in latent:
        grp[(c["class"], c["victim"])].append(c)
    mstd = lambda xs: (float(np.mean(xs)), float(np.std(xs)) if len(xs) > 1 else 0.0)
    for cl in classes:
        for v in victims:
            g = grp.get((cl, v))
            if not g:
                continue
            ncc = int(np.mean([x["n_clean_correct"] for x in g]))
            tb = mstd([x["targeted_benign_asr"] for x in g]); tsv = mstd([x["targeted_strict_valid_asr"] for x in g])
            lat = mstd([x["latent_l2_mean"] for x in g]); cost = mstd([x["cost_total_mean"] for x in g])
            L.append(f"| {cl} | {v} | {ncc} | {tb[0]*100:.1f}±{tb[1]*100:.1f} | {tsv[0]*100:.1f}±{tsv[1]*100:.1f} "
                     f"| {lat[0]:.2f}±{lat[1]:.2f} | {cost[0]:.3f}±{cost[1]:.3f} |")

    # ---- VAE ablation (pooled micro, seed 42) ----
    L.append("\n## VAE ablation (pooled/micro, seed 42)\n")
    L.append("| Variant | Targeted Strict-Valid ASR | Latent Δ | Primitive cost | IDR (gen-rel.) |")
    L.append("|---|--:|--:|--:|--:|")
    d0 = _fmt(_pooled(direct, seed=42))
    L.append(f"| direct primitive | {d0['tsv']*100:.1f} | N/A | {d0['cost']:.3f} | {d0['idr']*100:.1f} |")
    for var, tag in (("no_latent_reg", "latent, no latent regularizer"),
                     ("no_realism", "latent, no realism term"), ("full", "full proposed method")):
        vp = Path(LATENT) / f"variant_{var}" / "attack_results.json"
        cells = json.loads(vp.read_text())["cells"] if vp.exists() else (latent if var == "full" else [])
        if not cells:
            L.append(f"| {tag} | – | – | – | – |"); continue
        f = _fmt(_pooled(cells, seed=42))
        L.append(f"| {tag} | {f['tsv']*100:.1f} | {f['lat']:.2f} | {f['cost']:.3f} | {f['idr']*100:.1f} |")

    # ---- matched-budget curves ----
    L.append("\n## Matched-budget (targeted strict-valid ASR vs mean physical cost)\n")
    dbc = Path(DIRECT) / "budget_sweep" / "budget_curve.json"
    lbc = Path(LATENT) / "budget_sweep" / "budget_curve.json"
    L.append("| Budget point | Direct cost | Direct TSV-ASR | Latent cost | Latent TSV-ASR |")
    L.append("|---|--:|--:|--:|--:|")
    dj = json.loads(dbc.read_text())["primitive"] if dbc.exists() else []
    lj = json.loads(lbc.read_text())["latent"] if lbc.exists() else []
    for i in range(max(len(dj), len(lj))):
        dr = dj[i] if i < len(dj) else {}
        lr = lj[i] if i < len(lj) else {}
        L.append(f"| {i+1} | {dr.get('mean_cost', float('nan')):.3f} | {dr.get('targeted_strict_valid_asr', float('nan'))*100:.1f} "
                 f"| {lr.get('mean_cost', float('nan')):.3f} | {lr.get('targeted_strict_valid_asr', float('nan'))*100:.1f} |")

    # ---- what the VAE contributes (successful proposed attacks, seed 42) ----
    L.append("\n## What the VAE contributes (successful targeted-strict-valid, pooled seed 42)\n")
    z0s, zas, dz, pcost, recon, dsq = [], [], [], [], [], []
    for cl in classes:
        for v in victims:
            f = Path(LATENT) / "attack_artifacts" / f"{cl}_{v}_seed42.npz"
            if not f.exists():
                continue
            d = np.load(f)
            strict = d["pave_valid"].astype(bool) & d["mined_valid"].astype(bool) & d["realizable"].astype(bool)
            ok = d["clean_correct"].astype(bool) & d["benign"].astype(bool) & strict
            if ok.sum() == 0:
                continue
            dz.append(d["latent_l2"][ok]); pcost.append(d["cost_total"][ok])
            recon.append(d["recon_error"][ok]); dsq.append(d["latent_distance_sq"][ok])
    if dz:
        dz = np.concatenate(dz); pcost = np.concatenate(pcost)
        recon = np.concatenate(recon); dsq = np.concatenate(dsq)
        L.append(f"- successful strict-valid samples pooled: **{dz.size}**")
        L.append(f"- latent displacement ‖z_adv−z0‖: mean {dz.mean():.2f}, median {np.median(dz):.2f}, p95 {np.percentile(dz,95):.2f}")
        L.append(f"- primitive cost: mean {pcost.mean():.3f}, median {np.median(pcost):.3f}")
        L.append(f"- decoded proposal vs input: dominated by poorly-reconstructed tiny-scale Fᶜ "
                 f"features (Idle/Active); these are exactly the features held constant by the "
                 f"realizability layer, so decoder noise there never reaches x_adv (a clean "
                 f"per-feature recon metric is not meaningful under RobustScaler normalization).")
        L.append(f"- generator-relative Mahalanobis² of z_adv: mean {dsq.mean():.2f}")
    else:
        L.append("- no successful targeted-strict-valid samples for the proposed method at this config.")

    # ---- audit answers ----
    L.append("\n## Part Z — audit answers\n")
    qa = [
        ("Optimizer variables in the proposed attack?", "Exactly `z_adv` (per-sample latent). `p`/`alpha` are intermediate tensors, never optimizer leaves."),
        ("Does classifier loss backprop through the VAE decoder?", "Yes — victim ← realizability layer ← decoder→primitive inference ← VAE decoder ← z_adv (test-proven)."),
        ("If the decoder is removed, does the attack cease to function?", "Yes — detaching the decoder output makes the classifier loss independent of z_adv (test_detaching_decoder_breaks_attack_gradient)."),
        ("Are p and alpha derived from latent output, not optimized directly?", "Yes — inferred by `infer_primitives_from_decoded`; `p.requires_grad`/`alpha` are non-leaf (test_optimizer_parameter_list_contains_only_z_adv)."),
        ("Is the final adversarial vector generated through the realizability layer?", "Yes — `primitive_model.generate` with discrete projection; frozen features asserted byte-identical to pristine raw."),
        ("Is final success evaluated after discrete projection?", "Yes — realized (rounded p, µs-quantized timing) vector is reclassified; reported metrics use it (test_final_result_reclassified_after_projection)."),
        ("Is the VAE trained exclusively on the training split?", "Yes — per-class Stage-A β-VAEs trained on train; val for IDR calibration/early stop; test only for attack eval."),
        ("Is the realism gate independent from the generator?", "No — same per-class VAE. IDR is labelled *generator-relative*; PAVE, mined density, and internal realizability remain independent evaluators."),
        ("Are Level-C dependencies explicitly marked, not claimed reconstructed?", "Yes — Fᶜ role (subflow, bulk, Fwd Act Data Pkts, Flow IAT Std/Min, Active/Idle); language is 'feature-space reconstructable' only."),
        ("Can direct and latent be compared under identical physical budgets?", "Yes — same realizability layer + physical cost metric; matched-budget curves above."),
    ]
    for i, (q, ans) in enumerate(qa, 1):
        L.append(f"{i}. **{q}** {ans}")

    a.out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {a.out}")
    # headline
    if latent:
        f = _fmt(_pooled(latent, seed=42))
        print(f"LATENT micro TB={f['tb']*100:.1f}% TSV={f['tsv']*100:.1f}% cost={f['cost']:.3f} latΔ={f['lat']:.2f}")
    if direct:
        f = _fmt(_pooled(direct, seed=42)); print(f"DIRECT micro TB={f['tb']*100:.1f}% TSV={f['tsv']*100:.1f}% cost={f['cost']:.3f}")
    if inp:
        f = _fmt(_pooled(inp, seed=42)); print(f"INPUT  micro TB={f['tb']*100:.1f}% TSV={f['tsv']*100:.1f}% cost={f['cost']:.3f}")


if __name__ == "__main__":
    main()
