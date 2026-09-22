"""Assemble the latent-bottleneck comparison from every method's attack_results.json + artifacts.

Produces (under --output-dir):
* the 5-method main comparison table (targeted-Benign ASR, targeted strict-valid ASR, validity,
  feature/primitive cost, latent distance);
* per-class x victim targeted strict-valid ASR grids for each method;
* a validity breakdown (PAVE / mined / internal-consistency / strict) per method;
* gradient-norm diagnostics for representative high- and near-zero cells;
* a matched-budget table: cost-capped targeted strict-valid ASR for the three constrained methods.

Reads only JSON + npz; no model compute. Missing methods are skipped with a note.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# (label, output dir, internal-validity mask key in artifacts, is_latent)
METHODS = [
    ("Input PGD", "outputs/cicids2017_input_baseline", "realizable", False),
    ("Primitive-Direct", "outputs/cicids2017_primitive_attack", "realizable", False),
    ("VAE-Latent-Raw", "outputs/cicids2017_latent_raw", "mask_valid", True),
    ("VAE-Latent-Masked", "outputs/cicids2017_latent_masked", "mask_valid", True),
    ("VAE-Latent-Primitive", "outputs/cicids2017_vae_latent_primitive_strong", "realizable", True),
]
PROPOSED_BASE = ("VAE-Latent-Primitive (proposed cfg)", "outputs/cicids2017_vae_latent_attack")
BUDGET_METHODS = ["Primitive-Direct", "VAE-Latent-Masked", "VAE-Latent-Primitive"]
COST_CAPS = [0.1, 0.25, 0.5, 1.0, 2.0, float("inf")]


def _load(path):
    p = Path(path) / "attack_results.json"
    return json.loads(p.read_text()) if p.exists() else None


def _cells(res, seed):
    return [c for c in res["cells"] if seed is None or c["seed"] == seed]


def _wmean(cells, key):
    num = den = 0.0
    for c in cells:
        v = c.get(key)
        if v is not None and v == v:
            w = c["n_clean_correct"]
            num += w * v; den += w
    return num / den if den else float("nan")


def _pool(cells):
    denom = sum(c["n_clean_correct"] for c in cells)
    tb = sum(c.get("n_targeted_benign_success", 0) for c in cells)
    tsv = sum(c.get("n_targeted_strict_valid", 0) for c in cells)
    idr = _wmean(cells, "IDR")
    if idr != idr:
        idr = _wmean(cells, "IDR_generator_relative")
    return {
        "denom": denom,
        "tb_asr": tb / denom if denom else float("nan"),
        "tsv_asr": tsv / denom if denom else float("nan"),
        "validity": _wmean(cells, "strict_validity"),
        "pave": _wmean(cells, "pave_validity"),
        "mined": _wmean(cells, "mined_validity"),
        "cost": _wmean(cells, "cost_total_mean"),
        "latent": _wmean(cells, "latent_l2_mean"),
        "idr": idr,
    }


def _fmt_pct(x):
    return f"{x*100:.1f}" if x == x else "n/a"


def _matched_budget(res, internal_key, caps):
    """Cost-capped targeted strict-valid ASR pooled over cells (per-sample from artifacts)."""
    denom = 0
    succ_cost = []  # cost of each strict-valid-benign success (among clean-correct)
    for c in res["cells"]:
        z = np.load(c["artifact"])
        cc = z["clean_correct"].astype(bool)
        denom += int(cc.sum())
        strict = z["pave_valid"].astype(bool) & z["mined_valid"].astype(bool)
        if internal_key in z:
            strict &= z[internal_key].astype(bool)
        good = z["benign"].astype(bool) & strict & cc
        succ_cost.extend(z["cost_total"][good].tolist())
    succ_cost = np.asarray(succ_cost)
    return {f"{cap}": (float((succ_cost <= cap).sum()) / denom if denom else float("nan"))
            for cap in caps}, denom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/latent_bottleneck"))
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    seed = a.seed

    loaded = {label: (_load(d), key, lat) for label, d, key, lat in METHODS}
    pooled = {}
    for label, d, key, lat in METHODS:
        res = loaded[label][0]
        pooled[label] = _pool(_cells(res, seed)) if res else None

    out = {"seed": seed, "methods": {}}
    lines = ["# CICIDS2017 VAE-latent bottleneck — method comparison",
             "",
             f"Pooled over the (class x victim) grid at seed {seed}. Denominator = clean-correct "
             "malicious rows. The three VAE-latent variants share a matched search config; Input "
             "PGD and Primitive-Direct keep their own optimizers (see report).",
             "",
             "| Method | Targeted-Benign ASR | Targeted Strict-Valid ASR | Validity | Feature/primitive cost | Latent distance | IDR |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for label, d, key, lat in METHODS:
        p = pooled[label]
        if p is None:
            lines.append(f"| {label} | (not run) | | | | | |")
            continue
        out["methods"][label] = p
        latent = f"{p['latent']:.2f}" if lat and p["latent"] == p["latent"] else "N/A"
        lines.append(f"| {label} | {_fmt_pct(p['tb_asr'])} | {_fmt_pct(p['tsv_asr'])} | "
                     f"{_fmt_pct(p['validity'])} | {p['cost']:.3f} | {latent} | {_fmt_pct(p['idr'])} |")
    # proposed-config primitive row for reference
    pres = _load(PROPOSED_BASE[1])
    if pres:
        pp = _pool(_cells(pres, seed))
        out["methods"][PROPOSED_BASE[0]] = pp
        lines.append(f"| {PROPOSED_BASE[0]} | {_fmt_pct(pp['tb_asr'])} | {_fmt_pct(pp['tsv_asr'])} | "
                     f"{_fmt_pct(pp['validity'])} | {pp['cost']:.3f} | {pp['latent']:.2f} | {_fmt_pct(pp['idr'])} |")
    lines.append("")

    # per-class x victim TSV-ASR grids
    classes = ["DoS", "DDoS", "Recon", "BruteForce"]
    victims = ["mlp", "cnn", "lstm", "serial"]
    for label, d, key, lat in METHODS:
        res = loaded[label][0]
        if not res:
            continue
        lines += [f"## Per-class x victim targeted strict-valid ASR (%) — {label}", "",
                  "| class \\ victim | " + " | ".join(victims) + " |",
                  "|---" * (len(victims) + 1) + "|"]
        idx = {(c["class"], c["victim"]): c for c in _cells(res, seed)}
        for cl in classes:
            row = [cl]
            for vi in victims:
                c = idx.get((cl, vi))
                row.append(_fmt_pct(c["targeted_strict_valid_asr"]) if c else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # validity breakdown
    lines += ["## Validity breakdown (pooled, %)", "",
              "| Method | PAVE (Level-A) | mined density | strict (all gates) |",
              "|---|--:|--:|--:|"]
    for label, d, key, lat in METHODS:
        p = pooled[label]
        if p is None:
            continue
        lines.append(f"| {label} | {_fmt_pct(p['pave'])} | {_fmt_pct(p['mined'])} | {_fmt_pct(p['validity'])} |")
    lines.append("")

    # gradient-norm diagnostics
    lines += ["## Gradient-norm diagnostics (representative cells)", "",
              "| Method | cell | TSV-ASR % | dL/dx | dL/dp | dL/dalpha | dL/ddecoder | dL/dz |",
              "|---|---|--:|--:|--:|--:|--:|--:|"]
    rep = [("DoS", "lstm"), ("DoS", "serial"), ("DDoS", "cnn"), ("DDoS", "mlp"), ("Recon", "mlp")]
    grad_out = {}
    for label, d, key, lat in METHODS:
        res = loaded[label][0]
        if not res or not lat:
            continue
        idx = {(c["class"], c["victim"]): c for c in _cells(res, seed)}
        for cl, vi in rep:
            c = idx.get((cl, vi))
            if not c or not c.get("grad_norms"):
                continue
            g = c["grad_norms"]
            grad_out.setdefault(label, {})[f"{cl}/{vi}"] = {
                "tsv": c["targeted_strict_valid_asr"], "grad_norms": g}
            lines.append(f"| {label} | {cl}/{vi} | {_fmt_pct(c['targeted_strict_valid_asr'])} | "
                         f"{g.get('dL_dx_adv',0):.2e} | {g.get('dL_dp',0):.2e} | "
                         f"{g.get('dL_dalpha',0):.2e} | {g.get('dL_ddecoder_output',0):.2e} | "
                         f"{g.get('dL_dz_adv',0):.2e} |")
    lines.append("")

    # matched-budget
    lines += ["## Matched-budget: cost-capped targeted strict-valid ASR (%)", "",
              "Fraction of clean-correct rows that are strict-valid Benign AND whose realized "
              "feature cost is <= the cap. Same denominator per method.", "",
              "| Method | " + " | ".join(("cost<=inf" if c == float("inf") else f"cost<={c}") for c in COST_CAPS) + " |",
              "|---" * (len(COST_CAPS) + 1) + "|"]
    mb_out = {}
    for label, d, key, lat in METHODS:
        if label not in BUDGET_METHODS:
            continue
        res = loaded[label][0]
        if not res:
            continue
        caps, denom = _matched_budget(res, key, COST_CAPS)
        mb_out[label] = {"denom": denom, "caps": caps}
        cells = " | ".join(_fmt_pct(caps[f"{c}"]) for c in COST_CAPS)
        lines.append(f"| {label} | {cells} |")
    lines.append("")

    out["gradient_norms"] = grad_out
    out["matched_budget"] = mb_out
    (a.output_dir / "comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (a.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", a.output_dir / "comparison.md")


if __name__ == "__main__":
    main()
