"""Matched-budget comparison: targeted-Benign / targeted-strict-valid ASR vs perturbation
budget for the primitive-control attack, overlaid on the old aggregate attack's operating
point. Answers: is the primitive method better because of its representation/constraints,
or merely because it spends a larger budget?

The budget is swept by tightening the absolute primitive ceilings (p_max, alpha_max), which
lowers the achievable normalized cost. Each level reports (mean cost, targeted-Benign ASR,
targeted strict-valid ASR) pooled over class x victim (1 seed for speed).

Usage:
    PYTHONPATH=".;src" python scripts/budget_sweep_primitive.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from attack.run_cicids2017_primitive_attack import run, VICTIMS
from vae.cicids2017_stage_a import ATTACK_CLASSES

LEVELS = [
    {"p_max": 40.0, "alpha_max": 2.0},
    {"p_max": 120.0, "alpha_max": 5.0},
    {"p_max": 400.0, "alpha_max": 20.0},
    {"p_max": 1460.0, "alpha_max": 100.0},
]


def _pooled(cells):
    d = {"denom": 0, "tb": 0, "tsv": 0, "cost": []}
    for c in cells:
        d["denom"] += c["n_clean_correct"]
        d["tb"] += c["n_targeted_benign_success"]
        d["tsv"] += c["n_targeted_strict_valid"]
        if c["cost_total_mean"] == c["cost_total_mean"]:
            d["cost"].append(c["cost_total_mean"])
    return d


def main() -> None:
    out_root = Path("outputs/cicids2017_primitive_attack/budget_sweep")
    rows = []
    for lvl in LEVELS:
        od = out_root / f"p{int(lvl['p_max'])}_a{int(lvl['alpha_max'])}"
        res = run(classes=list(ATTACK_CLASSES), victims=list(VICTIMS), device="cpu",
                  test_limit=512, steps=40, lr=0.1, p_max=lvl["p_max"], alpha_max=lvl["alpha_max"],
                  mtu_cap=0.0, cost_weight=0.01, stage_a_dir=None, output_dir=od,
                  seeds=[42], init_noise=0.5)
        d = _pooled(res["cells"])
        rows.append({"p_max": lvl["p_max"], "alpha_max": lvl["alpha_max"],
                     "mean_cost": float(np.mean(d["cost"])),
                     "targeted_benign_asr": d["tb"] / d["denom"],
                     "targeted_strict_valid_asr": d["tsv"] / d["denom"]})
        print(json.dumps(rows[-1]), flush=True)

    # old aggregate operating point
    old = Path("outputs/cicids2017_vae_attacks_masked/attack_results.json")
    old_pt = None
    if old.exists():
        leg = json.loads(old.read_text())
        lr = [vd["A4"] for cd in leg["classes"].values() for vd in cd["victims"].values()]
        old_pt = {"mean_cost": float(np.nanmean([m["mean_normalized_cost"] for m in lr])),
                  "targeted_benign_asr": float(np.nanmean([m.get("targeted_benign_rate", np.nan) for m in lr])),
                  "targeted_strict_valid_asr": float(np.nanmean([m.get("ASR_L0_L1_L2", np.nan) for m in lr]))}

    md = ["# Matched-budget comparison (primitive sweep vs old aggregate)\n",
          "| Method | p_max | α_max | mean cost | Targeted-Benign ASR | Targeted Strict-Valid ASR |",
          "|---|--:|--:|--:|--:|--:|"]
    for r in rows:
        md.append(f"| primitive | {r['p_max']:.0f} | {r['alpha_max']:.0f} | {r['mean_cost']:.3f} "
                  f"| {r['targeted_benign_asr']*100:.1f} | {r['targeted_strict_valid_asr']*100:.1f} |")
    if old_pt:
        md.append(f"| old aggregate (A4) | - | - | {old_pt['mean_cost']:.3f} "
                  f"| {old_pt['targeted_benign_asr']*100:.1f} | {old_pt['targeted_strict_valid_asr']*100:.1f} |")
    out = out_root / "budget_curve.md"
    out.write_text("\n".join(md), encoding="utf-8")
    (out_root / "budget_curve.json").write_text(json.dumps({"primitive": rows, "old": old_pt}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
