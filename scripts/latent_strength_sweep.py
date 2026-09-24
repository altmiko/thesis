"""Search-strength sweep for the VAE-Latent-Primitive attack (ceiling estimation).

Sweeps one axis at a time from a fixed baseline and reports POOLED targeted strict-valid ASR
(and targeted-Benign ASR, mean cost, mean latent move) over all (class, victim) cells at a
single seed. The point is NOT to maximise ASR but to bound what the current latent attack can
reach; where extra search does not help, that is evidence about *where* capability is lost.

Also records gradient norms (||dL/dx_adv||, ||dL/dp||, ||dL/dalpha||, ||dL/ddecoder||,
||dL/dz_adv||) for representative high- and near-zero-performing cells from the baseline run.

Run (from repo root):
    PYTHONPATH="src;." python scripts/latent_strength_sweep.py --test-limit 256
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from attack.run_cicids2017_vae_latent_attack import run
from attack.vae_latent_primitive import LatentAttackConfig

BASELINE = dict(steps=120, learning_rate=0.08, objective="cw", kappa=0.0, epsilon_z=10.0,
                restarts=1, lr_schedule="constant", lambda_latent=0.005, lambda_cost=0.05,
                lambda_realism=0.001)

SWEEPS = {
    "epsilon_z": [5.0, 10.0, 20.0, 40.0],
    "restarts": [1, 4],
    "steps": [60, 120, 240],
    "lambda_latent": [0.0, 0.005, 0.05],
    "lambda_cost": [0.0, 0.05, 0.2],
    "kappa": [0.0, 5.0, 15.0],
    "lr_schedule": ["constant", "cosine"],
}

REPRESENTATIVE = [("DDoS", "cnn"), ("DDoS", "mlp"), ("Recon", "mlp")]


def _pool(cells):
    denom = sum(c["n_clean_correct"] for c in cells)
    tb = sum(c["n_targeted_benign_success"] for c in cells)
    tsv = sum(c["n_targeted_strict_valid"] for c in cells)
    cost = [c["cost_total_mean"] for c in cells if c["cost_total_mean"] == c["cost_total_mean"]]
    lat = [c["latent_l2_mean"] for c in cells if c["latent_l2_mean"] == c["latent_l2_mean"]]
    return {"denom": denom,
            "targeted_benign_asr": tb / denom if denom else float("nan"),
            "targeted_strict_valid_asr": tsv / denom if denom else float("nan"),
            "mean_cost": sum(cost) / len(cost) if cost else float("nan"),
            "mean_latent_l2": sum(lat) / len(lat) if lat else float("nan")}


def _run_cfg(overrides, *, classes, victims, seed, test_limit, scratch, log_grad=False):
    cfg = LatentAttackConfig(**{**BASELINE, **overrides}, log_grad_norms=log_grad)
    out = scratch / "cfg"
    if out.exists():
        shutil.rmtree(out)
    res = run(classes=classes, victims=victims, device="cpu", test_limit=test_limit,
              cost_weight=0.01,
              calibration_path=Path("artifacts/primattack/budget_calibration.json"),
              budget_name="maximum-evaluated", stage_a_dir=None,
              output_dir=out, seeds=[seed], config=cfg, variant="full")
    cells = res["cells"]
    # free disk: keep only the json summary, drop per-cell npz artifacts.
    art = out / "attack_artifacts"
    if art.exists():
        shutil.rmtree(art)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", default="DoS,DDoS,Recon,BruteForce")
    ap.add_argument("--victims", default="mlp,cnn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-limit", type=int, default=256)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/latent_strength_sweep"))
    a = ap.parse_args()
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    victims = [v.strip() for v in a.victims.split(",") if v.strip()]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    scratch = a.output_dir / "scratch"; scratch.mkdir(exist_ok=True)

    # 1) baseline run WITH grad-norm logging (gives every representative cell's grad norms).
    base_cells = _run_cfg({}, classes=classes, victims=victims, seed=a.seed,
                          test_limit=a.test_limit, scratch=scratch, log_grad=True)
    base_pool = _pool(base_cells)
    grad = {}
    for cl, vi in REPRESENTATIVE:
        m = [c for c in base_cells if c["class"] == cl and c["victim"] == vi]
        if m:
            grad[f"{cl}/{vi}"] = {"targeted_strict_valid_asr": m[0]["targeted_strict_valid_asr"],
                                  "grad_norms": m[0]["grad_norms"]}
    (a.output_dir / "grad_norms.json").write_text(json.dumps(grad, indent=2), encoding="utf-8")

    # 2) per-axis sweep.
    results = {"baseline_config": BASELINE, "seed": a.seed, "test_limit": a.test_limit,
               "classes": classes, "victims": victims, "baseline_pooled": base_pool, "sweeps": {}}
    for axis, values in SWEEPS.items():
        rows = []
        for v in values:
            if v == BASELINE.get(axis):
                rows.append({"value": v, **base_pool})
                continue
            cells = _run_cfg({axis: v}, classes=classes, victims=victims, seed=a.seed,
                             test_limit=a.test_limit, scratch=scratch)
            rows.append({"value": v, **_pool(cells)})
            print(f"[{axis}={v}] TSV={rows[-1]['targeted_strict_valid_asr']*100:.1f}% "
                  f"TB={rows[-1]['targeted_benign_asr']*100:.1f}% cost={rows[-1]['mean_cost']:.3f}",
                  flush=True)
        results["sweeps"][axis] = rows
        (a.output_dir / "sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # 3) markdown.
    lines = ["# VAE-Latent-Primitive search-strength sweep",
             "",
             f"Pooled over {len(classes)}x{len(victims)} cells, seed {a.seed}, "
             f"test_limit {a.test_limit}/class. Baseline: " +
             ", ".join(f"{k}={v}" for k, v in BASELINE.items()) + ".",
             "",
             f"**Baseline pooled**: targeted strict-valid ASR "
             f"{base_pool['targeted_strict_valid_asr']*100:.1f}%, targeted-Benign "
             f"{base_pool['targeted_benign_asr']*100:.1f}%, mean cost {base_pool['mean_cost']:.3f}, "
             f"mean latent move {base_pool['mean_latent_l2']:.2f}.", ""]
    for axis, rows in results["sweeps"].items():
        lines += [f"## {axis}", "", "| value | TSV-ASR % | TB-ASR % | mean cost | mean latent |",
                  "|---|--:|--:|--:|--:|"]
        for r in rows:
            lines.append(f"| {r['value']} | {r['targeted_strict_valid_asr']*100:.1f} | "
                         f"{r['targeted_benign_asr']*100:.1f} | {r['mean_cost']:.3f} | "
                         f"{r['mean_latent_l2']:.2f} |")
        lines.append("")
    lines += ["## Gradient norms (representative cells, baseline config)", "",
              "| cell | TSV-ASR % | dL/dx | dL/dp | dL/dalpha | dL/ddecoder | dL/dz |",
              "|---|--:|--:|--:|--:|--:|--:|"]
    for cell, d in grad.items():
        g = d["grad_norms"] or {}
        lines.append(f"| {cell} | {d['targeted_strict_valid_asr']*100:.1f} | "
                     f"{g.get('dL_dx_adv',0):.2e} | {g.get('dL_dp',0):.2e} | "
                     f"{g.get('dL_dalpha',0):.2e} | {g.get('dL_ddecoder_output',0):.2e} | "
                     f"{g.get('dL_dz_adv',0):.2e} |")
    (a.output_dir / "sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(scratch, ignore_errors=True)
    print("wrote", a.output_dir / "sweep.md")


if __name__ == "__main__":
    main()
