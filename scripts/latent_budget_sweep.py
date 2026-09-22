"""Matched-budget curve for the VAE-latent attack: sweep lambda_cost to trace
(mean physical cost) -> (targeted strict-valid ASR), for overlay on the direct-primitive
budget curve. 1 seed, reduced test-limit for speed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from attack.run_cicids2017_vae_latent_attack import run, VICTIMS
from attack.vae_latent_primitive import LatentAttackConfig
from vae.cicids2017_stage_a import ATTACK_CLASSES

LAMBDAS = [1.0, 0.2, 0.05, 0.0]  # high cost-penalty -> low budget ... none -> high budget


def _pooled(cells):
    denom = sum(c["n_clean_correct"] for c in cells)
    tsv = sum(c["n_targeted_strict_valid"] for c in cells)
    tb = sum(c["n_targeted_benign_success"] for c in cells)
    cost = [c["cost_total_mean"] for c in cells if c["cost_total_mean"] == c["cost_total_mean"]]
    return {"mean_cost": float(np.mean(cost)), "targeted_benign_asr": tb / denom,
            "targeted_strict_valid_asr": tsv / denom}


def main() -> None:
    out_root = Path("outputs/cicids2017_vae_latent_attack/budget_sweep")
    rows = []
    for lc in LAMBDAS:
        cfg = LatentAttackConfig(steps=120, learning_rate=0.08, objective="cw",
                                 lambda_latent=0.005, lambda_cost=lc, lambda_realism=0.001,
                                 epsilon_z=10.0, init_noise=0.3)
        res = run(classes=list(ATTACK_CLASSES), victims=list(VICTIMS), device="cpu",
                  test_limit=512, cost_weight=0.01, p_max=1460.0, alpha_max=100.0, mtu_cap=0.0,
                  stage_a_dir=None, output_dir=out_root / f"lc{lc}", seeds=[42], config=cfg,
                  variant=f"lc{lc}")
        pt = _pooled(res["cells"]); pt["lambda_cost"] = lc
        rows.append(pt); print(json.dumps(pt), flush=True)
    (out_root).mkdir(parents=True, exist_ok=True)
    (out_root / "budget_curve.json").write_text(json.dumps({"latent": rows}, indent=2))
    md = ["# VAE-latent matched-budget curve\n", "| λ_cost | mean cost | Targeted-Benign ASR | Targeted Strict-Valid ASR |",
          "|--:|--:|--:|--:|"]
    for r in rows:
        md.append(f"| {r['lambda_cost']} | {r['mean_cost']:.3f} | {r['targeted_benign_asr']*100:.1f} | {r['targeted_strict_valid_asr']*100:.1f} |")
    (out_root / "budget_curve.md").write_text("\n".join(md))
    print(f"wrote {out_root/'budget_curve.md'}")


if __name__ == "__main__":
    main()
