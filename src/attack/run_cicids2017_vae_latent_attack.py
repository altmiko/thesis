"""VAE Latent-Space Primitive-Constrained Attack on CICIDS2017-DistriNet (proposed method).

Optimizes z_adv in the per-class beta-VAE latent space; the decoder proposes a movement that
is collapsed into the realizable primitives (p, alpha) and passed through the SAME
realizability layer used by the direct primitive baseline. Success is measured AFTER discrete
projection. Separate output tree from the primitive baseline (never mixed).

Realism note (Part L): the attack VAE and the IDR/Mahalanobis gate are the SAME per-class VAE,
so the IDR score here is *generator-relative*, not independent evidence. PAVE (Level-A), the
mined density engine, and the internal realizability validator remain independent evaluators.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel, SCALER_ATOL
from attack.realizability.validator import RealizabilityValidator
from attack.run_cicids2017_primitive_attack import (
    VICTIMS, _class_rows, evaluate_cell, train_envelope,
    _LENGTH_COLS, _TIMING_COLS, _RATE_COLS,
)
from attack.vae_latent_primitive import LatentAttackConfig, LatentPrimitiveAttack
from datasets.cicids2017 import CICIDS2017Adapter
from evaluation.pave_style_validator import PAVEStyleValidator
from experiments.ablations import build_ablation
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import ATTACK_CLASSES, load_stage_a

VARIANTS = {
    "full": {},
    "no_latent_reg": {"lambda_latent": 0.0},
    "no_realism": {"lambda_realism": 0.0},
    "ce": {"objective": "ce"},
}


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _load_realism(idr_path: Path, device: str) -> dict[str, torch.Tensor]:
    s = np.load(idr_path)
    return {"mean": torch.tensor(s["mean"], dtype=torch.float32, device=device),
            "precision": torch.tensor(s["precision"], dtype=torch.float32, device=device),
            "threshold_sq": torch.tensor(float(s["threshold_sq"]), dtype=torch.float32, device=device)}


def run(*, classes, victims, device, test_limit, cost_weight, p_max, alpha_max, mtu_cap,
        stage_a_dir, output_dir, seeds, config: LatentAttackConfig, variant="full"):
    adapter = CICIDS2017Adapter()
    repo = adapter.repo_root
    manifest = adapter.feature_manifest(); transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    model = CICIDS2017PrimitiveModel(manifest); val = RealizabilityValidator(model)
    attack = LatentPrimitiveAttack(model, config)
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)
    groups_idx = {"padding": torch.tensor([model.i[n] for n in _LENGTH_COLS], device=device),
                  "timing": torch.tensor([model.i[n] for n in _TIMING_COLS], device=device),
                  "rate": torch.tensor([model.i[n] for n in _RATE_COLS], device=device)}

    test = adapter.load_split("test")
    raw_test = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    raw_train = np.load(adapter._processed / "X_train_pristine.npy", mmap_mode="r")
    layer1_fit = np.ascontiguousarray(raw_train[:200000], dtype=np.float32)
    layer2_path = repo / "constraints" / adapter.name / "mined.json"
    stage_a_dir = stage_a_dir or (repo / "outputs" / "cicids2017_vae_attacks" / "stage_a")
    victim_dir = repo / "outputs" / "cicids2017distrinet" / "models"
    artifact_dir = output_dir / "attack_artifacts"; artifact_dir.mkdir(parents=True, exist_ok=True)

    pave = PAVEStyleValidator(integer_tolerance=SCALER_ATOL, range_tolerance=SCALER_ATOL).fit(
        np.asarray(raw_train, dtype=np.float64), manifest.names, schema=manifest)
    envelope = train_envelope(raw_train, model.i)
    bounds_cfg = {"p_max": p_max, "alpha_max": alpha_max, "mtu_cap": mtu_cap,
                  **{f"env_{k}": v for k, v in envelope.items()}}

    results = {
        "dataset": adapter.name, "method_id": "vae_latent_primitive",
        "attack": "VAE Latent-Space Primitive-Constrained Attack (z_adv optimized; grad flows through decoder)",
        "variant": variant, "threat_model": "targeted Attack->Benign",
        "denominator": "clean-correct malicious test rows per (class, victim)",
        "vae_role": "GENERATOR (encoder+decoder in the classifier-gradient path); realism gate is "
                    "the same VAE -> IDR is generator-relative, not independent",
        "config": {**config.__dict__, "test_limit_per_class": test_limit, "p_max": p_max,
                   "alpha_max": alpha_max, "mtu_cap": mtu_cap, "seeds": seeds},
        "feature_roles": {n: {"role": r.tag, "reason": why} for n, (r, why) in model.roles().items()},
        "cells": [],
    }

    for class_name in classes:
        cid = mapping.name_to_id[class_name]
        idx = _class_rows(test.y, cid, test_limit, 42 + cid)
        raw_np = np.ascontiguousarray(np.asarray(raw_test[idx]), dtype=np.float32)
        raw = torch.tensor(raw_np, device=device)
        vae, _ = load_stage_a(adapter, stage_a_dir / f"vae_{class_name}.pt", device=device)
        idr_path = stage_a_dir / f"idr_{class_name}.npz"
        realism = _load_realism(idr_path, device)
        engine = build_ablation("A4", adapter, encoder_input_transform="asinh",
                                layer1_fit_x_raw=layer1_fit, layer2_path=layer2_path).engine
        bounds = model.per_flow_bounds(raw, bounds_cfg)
        for vname in victims:
            victim = load_category_victim(victim_dir / f"{vname}_category.pt", device=device)
            for seed in seeds:
                _seed(seed)
                res = attack.attack(vae, victim, raw, center, scale, bounds,
                                    target_class=0, realism=realism)
                adv_raw = res.x_adv_realized_raw
                fidx = torch.tensor([model.i[n] for n in val.frozen_names], device=device)
                assert torch.allclose(adv_raw[:, fidx], raw[:, fidx], atol=SCALER_ATOL, rtol=1e-4), \
                    "frozen feature changed in latent attack realization"
                masks, cost, yc, ya = evaluate_cell(model, val, victim, vae, engine, pave, raw,
                                                    adv_raw, center, scale, cid, idr_path, groups_idx)
                ap = artifact_dir / f"{class_name}_{vname}_seed{seed}.npz"
                np.savez_compressed(
                    ap, X_clean_raw=raw_np,
                    X_adv_raw=adv_raw.cpu().numpy().astype(np.float32),
                    p=res.controls_realized["p"].cpu().numpy().astype(np.float32),
                    alpha=res.controls_realized["alpha"].cpu().numpy().astype(np.float32),
                    p_cont=res.controls_continuous["p"].cpu().numpy().astype(np.float32),
                    alpha_cont=res.controls_continuous["alpha"].cpu().numpy().astype(np.float32),
                    z0=res.z0.cpu().numpy().astype(np.float32),
                    z_adv=res.z_adv.cpu().numpy().astype(np.float32),
                    latent_l2=res.latent_l2.cpu().numpy().astype(np.float32),
                    recon_error=res.reconstruction_error.cpu().numpy().astype(np.float32),
                    latent_distance_sq=(res.latent_distance_sq.cpu().numpy().astype(np.float32)
                                        if res.latent_distance_sq is not None else np.zeros(len(idx), np.float32)),
                    timing_active=res.timing_active.cpu().numpy(),
                    y_true=np.full(len(idx), cid, dtype=np.int64),
                    y_pred_clean=yc.cpu().numpy().astype(np.int64),
                    y_pred_adv=ya.cpu().numpy().astype(np.int64),
                    cost_total=cost["total"].cpu().numpy().astype(np.float32),
                    cost_padding=cost["padding"].cpu().numpy().astype(np.float32),
                    cost_timing=cost["timing"].cpu().numpy().astype(np.float32),
                    **{k: m.cpu().numpy() for k, m in masks.items()},
                )
                denom = int(masks["clean_correct"].sum()); cc = masks["clean_correct"]
                strict = masks["pave_valid"] & masks["mined_valid"] & masks["realizable"]
                rate = lambda m: (float((m & cc).sum()) / denom) if denom else float("nan")
                ll = res.latent_l2.cpu().numpy()[cc.cpu().numpy()]
                cell = {
                    "class": class_name, "victim": vname, "seed": seed, "variant": variant,
                    "artifact": str(ap), "n_total": len(idx), "n_clean_correct": denom,
                    "n_targeted_benign_success": int((masks["benign"] & cc).sum()),
                    "n_targeted_strict_valid": int((masks["benign"] & strict & cc).sum()),
                    "untargeted_asr": rate(masks["evasion"]),
                    "targeted_benign_asr": rate(masks["benign"]),
                    "targeted_strict_valid_asr": rate(masks["benign"] & strict),
                    "strict_validity": rate(strict), "pave_validity": rate(masks["pave_valid"]),
                    "mined_validity": rate(masks["mined_valid"]),
                    "realizability_aware_validity": rate(masks["realizable"]),
                    "IDR_generator_relative": rate(masks["in_dist"]),
                    "cost_total_mean": float(cost["total"][cc].mean()) if denom else float("nan"),
                    "latent_l2_mean": float(np.mean(ll)) if ll.size else float("nan"),
                    "latent_l2_median": float(np.median(ll)) if ll.size else float("nan"),
                    "grad_norms": res.grad_norms,
                }
                results["cells"].append(cell)
                print(json.dumps({k: cell[k] for k in
                                  ("class", "victim", "seed", "n_clean_correct", "targeted_benign_asr",
                                   "targeted_strict_valid_asr", "strict_validity", "latent_l2_mean",
                                   "cost_total_mean")}), flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "attack_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    ap.add_argument("--victims", default=",".join(VICTIMS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test-limit", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--objective", default="cw", choices=["cw", "ce"])
    ap.add_argument("--epsilon-z", type=float, default=10.0)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"])
    ap.add_argument("--log-grad-norms", action="store_true")
    ap.add_argument("--lambda-latent", type=float, default=0.005)
    ap.add_argument("--lambda-cost", type=float, default=0.05)
    ap.add_argument("--lambda-realism", type=float, default=0.001)
    ap.add_argument("--p-max", type=float, default=1460.0)
    ap.add_argument("--alpha-max", type=float, default=100.0)
    ap.add_argument("--mtu-cap", type=float, default=0.0)
    ap.add_argument("--cost-weight", type=float, default=0.01)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--stage-a-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/cicids2017_vae_latent_attack"))
    a = ap.parse_args()
    overrides = VARIANTS[a.variant]
    cfg = LatentAttackConfig(
        steps=a.steps, learning_rate=a.learning_rate, objective=overrides.get("objective", a.objective),
        kappa=a.kappa, epsilon_z=a.epsilon_z, restarts=a.restarts, lr_schedule=a.lr_schedule,
        lambda_latent=overrides.get("lambda_latent", a.lambda_latent),
        lambda_cost=a.lambda_cost, lambda_realism=overrides.get("lambda_realism", a.lambda_realism),
        log_grad_norms=a.log_grad_norms)
    classes = [x.strip() for x in a.classes.split(",") if x.strip()]
    victims = [x.strip() for x in a.victims.split(",") if x.strip()]
    seeds = [int(x) for x in str(a.seeds).split(",") if str(x).strip()]
    run(classes=classes, victims=victims, device=a.device, test_limit=a.test_limit,
        cost_weight=a.cost_weight, p_max=a.p_max, alpha_max=a.alpha_max, mtu_cap=a.mtu_cap,
        stage_a_dir=a.stage_a_dir, output_dir=a.output_dir, seeds=seeds, config=cfg, variant=a.variant)


if __name__ == "__main__":
    main()
