"""VAE Latent-Space Primitive-Constrained Attack on CICIDS2017-DistriNet (proposed method).

Optimizes z_adv in the per-class beta-VAE latent space; the decoder proposes a movement that
is collapsed into the realizable primitives (p, alpha) and passed through the SAME
realizability layer used by the direct primitive baseline. Success is measured AFTER discrete
projection. Separate output tree from the primitive baseline (never mixed).

Realism note: the attack VAE and the IDR/Mahalanobis gate are the SAME per-class VAE,
so the IDR score here is generator-relative, not independent evidence. Structural validity
is validator_v2 ``hybrid_valid`` only; realizability checks are diagnostic.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from attack.primattack_budget import BUDGET_NAMES, class_calibration, load_calibration
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel, SCALER_ATOL
from attack.realizability.validator import RealizabilityValidator
from attack.run_cicids2017_primitive_attack import (
    VICTIMS, _class_rows, evaluate_cell,
    _LENGTH_COLS, _TIMING_COLS, _RATE_COLS,
)
from attack.run_cicids2017_vae_attacks import _idr_mask
from attack.vae_latent_primitive import LatentAttackConfig, LatentPrimitiveAttack
from datasets.cicids2017 import CICIDS2017Adapter
from experiments.provenance import (
    artifact_provenance_arrays,
    build_provenance,
    deterministic_runtime,
    ensure_fresh_output_dir,
    load_row_ids,
)
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import ATTACK_CLASSES, load_stage_a

VARIANTS = {
    "full": {},
    "no_latent_reg": {"lambda_latent": 0.0},
    "no_realism": {"lambda_realism": 0.0},
    "ce": {"objective": "ce"},
}


def _seed(seed: int) -> None:
    deterministic_runtime(seed)

def _load_realism(idr_path: Path, device: str) -> dict[str, torch.Tensor]:
    s = np.load(idr_path)
    return {"mean": torch.tensor(s["mean"], dtype=torch.float32, device=device),
            "precision": torch.tensor(s["precision"], dtype=torch.float32, device=device),
            "threshold_sq": torch.tensor(float(s["threshold_sq"]), dtype=torch.float32, device=device)}


def run(*, classes, victims, device, test_limit, cost_weight, calibration_path, budget_name,
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
    calibration = load_calibration(calibration_path)
    stage_a_dir = stage_a_dir or (repo / "outputs" / "cicids2017_vae_stage_a")
    victim_dir = repo / "outputs" / "cicids2017distrinet" / "models"
    method_id = "vae_latent_primitive" if variant == "full" else f"vae_latent_primitive_{variant}"
    run_config = {**config.__dict__, "test_limit_per_class": test_limit,
                  "budget_name": budget_name, "calibration_path": str(calibration_path),
                  "calibration_fit_split": calibration["fit_split"], "seeds": seeds,
                  "attack_batch_size_per_class": test_limit,
                  "alpha_mapping": "exp(relu(mean_log_timing_ratio))"}
    checkpoint_paths = {
        **{f"victim_{name}": victim_dir / f"{name}_category.pt" for name in victims},
        **{f"vae_{name}": stage_a_dir / f"vae_{name}.pt" for name in classes},
        **{f"idr_{name}": stage_a_dir / f"idr_{name}.npz" for name in classes},
        "budget_calibration": calibration_path,
    }
    provenance = build_provenance(
        repo_root=repo, dataset=adapter.name, method_id=method_id, config=run_config,
        preprocessing_manifest=adapter._processed / "preprocessing_manifest.json",
        scaler=adapter._processed / "scaler.pkl", checkpoints=checkpoint_paths,
    )
    ensure_fresh_output_dir(output_dir)
    artifact_dir = output_dir / "attack_artifacts"; artifact_dir.mkdir()
    (output_dir / "run_manifest.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    all_row_ids = load_row_ids(adapter._processed, "test")

    results = {
        "dataset": adapter.name, "method_id": method_id,
        "attack": "VAE Latent-Space Primitive-Constrained Attack (z_adv optimized; grad flows through decoder)",
        "variant": variant, "threat_model": "targeted Attack->Benign",
        "denominator": "clean-correct malicious test rows per (class, victim)",
        "vae_role": "GENERATOR (encoder+decoder in the classifier-gradient path); realism gate is "
                    "the same VAE -> IDR is generator-relative, not independent",
        "strict_valid_definition": "validator_v2 (hybrid_valid) only",
        "config": run_config,
        "provenance": provenance,
        "feature_roles": {n: {"role": r.tag, "reason": why} for n, (r, why) in model.roles().items()},
        "cells": [],
    }

    for class_name in classes:
        cid = mapping.name_to_id[class_name]
        class_cfg = class_calibration(calibration, class_name, budget_name)
        idx = _class_rows(test.y, cid, test_limit, 42 + cid)
        raw_np = np.ascontiguousarray(np.asarray(raw_test[idx]), dtype=np.float32)
        raw = torch.tensor(raw_np, device=device)
        vae, _ = load_stage_a(
            adapter, stage_a_dir / f"vae_{class_name}.pt",
            expected_class_name=class_name, device=device,
        )
        idr_path = stage_a_dir / f"idr_{class_name}.npz"
        realism = _load_realism(idr_path, device)
        bounds = model.per_flow_bounds(raw, class_cfg.bounds_config())
        for vname in victims:
            victim = load_category_victim(
                victim_dir / f"{vname}_category.pt", adapter=adapter,
                expected_model_type=vname, device=device,
            )
            for seed in seeds:
                _seed(seed)
                res = attack.attack(vae, victim, raw, center, scale, bounds,
                                    target_class=0, realism=realism)
                adv_raw = res.x_adv_realized_raw
                fidx = torch.tensor([model.i[n] for n in val.frozen_names], device=device)
                assert torch.allclose(adv_raw[:, fidx], raw[:, fidx], atol=SCALER_ATOL, rtol=1e-4), \
                    "frozen feature changed in latent attack realization"
                masks, cost, yc, ya = evaluate_cell(model, val, victim, raw,
                                                    adv_raw, center, scale, cid, groups_idx)
                in_dist = _idr_mask(vae, (adv_raw - center) / scale, idr_path)
                ap = artifact_dir / f"{class_name}_{vname}_seed{seed}.npz"
                strict = masks["domain_valid"]
                checkpoint_ids = {
                    key: provenance["checkpoints"][key]["sha256"]
                    for key in (f"victim_{vname}", f"vae_{class_name}", f"idr_{class_name}")
                }
                np.savez_compressed(
                    ap, X_clean_raw=raw_np,
                    X_adv_raw=adv_raw.cpu().numpy().astype(np.float32),
                    X_adv_continuous_raw=res.x_adv_continuous_raw.cpu().numpy().astype(np.float32),
                    decoded_base_raw=res.decoded_base_raw.cpu().numpy().astype(np.float32),
                    decoded_adv_raw=res.decoded_adv_raw.cpu().numpy().astype(np.float32),
                    p=res.controls_realized["p"].cpu().numpy().astype(np.float32),
                    delay=res.controls_realized["delay"].cpu().numpy().astype(np.float32),
                    shape=res.controls_realized["shape"].cpu().numpy().astype(np.float32),
                    p_cont=res.controls_continuous["p"].cpu().numpy().astype(np.float32),
                    delay_cont=res.controls_continuous["delay"].cpu().numpy().astype(np.float32),
                    shape_cont=res.controls_continuous["shape"].cpu().numpy().astype(np.float32),
                    z0=res.z0.cpu().numpy().astype(np.float32),
                    z_adv=res.z_adv.cpu().numpy().astype(np.float32),
                    latent_l2=res.latent_l2.cpu().numpy().astype(np.float32),
                    recon_error=res.reconstruction_error.cpu().numpy().astype(np.float32),
                    latent_distance_sq=(res.latent_distance_sq.cpu().numpy().astype(np.float32)
                                        if res.latent_distance_sq is not None else np.zeros(len(idx), np.float32)),
                    timing_active=res.timing_active.cpu().numpy(),
                    y_true=np.full(len(idx), cid, dtype=np.int64),
                    true_label=np.full(len(idx), cid, dtype=np.int64),
                    y_pred_clean=yc.cpu().numpy().astype(np.int64),
                    clean_prediction=yc.cpu().numpy().astype(np.int64),
                    y_pred_adv=ya.cpu().numpy().astype(np.int64),
                    final_adversarial_prediction=ya.cpu().numpy().astype(np.int64),
                    clean_logits=res.clean_logits.cpu().numpy().astype(np.float32),
                    continuous_adversarial_logits=res.continuous_logits.cpu().numpy().astype(np.float32),
                    final_adversarial_logits=res.realized_logits.cpu().numpy().astype(np.float32),
                    cost_total=cost["total"].cpu().numpy().astype(np.float32),
                    cost_padding=cost["padding"].cpu().numpy().astype(np.float32),
                    cost_timing=cost["timing"].cpu().numpy().astype(np.float32),
                    target_success_flag=masks["targeted_success"].cpu().numpy(),
                    strict_valid=strict.cpu().numpy(),
                    **artifact_provenance_arrays(
                        provenance, row_ids=all_row_ids[idx], class_name=class_name,
                        victim=vname, method_id=method_id, seed=seed,
                        checkpoint_ids=checkpoint_ids,
                    ),
                    in_dist=in_dist.cpu().numpy(),
                    **{k: m.cpu().numpy() for k, m in masks.items()},
                )
                denom = int(masks["clean_correct"].sum()); cc = masks["clean_correct"]
                strict = masks["domain_valid"]
                rate = lambda m: (float((m & cc).sum()) / denom) if denom else float("nan")
                ll = res.latent_l2.cpu().numpy()[cc.cpu().numpy()]
                cell = {
                    "class": class_name, "victim": vname, "seed": seed, "variant": variant,
                    "artifact": str(ap), "n_total": len(idx), "n_clean_correct": denom,
                    "n_targeted_benign_success": int((masks["targeted_success"] & cc).sum()),
                    "n_targeted_strict_valid": int((masks["targeted_success"] & strict & cc).sum()),
                    "untargeted_asr": rate(masks["evasion"]),
                    "targeted_benign_asr": rate(masks["targeted_success"]),
                    "targeted_strict_valid_asr": rate(masks["targeted_success"] & strict),
                    "strict_validity": rate(strict),
                    "mined_validity": rate(masks["domain_valid"]),
                    "realizability_aware_validity_diagnostic": rate(
                        masks["primitive_transform_consistent"]
                    ),
                    "IDR_generator_relative": rate(in_dist),
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
    ap.add_argument("--calibration", type=Path,
                    default=Path("artifacts/primattack/budget_calibration.json"))
    ap.add_argument("--budget", choices=BUDGET_NAMES, default="maximum-evaluated")
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
        cost_weight=a.cost_weight, calibration_path=a.calibration, budget_name=a.budget,
        stage_a_dir=a.stage_a_dir, output_dir=a.output_dir, seeds=seeds, config=cfg,
        variant=a.variant)


if __name__ == "__main__":
    main()
