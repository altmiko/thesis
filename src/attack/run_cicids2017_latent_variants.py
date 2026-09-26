"""VAE latent decoder-movement attacks on CICIDS2017-DistriNet (masked / raw variants).

Two genuine VAE latent attacks whose decoder movement is NOT compressed into (p, delay, shape):

* ``--variant masked`` -> ``VAE-Latent-Masked``: decoder movement applied to every PERTURBABLE
  feature of the CICIDS2017 perturbation mask; FROZEN copied from pristine input; DERIVED_EXACT
  recomputed from parents; Layer-0 domain clamp. Masked feature-space attack (NOT Level-C).
* ``--variant raw`` -> ``VAE-Latent-Raw``: DIAGNOSTIC only. Decoder movement on all 79 features
  with a minimal Layer-0 domain clamp. Measures raw decoder/manifold evasion power before the
  perturbation mask or the primitive realizability layer. Not a proposed attack.

``z_adv`` is the only optimizer leaf; the decoder stays in the classifier-gradient path; feature
values are never optimized directly. Structural validity is validator_v2 ``hybrid_valid`` only;
IDR remains a separate realism metric. Mask frozen/derived checks remain diagnostics, not gates.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from attack.masks import get_dataset_mask
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel, SCALER_ATOL, _INTEGER_FEATURES
from attack.run_cicids2017_primitive_attack import (
    VICTIMS, _class_rows, _LENGTH_COLS, _TIMING_COLS, _RATE_COLS)
from attack.run_cicids2017_vae_attacks import _idr_mask
from attack.vae_latent_primitive import LatentAttackConfig
from attack.vae_latent_variants import LatentMaskedAttack, LatentRawAttack
from datasets.cicids2017 import CICIDS2017Adapter
from validation.attack_interface import structural_masks
from experiments.provenance import (
    artifact_provenance_arrays,
    build_provenance,
    deterministic_runtime,
    ensure_fresh_output_dir,
    load_row_ids,
)
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import ATTACK_CLASSES, load_stage_a


def _seed(seed: int) -> None:
    deterministic_runtime(seed)


def _load_realism(idr_path: Path, device: str) -> dict[str, torch.Tensor]:
    s = np.load(idr_path)
    return {"mean": torch.tensor(s["mean"], dtype=torch.float32, device=device),
            "precision": torch.tensor(s["precision"], dtype=torch.float32, device=device),
            "threshold_sq": torch.tensor(float(s["threshold_sq"]), dtype=torch.float32, device=device)}


def _decompose_cost(adv_raw, raw, scale, groups_idx):
    F = raw.shape[1]
    out = {"total": (((adv_raw - raw).abs() / scale).sum(1) / F)}
    for name, idx in groups_idx.items():
        out[name] = ((adv_raw[:, idx] - raw[:, idx]).abs() / scale[idx]).sum(1) / F
    return out


def evaluate_variant(*, rmask, victim, base_vae, raw, adv_raw, center, scale,
                     class_id, idr_path, groups_idx):
    """Variant masks; strict validity is validator_v2 hybrid_valid only."""
    with torch.no_grad():
        x_clean = (raw - center) / scale
        x_adv = (adv_raw - center) / scale
        clean_pred = victim(x_clean).argmax(1)
        adv_pred = victim(x_adv).argmax(1)
        clean_correct = clean_pred == class_id
        evasion = adv_pred != class_id
        benign = adv_pred == 0

        mined = torch.tensor(structural_masks(adv_raw.detach().cpu().numpy(),
                                              source_raw=raw.detach().cpu().numpy())["hybrid_valid"],
                             device=raw.device)
        in_dist = _idr_mask(base_vae, x_adv, idr_path)
        frozen_ok = ~rmask.frozen_violation_mask(adv_raw, raw, atol=SCALER_ATOL, rtol=1e-4)
        derived_ok = ~rmask.derived_consistency_mask(adv_raw, scale=scale, tol=1e-3)
        changed = ((adv_raw - raw).abs() > (1e-5 + 1e-4 * raw.abs())).sum(1)
        cost = _decompose_cost(adv_raw, raw, scale, groups_idx)

    masks = {
        "clean_correct": clean_correct, "evasion": evasion, "benign": benign,
        "mined_valid": mined, "in_dist": in_dist,
        "frozen_ok": frozen_ok, "derived_ok": derived_ok,
        "mask_valid": frozen_ok & derived_ok,
    }
    return masks, cost, changed, clean_pred, adv_pred


def _build_attack(variant, rmask, projector, cfg, model):
    if variant == "raw":
        return LatentRawAttack(projector, cfg)
    if variant == "masked":
        int_names = [n for n in rmask.mask.perturbable if n in _INTEGER_FEATURES]
        int_idx = [model.i[n] for n in int_names]
        return LatentMaskedAttack(rmask, projector, cfg, integer_perturbable_idx=int_idx)
    raise ValueError(f"unknown variant {variant!r}")


def run(*, variant, classes, victims, device, test_limit, stage_a_dir, output_dir, seeds,
        config: LatentAttackConfig):
    adapter = CICIDS2017Adapter()
    repo = adapter.repo_root
    manifest = adapter.feature_manifest(); transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    model = CICIDS2017PrimitiveModel(manifest)
    rmask = get_dataset_mask(adapter.name).resolve(manifest)
    projector = rmask.generator_projector()
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)
    groups_idx = {"padding": torch.tensor([model.i[n] for n in _LENGTH_COLS], device=device),
                  "timing": torch.tensor([model.i[n] for n in _TIMING_COLS], device=device),
                  "rate": torch.tensor([model.i[n] for n in _RATE_COLS], device=device)}
    attack = _build_attack(variant, rmask, projector, config, model)

    test = adapter.load_split("test")
    raw_test = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    raw_train = np.load(adapter._processed / "X_train_pristine.npy", mmap_mode="r")
    stage_a_dir = stage_a_dir or (repo / "outputs" / "cicids2017_vae_stage_a")
    victim_dir = repo / "outputs" / "cicids2017distrinet" / "models"


    method_id = f"vae_latent_{variant}"
    desc = {"masked": "VAE-Latent-Masked (z_adv optimized; decoder movement on PERTURBABLE features; "
                      "frozen copied, DERIVED_EXACT recomputed, Layer-0 domain clamp)",
            "raw": "VAE-Latent-Raw (DIAGNOSTIC; z_adv optimized; decoder movement on all 79 features; "
                   "minimal Layer-0 domain clamp only)"}[variant]
    run_config = {**config.__dict__, "test_limit_per_class": test_limit, "seeds": seeds,
                  "attack_batch_size_per_class": test_limit}
    checkpoint_paths = {
        **{f"victim_{name}": victim_dir / f"{name}_category.pt" for name in victims},
        **{f"vae_{name}": stage_a_dir / f"vae_{name}.pt" for name in classes},
        **{f"idr_{name}": stage_a_dir / f"idr_{name}.npz" for name in classes},
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
        "dataset": adapter.name, "method_id": method_id, "attack": desc, "variant": variant,
        "threat_model": "targeted Attack->Benign",
        "denominator": "clean-correct malicious test rows per (class, victim)",
        "vae_role": "GENERATOR (encoder+decoder in the classifier-gradient path); realism gate is "
                    "the same VAE -> IDR is generator-relative",
        "strict_valid_definition": "validator_v2 (hybrid_valid) only",
        "perturbable_features": list(rmask.mask.perturbable),
        "config": run_config,
        "provenance": provenance,
        "cells": [],
    }

    for class_name in classes:
        cid = mapping.name_to_id[class_name]
        idx = _class_rows(test.y, cid, test_limit, 42 + cid)
        raw_np = np.ascontiguousarray(np.asarray(raw_test[idx]), dtype=np.float32)
        raw = torch.tensor(raw_np, device=device)
        vae, _ = load_stage_a(
            adapter, stage_a_dir / f"vae_{class_name}.pt",
            expected_class_name=class_name, device=device,
        )
        idr_path = stage_a_dir / f"idr_{class_name}.npz"
        realism = _load_realism(idr_path, device)
        for vname in victims:
            victim = load_category_victim(
                victim_dir / f"{vname}_category.pt", adapter=adapter,
                expected_model_type=vname, device=device,
            )
            for seed in seeds:
                _seed(seed)
                res = attack.attack(vae, victim, raw, center, scale, target_class=0, realism=realism)
                adv_raw = res.x_adv_realized_raw
                if variant == "masked":
                    fidx = rmask._frozen_t.to(device)
                    assert torch.allclose(adv_raw[:, fidx], raw[:, fidx], atol=SCALER_ATOL, rtol=1e-4), \
                        "frozen feature changed in masked latent attack"
                masks, cost, changed, yc, ya = evaluate_variant(
                    rmask=rmask, victim=victim, base_vae=vae, raw=raw,
                    adv_raw=adv_raw, center=center, scale=scale, class_id=cid, idr_path=idr_path,
                    groups_idx=groups_idx)
                ap = artifact_dir / f"{class_name}_{vname}_seed{seed}.npz"
                ll = res.latent_l2.cpu().numpy()
                strict = masks["mined_valid"]
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
                    z0=res.z0.cpu().numpy().astype(np.float32),
                    z_adv=res.z_adv.cpu().numpy().astype(np.float32),
                    latent_l2=ll.astype(np.float32),
                    changed_features=changed.cpu().numpy().astype(np.int64),
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
                    latent_distance_sq=(res.latent_distance_sq.cpu().numpy().astype(np.float32)
                                        if res.latent_distance_sq is not None else np.zeros(len(idx), np.float32)),
                    target_success_flag=masks["benign"].cpu().numpy(),
                    strict_valid=strict.cpu().numpy(),
                    **artifact_provenance_arrays(
                        provenance, row_ids=all_row_ids[idx], class_name=class_name,
                        victim=vname, method_id=method_id, seed=seed,
                        checkpoint_ids=checkpoint_ids,
                    ),
                    **{k: m.cpu().numpy() for k, m in masks.items()},
                )
                denom = int(masks["clean_correct"].sum()); cc = masks["clean_correct"]
                strict = masks["mined_valid"]
                rate = lambda m: (float((m & cc).sum()) / denom) if denom else float("nan")
                llc = ll[cc.cpu().numpy()]
                cell = {
                    "class": class_name, "victim": vname, "seed": seed, "variant": variant,
                    "artifact": str(ap), "n_total": len(idx), "n_clean_correct": denom,
                    "n_targeted_benign_success": int((masks["benign"] & cc).sum()),
                    "n_targeted_strict_valid": int((masks["benign"] & strict & cc).sum()),
                    "untargeted_asr": rate(masks["evasion"]),
                    "targeted_benign_asr": rate(masks["benign"]),
                    "targeted_strict_valid_asr": rate(masks["benign"] & strict),
                    "strict_validity": rate(strict),
                    "mined_validity": rate(masks["mined_valid"]),
                    "mask_validity_diagnostic": rate(masks["mask_valid"]),
                    "frozen_ok_rate": rate(masks["frozen_ok"]),
                    "derived_ok_rate": rate(masks["derived_ok"]),
                    "dependency_validity": rate(masks["derived_ok"]),
                    "IDR_generator_relative": rate(masks["in_dist"]),
                    "cost_total_mean": float(cost["total"][cc].mean()) if denom else float("nan"),
                    "mean_changed_features": float(changed[cc].float().mean()) if denom else float("nan"),
                    "latent_l2_mean": float(np.mean(llc)) if llc.size else float("nan"),
                    "latent_l2_median": float(np.median(llc)) if llc.size else float("nan"),
                    "grad_norms": res.grad_norms,
                }
                results["cells"].append(cell)
                print(json.dumps({k: cell[k] for k in
                                  ("class", "victim", "seed", "n_clean_correct", "targeted_benign_asr",
                                   "targeted_strict_valid_asr", "strict_validity", "mask_validity_diagnostic",
                                   "latent_l2_mean", "cost_total_mean")}), flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "attack_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=["masked", "raw"])
    ap.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    ap.add_argument("--victims", default=",".join(VICTIMS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test-limit", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--objective", default="cw", choices=["cw", "ce"])
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--epsilon-z", type=float, default=10.0)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"])
    ap.add_argument("--lambda-latent", type=float, default=0.005)
    ap.add_argument("--lambda-cost", type=float, default=0.05)
    ap.add_argument("--lambda-realism", type=float, default=0.001)
    ap.add_argument("--log-grad-norms", action="store_true")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--stage-a-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    a = ap.parse_args()
    cfg = LatentAttackConfig(
        steps=a.steps, learning_rate=a.learning_rate, objective=a.objective, kappa=a.kappa,
        epsilon_z=a.epsilon_z, restarts=a.restarts, lr_schedule=a.lr_schedule,
        lambda_latent=a.lambda_latent, lambda_cost=a.lambda_cost, lambda_realism=a.lambda_realism,
        log_grad_norms=a.log_grad_norms)
    out = a.output_dir or Path(f"outputs/cicids2017_latent_{a.variant}")
    run(variant=a.variant, classes=[c.strip() for c in a.classes.split(",") if c.strip()],
        victims=[v.strip() for v in a.victims.split(",") if v.strip()], device=a.device,
        test_limit=a.test_limit, stage_a_dir=a.stage_a_dir, output_dir=out,
        seeds=[int(s) for s in str(a.seeds).split(",") if str(s).strip()], config=cfg)


if __name__ == "__main__":
    main()
