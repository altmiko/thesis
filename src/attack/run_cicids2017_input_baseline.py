"""Input-space targeted PGD baseline on CICIDS2017-DistriNet.

Unconstrained targeted PGD in scaled feature space (L-inf ball), toward Benign. This is the
classical adversarial baseline: it edits all 79 features freely with NO realizability model,
so it exposes what "success" costs in validity. Evaluated with the SAME victims, test split,
clean-correct denominator, validators, and target class as the other methods -- the realized
vector is simply the (clamped) PGD output, and its frozen/dependency/discreteness validity is
expected to be low. Separate output tree.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel, SCALER_ATOL
from attack.realizability.validator import RealizabilityValidator
from attack.run_cicids2017_primitive_attack import (
    VICTIMS, _class_rows, evaluate_cell, _LENGTH_COLS, _TIMING_COLS, _RATE_COLS)
from datasets.cicids2017 import CICIDS2017Adapter
from evaluation.pave_style_validator import PAVEStyleValidator
from experiments.ablations import build_ablation
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import ATTACK_CLASSES, load_stage_a


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def targeted_pgd(victim, x0_scaled, target, *, epsilon, steps, alpha):
    x = x0_scaled.clone()
    x_adv = (x + torch.empty_like(x).uniform_(-epsilon, epsilon)).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(victim(x_adv), target)  # minimize CE toward benign target
        grad, = torch.autograd.grad(loss, x_adv)
        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = x_adv.detach()
    return x_adv


def run(*, classes, victims, device, test_limit, epsilon, steps, alpha, stage_a_dir,
        output_dir, seeds):
    adapter = CICIDS2017Adapter(); repo = adapter.repo_root
    manifest = adapter.feature_manifest(); transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    model = CICIDS2017PrimitiveModel(manifest); val = RealizabilityValidator(model)
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

    results = {"dataset": adapter.name, "method_id": "input_pgd",
               "attack": "Input-space targeted PGD (L-inf, unconstrained; NO realizability model)",
               "threat_model": "targeted Attack->Benign", "denominator": "clean-correct malicious test rows",
               "config": {"epsilon": epsilon, "steps": steps, "alpha": alpha,
                          "test_limit_per_class": test_limit, "seeds": seeds}, "cells": []}
    for class_name in classes:
        cid = mapping.name_to_id[class_name]
        idx = _class_rows(test.y, cid, test_limit, 42 + cid)
        raw_np = np.ascontiguousarray(np.asarray(raw_test[idx]), dtype=np.float32)
        raw = torch.tensor(raw_np, device=device)
        base_vae, _ = load_stage_a(adapter, stage_a_dir / f"vae_{class_name}.pt", device=device)
        idr_path = stage_a_dir / f"idr_{class_name}.npz"
        engine = build_ablation("A4", adapter, encoder_input_transform="asinh",
                                layer1_fit_x_raw=layer1_fit, layer2_path=layer2_path).engine
        for vname in victims:
            victim = load_category_victim(victim_dir / f"{vname}_category.pt", device=device)
            for seed in seeds:
                _seed(seed)
                x0 = (raw - center) / scale
                target = torch.zeros(raw.shape[0], dtype=torch.long, device=device)
                x_adv = targeted_pgd(victim, x0, target, epsilon=epsilon, steps=steps, alpha=alpha)
                adv_raw = (x_adv * scale + center).detach()
                masks, cost, yc, ya = evaluate_cell(model, val, victim, base_vae, engine, pave, raw,
                                                    adv_raw, center, scale, cid, idr_path, groups_idx)
                ap = artifact_dir / f"{class_name}_{vname}_seed{seed}.npz"
                np.savez_compressed(
                    ap, X_clean_raw=raw_np, X_adv_raw=adv_raw.cpu().numpy().astype(np.float32),
                    y_true=np.full(len(idx), cid, dtype=np.int64),
                    y_pred_clean=yc.cpu().numpy().astype(np.int64),
                    y_pred_adv=ya.cpu().numpy().astype(np.int64),
                    cost_total=cost["total"].cpu().numpy().astype(np.float32),
                    **{k: m.cpu().numpy() for k, m in masks.items()})
                denom = int(masks["clean_correct"].sum()); cc = masks["clean_correct"]
                strict = masks["pave_valid"] & masks["mined_valid"] & masks["realizable"]
                rate = lambda m: (float((m & cc).sum()) / denom) if denom else float("nan")
                cell = {"class": class_name, "victim": vname, "seed": seed, "artifact": str(ap),
                        "n_total": len(idx), "n_clean_correct": denom,
                        "n_targeted_benign_success": int((masks["benign"] & cc).sum()),
                        "n_targeted_strict_valid": int((masks["benign"] & strict & cc).sum()),
                        "untargeted_asr": rate(masks["evasion"]),
                        "targeted_benign_asr": rate(masks["benign"]),
                        "targeted_strict_valid_asr": rate(masks["benign"] & strict),
                        "strict_validity": rate(strict), "pave_validity": rate(masks["pave_valid"]),
                        "mined_validity": rate(masks["mined_valid"]),
                        "realizability_aware_validity": rate(masks["realizable"]),
                        "IDR": rate(masks["in_dist"]),
                        "cost_total_mean": float(cost["total"][cc].mean()) if denom else float("nan")}
                results["cells"].append(cell)
                print(json.dumps({k: cell[k] for k in ("class", "victim", "seed", "targeted_benign_asr",
                      "targeted_strict_valid_asr", "strict_validity", "cost_total_mean")}), flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "attack_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    ap.add_argument("--victims", default=",".join(VICTIMS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test-limit", type=int, default=1024)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--stage-a-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/cicids2017_input_baseline"))
    a = ap.parse_args()
    run(classes=[c.strip() for c in a.classes.split(",") if c.strip()],
        victims=[v.strip() for v in a.victims.split(",") if v.strip()],
        device=a.device, test_limit=a.test_limit, epsilon=a.epsilon, steps=a.steps, alpha=a.alpha,
        stage_a_dir=a.stage_a_dir, output_dir=a.output_dir,
        seeds=[int(x) for x in str(a.seeds).split(",") if str(x).strip()])


if __name__ == "__main__":
    main()
