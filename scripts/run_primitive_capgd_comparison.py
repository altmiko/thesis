"""Full paired CAPGD-over-PrimAttack-primitives experiment.

CAPGD optimizes only PrimAttack's two normalized controls (padding and timing).
Every other component is PrimAttack's canonical implementation: capability gates,
per-flow p75 bounds, differentiable transform, projection, quantization,
validator-v2, primitive feasibility, and the flow-level semantic proxy.

The protocol matches ``scripts/run_capgd_primattack_comparison.py``: the same frozen
clean-correct rows (800 per victim/class), the same three seeds, and pairing by
ordered ``sample_id`` against the committed PrimAttack artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from joblib import parallel_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _path in (str(REPO_ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from attack.flow_semantics import FlowSemanticValidator, SemanticStatus  # noqa: E402
from attack.primattack_budget import class_calibration, load_calibration  # noqa: E402
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel  # noqa: E402
from attack.realizability.validator import RealizabilityValidator  # noqa: E402
from comparisons.primitive_capgd import run_primitive_capgd  # noqa: E402
from datasets.cicids2017 import CICIDS2017Adapter  # noqa: E402
from experiments.provenance import (  # noqa: E402
    build_provenance,
    deterministic_runtime,
    ensure_fresh_output_dir,
)
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402
from validation.attack_interface import structural_masks  # noqa: E402

VICTIM_CKPT = {
    "mlp": REPO_ROOT / "outputs/cicids2017distrinet/models/mlp_category.pt",
    "cnn": REPO_ROOT / "outputs/cicids2017distrinet/models/cnn_category.pt",
    "ft_transformer": REPO_ROOT / "outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt",
}
DEFAULT_VICTIMS = ("mlp", "cnn", "ft_transformer")
DEFAULT_CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")
DEFAULT_SEEDS = (42, 123, 2024)
PRIM_REFERENCE = "prim_search_joint_p75"
METHOD_PREFIX = "primitive_capgd"


def _sha(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(ids.astype(str)).encode()).hexdigest()


def _rate(mask: np.ndarray) -> float:
    return float(np.asarray(mask, bool).mean()) if len(mask) else float("nan")


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    victims = _parse_csv(args.victims)
    classes = _parse_csv(args.classes)
    seeds = [int(value) for value in _parse_csv(args.seeds)]
    steps_list = [int(value) for value in _parse_csv(args.steps)]
    if set(victims) - set(VICTIM_CKPT):
        raise ValueError(f"unknown victims: {sorted(set(victims) - set(VICTIM_CKPT))}")
    if set(classes) - set(DEFAULT_CLASSES):
        raise ValueError(f"classes must be drawn from {DEFAULT_CLASSES}")

    out = ensure_fresh_output_dir(args.output_dir)
    artifacts = out / "artifacts"
    artifacts.mkdir()

    adapter = CICIDS2017Adapter(REPO_ROOT)
    mapping = adapter.class_mapping()
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    center = torch.as_tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.as_tensor(transform.scale, dtype=torch.float32, device=device)
    primitive_model = CICIDS2017PrimitiveModel(manifest)
    realizability = RealizabilityValidator(primitive_model)
    calibration_path = REPO_ROOT / "artifacts/primattack/budget_calibration.json"
    calibration = load_calibration(calibration_path)
    semantics = FlowSemanticValidator(primitive_model, calibration)

    processed = adapter._processed
    raw_test = np.load(processed / "X_test_pristine.npy", mmap_mode="r")
    y_test = np.load(processed / "y_test_cat.npy").astype(np.int64)
    metadata = pd.read_parquet(
        processed / "test.parquet", columns=["sample_id", "Src IP", "Dst IP"]
    )
    all_ids = metadata["sample_id"].astype(str).to_numpy(dtype="U128")
    selection_source = json.loads(
        (REPO_ROOT / "outputs/full_adv_eval/selection.json").read_text(encoding="utf-8")
    )

    limit = args.limit_per_cell
    selection: dict[str, dict[str, dict]] = {}
    for victim_name in victims:
        selection[victim_name] = {}
        for class_name in classes:
            source = selection_source[victim_name][class_name]
            idx = np.asarray(source["positional_idx"], np.int64)
            ids = np.asarray(source["sample_ids"], dtype="U128")
            if limit is not None:
                idx = idx[:limit]
                ids = ids[:limit]
            class_id = int(source["class_id"])
            if not np.array_equal(all_ids[idx], ids):
                raise ValueError(f"{victim_name}/{class_name}: sample-id/order mismatch")
            if not bool((y_test[idx] == class_id).all()):
                raise ValueError(f"{victim_name}/{class_name}: true-label mismatch")
            selection[victim_name][class_name] = {
                "class_id": class_id,
                "positional_idx": idx.tolist(),
                "sample_ids": ids.tolist(),
                "n_used": int(len(idx)),
                "sha256_sample_ids": _sha(ids),
            }
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    config = {
        "dataset": adapter.name,
        "method": METHOD_PREFIX,
        "description": "upstream TabularBench CAPGD over normalized PrimAttack controls",
        "goal": "untargeted",
        "victims": victims,
        "classes": classes,
        "seeds": seeds,
        "steps": steps_list,
        "n_per_class": None if limit is None else limit,
        "controls": ["q_padding", "q_delay", "q_shape"],
        "control_box": "[0,1]^3 mapped per row to PrimAttack's p75 hard control box",
        "capgd": {"norm": "Linf", "epsilon": 1.0, "eps_margin": 0.0, "n_restarts": 2, "loss": "ce", "rho": 0.75},
        "budget": "maximum-evaluated (p75)",
        "budget_calibration": str(calibration_path),
        "calibration_fit_split": calibration["fit_split"],
        "primattack_reference": PRIM_REFERENCE,
        "eligibility": "frozen clean-correct rows from full_adv_eval selection",
        "denominator": "all selected clean-correct rows",
        "validity": "validator_v2 hybrid_valid; feasibility = primitive consistency + budget; semantic = flow proxy PASS",
    }
    checkpoints = {f"victim_{name}": VICTIM_CKPT[name] for name in victims}
    checkpoints["budget_calibration"] = calibration_path
    provenance = build_provenance(
        repo_root=REPO_ROOT,
        dataset=adapter.name,
        method_id=METHOD_PREFIX,
        config=config,
        preprocessing_manifest=processed / "preprocessing_manifest.json",
        scaler=processed / "scaler.pkl",
        checkpoints=checkpoints,
    )
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    cells: list[dict] = []
    started = time.perf_counter()
    for victim_name in victims:
        victim = load_category_victim(
            VICTIM_CKPT[victim_name], adapter=adapter,
            expected_model_type=victim_name, device=device,
        )
        for class_name in classes:
            selected = selection[victim_name][class_name]
            class_id = int(selected["class_id"])
            idx = np.asarray(selected["positional_idx"], np.int64)
            ids = np.asarray(selected["sample_ids"], dtype="U128")
            raw_np = np.ascontiguousarray(np.asarray(raw_test[idx]), dtype=np.float32)
            raw = torch.as_tensor(raw_np, device=device)
            labels_np = np.full(len(idx), class_id, dtype=np.int64)
            labels_t = torch.full((len(idx),), class_id, dtype=torch.long, device=device)
            with torch.no_grad():
                clean_pred = victim((raw - center) / scale).argmax(1).cpu().numpy().astype(np.int64)
            if not bool((clean_pred == class_id).all()):
                raise ValueError(f"{victim_name}/{class_name}: frozen selection not clean-correct")

            caps = primitive_model.infer_capabilities(raw)
            class_cfg = class_calibration(calibration, class_name, "maximum-evaluated")
            bounds = primitive_model.per_flow_bounds(
                raw, class_cfg.bounds_config(), capabilities=caps
            )
            source_meta = {
                "Src IP": metadata.iloc[idx]["Src IP"].astype(str).to_numpy(),
                "Dst IP": metadata.iloc[idx]["Dst IP"].astype(str).to_numpy(),
            }

            for steps in steps_list:
                method = f"{METHOD_PREFIX}_{steps}step"
                for seed in seeds:
                    deterministic_runtime(seed)
                    t0 = time.perf_counter()
                    with parallel_backend("threading"):
                        result = run_primitive_capgd(
                            repo_root=REPO_ROOT, primitive_model=primitive_model,
                            victim=victim, raw=raw, bounds=bounds, capabilities=caps,
                            center=center, scale=scale, true_labels=labels_t, seed=seed,
                            steps=steps,
                        )
                    elapsed = time.perf_counter() - t0
                    adv_raw = result.adversarial_raw
                    if not bool(np.isfinite(adv_raw.cpu().numpy()).all()):
                        raise FloatingPointError("primitive CAPGD produced NaN or Inf")
                    with torch.no_grad():
                        adv_pred = victim((adv_raw - center) / scale).argmax(1).cpu().numpy().astype(np.int64)
                    evasion = adv_pred != class_id
                    targeted = adv_pred == 0
                    validator = structural_masks(adv_raw.cpu().numpy())
                    domain = np.asarray(validator["hybrid_valid"], bool)
                    transform_report = realizability.validate(adv_raw, raw)
                    transform_ok = transform_report.valid.cpu().numpy()
                    semantic = semantics.evaluate(
                        raw, adv_raw, result.requested, result.projected, bounds,
                        class_name=class_name, budget=class_cfg.budget,
                        original_labels=labels_np, adversarial_labels=labels_np.copy(),
                        original_metadata=source_meta, adversarial_metadata=source_meta,
                    )
                    primitive_feasible = semantic.primitive_feasible & transform_ok
                    sem_pass = semantic.semantic_status == SemanticStatus.PASS.value

                    artifact = artifacts / f"{victim_name}__{class_name}__{method}__seed{seed}.npz"
                    np.savez_compressed(
                        artifact, sample_id=ids, positional_idx=idx,
                        true_class=labels_np, clean_pred=clean_pred, adv_pred=adv_pred,
                        clean_correct=clean_pred == class_id, evasion=evasion,
                        targeted_success=targeted, domain_valid=domain,
                        primitive_feasible=primitive_feasible, semantic_pass=sem_pass,
                        semantic_status=semantic.semantic_status,
                        primitive_transform_consistent=transform_ok,
                        q_padding=result.normalized_controls[:, 0].cpu().numpy().astype(np.float32),
                        q_delay=result.normalized_controls[:, 1].cpu().numpy().astype(np.float32),
                        q_shape=result.normalized_controls[:, 2].cpu().numpy().astype(np.float32),
                        p_requested=result.requested["p"].cpu().numpy().astype(np.float32),
                        delay_requested=result.requested["delay"].cpu().numpy().astype(np.float32),
                        shape_requested=result.requested["shape"].cpu().numpy().astype(np.float32),
                        p_projected=result.projected["p"].cpu().numpy().astype(np.float32),
                        delay_projected=result.projected["delay"].cpu().numpy().astype(np.float32),
                        shape_projected=result.projected["shape"].cpu().numpy().astype(np.float32),
                        p_hi=bounds["p"].cpu().numpy().astype(np.float32),
                        delay_hi=bounds["delay"].cpu().numpy().astype(np.float32),
                        shape_hi=bounds["shape"].cpu().numpy().astype(np.float32),
                        elapsed_seconds=np.asarray(elapsed, np.float64),
                        method=np.asarray(method), victim=np.asarray(victim_name),
                        attack_class=np.asarray(class_name), seed=np.asarray(seed, np.int64),
                        goal=np.asarray("untargeted"), steps=np.asarray(steps, np.int64),
                    )
                    cell = {
                        "victim": victim_name, "class": class_name, "method": method,
                        "steps": steps, "seed": seed, "n": len(idx),
                        "sha256_sample_ids": selected["sha256_sample_ids"],
                        "raw_untargeted_asr": _rate(evasion),
                        "valid_untargeted_asr": _rate(evasion & domain),
                        "feasible_untargeted_asr": _rate(evasion & domain & primitive_feasible),
                        "semantic_untargeted_asr": _rate(evasion & domain & primitive_feasible & sem_pass),
                        "targeted_benign_asr": _rate(targeted),
                        "valid_targeted_benign_asr": _rate(targeted & domain),
                        "domain_validity": _rate(domain),
                        "primitive_feasibility": _rate(primitive_feasible),
                        "semantic_pass_rate": _rate(sem_pass),
                        "median_q_padding": float(np.median(result.normalized_controls[:, 0].cpu().numpy())),
                        "median_q_delay": float(np.median(result.normalized_controls[:, 1].cpu().numpy())),
                        "median_q_shape": float(np.median(result.normalized_controls[:, 2].cpu().numpy())),
                        "elapsed_seconds": elapsed, "artifact": str(artifact),
                    }
                    cells.append(cell)
                    print(json.dumps(cell), flush=True)

    (out / "cells.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")
    summary = {
        "dataset": adapter.name, "method_id": METHOD_PREFIX, "cells": cells,
        "elapsed_seconds": time.perf_counter() - started,
        "comparison_reference": f"outputs/full_adv_eval_primattack_v2 artifacts for {PRIM_REFERENCE}",
    }
    (out / "attack_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--victims", default=",".join(DEFAULT_VICTIMS))
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--steps", default="10,40")
    parser.add_argument("--limit-per-cell", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs/comparisons/primitive_capgd_vs_primattack",
    )
    args = parser.parse_args()
    if args.limit_per_cell is not None and args.limit_per_cell <= 0:
        parser.error("limit-per-cell must be positive")
    out = run(args)
    print(f"primitive CAPGD artifacts written to {out}")


if __name__ == "__main__":
    main()
