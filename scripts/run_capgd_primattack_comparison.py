"""Run upstream TabularBench CAPGD on the frozen PrimAttack comparison rows.

The output is directly pairable with ``outputs/full_adv_eval`` by
(victim, attack_class, seed, sample_id).  CAPGD is evaluated with the same
validator-v2 ``hybrid_valid`` gate used for PrimAttack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Required by deterministic CUDA matrix multiplications. It must be set before
# PyTorch initializes CUDA.
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

from comparisons.capgd_cicids2017 import (  # noqa: E402
    CAPGD_METHOD_ID,
    TABULARBENCH_COMMIT,
    RawCICIDSVictim,
    build_capgd_resources,
    evaluate_capgd_output,
    finalize_capgd_output,
    make_capgd,
    save_constraint_manifest,
)
from datasets.cicids2017 import CICIDS2017Adapter  # noqa: E402
from experiments.provenance import (  # noqa: E402
    artifact_provenance_arrays,
    build_provenance,
    deterministic_runtime,
    ensure_fresh_output_dir,
    sha256_file,
)
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402

VICTIM_CKPT = {
    "mlp": REPO_ROOT / "outputs/cicids2017distrinet/models/mlp_category.pt",
    "cnn": REPO_ROOT / "outputs/cicids2017distrinet/models/cnn_category.pt",
    "ft_transformer": REPO_ROOT / "outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt",
}
DEFAULT_CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")
DEFAULT_VICTIMS = ("mlp", "cnn", "ft_transformer")
DEFAULT_SEEDS = (42, 123, 2024)
BENIGN_ID = 0


def _sha_ids(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(ids.astype(str).tolist()).encode("utf-8")).hexdigest()


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _rate(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=bool).mean()) if len(mask) else float("nan")


def _load_selection(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing frozen selection: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_selected_rows(
    selection: dict,
    *,
    victims: list[str],
    classes: list[str],
    limit: int | None,
    y_test: np.ndarray,
    sample_ids: np.ndarray,
) -> dict:
    used: dict[str, dict[str, dict]] = {}
    for victim in victims:
        if victim not in selection:
            raise KeyError(f"victim {victim!r} absent from frozen selection")
        used[victim] = {}
        for class_name in classes:
            source = selection[victim][class_name]
            idx = np.asarray(source["positional_idx"], dtype=np.int64)
            ids = np.asarray(source["sample_ids"], dtype="U128")
            if limit is not None:
                idx = idx[:limit]
                ids = ids[:limit]
            class_id = int(source["class_id"])
            if not np.array_equal(sample_ids[idx], ids):
                raise ValueError(f"{victim}/{class_name}: sample ID/order mismatch")
            if not bool((y_test[idx] == class_id).all()):
                raise ValueError(f"{victim}/{class_name}: true-label mismatch")
            used[victim][class_name] = {
                "class_id": class_id,
                "n_eligible_total": int(source["n_eligible_total"]),
                "n_used": int(len(idx)),
                "positional_idx": idx.tolist(),
                "sample_ids": ids.tolist(),
                "sha256_sample_ids": _sha_ids(ids),
            }
    return used


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    victims = _parse_csv(args.victims)
    classes = _parse_csv(args.classes)
    seeds = [int(value) for value in _parse_csv(args.seeds)]
    unknown_victims = set(victims) - set(VICTIM_CKPT)
    if unknown_victims:
        raise ValueError(f"unknown victims: {sorted(unknown_victims)}")
    if set(classes) - set(DEFAULT_CLASSES):
        raise ValueError(f"classes must be drawn from {DEFAULT_CLASSES}")

    output_dir = ensure_fresh_output_dir(args.output_dir)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir()

    adapter = CICIDS2017Adapter(REPO_ROOT)
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    resources = build_capgd_resources(REPO_ROOT, adapter=adapter)
    save_constraint_manifest(resources, output_dir / "constraint_manifest.json")

    processed = adapter._processed
    raw_test = np.load(processed / "X_test_pristine.npy", mmap_mode="r")
    y_test = np.load(processed / "y_test_cat.npy").astype(np.int64)
    metadata = pd.read_parquet(processed / "test.parquet", columns=["sample_id"])
    sample_ids = metadata["sample_id"].astype(str).to_numpy(dtype="U128")
    if not (len(raw_test) == len(y_test) == len(sample_ids)):
        raise ValueError("test arrays/parquet length mismatch")

    frozen_selection = _load_selection(args.selection)
    selection = _verify_selected_rows(
        frozen_selection,
        victims=victims,
        classes=classes,
        limit=args.limit_per_cell,
        y_test=y_test,
        sample_ids=sample_ids,
    )
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    external_files = [
        REPO_ROOT / "external/tabularbench/tabularbench/attacks/capgd/capgd.py",
        REPO_ROOT / "external/tabularbench/tabularbench/attacks/objective_calculator.py",
        REPO_ROOT / "external/tabularbench/tabularbench/constraints/constraints_checker.py",
    ]
    config = {
        "dataset": adapter.name,
        "method": CAPGD_METHOD_ID,
        "goal": "untargeted",
        "victims": victims,
        "classes": classes,
        "seeds": seeds,
        "norm": args.norm,
        "epsilon": args.epsilon,
        "steps": args.steps,
        "n_restarts": 2,
        "loss": "ce",
        "rho": 0.75,
        "eps_margin": 0.01,
        "batch_size": args.batch_size,
        "limit_per_cell": args.limit_per_cell,
        "selection_source": str(args.selection),
        "eligibility": "frozen clean-correct rows selected by full_adv_eval",
        "denominator": "all selected clean-correct rows",
        "validity": "validator_v2 hybrid_valid",
        "internal_success": "misclassification AND TabularBench constraints AND distance",
        "mutable_policy": "CICIDS config mask: 9 direct features plus 7 repaired derived targets",
        "tabularbench": {
            "commit": TABULARBENCH_COMMIT,
            "source_files": {
                str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in external_files
            },
        },
    }
    checkpoints = {f"victim_{name}": VICTIM_CKPT[name] for name in victims}
    provenance = build_provenance(
        repo_root=REPO_ROOT,
        dataset=adapter.name,
        method_id=CAPGD_METHOD_ID,
        config=config,
        preprocessing_manifest=processed / "preprocessing_manifest.json",
        scaler=processed / "scaler.pkl",
        checkpoints=checkpoints,
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    center = np.asarray(transform.center, dtype=np.float32)
    scale = np.asarray(transform.scale, dtype=np.float32)
    center_t = torch.as_tensor(center, device=device)
    scale_t = torch.as_tensor(scale, device=device)
    cells: list[dict] = []
    started = time.perf_counter()

    for victim_name in victims:
        victim = load_category_victim(
            VICTIM_CKPT[victim_name],
            adapter=adapter,
            expected_model_type=victim_name,
            device=device,
        )
        raw_victim = RawCICIDSVictim(victim, center, scale).to(device).eval()

        for class_name in classes:
            selected = selection[victim_name][class_name]
            class_id = int(selected["class_id"])
            if mapping.name_to_id[class_name] != class_id:
                raise ValueError(f"class mapping mismatch for {class_name}")
            idx = np.asarray(selected["positional_idx"], dtype=np.int64)
            ids = np.asarray(selected["sample_ids"], dtype="U128")
            clean_np = np.ascontiguousarray(np.asarray(raw_test[idx]), dtype=np.float32)
            labels_np = np.full(len(idx), class_id, dtype=np.int64)

            clean_t = torch.as_tensor(clean_np, device=device)
            with torch.no_grad():
                clean_pred = raw_victim(clean_t).argmax(1).cpu().numpy().astype(np.int64)
            if not bool((clean_pred == class_id).all()):
                raise ValueError(f"{victim_name}/{class_name}: frozen selection is not clean-correct")

            for seed in seeds:
                deterministic_runtime(seed)
                attack = make_capgd(
                    resources,
                    raw_victim,
                    device=device,
                    seed=seed,
                    norm=args.norm,
                    eps=args.epsilon,
                    steps=args.steps,
                )
                chunks: list[np.ndarray] = []
                cell_started = time.perf_counter()
                for start in range(0, len(clean_np), args.batch_size):
                    stop = min(start + args.batch_size, len(clean_np))
                    batch_clean = torch.as_tensor(clean_np[start:stop], device=device)
                    batch_y = torch.full(
                        (stop - start,), class_id, dtype=torch.long, device=device
                    )
                    # ObjectiveCalculator hardcodes joblib n_jobs=-1. A threading
                    # backend keeps its exact computation while avoiding Windows
                    # subprocesses that cannot inherit the NumPy-2 compatibility alias.
                    with parallel_backend("threading"):
                        candidate = attack(batch_clean, batch_y)
                    candidate = finalize_capgd_output(resources, batch_clean, candidate)
                    chunks.append(candidate.detach().cpu().numpy().astype(np.float32))
                elapsed = time.perf_counter() - cell_started
                adv_np = np.concatenate(chunks, axis=0)

                evaluated = evaluate_capgd_output(
                    resources,
                    clean_np,
                    adv_np,
                    norm=args.norm,
                    eps=args.epsilon,
                )
                adv_t = torch.as_tensor(adv_np, device=device)
                with torch.no_grad():
                    adv_pred = raw_victim(adv_t).argmax(1).cpu().numpy().astype(np.int64)
                evasion = adv_pred != class_id
                targeted = adv_pred == BENIGN_ID
                hybrid = np.asarray(evaluated["hybrid_valid"], dtype=bool)
                internal = np.asarray(evaluated["internal_constraint_valid"], dtype=bool)
                distance_ok = np.asarray(evaluated["distance_ok"], dtype=bool)
                constrained_success = evasion & hybrid & internal & distance_ok
                delta_raw = adv_np - clean_np
                delta_scaled = delta_raw / scale[None, :]
                changed = np.abs(delta_raw) > (1e-5 + 1e-4 * np.abs(clean_np))

                artifact = artifact_dir / (
                    f"{victim_name}__{class_name}__capgd__seed{seed}.npz"
                )
                checkpoint_ids = {
                    f"victim_{victim_name}": provenance["checkpoints"][f"victim_{victim_name}"]["sha256"]
                }
                np.savez_compressed(
                    artifact,
                    X_clean_raw=clean_np,
                    X_adv_raw=adv_np,
                    sample_id=ids,
                    positional_idx=idx,
                    true_class=labels_np,
                    clean_pred=clean_pred,
                    adv_pred=adv_pred,
                    clean_correct=clean_pred == labels_np,
                    evasion=evasion,
                    targeted_success=targeted,
                    internal_constraint_valid=internal,
                    distance=np.asarray(evaluated["distance"], dtype=np.float32),
                    distance_ok=distance_ok,
                    schema_valid=np.asarray(evaluated["schema_valid"], dtype=bool),
                    extractor_valid=np.asarray(evaluated["extractor_valid"], dtype=bool),
                    protocol_valid=np.asarray(evaluated["protocol_valid"], dtype=bool),
                    mined_valid=np.asarray(evaluated["mined_valid"], dtype=bool),
                    hard_structural_valid=np.asarray(evaluated["hard_structural_valid"], dtype=bool),
                    domain_valid=hybrid,
                    in_distribution=np.asarray(evaluated["in_distribution"], dtype=bool),
                    plausibility_score=np.asarray(evaluated["plausibility_score"], dtype=np.float32),
                    constrained_success=constrained_success,
                    l2_robust_scaled=np.linalg.norm(delta_scaled, axis=1).astype(np.float32),
                    linf_robust_scaled=np.max(np.abs(delta_scaled), axis=1).astype(np.float32),
                    changed_feature_count=changed.sum(axis=1).astype(np.int16),
                    attack=np.asarray(CAPGD_METHOD_ID),
                    attack_class=np.asarray(class_name),
                    goal=np.asarray("untargeted"),
                    norm=np.asarray(args.norm),
                    epsilon=np.asarray(args.epsilon, dtype=np.float32),
                    elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
                    rules_rejecting_json=np.asarray(json.dumps(evaluated["rules_rejecting"], sort_keys=True)),
                    **artifact_provenance_arrays(
                        provenance,
                        row_ids=ids,
                        class_name=class_name,
                        victim=victim_name,
                        method_id=CAPGD_METHOD_ID,
                        seed=seed,
                        checkpoint_ids=checkpoint_ids,
                    ),
                )

                cell = {
                    "victim": victim_name,
                    "class": class_name,
                    "seed": seed,
                    "n": len(clean_np),
                    "sha256_sample_ids": selected["sha256_sample_ids"],
                    "raw_untargeted_asr": _rate(evasion),
                    "validator_valid_untargeted_asr": _rate(evasion & hybrid),
                    "capgd_constrained_asr": _rate(constrained_success),
                    "targeted_benign_asr": _rate(targeted),
                    "hybrid_validity_rate": _rate(hybrid),
                    "internal_constraint_validity_rate": _rate(internal),
                    "distance_validity_rate": _rate(distance_ok),
                    "in_distribution_rate": _rate(np.asarray(evaluated["in_distribution"])),
                    "median_distance": float(np.median(evaluated["distance"])),
                    "median_changed_features": float(np.median(changed.sum(axis=1))),
                    "elapsed_seconds": elapsed,
                    "artifact": str(artifact),
                    "rules_rejecting": evaluated["rules_rejecting"],
                }
                cells.append(cell)
                print(json.dumps(cell), flush=True)

    (output_dir / "cells.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")
    summary = {
        "dataset": adapter.name,
        "method_id": CAPGD_METHOD_ID,
        "cells": cells,
        "elapsed_seconds": time.perf_counter() - started,
        "comparison_reference": "outputs/full_adv_eval_primattack_v2 artifacts for prim_search_joint_p75",
    }
    (output_dir / "attack_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--victims", default=",".join(DEFAULT_VICTIMS))
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--norm", choices=("L2", "Linf"), default="L2")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-per-cell", type=int, default=None)
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPO_ROOT / "outputs/full_adv_eval/selection.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/comparisons/capgd_vs_primattack",
    )
    args = parser.parse_args()
    if args.epsilon <= 0 or args.steps <= 0 or args.batch_size <= 0:
        parser.error("epsilon, steps, and batch-size must be positive")
    if args.limit_per_cell is not None and args.limit_per_cell <= 0:
        parser.error("limit-per-cell must be positive")
    out = run(args)
    print(f"CAPGD artifacts written to {out}")


if __name__ == "__main__":
    main()
