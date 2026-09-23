"""Calibrated primitive-domain white-box attack on CICIDS2017-DistriNet.

Only two controls are optimized: forward packet-length augmentation ``p`` and forward timing
Dilation ``alpha``. The canonical primitive map recomputes dependent features, final controls
are projected into a train-calibrated hard budget, and the realized vector is reclassified.
Domain validity, primitive feasibility, and flow-level semantic preservation remain separate.
No packet-level realization or replay is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from attack.flow_semantics import FlowSemanticValidator, SemanticStatus
from attack.primattack_budget import (
    BUDGET_NAMES,
    class_calibration,
    load_calibration,
)
from attack.realizability.base import NullPacketBackend
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel, SCALER_ATOL
from attack.realizability.validator import RealizabilityValidator
from attack.run_cicids2017_vae_attacks import _idr_mask
from datasets.cicids2017 import CICIDS2017Adapter
from validation.attack_interface import structural_masks
from experiments.provenance import (
    artifact_provenance_arrays,
    build_provenance,
    deterministic_runtime,
    ensure_fresh_output_dir,
)
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import ATTACK_CLASSES, load_stage_a

VICTIMS = ("mlp", "cnn", "lstm", "serial")
PRIMITIVE_MODES = ("timing-only", "padding-only", "joint")
OPTIMIZERS = ("optimized", "random-feasible")
_LENGTH_COLS = (
    "Total Length of Fwd Packet", "Fwd Packet Length Min", "Fwd Packet Length Max",
    "Fwd Packet Length Mean", "Fwd Segment Size Avg", "Packet Length Min",
    "Packet Length Max", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "Average Packet Size",
)
_TIMING_COLS = (
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Flow Duration", "Flow IAT Mean", "Flow IAT Max",
)
_RATE_COLS = ("Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s")


def _class_rows(y: np.ndarray, class_id: int, limit: int | None, seed: int) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(y) == class_id)
    if limit is not None and len(idx) > limit:
        pick = np.random.default_rng(seed).choice(len(idx), limit, replace=False)
        idx = idx[np.sort(pick)]
    return idx


def optimize_primitives(
    model, victim, raw, center, scale, bounds, caps, *, steps, lr, cost_weight,
    init_noise, seed,
):
    """Adam over only two unconstrained leaves, mapped into the hard primitive box."""
    n = raw.shape[0]
    generator = torch.Generator(device=raw.device).manual_seed(seed)
    u = (
        -2.0
        + init_noise * torch.randn(n, generator=generator, device=raw.device, dtype=raw.dtype)
    ).requires_grad_(True)
    v = (
        -2.0
        + init_noise * torch.randn(n, generator=generator, device=raw.device, dtype=raw.dtype)
    ).requires_grad_(True)
    optimizer = torch.optim.Adam([u, v], lr=lr)
    target = torch.zeros(n, dtype=torch.long, device=raw.device)
    pad_active = caps.pad_allowed.to(raw.dtype)
    timing_active = caps.timing_allowed.to(raw.dtype)

    def controls() -> dict[str, torch.Tensor]:
        return {
            "p": bounds["p"] * torch.sigmoid(u) * pad_active,
            "alpha": 1.0
            + (bounds["alpha"] - 1.0) * torch.sigmoid(v) * timing_active,
        }

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        requested = controls()
        transformed = model.generate(raw, requested, quantize=False)
        logits = victim((transformed - center) / scale)
        loss = (
            F.cross_entropy(logits, target, reduction="none")
            + cost_weight
            * (torch.sigmoid(u) * pad_active + torch.sigmoid(v) * timing_active)
        )
        loss.sum().backward()
        optimizer.step()
    with torch.no_grad():
        requested = controls()
    return {name: value.detach() for name, value in requested.items()}


def random_feasible_primitives(bounds: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """Uniform random control inside the exact same hard box as optimized PrimAttack."""
    first = bounds["p"]
    generator = torch.Generator(device=first.device).manual_seed(seed)
    p_fraction = torch.rand(first.shape, generator=generator, device=first.device, dtype=first.dtype)
    alpha_fraction = torch.rand(
        first.shape, generator=generator, device=first.device, dtype=first.dtype
    )
    return {
        "p": bounds["p"] * p_fraction,
        "alpha": 1.0 + (bounds["alpha"] - 1.0) * alpha_fraction,
    }


def _apply_primitive_mode(
    bounds: dict[str, torch.Tensor], mode: str
) -> dict[str, torch.Tensor]:
    if mode not in PRIMITIVE_MODES:
        raise ValueError(f"primitive mode must be one of {PRIMITIVE_MODES}")
    result = {name: value.clone() for name, value in bounds.items()}
    if mode == "timing-only":
        result["p"].zero_()
        result["p_numeric"].zero_()
    elif mode == "padding-only":
        result["alpha"].fill_(1.0)
        result["alpha_numeric"].fill_(1.0)
    return result


def _decompose_cost(adv_raw, raw, scale, groups_idx):
    per = ((adv_raw - raw).abs() / scale).sum(1)
    feature_count = raw.shape[1]
    result = {"total": per / feature_count}
    for name, idx in groups_idx.items():
        result[name] = (
            (adv_raw[:, idx] - raw[:, idx]).abs() / scale[idx]
        ).sum(1) / feature_count
    return result


def evaluate_cell(model, val, victim, base_vae, raw, adv_raw, center, scale,
                  class_id, idr_path, groups_idx):
    """Evaluate classifier, validator_v2, realism, and internal primitive consistency."""
    with torch.no_grad():
        x_clean = (raw - center) / scale
        x_adv = (adv_raw - center) / scale
        clean_pred = victim(x_clean).argmax(1)
        adv_pred = victim(x_adv).argmax(1)
        validator = structural_masks(adv_raw.detach().cpu().numpy())
        in_dist = _idr_mask(base_vae, x_adv, idr_path)
        report = val.validate(adv_raw, raw)
        categories = report.categories
        cost = _decompose_cost(adv_raw, raw, scale, groups_idx)

    masks = {
        "clean_correct": clean_pred == class_id,
        "evasion": adv_pred != class_id,
        "targeted_success": adv_pred == 0,
        "domain_valid": torch.as_tensor(validator["hybrid_valid"], device=raw.device),
        "hard_structural_valid": torch.as_tensor(
            validator["hard_structural_valid"], device=raw.device
        ),
        "validator_in_distribution": torch.as_tensor(
            validator["in_distribution"], device=raw.device
        ),
        "in_dist": in_dist,
        "dependency_ok": ~categories["algebraic_dependency_fail"],
        "packet_summary_ok": ~categories["packet_summary_fail"],
        "timing_consistency_ok": ~categories["timing_fail"],
        "rate_consistency_ok": ~categories["negative_rate_fail"],
        "discreteness_ok": ~categories["discreteness_fail"],
        "frozen_ok": ~categories["frozen_fail"],
        "primitive_transform_consistent": report.valid,
    }
    return masks, cost, clean_pred, adv_pred


def _joined(reasons: list[list[str]]) -> np.ndarray:
    return np.asarray(["|".join(values) for values in reasons], dtype="U512")


def _rate(mask: torch.Tensor, eligible: torch.Tensor) -> float:
    denominator = int(eligible.sum())
    return float((mask & eligible).sum()) / denominator if denominator else float("nan")


def run(
    *, classes, victims, device, test_limit, steps, lr, cost_weight, stage_a_dir,
    output_dir, seeds, init_noise, calibration_path, budget_name,
    primitive_mode="joint", optimizer_name="optimized",
):
    if budget_name not in BUDGET_NAMES:
        raise ValueError(f"budget name must be one of {BUDGET_NAMES}")
    if primitive_mode not in PRIMITIVE_MODES:
        raise ValueError(f"primitive mode must be one of {PRIMITIVE_MODES}")
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")

    adapter = CICIDS2017Adapter()
    repo = adapter.repo_root
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    model = CICIDS2017PrimitiveModel(manifest)
    realizability = RealizabilityValidator(model)
    calibration = load_calibration(calibration_path)
    semantic_validator = FlowSemanticValidator(model, calibration)
    packet_backend = NullPacketBackend()
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)
    groups_idx = {
        "padding": torch.tensor([model.i[name] for name in _LENGTH_COLS], device=device),
        "timing": torch.tensor([model.i[name] for name in _TIMING_COLS], device=device),
        "rate": torch.tensor([model.i[name] for name in _RATE_COLS], device=device),
    }

    test = adapter.load_split("test")
    raw_test = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    metadata = pd.read_parquet(
        adapter._processed / "test.parquet",
        columns=["sample_id", "Src IP", "Dst IP"],
    )
    all_row_ids = metadata["sample_id"].astype(str).to_numpy(dtype="U128")
    stage_a_dir = stage_a_dir or (repo / "outputs" / "cicids2017_vae_stage_a")
    victim_dir = repo / "outputs" / "cicids2017distrinet" / "models"
    method_id = "primitive_direct" if optimizer_name == "optimized" else "primitive_random"
    config = {
        "test_limit_per_class": test_limit,
        "attack_steps": steps,
        "learning_rate": lr,
        "cost_weight": cost_weight,
        "seeds": seeds,
        "init_noise": init_noise,
        "budget_name": budget_name,
        "primitive_mode": primitive_mode,
        "optimizer": optimizer_name,
        "calibration_path": str(calibration_path),
        "calibration_fit_split": calibration["fit_split"],
        "scaler_atol": SCALER_ATOL,
        "dur_floor_us": model.dur_floor_us,
    }
    checkpoint_paths = {
        **{f"victim_{name}": victim_dir / f"{name}_category.pt" for name in victims},
        **{f"vae_{name}": stage_a_dir / f"vae_{name}.pt" for name in classes},
        **{f"idr_{name}": stage_a_dir / f"idr_{name}.npz" for name in classes},
        "budget_calibration": calibration_path,
    }
    provenance = build_provenance(
        repo_root=repo,
        dataset=adapter.name,
        method_id=method_id,
        config=config,
        preprocessing_manifest=adapter._processed / "preprocessing_manifest.json",
        scaler=adapter._processed / "scaler.pkl",
        checkpoints=checkpoint_paths,
    )
    ensure_fresh_output_dir(output_dir)
    artifact_dir = output_dir / "attack_artifacts"
    artifact_dir.mkdir()
    (output_dir / "run_manifest.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    results = {
        "dataset": adapter.name,
        "method_id": method_id,
        "attack": "PrimAttack: differentiable primitive-domain white-box attack",
        "threat_model": "targeted malicious-to-Benign flow-level primitive modification",
        "denominator": "eligible clean-correct malicious test rows",
        "domain_validity_definition": "validator_v2 hybrid_valid",
        "primitive_feasibility_definition": (
            "projected controls and realized costs inside calibrated hard primitive budget "
            "and internally consistent primitive transform"
        ),
        "sp_asr_definition": (
            "targeted success AND domain validity AND primitive feasibility AND "
            "flow-level semantic-preservation proxy PASS"
        ),
        "real_world_functionality_preservation": "NOT ESTABLISHED",
        "packet_level_verification": {
            "available": packet_backend.available(),
            "reason": packet_backend.reason,
        },
        "config": config,
        "provenance": provenance,
        "primitives": [spec.__dict__ for spec in model.primitives()],
        "feature_roles": {
            name: {"role": role.value, "reason": reason}
            for name, (role, reason) in model.roles().items()
        },
        "cells": [],
    }

    for class_name in classes:
        class_id = mapping.name_to_id[class_name]
        class_cfg = class_calibration(calibration, class_name, budget_name)
        idx = _class_rows(test.y, class_id, test_limit, 42 + class_id)
        raw_np = np.ascontiguousarray(np.asarray(raw_test[idx]), dtype=np.float32)
        raw = torch.tensor(raw_np, device=device)
        source_metadata = {
            "Src IP": metadata.iloc[idx]["Src IP"].astype(str).to_numpy(),
            "Dst IP": metadata.iloc[idx]["Dst IP"].astype(str).to_numpy(),
        }
        labels = np.full(len(idx), class_id, dtype=np.int64)
        base_vae, _ = load_stage_a(
            adapter,
            stage_a_dir / f"vae_{class_name}.pt",
            expected_class_name=class_name,
            device=device,
        )
        idr_path = stage_a_dir / f"idr_{class_name}.npz"
        caps = model.infer_capabilities(raw)
        bounds = model.per_flow_bounds(
            raw, class_cfg.bounds_config(), capabilities=caps
        )
        bounds = _apply_primitive_mode(bounds, primitive_mode)

        for victim_name in victims:
            victim = load_category_victim(
                victim_dir / f"{victim_name}_category.pt",
                adapter=adapter,
                expected_model_type=victim_name,
                device=device,
            )
            for seed in seeds:
                deterministic_runtime(seed)
                if optimizer_name == "optimized":
                    requested = optimize_primitives(
                        model, victim, raw, center, scale, bounds, caps,
                        steps=steps, lr=lr, cost_weight=cost_weight,
                        init_noise=init_noise, seed=seed,
                    )
                else:
                    requested = random_feasible_primitives(bounds, seed)
                projected = model.project_controls(raw, requested, bounds)
                adv_raw = model.generate(raw, projected, quantize=True).detach()
                frozen_idx = torch.tensor(
                    [model.i[name] for name in realizability.frozen_names], device=device
                )
                assert torch.allclose(
                    adv_raw[:, frozen_idx], raw[:, frozen_idx], atol=SCALER_ATOL, rtol=1e-4
                ), "frozen feature changed under primitive map"
                masks, cost, clean_prediction, adversarial_prediction = evaluate_cell(
                    model, realizability, victim, base_vae, raw, adv_raw,
                    center, scale, class_id, idr_path, groups_idx,
                )
                semantic = semantic_validator.evaluate(
                    raw,
                    adv_raw,
                    requested,
                    projected,
                    bounds,
                    class_name=class_name,
                    budget=class_cfg.budget,
                    original_labels=labels,
                    adversarial_labels=labels.copy(),
                    original_metadata=source_metadata,
                    adversarial_metadata=source_metadata,
                )
                primitive_feasible_np = (
                    semantic.primitive_feasible
                    & masks["primitive_transform_consistent"].cpu().numpy()
                )
                primitive_feasible = torch.as_tensor(
                    primitive_feasible_np, device=device
                )
                semantic_pass = torch.as_tensor(semantic.semantic_pass, device=device)
                eligible = masks["clean_correct"]
                domain_valid = masks["domain_valid"]
                targeted = masks["targeted_success"]
                final_finite = np.isfinite(adv_raw.cpu().numpy()).all(axis=1)
                if not bool(final_finite.all()):
                    raise FloatingPointError("PrimAttack generated NaN or Inf")

                with torch.no_grad():
                    clean_logits = victim((raw - center) / scale)
                    adversarial_logits = victim((adv_raw - center) / scale)
                domain_reasons = np.where(
                    domain_valid.cpu().numpy(), "", "VALIDATOR_V2_HYBRID_REJECTED"
                ).astype("U128")
                primitive_reasons = _joined(semantic.primitive_violation_reasons)
                semantic_failures = _joined(semantic.failure_reasons)
                semantic_not_testable = _joined(semantic.not_testable_reasons)
                changed_features = np.asarray(
                    ["|".join(names) for names in semantic.features_changed], dtype="U4096"
                )
                artifact_path = (
                    artifact_dir / f"{class_name}_{victim_name}_seed{seed}.npz"
                )
                checkpoint_ids = {
                    key: provenance["checkpoints"][key]["sha256"]
                    for key in (
                        f"victim_{victim_name}",
                        f"vae_{class_name}",
                        f"idr_{class_name}",
                        "budget_calibration",
                    )
                }
                costs = semantic.costs
                np.savez_compressed(
                    artifact_path,
                    X_clean_raw=raw_np,
                    X_adv_raw=adv_raw.cpu().numpy().astype(np.float32),
                    X_adv_scaled=((adv_raw - center) / scale).cpu().numpy().astype(np.float32),
                    sample_id=all_row_ids[idx],
                    attack_class=np.full(len(idx), class_name),
                    victim_model=np.full(len(idx), victim_name),
                    optimizer=np.full(len(idx), optimizer_name),
                    budget_name=np.full(len(idx), budget_name),
                    primitive_mode=np.full(len(idx), primitive_mode),
                    original_prediction=clean_prediction.cpu().numpy(),
                    adversarial_prediction=adversarial_prediction.cpu().numpy(),
                    target_class=np.zeros(len(idx), dtype=np.int64),
                    targeted_success=targeted.cpu().numpy(),
                    domain_valid=domain_valid.cpu().numpy(),
                    domain_violation_reasons=domain_reasons,
                    primitive_feasible=primitive_feasible_np,
                    primitive_violation_reasons=primitive_reasons,
                    semantic_status=semantic.semantic_status,
                    semantic_failure_reasons=semantic_failures,
                    semantic_not_testable_reasons=semantic_not_testable,
                    required_tests_passed=semantic.required_tests_passed,
                    required_tests_failed=semantic.required_tests_failed,
                    tests_not_testable=semantic.tests_not_testable,
                    original_duration=costs.original_duration,
                    adversarial_duration=costs.adversarial_duration,
                    delta_duration=costs.delta_duration,
                    relative_duration_change=costs.relative_duration_change,
                    original_byte_quantity=costs.original_byte_quantity,
                    adversarial_byte_quantity=costs.adversarial_byte_quantity,
                    added_byte_quantity=costs.added_byte_quantity,
                    relative_byte_change=costs.relative_byte_change,
                    original_rate=costs.original_rate,
                    adversarial_rate=costs.adversarial_rate,
                    rate_retention=costs.rate_retention,
                    padding_percent_of_forward_mean=costs.padding_percent_of_forward_mean,
                    normalized_padding_magnitude=costs.normalized_padding_magnitude,
                    normalized_timing_magnitude=costs.normalized_timing_magnitude,
                    primitive_p_requested=requested["p"].cpu().numpy(),
                    primitive_alpha_requested=requested["alpha"].cpu().numpy(),
                    primitive_p_projected=projected["p"].cpu().numpy(),
                    primitive_alpha_projected=projected["alpha"].cpu().numpy(),
                    primitive_values_requested=np.asarray([
                        json.dumps({"p": float(p), "alpha": float(alpha)})
                        for p, alpha in zip(
                            requested["p"].cpu().numpy(), requested["alpha"].cpu().numpy()
                        )
                    ]),
                    primitive_values_projected=np.asarray([
                        json.dumps({"p": float(p), "alpha": float(alpha)})
                        for p, alpha in zip(
                            projected["p"].cpu().numpy(), projected["alpha"].cpu().numpy()
                        )
                    ]),
                    p_hi=bounds["p"].cpu().numpy(),
                    alpha_hi=bounds["alpha"].cpu().numpy(),
                    features_changed=changed_features,
                    number_features_changed=semantic.number_features_changed,
                    clean_correct=eligible.cpu().numpy(),
                    primitive_transform_consistent=masks[
                        "primitive_transform_consistent"
                    ].cpu().numpy(),
                    in_distribution=masks["in_dist"].cpu().numpy(),
                    clean_logits=clean_logits.cpu().numpy(),
                    adversarial_logits=adversarial_logits.cpu().numpy(),
                    cost_total=cost["total"].cpu().numpy(),
                    cost_padding=cost["padding"].cpu().numpy(),
                    cost_timing=cost["timing"].cpu().numpy(),
                    cost_rate=cost["rate"].cpu().numpy(),
                    pad_semantic_allowed=caps.pad_allowed.cpu().numpy(),
                    timing_semantic_allowed=caps.timing_allowed.cpu().numpy(),
                    pad_disable_reason=np.asarray(caps.pad_reason),
                    timing_disable_reason=np.asarray(caps.timing_reason),
                    **artifact_provenance_arrays(
                        provenance,
                        row_ids=all_row_ids[idx],
                        class_name=class_name,
                        victim=victim_name,
                        method_id=method_id,
                        seed=seed,
                        checkpoint_ids=checkpoint_ids,
                    ),
                )

                eligible_np = eligible.cpu().numpy()
                semantic_pass_np = semantic.semantic_status == SemanticStatus.PASS.value
                semantic_fail_np = semantic.semantic_status == SemanticStatus.FAIL.value
                semantic_nt_np = (
                    semantic.semantic_status == SemanticStatus.NOT_FULLY_TESTABLE.value
                )
                values = lambda array: np.asarray(array)[eligible_np]
                cell = {
                    "class": class_name,
                    "victim": victim_name,
                    "seed": seed,
                    "budget_name": budget_name,
                    "primitive_mode": primitive_mode,
                    "optimizer": optimizer_name,
                    "artifact": str(artifact_path),
                    "n_total": len(idx),
                    "eligible_original_samples": int(eligible.sum()),
                    "n_targeted_success": int((targeted & eligible).sum()),
                    "raw_targeted_asr": _rate(targeted, eligible),
                    "valid_targeted_asr": _rate(targeted & domain_valid, eligible),
                    "primitive_feasible_targeted_asr": _rate(
                        targeted & domain_valid & primitive_feasible, eligible
                    ),
                    "sp_asr": _rate(
                        targeted & domain_valid & primitive_feasible & semantic_pass,
                        eligible,
                    ),
                    "semantic_pass_rate": float(values(semantic_pass_np).mean()),
                    "semantic_fail_rate": float(values(semantic_fail_np).mean()),
                    "not_fully_testable_rate": float(values(semantic_nt_np).mean()),
                    "semantic_testability_rate": float(values(~semantic_nt_np).mean()),
                    "semantic_pass_count": int(values(semantic_pass_np).sum()),
                    "semantic_fail_count": int(values(semantic_fail_np).sum()),
                    "not_fully_testable_count": int(values(semantic_nt_np).sum()),
                    "median_relative_duration_change": float(
                        np.median(values(costs.relative_duration_change))
                    ),
                    "median_relative_byte_change": float(
                        np.median(values(costs.relative_byte_change))
                    ),
                    "median_rate_retention": float(np.median(values(costs.rate_retention))),
                    "median_number_features_changed": float(
                        np.median(values(semantic.number_features_changed))
                    ),
                    "primitive_feasibility_rate": float(values(primitive_feasible_np).mean()),
                    "domain_validity_rate": _rate(domain_valid, eligible),
                    "calibrated_budget": class_cfg.budget.__dict__,
                }
                results["cells"].append(cell)
                print(json.dumps(cell, default=str), flush=True)
        (output_dir / "attack_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    parser.add_argument("--victims", default=",".join(VICTIMS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-limit", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--cost-weight", type=float, default=0.01)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--init-noise", type=float, default=0.5)
    parser.add_argument("--stage-a-dir", type=Path, default=None)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("artifacts/primattack/budget_calibration.json"),
    )
    parser.add_argument("--budget", choices=BUDGET_NAMES, default="maximum-evaluated")
    parser.add_argument("--primitive-mode", choices=PRIMITIVE_MODES, default="joint")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="optimized")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/primattack_calibrated")
    )
    args = parser.parse_args()
    run(
        classes=[value.strip() for value in args.classes.split(",") if value.strip()],
        victims=[value.strip() for value in args.victims.split(",") if value.strip()],
        device=args.device,
        test_limit=args.test_limit,
        steps=args.steps,
        lr=args.learning_rate,
        cost_weight=args.cost_weight,
        stage_a_dir=args.stage_a_dir,
        output_dir=args.output_dir,
        seeds=[int(value) for value in args.seeds.split(",") if value.strip()],
        init_noise=args.init_noise,
        calibration_path=args.calibration,
        budget_name=args.budget,
        primitive_mode=args.primitive_mode,
        optimizer_name=args.optimizer,
    )


if __name__ == "__main__":
    main()
