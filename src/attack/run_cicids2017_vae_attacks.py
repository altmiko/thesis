"""Run CICIDS2017 typed-VAE attacks against the four trained NN victims.

Offline thesis experiment only. A1-A6 are assembled through ``build_ablation``;
per-class perturbability uses the existing train-only CFF top-25% masks. Generation
and independent validation are separate. Reported denominators are clean-correct test
rows only.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attack.masks import get_dataset_mask
from attack.residual_head import ResidualAttackGenerator, ResidualHead
from attack.train_attack_head import StageBConfig, VictimGuidedTrainer
from datasets.cicids2017 import CICIDS2017Adapter
from experiments.ablations import build_ablation
from validation.attack_interface import structural_masks
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import ATTACK_CLASSES, load_stage_a

ABLATIONS = ("A1", "A2", "A3", "A4", "A5", "A6")
VICTIMS = ("mlp", "cnn", "ft_transformer")


def _seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subset(x: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(x) <= limit:
        return np.ascontiguousarray(x, dtype=np.float32)
    idx = np.random.default_rng(seed).choice(len(x), limit, replace=False)
    idx.sort()
    return np.ascontiguousarray(x[idx], dtype=np.float32)


def _class_x(split, class_id: int, limit: int | None = None, seed: int = 42) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(split.y) == class_id)
    if limit is not None and len(idx) > limit:
        pick = np.random.default_rng(seed).choice(len(idx), limit, replace=False)
        idx = idx[np.sort(pick)]
    return np.ascontiguousarray(np.asarray(split.x[idx]), dtype=np.float32)


def load_cff_mask(repo_root: Path, manifest, class_name: str, tier: str = "top25") -> torch.Tensor:
    cff = repo_root / "outputs" / "cff_cicids2017distrinet"
    order = json.loads((cff / "feature_order.json").read_text(encoding="utf-8"))
    manifest.assert_names_match(order)
    mask = np.load(cff / "masks" / f"{class_name}_{tier}.npy").astype(bool)
    if mask.shape != (manifest.n_features,):
        raise ValueError(f"{class_name}/{tier}: mask shape {mask.shape} != {(manifest.n_features,)}")
    mask[np.asarray(manifest.derived_indices(), dtype=np.int64)] = False
    return torch.tensor(mask, dtype=torch.bool)


def _build_generator(bundle, device: str, *, resolved=None, mutable_mask=None, latent_dim: int = 16) -> ResidualAttackGenerator:
    """Compose the attack generator.

    Config-mask path (``resolved`` given): the Layer-0 projector keeps value-type
    domain clamps + frozen preservation but drops the manifest's (possibly *inverse*)
    derived recompute, which would otherwise clobber a directly-perturbable feature.
    The verified dependencies are recomputed separately by :func:`_apply_dependencies`.
    Legacy CFF path (``resolved is None``): unchanged behaviour.
    """
    if resolved is not None:
        mode = bundle.config.residual_mode
        head = ResidualHead(latent_dim, resolved.n_features) if mode != "latent" else None
        generator = ResidualAttackGenerator(
            bundle.vae,
            resolved.generator_projector(),
            resolved.perturbable_mask(),
            head=head,
            mode=mode,
            apply_layer0=(0 in bundle.engine.active_layers),
        )
    elif bundle.generator is not None:
        generator = bundle.generator
    else:
        generator = ResidualAttackGenerator(
            bundle.vae,
            bundle.engine.layer0,
            mutable_mask,
            mode="latent",
            apply_layer0=(0 in bundle.engine.active_layers),
        )
    return generator.to(device)


def _apply_dependencies(resolved, vae, x_scaled: torch.Tensor, x_orig_scaled: torch.Tensor) -> torch.Tensor:
    """Differentiable dependency stage in raw space: restore FROZEN features from the
    source sample and recompute DERIVED_EXACT features from perturbed parents, then map
    back to model space. Gradients flow classifier -> derived -> perturbable parents."""
    raw = vae.continuous_scaled_to_raw(x_scaled)
    raw_orig = vae.continuous_scaled_to_raw(x_orig_scaled)
    raw = resolved.apply(raw, raw_orig)
    return vae.continuous_raw_to_scaled(raw)


def _assert_frozen_unchanged(resolved, vae, x_orig_scaled, x_adv_scaled, *, atol: float = 1e-5, rtol: float = 1e-4) -> int:
    with torch.no_grad():
        raw_adv = vae.continuous_scaled_to_raw(x_adv_scaled)
        raw_orig = vae.continuous_scaled_to_raw(x_orig_scaled)
        violations = int(resolved.frozen_violation_mask(raw_adv, raw_orig, atol=atol, rtol=rtol).sum())
    assert violations == 0, f"{violations} samples changed a FROZEN feature"
    return violations


def targeted_latent_attack(
    generator: ResidualAttackGenerator,
    engine,
    victim,
    x_scaled: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    latent_epsilon: float,
    lambda_delta: float = 0.05,
    lambda_constraints: float = 0.01,
    resolved=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Targeted benign latent PGD/Adam around each source posterior mean."""
    generator.eval()
    with torch.no_grad():
        z0 = generator.encode_mu(x_scaled)
    z = z0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    target = torch.zeros(len(x_scaled), dtype=torch.long, device=x_scaled.device)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        x_adv, _ = generator.generate(x_scaled, z)
        if resolved is not None:
            x_adv = _apply_dependencies(resolved, generator.vae, x_adv, x_scaled)
        logits = victim(x_adv)
        attack_loss = F.cross_entropy(logits, target)
        delta_loss = (x_adv - x_scaled).abs().mean()
        raw_adv = generator.vae.continuous_scaled_to_raw(x_adv)
        penalties = engine.penalty(raw_adv, w1=1.0, w2=1.0)
        loss = attack_loss + lambda_delta * delta_loss + lambda_constraints * penalties["total"]
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            z.clamp_(z0 - latent_epsilon, z0 + latent_epsilon)

    with torch.no_grad():
        x_adv, _ = generator.generate(x_scaled, z)
        if resolved is not None:
            x_adv = _apply_dependencies(resolved, generator.vae, x_adv, x_scaled)
    return x_adv.detach(), z.detach()


def _idr_mask(model, x_scaled: torch.Tensor, idr_path: Path) -> torch.Tensor:
    stats = np.load(idr_path)
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=x_scaled.device)
    precision = torch.tensor(stats["precision"], dtype=torch.float32, device=x_scaled.device)
    threshold = float(stats["threshold_sq"])
    with torch.no_grad():
        latent, _ = model.encode(x_scaled)
        delta = latent - mean.unsqueeze(0)
        distance_sq = torch.einsum("ni,ij,nj->n", delta, precision, delta)
    return distance_sq <= threshold


def _evaluate(
    victim,
    model,
    validator,
    transform,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    class_id: int,
    idr_path: Path,
    resolved=None,
) -> dict:
    with torch.no_grad():
        clean_pred = victim(x_clean).argmax(dim=1)
        adv_pred = victim(x_adv).argmax(dim=1)
        raw_clean = model.continuous_scaled_to_raw(x_clean)
        raw_adv = model.continuous_scaled_to_raw(x_adv)
        clean_valid = validator.validate(raw_clean)
        valid = validator.validate(raw_adv)
        clean_in_distribution = _idr_mask(model, x_clean, idr_path)
        in_distribution = _idr_mask(model, x_adv, idr_path)
        clean_correct = clean_pred == class_id
        evasion = adv_pred != class_id
        benign_target = adv_pred == 0
        denom = int(clean_correct.sum())
        scale = torch.tensor(transform.scale, dtype=torch.float32, device=x_adv.device)
        per_sample_cost = ((raw_adv - raw_clean).abs() / scale.unsqueeze(0)).mean(dim=1)
        if resolved is not None:
            frozen_violation = resolved.frozen_violation_mask(raw_adv, raw_clean)
            derived_fail = resolved.derived_consistency_mask(raw_adv, scale=scale)
            derived_counts = resolved.derived_consistency_counts(
                raw_adv, scale=scale, row_mask=clean_correct
            )
            perturb_freq = resolved.per_feature_perturbation_frequency(
                raw_adv, raw_clean, clean_correct
            )

    def conditional_rate(mask: torch.Tensor) -> float:
        if denom == 0:
            return float("nan")
        return float((mask & clean_correct).sum().item() / denom)

    engine_l2 = valid["pass_l0_l1_l2"]
    per_constraint = {
        name: 1.0 - conditional_rate(mask)
        for name, mask in valid["per_constraint"].items()
    }
    # validator_v2 is the HEADLINE structural verdict (joint_valid -> True_IDSR,
    # adv_validity_rate). The engine masks (pass_l0 / pass_l0_l1 / engine_l2) remain
    # the A0-A6 ablation ladder (ASR_L0 / ASR_L0_L1 / ASR_L0_L1_L2).
    v2_hybrid = torch.tensor(
        structural_masks(raw_adv.detach().cpu().numpy())["hybrid_valid"],
        device=raw_adv.device)
    joint_valid = v2_hybrid
    eligible_cost = per_sample_cost[clean_correct]
    result = {
        "denominator_clean_correct": denom,
        "clean_accuracy_on_attack_class": float(clean_correct.float().mean()),
        "clean_rate_L0": conditional_rate(clean_valid["pass_l0"]),
        "clean_rate_L0_L1": conditional_rate(clean_valid["pass_l0_l1"]),
        "clean_rate_L0_L1_L2": conditional_rate(clean_valid["pass_l0_l1_l2"]),
        "clean_IDR": conditional_rate(clean_in_distribution),
        "ASR_raw": conditional_rate(evasion),
        "targeted_benign_rate": conditional_rate(benign_target),
        "ASR_L0": conditional_rate(evasion & valid["pass_l0"]),
        "ASR_L0_L1": conditional_rate(evasion & valid["pass_l0_l1"]),
        "ASR_L0_L1_L2": conditional_rate(evasion & engine_l2),
        "IDR": conditional_rate(in_distribution),
        "True_IDSR": conditional_rate(evasion & joint_valid & in_distribution),
        "ASR_v2_hybrid_valid": conditional_rate(evasion & joint_valid),
        "adv_validity_rate_engine_L0_L1_L2": conditional_rate(engine_l2),
        "mean_normalized_cost": float(eligible_cost.mean()) if denom else float("nan"),
        "median_normalized_cost": float(eligible_cost.median()) if denom else float("nan"),
        "constraint_violation_rates": per_constraint,
    }
    if resolved is not None:
        result["adv_validity_rate"] = conditional_rate(joint_valid)
        result["invalid_sample_count"] = int(((~joint_valid) & clean_correct).sum())
        result["frozen_feature_violations"] = int((frozen_violation & clean_correct).sum())
        result["derived_consistency_failures"] = int((derived_fail & clean_correct).sum())
        result["per_feature_perturbation_frequency"] = perturb_freq
        result["derived_consistency_failures_by_feature"] = derived_counts
    return result


def run(
    *,
    classes: list[str],
    victims: list[str],
    ablations: list[str],
    device: str,
    test_limit: int,
    layer1_fit_limit: int,
    stage_b_train_limit: int,
    steps: int,
    learning_rate: float,
    latent_epsilon: float,
    constraint_weight: float,
    mask_tier: str,
    mask_source: str,
    stage_a_dir: Path | None,
    output_dir: Path,
) -> dict:
    _seed(42)
    adapter = CICIDS2017Adapter()
    repo_root = adapter.repo_root
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    resolved = get_dataset_mask(adapter.name).resolve(manifest) if mask_source == "config" else None
    if resolved is not None:
        print(resolved.diagnostics(), flush=True)
    train = adapter.load_split("train")
    test = adapter.load_split("test")
    raw_train = np.load(adapter._processed / "X_train_pristine.npy", mmap_mode="r")
    layer1_fit_raw = _subset(raw_train, layer1_fit_limit, 42)
    layer2_path = repo_root / "old_constraints" / adapter.name / "mined.json"
    stage_a_dir = stage_a_dir or (output_dir / "stage_a")
    victim_dir = repo_root / "outputs" / "cicids2017distrinet" / "models"
    artifact_dir = output_dir / "attack_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "dataset": adapter.name,
        "test_limit_per_class": test_limit,
        "layer1_fit_rows": len(layer1_fit_raw),
        "stage_b_train_limit_per_class": stage_b_train_limit,
        "attack_steps": steps,
        "attack_learning_rate": learning_rate,
        "latent_epsilon": latent_epsilon,
        "constraint_weight": constraint_weight,
        "mask_tier": mask_tier,
        "mask_source": mask_source,
        "classes": {},
    }
    for class_name in classes:
        class_id = mapping.name_to_id[class_name]
        checkpoint_path = stage_a_dir / f"vae_{class_name}.pt"
        idr_path = stage_a_dir / f"idr_{class_name}.npz"
        base_model, checkpoint = load_stage_a(
            adapter, checkpoint_path, expected_class_name=class_name, device=device
        )
        base_state = checkpoint["state_dict"]
        if resolved is not None:
            mutable_mask = resolved.perturbable_mask().to(device)
        else:
            mutable_mask = load_cff_mask(repo_root, manifest, class_name, tier=mask_tier).to(device)
        x_test_np = _class_x(test, class_id, test_limit, 42 + class_id)
        x_train_np = _class_x(train, class_id, stage_b_train_limit, 142 + class_id)
        x_test = torch.tensor(x_test_np, dtype=torch.float32, device=device)
        class_results: dict = {
            "class_id": class_id,
            "test_rows": len(x_test_np),
            "mutable_features": int(mutable_mask.sum()),
            "victims": {},
        }

        # Full independent validator used for every ablation, regardless of which
        # layers were active during generation.
        validator_bundle = build_ablation(
            "A4",
            adapter,
            encoder_input_transform="asinh",
            layer1_fit_x_raw=layer1_fit_raw,
            mutable_mask=mutable_mask,
            layer2_path=layer2_path,
        )
        validator = validator_bundle.engine

        for victim_name in victims:
            victim = load_category_victim(
                victim_dir / f"{victim_name}_category.pt",
                adapter=adapter, expected_model_type=victim_name, device=device,
            )
            victim_results: dict = {}
            for ablation in ablations:
                _seed(42)
                bundle = build_ablation(
                    ablation,
                    adapter,
                    encoder_input_transform="asinh",
                    layer1_fit_x_raw=layer1_fit_raw,
                    mutable_mask=mutable_mask,
                    layer2_path=layer2_path,
                )
                bundle.vae.load_state_dict(base_state, strict=True)
                bundle.vae.to(device).eval()
                for parameter in bundle.vae.parameters():
                    parameter.requires_grad_(False)
                generator = _build_generator(bundle, device, resolved=resolved, mutable_mask=mutable_mask)

                stage_b_history = None
                if ablation == "A6":
                    trainer = VictimGuidedTrainer(
                        generator,
                        victim,
                        transform,
                        bundle.engine,
                        StageBConfig(
                            epochs=5,
                            batch_size=256,
                            lr=1e-3,
                            lambda_attack=1.0,
                            lambda_delta=0.1,
                            lambda_c1=constraint_weight,
                            lambda_c2=constraint_weight,
                            seed=42,
                        ),
                        device=device,
                    )
                    stage_b_history = trainer.fit(x_train_np)["history"]
                    head_dir = output_dir / "stage_b"
                    head_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "state_dict": generator.head.state_dict(),
                            "class_name": class_name,
                            "victim": victim_name,
                            "manifest_hash": manifest.content_hash,
                            "mask_tier": mask_tier,
                            "history": stage_b_history,
                        },
                        head_dir / f"head_{class_name}_{victim_name}.pt",
                    )

                x_adv_parts: list[torch.Tensor] = []
                for start in range(0, len(x_test), 256):
                    xb = x_test[start : start + 256]
                    adv, _ = targeted_latent_attack(
                        generator,
                        bundle.engine,
                        victim,
                        xb,
                        steps=steps,
                        learning_rate=learning_rate,
                        latent_epsilon=latent_epsilon,
                        lambda_constraints=constraint_weight,
                        resolved=resolved,
                    )
                    x_adv_parts.append(adv)
                x_adv = torch.cat(x_adv_parts, dim=0)
                if resolved is not None:
                    _assert_frozen_unchanged(resolved, base_model, x_test, x_adv)
                metrics = _evaluate(
                    victim,
                    base_model,
                    validator,
                    transform,
                    x_test,
                    x_adv,
                    class_id,
                    idr_path,
                    resolved,
                )
                with torch.no_grad():
                    y_pred_clean = victim(x_test).argmax(dim=1)
                    y_pred_adv = victim(x_adv).argmax(dim=1)
                artifact_path = (
                    artifact_dir / f"{class_name}_{victim_name}_{ablation}.npz"
                )
                np.savez_compressed(
                    artifact_path,
                    X_clean=x_test.detach().cpu().numpy().astype(np.float32, copy=False),
                    X_adv=x_adv.detach().cpu().numpy().astype(np.float32, copy=False),
                    y_true=np.full(len(x_test), class_id, dtype=np.int64),
                    y_pred_clean=y_pred_clean.detach().cpu().numpy().astype(np.int64, copy=False),
                    y_pred_adv=y_pred_adv.detach().cpu().numpy().astype(np.int64, copy=False),
                    attack_name=np.asarray(
                        f"vae-targeted-{class_name}-{victim_name}-{ablation}"
                    ),
                )
                metrics["artifact_path"] = str(artifact_path)
                metrics["active_generation_layers"] = sorted(bundle.engine.active_layers)
                if stage_b_history is not None:
                    metrics["stage_b_history"] = stage_b_history
                victim_results[ablation] = metrics
                print(json.dumps({"class": class_name, "victim": victim_name, "ablation": ablation, **{k: v for k, v in metrics.items() if not isinstance(v, dict)}}), flush=True)
            class_results["victims"][victim_name] = victim_results
        results["classes"][class_name] = class_results
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "attack_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    parser.add_argument("--victims", default=",".join(VICTIMS))
    parser.add_argument("--ablations", default=",".join(ABLATIONS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-limit", type=int, default=1024)
    parser.add_argument("--layer1-fit-limit", type=int, default=200000)
    parser.add_argument("--stage-b-train-limit", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--latent-epsilon", type=float, default=20.0)
    parser.add_argument("--constraint-weight", type=float, default=0.1)
    parser.add_argument(
        "--mask-tier",
        choices=("top10", "top25", "top50", "eligible"),
        default="eligible",
    )
    parser.add_argument("--mask-source", choices=("config", "cff"), default="config")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cicids2017_vae_attacks"))
    parser.add_argument(
        "--stage-a-dir",
        type=Path,
        default=None,
        help="Existing Stage-A checkpoints; defaults to <output-dir>/stage_a.",
    )
    args = parser.parse_args()
    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    victims = [x.strip() for x in args.victims.split(",") if x.strip()]
    ablations = [x.strip() for x in args.ablations.split(",") if x.strip()]
    unknown_classes = sorted(set(classes) - set(ATTACK_CLASSES))
    unknown_victims = sorted(set(victims) - set(VICTIMS))
    unknown_ablations = sorted(set(ablations) - set(ABLATIONS))
    if unknown_classes or unknown_victims or unknown_ablations:
        raise ValueError(
            f"unknown selections classes={unknown_classes}, victims={unknown_victims}, "
            f"ablations={unknown_ablations}"
        )
    run(
        classes=classes,
        victims=victims,
        ablations=ablations,
        device=args.device,
        test_limit=args.test_limit,
        layer1_fit_limit=args.layer1_fit_limit,
        stage_b_train_limit=args.stage_b_train_limit,
        steps=args.steps,
        learning_rate=args.learning_rate,
        latent_epsilon=args.latent_epsilon,
        constraint_weight=args.constraint_weight,
        mask_tier=args.mask_tier,
        mask_source=args.mask_source,
        stage_a_dir=args.stage_a_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
