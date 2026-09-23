"""Realizability-aware primitive-control attack on CICIDS2017-DistriNet.

The attacker optimizes two attacker-controllable primitives per flow -- forward
packet-length augmentation ``p`` and forward timing dilation ``alpha`` -- via a
differentiable primitive->feature map (:mod:`attack.realizability.cicids2017`). Every
aggregate/derived CICFlowMeter feature affected by a primitive is deterministically
recomputed; unrelated features are copied from the ORIGINAL RAW test row (pristine), so no
inverse-transform / float round-trip artifact is introduced.

Pipeline (per section 1 of the design):

    pristine raw flow
      -> differentiable optimization over (p, alpha)          [continuous]
      -> discrete realizability projection (round p; us-quantize timing)
      -> complete dependency recomputation                    [model.generate(quantize=True)]
      -> victim classifier + independent validators
      -> metrics reported AFTER projection

Threat model: Attack -> Benign (targeted). Untargeted evasion is reported too, but the
headline is the targeted-benign / targeted-strict-valid rate over the clean-correct
denominator.

NOTE ON THE VAE: the perturbation is the primitive pair, optimized directly in
primitive space; the classifier gradient does NOT flow through any VAE encoder/decoder.
The Stage-A per-class beta-VAE is used ONLY as the realism gate (val-anchored Mahalanobis
in-distribution test / True-IDSR). This is therefore a *differentiable primitive-domain
attack with a VAE realism gate*, not a latent-VAE attack -- see the audit report.

Offline thesis experiment only.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attack.realizability.base import NullPacketBackend
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel, SCALER_ATOL
from attack.realizability.validator import RealizabilityValidator
from attack.run_cicids2017_vae_attacks import _idr_mask
from datasets.cicids2017 import CICIDS2017Adapter
from evaluation.pave_style_validator import PAVEStyleValidator
from experiments.ablations import build_ablation
from experiments.provenance import (
    artifact_provenance_arrays,
    build_provenance,
    deterministic_runtime,
    ensure_fresh_output_dir,
    load_row_ids,
)
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import ATTACK_CLASSES, load_stage_a

VICTIMS = ("mlp", "cnn", "lstm", "serial")

# Feature groups for the additive normalized-cost decomposition (roles, see model).
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

# Train-mined feature envelopes used for the data-driven per-flow padding/timing headroom.
_ENV_TIMING = ("Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Std", "Fwd IAT Mean", "Flow Duration")
_ENV_LENGTH = ("Fwd Packet Length Max", "Fwd Packet Length Min",
               "Fwd Packet Length Mean", "Total Length of Fwd Packet")


def _seed(seed: int) -> None:
    deterministic_runtime(seed)


def _class_rows(y: np.ndarray, class_id: int, limit: int | None, seed: int) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(y) == class_id)
    if limit is not None and len(idx) > limit:
        pick = np.random.default_rng(seed).choice(len(idx), limit, replace=False)
        idx = idx[np.sort(pick)]
    return idx


def train_envelope(raw_train: np.ndarray, i: dict[str, int]) -> dict[str, float]:
    """Train-only (leakage-safe) upper envelope for the controlled features."""
    feats = set(_ENV_TIMING) | set(_ENV_LENGTH)
    return {n: float(np.asarray(raw_train[:, i[n]]).max()) for n in feats}


def optimize_primitives(model, victim, raw, center, scale, bounds, *, steps, lr, cost_weight,
                        init_noise, seed):
    """Adam over per-flow (p, alpha) toward Benign; primitives kept in [identity, per-flow cap]
    by a sigmoid reparameterization (hard bounds act on the CONTROL space, not derived features)."""
    n = raw.shape[0]
    g = torch.Generator(device=raw.device).manual_seed(seed)
    u = (-2.0 + init_noise * torch.randn(n, generator=g, device=raw.device, dtype=raw.dtype)).requires_grad_(True)
    v = (-2.0 + init_noise * torch.randn(n, generator=g, device=raw.device, dtype=raw.dtype)).requires_grad_(True)
    opt = torch.optim.Adam([u, v], lr=lr)
    target = torch.zeros(n, dtype=torch.long, device=raw.device)
    p_hi, alpha_hi = bounds["p"], bounds["alpha"]
    # A2: forward timing dilation is meaningless for single-forward-packet flows -- disable it
    # AT OPTIMIZATION TIME so the optimizer wastes no effort and reports no learned alpha.
    timing_active = model.active_mask(raw, "alpha").to(raw.dtype)
    pad_active = model.active_mask(raw, "p").to(raw.dtype)

    def controls():
        return {"p": p_hi * torch.sigmoid(u) * pad_active,
                "alpha": 1.0 + (alpha_hi - 1.0) * torch.sigmoid(v) * timing_active}

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        ctl = controls()
        logits = victim((model.generate(raw, ctl) - center) / scale)
        loss_per_sample = (
            F.cross_entropy(logits, target, reduction="none")
            + cost_weight * (
                torch.sigmoid(u) * pad_active + torch.sigmoid(v) * timing_active
            )
        )
        loss_per_sample.sum().backward()
        opt.step()
    with torch.no_grad():
        ctl = controls()
    return {k: t.detach() for k, t in ctl.items()}


def _decompose_cost(adv_raw, raw, scale, groups_idx):
    per = ((adv_raw - raw).abs() / scale).sum(1)  # summed normalized L1 (not yet /F)
    F = raw.shape[1]
    out = {"total": (per / F)}
    for name, idx in groups_idx.items():
        out[name] = ((adv_raw[:, idx] - raw[:, idx]).abs() / scale[idx]).sum(1) / F
    return out


def evaluate_cell(model, val, victim, base_vae, engine, pave, raw, adv_raw, center, scale,
                  class_id, idr_path, groups_idx):
    """Return per-sample masks + arrays (analyzer computes every rate/CI from these)."""
    with torch.no_grad():
        x_clean = (raw - center) / scale
        x_adv = (adv_raw - center) / scale
        clean_pred = victim(x_clean).argmax(1)
        adv_pred = victim(x_adv).argmax(1)
        clean_correct = clean_pred == class_id
        evasion = adv_pred != class_id
        benign = adv_pred == 0

        pave_valid = torch.tensor(np.asarray(pave.validate_batch(adv_raw.cpu().numpy())["valid_mask"]),
                                  device=raw.device)
        mined = engine.validate(adv_raw)["pass_l0_l1_l2"]
        in_dist = _idr_mask(base_vae, x_adv, idr_path)
        rep = val.validate(adv_raw, raw)
        cats = rep.categories
        cost = _decompose_cost(adv_raw, raw, scale, groups_idx)

    masks = {
        "clean_correct": clean_correct, "evasion": evasion, "benign": benign,
        "pave_valid": pave_valid, "mined_valid": mined, "in_dist": in_dist,
        "dep_ok": ~cats["algebraic_dependency_fail"],
        "packet_ok": ~cats["packet_summary_fail"],
        "timing_ok": ~cats["timing_fail"],
        "rate_ok": ~cats["negative_rate_fail"],
        "disc_ok": ~cats["discreteness_fail"],
        "frozen_ok": ~cats["frozen_fail"],
        "realizable": rep.valid,
    }
    return masks, cost, clean_pred, adv_pred


def run(*, classes, victims, device, test_limit, steps, lr, p_max, alpha_max, mtu_cap,
        cost_weight, stage_a_dir, output_dir, seeds, init_noise):
    adapter = CICIDS2017Adapter()
    repo = adapter.repo_root
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    model = CICIDS2017PrimitiveModel(manifest)
    val = RealizabilityValidator(model)
    packet_backend = NullPacketBackend()
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)
    groups_idx = {
        "padding": torch.tensor([model.i[n] for n in _LENGTH_COLS], device=device),
        "timing": torch.tensor([model.i[n] for n in _TIMING_COLS], device=device),
        "rate": torch.tensor([model.i[n] for n in _RATE_COLS], device=device),
    }

    test = adapter.load_split("test")
    raw_test = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    raw_train = np.load(adapter._processed / "X_train_pristine.npy", mmap_mode="r")
    layer1_fit = np.ascontiguousarray(raw_train[:200000], dtype=np.float32)
    layer2_path = repo / "constraints" / adapter.name / "mined.json"
    stage_a_dir = stage_a_dir or (repo / "outputs" / "cicids2017_vae_stage_a")
    victim_dir = repo / "outputs" / "cicids2017distrinet" / "models"

    pave = PAVEStyleValidator(integer_tolerance=SCALER_ATOL, range_tolerance=SCALER_ATOL).fit(
        np.asarray(raw_train, dtype=np.float64), manifest.names, schema=manifest)
    envelope = train_envelope(raw_train, model.i)
    bounds_cfg = {"p_max": p_max, "alpha_max": alpha_max, "mtu_cap": mtu_cap,
                  **{f"env_{k}": v for k, v in envelope.items()}}
    config = {"test_limit_per_class": test_limit, "attack_steps": steps, "learning_rate": lr,
              "p_max": p_max, "alpha_max": alpha_max, "mtu_cap": mtu_cap,
              "cost_weight": cost_weight, "seeds": seeds, "init_noise": init_noise,
              "scaler_atol": SCALER_ATOL, "dur_floor_us": model.dur_floor_us,
              "attack_batch_size_per_class": test_limit}
    checkpoint_paths = {
        **{f"victim_{name}": victim_dir / f"{name}_category.pt" for name in victims},
        **{f"vae_{name}": stage_a_dir / f"vae_{name}.pt" for name in classes},
        **{f"idr_{name}": stage_a_dir / f"idr_{name}.npz" for name in classes},
    }
    provenance = build_provenance(
        repo_root=repo, dataset=adapter.name, method_id="primitive_direct", config=config,
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
        "dataset": adapter.name,
        "method_id": "primitive_direct",
        "attack": "Direct Primitive-Domain Attack (optimize p, alpha directly; realizability layer; NO VAE in gradient path)",
        "threat_model": "targeted Attack->Benign (untargeted also reported)",
        "denominator": "clean-correct malicious test rows per (class, victim)",
        "vae_role": "realism gate only (val-anchored Mahalanobis IDR); NOT in the attack gradient path",
        "packet_level_verification": {"available": packet_backend.available(),
                                       "reason": packet_backend.reason},
        "strict_valid_definition": "PAVE & mined & primitive-realizability evaluator",
        "config": config,
        "provenance": provenance,
        "primitives": [pspec.__dict__ for pspec in model.primitives()],
        "feature_roles": {n: {"role": r.value, "reason": why} for n, (r, why) in model.roles().items()},
        "cells": [],
    }

    for class_name in classes:
        cid = mapping.name_to_id[class_name]
        # Fixed evaluation rows across seeds (isolates optimization variance from sampling).
        idx = _class_rows(test.y, cid, test_limit, 42 + cid)
        raw_np = np.ascontiguousarray(np.asarray(raw_test[idx]), dtype=np.float32)
        raw = torch.tensor(raw_np, device=device)
        base_vae, _ = load_stage_a(
            adapter, stage_a_dir / f"vae_{class_name}.pt",
            expected_class_name=class_name, device=device,
        )
        idr_path = stage_a_dir / f"idr_{class_name}.npz"
        engine = build_ablation("A4", adapter, encoder_input_transform="asinh",
                                layer1_fit_x_raw=layer1_fit, layer2_path=layer2_path).engine
        bounds = model.per_flow_bounds(raw, bounds_cfg)
        for vname in victims:
            victim = load_category_victim(
                victim_dir / f"{vname}_category.pt", adapter=adapter,
                expected_model_type=vname, device=device,
            )
            for seed in seeds:
                _seed(seed)
                ctl = optimize_primitives(model, victim, raw, center, scale, bounds,
                                          steps=steps, lr=lr, cost_weight=cost_weight,
                                          init_noise=init_noise, seed=seed)
                proj = model.project_controls(raw, ctl)
                adv_raw = model.generate(raw, proj, quantize=True).detach()
                # hard guarantee: every frozen feature is byte-identical to the pristine source
                fidx = torch.tensor([model.i[n] for n in val.frozen_names], device=device)
                assert torch.allclose(adv_raw[:, fidx], raw[:, fidx], atol=SCALER_ATOL, rtol=1e-4), \
                    "frozen feature changed under primitive map"
                masks, cost, yc, ya = evaluate_cell(
                    model, val, victim, base_vae, engine, pave, raw, adv_raw,
                    center, scale, cid, idr_path, groups_idx)
                ap = artifact_dir / f"{class_name}_{vname}_seed{seed}.npz"
                strict = masks["pave_valid"] & masks["mined_valid"] & masks["realizable"]
                checkpoint_ids = {
                    key: provenance["checkpoints"][key]["sha256"]
                    for key in (f"victim_{vname}", f"vae_{class_name}", f"idr_{class_name}")
                }
                with torch.no_grad():
                    clean_logits = victim((raw - center) / scale)
                    final_logits = victim((adv_raw - center) / scale)
                np.savez_compressed(
                    ap,
                    X_clean_raw=raw_np,
                    X_adv_raw=adv_raw.cpu().numpy().astype(np.float32),
                    X_adv_scaled=((adv_raw - center) / scale).cpu().numpy().astype(np.float32),
                    p=proj["p"].cpu().numpy().astype(np.float32),
                    alpha=proj["alpha"].cpu().numpy().astype(np.float32),
                    p_cont=ctl["p"].cpu().numpy().astype(np.float32),
                    alpha_cont=ctl["alpha"].cpu().numpy().astype(np.float32),
                    p_hi=bounds["p"].cpu().numpy().astype(np.float32),
                    alpha_hi=bounds["alpha"].cpu().numpy().astype(np.float32),
                    y_true=np.full(len(idx), cid, dtype=np.int64),
                    true_label=np.full(len(idx), cid, dtype=np.int64),
                    y_pred_clean=yc.cpu().numpy().astype(np.int64),
                    clean_prediction=yc.cpu().numpy().astype(np.int64),
                    y_pred_adv=ya.cpu().numpy().astype(np.int64),
                    final_adversarial_prediction=ya.cpu().numpy().astype(np.int64),
                    clean_logits=clean_logits.cpu().numpy().astype(np.float32),
                    final_adversarial_logits=final_logits.cpu().numpy().astype(np.float32),
                    cost_total=cost["total"].cpu().numpy().astype(np.float32),
                    cost_padding=cost["padding"].cpu().numpy().astype(np.float32),
                    cost_timing=cost["timing"].cpu().numpy().astype(np.float32),
                    cost_rate=cost["rate"].cpu().numpy().astype(np.float32),
                    timing_active=model.active_mask(raw, "alpha").cpu().numpy(),
                    target_success_flag=masks["benign"].cpu().numpy(),
                    strict_valid=strict.cpu().numpy(),
                    **artifact_provenance_arrays(
                        provenance, row_ids=all_row_ids[idx], class_name=class_name,
                        victim=vname, method_id="primitive_direct", seed=seed,
                        checkpoint_ids=checkpoint_ids,
                    ),
                    **{k: m.cpu().numpy() for k, m in masks.items()},
                )
                denom = int(masks["clean_correct"].sum())
                cc = masks["clean_correct"]
                strict = masks["pave_valid"] & masks["mined_valid"] & masks["realizable"]
                rate = lambda m: (float((m & cc).sum()) / denom) if denom else float("nan")
                cell = {
                    "class": class_name, "victim": vname, "seed": seed,
                    "artifact": str(ap),
                    "n_total": len(idx), "n_clean_correct": denom,
                    "n_untargeted_success": int((masks["evasion"] & cc).sum()),
                    "n_targeted_benign_success": int((masks["benign"] & cc).sum()),
                    "n_strict_valid": int((strict & cc).sum()),
                    "n_targeted_strict_valid": int((masks["benign"] & strict & cc).sum()),
                    "untargeted_asr": rate(masks["evasion"]),
                    "targeted_benign_asr": rate(masks["benign"]),
                    "untargeted_strict_valid_asr": rate(masks["evasion"] & strict),
                    "targeted_strict_valid_asr": rate(masks["benign"] & strict),
                    "pave_validity": rate(masks["pave_valid"]),
                    "mined_validity": rate(masks["mined_valid"]),
                    "dependency_validity": rate(masks["dep_ok"]),
                    "discreteness_validity": rate(masks["disc_ok"]),
                    "realizability_aware_validity": rate(masks["realizable"]),
                    "strict_validity": rate(strict),
                    "IDR": rate(masks["in_dist"]),
                    "true_idsr": rate(masks["benign"] & strict & masks["in_dist"]),
                    "cost_total_mean": float(cost["total"][cc].mean()) if denom else float("nan"),
                    "cost_padding_mean": float(cost["padding"][cc].mean()) if denom else float("nan"),
                    "cost_timing_mean": float(cost["timing"][cc].mean()) if denom else float("nan"),
                    "p_median": float(np.median(proj["p"].cpu().numpy()[cc.cpu().numpy()])) if denom else float("nan"),
                    "alpha_median": float(np.median(proj["alpha"].cpu().numpy()[cc.cpu().numpy()])) if denom else float("nan"),
                    "fail_counts": {k: int((~masks[k] & cc).sum()) for k in
                                    ("dep_ok", "packet_ok", "timing_ok", "rate_ok", "disc_ok", "frozen_ok")},
                }
                results["cells"].append(cell)
                print(json.dumps({k: cell[k] for k in
                                  ("class", "victim", "seed", "n_clean_correct", "untargeted_asr",
                                   "targeted_benign_asr", "targeted_strict_valid_asr",
                                   "strict_validity", "cost_total_mean")}), flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "attack_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    ap.add_argument("--victims", default=",".join(VICTIMS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test-limit", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--p-max", type=float, default=1460.0, help="absolute fwd padding ceiling (bytes)")
    ap.add_argument("--alpha-max", type=float, default=100.0, help="absolute timing dilation ceiling")
    ap.add_argument("--mtu-cap", type=float, default=0.0,
                    help="optional per-packet resulting-length cap (0=disabled; use train envelope only)")
    ap.add_argument("--cost-weight", type=float, default=0.01)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--init-noise", type=float, default=0.5)
    ap.add_argument("--stage-a-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/cicids2017_primitive_attack"))
    a = ap.parse_args()
    classes = [x.strip() for x in a.classes.split(",") if x.strip()]
    victims = [x.strip() for x in a.victims.split(",") if x.strip()]
    seeds = [int(x) for x in str(a.seeds).split(",") if str(x).strip()]
    run(classes=classes, victims=victims, device=a.device, test_limit=a.test_limit, steps=a.steps,
        lr=a.learning_rate, p_max=a.p_max, alpha_max=a.alpha_max, mtu_cap=a.mtu_cap,
        cost_weight=a.cost_weight, stage_a_dir=a.stage_a_dir, output_dir=a.output_dir,
        seeds=seeds, init_noise=a.init_noise)


if __name__ == "__main__":
    main()
