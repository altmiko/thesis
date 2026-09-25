"""Controlled PrimAttack optimizer ablation: Hybrid (default) vs Prim-PGD vs Prim-C&W.

Every method runs through ``attack.primitive_optimizer.RealizedSearch``, so all three share:
the frozen clean-correct source rows (read from the canonical campaign's ``selection.json``
and re-verified), the victims, the primitive controls ``(p, delay, shape)``, the calibrated
per-flow hard box of each budget (p50 / p75 / envelope-only "unbounded"), the canonical
feature recomputation with integer quantization, the validator_v2 ``hybrid_valid`` gate,
the targeted->Benign success predicate, the incumbent ordering, and one per-flow cap on
victim forward evaluations (realized + surrogate). Mode is ``joint`` (all three controls).

Per (victim, class, budget, seed, method) a per-row npz and one ``cells.json`` entry are
written under ``--output-dir`` (default ``outputs/primattack_optimizer_ablation/<dataset>``).
An existing output directory is never overwritten; pass ``--resume`` to finish it.

    python scripts/run_primattack_optimizer_ablation.py --dataset cicids2017 --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from attack.flow_semantics import FlowSemanticValidator, SemanticStatus  # noqa: E402
from attack.primattack_budget import (  # noqa: E402
    class_calibration, load_calibration, unbounded_calibration,
)
from attack.primitive_optimizer import (  # noqa: E402
    CANDIDATE_NAMES, hybrid_valid_gate, optimize_primitive_candidates, optimize_primitive_cw,
    optimize_primitive_pgd,
)
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel  # noqa: E402
from attack.realizability.validator import RealizabilityValidator  # noqa: E402
from attack.run_cicids2017_primitive_attack import (  # noqa: E402
    _LENGTH_COLS, _RATE_COLS, _TIMING_COLS, _apply_primitive_mode, _decompose_cost, evaluate_cell,
)
from datasets import get_adapter  # noqa: E402
from experiments.provenance import deterministic_runtime  # noqa: E402
from run_full_adversarial_eval import (  # noqa: E402
    BENIGN_ID, BUDGET_LABEL, DATASET_DEFAULTS, victim_checkpoint,
)
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402
from vae.cicids2017_stage_a import ATTACK_CLASSES  # noqa: E402

METHODS = ("hybrid", "pgd", "cw")
MODE = "joint"


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _sha_ids(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, ids.tolist())).encode("utf-8")).hexdigest()


def method_configs(args) -> dict[str, dict]:
    b = args.eval_budget
    pgd_steps = (b - 1) // (2 * args.pgd_restarts)
    cw_steps = (b - 1) // (2 * args.cw_stages)
    return {
        "hybrid": {"steps": args.hybrid_steps, "learning_rate": args.hybrid_lr,
                   "restarts": None, "canonical_restarts": 2, "momentum": 0.75,
                   "checkpoint_interval": max(5, args.hybrid_steps // 4), "stall_factor": 0.5},
        "pgd": {"steps": pgd_steps, "step_size": args.pgd_step, "restarts": args.pgd_restarts,
                "momentum": args.pgd_momentum},
        "cw": {"steps": cw_steps, "learning_rate": args.cw_lr, "stages": args.cw_stages,
               "c_init": args.cw_c, "kappa": args.cw_kappa, "betas": [0.9, 0.999]},
    }


def run_method(method, cfg, args_tuple, *, seed, gate, budget):
    if method == "hybrid":
        return optimize_primitive_candidates(
            *args_tuple, steps=cfg["steps"], learning_rate=cfg["learning_rate"], seed=seed,
            restarts=cfg["restarts"], validity_fn=gate, eval_budget=budget)
    if method == "pgd":
        return optimize_primitive_pgd(
            *args_tuple, steps=cfg["steps"], step_size=cfg["step_size"],
            restarts=cfg["restarts"], momentum=cfg["momentum"], seed=seed,
            validity_fn=gate, eval_budget=budget)
    if method == "cw":
        return optimize_primitive_cw(
            *args_tuple, steps=cfg["steps"], learning_rate=cfg["learning_rate"],
            stages=cfg["stages"], c_init=cfg["c_init"], kappa=cfg["kappa"],
            betas=tuple(cfg["betas"]), validity_fn=gate, eval_budget=budget)
    raise KeyError(method)


def _stats(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return float("nan"), float("nan")
    return float(values.mean()), float(np.median(values))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cicids2017")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--selection-from", type=Path, default=None,
                    help="frozen selection.json (default: outputs/adv_campaign_noidr/<dataset>)")
    ap.add_argument("--split", choices=("test", "val"), default="test",
                    help="test: frozen canonical selection; val: fresh seeded clean-correct "
                         "selection (hyperparameter tuning only)")
    ap.add_argument("--n-per-class", type=int, default=800, help="val split only")
    ap.add_argument("--selection-seed", type=int, default=42, help="val split only")
    ap.add_argument("--seeds", default="42,123,2024")
    ap.add_argument("--match-victim-seed", action="store_true")
    ap.add_argument("--victims", default=None)
    ap.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    ap.add_argument("--budgets", default="intermediate,maximum-evaluated,unbounded")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--eval-budget", type=int, default=256,
                    help="per-flow cap on victim forward evaluations (realized + surrogate), "
                         "including the shared identity evaluation")
    ap.add_argument("--hybrid-steps", type=int, default=40)
    ap.add_argument("--hybrid-lr", type=float, default=0.1)
    ap.add_argument("--pgd-restarts", type=int, default=3)
    ap.add_argument("--pgd-step", type=float, default=0.05)
    ap.add_argument("--pgd-momentum", type=float, default=0.75)
    ap.add_argument("--cw-stages", type=int, default=3)
    ap.add_argument("--cw-lr", type=float, default=0.05)
    ap.add_argument("--cw-c", type=float, default=1.0)
    ap.add_argument("--cw-kappa", type=float, default=0.0)
    ap.add_argument("--limit-rows", type=int, default=None, help="smoke tests only")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    adapter = get_adapter(args.dataset)
    dataset = adapter.name
    defaults = DATASET_DEFAULTS[dataset]
    out = args.output_dir or (REPO_ROOT / "outputs/primattack_optimizer_ablation" / dataset)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"{out} exists; refusing to overwrite (use --resume)")
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    device = args.device
    methods = _csv(args.methods)
    if set(methods) - set(METHODS):
        raise ValueError(f"unknown methods {sorted(set(methods) - set(METHODS))}")
    budgets, classes = _csv(args.budgets), _csv(args.classes)
    victims = _csv(args.victims or defaults["victims"])
    base_seeds = [int(s) for s in _csv(args.seeds)]
    cfgs = method_configs(args)

    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)
    model = CICIDS2017PrimitiveModel(manifest)
    realizability = RealizabilityValidator(model)
    calibration_path = defaults["calibration"]
    calibration = load_calibration(calibration_path)
    if calibration.get("dataset") != dataset or calibration.get("fit_split") != "train":
        raise ValueError("calibration must be the train-fit artifact of this dataset")
    semantic_validator = FlowSemanticValidator(model, calibration)
    gate = hybrid_valid_gate(dataset)
    groups_idx = {
        name: torch.tensor([model.i[c] for c in cols], device=device)
        for name, cols in (("padding", _LENGTH_COLS), ("timing", _TIMING_COLS),
                           ("rate", _RATE_COLS))
    }

    processed = adapter._processed
    split = args.split
    raw_all = np.load(processed / f"X_{split}_pristine.npy", mmap_mode="r")
    y = np.load(processed / f"y_{split}_cat.npy").astype(np.int64)
    meta = pd.read_parquet(processed / f"{split}.parquet",
                           columns=["sample_id", "Src IP", "Dst IP"])
    sample_ids_all = meta["sample_id"].astype(str).to_numpy(dtype="U128")
    raw_all_t = torch.tensor(np.ascontiguousarray(raw_all), dtype=torch.float32, device=device)

    if split == "test":
        sel_path = args.selection_from or (REPO_ROOT / "outputs/adv_campaign_noidr" / dataset
                                           / "selection.json")
        frozen = json.loads(sel_path.read_text(encoding="utf-8"))
    else:
        sel_path, frozen = None, {}

    victim_info, rows = {}, {}
    for vname in victims:
        ckpt, arch, train_seed = victim_checkpoint(dataset, vname)
        seeds = [train_seed] if (args.match_victim_seed and train_seed is not None) else base_seeds
        victim = load_category_victim(ckpt, adapter=adapter, expected_model_type=arch,
                                      device=device)
        victim_info[vname] = {"checkpoint": str(ckpt), "arch": arch, "train_seed": train_seed,
                              "attack_seeds": seeds, "victim": victim}
        if split != "test":
            # Canonical driver's rule on this split: one seeded permutation of the class rows,
            # first n clean-correct, re-sorted to split order.
            with torch.no_grad():
                pred = torch.cat([victim((raw_all_t[s:s + 16384] - center) / scale).argmax(1)
                                  for s in range(0, len(y), 16384)]).cpu().numpy()
            frozen[vname] = {}
            for cname in classes:
                cid = int(mapping.name_to_id[cname])
                order = np.random.default_rng(args.selection_seed + cid).permutation(
                    np.flatnonzero(y == cid))
                idx = np.sort(order[pred[order] == cid][: args.n_per_class])
                frozen[vname][cname] = {
                    "positional_idx": idx.tolist(), "sample_ids": sample_ids_all[idx].tolist(),
                    "sha256_sample_ids": _sha_ids(sample_ids_all[idx]), "n_used": int(len(idx)),
                }
        for cname in classes:
            entry = frozen[vname][cname]
            idx = np.asarray(entry["positional_idx"], dtype=np.int64)
            if args.limit_rows:
                idx = idx[: args.limit_rows]
            sids = sample_ids_all[idx]
            if not args.limit_rows:
                if _sha_ids(sids) != entry["sha256_sample_ids"]:
                    raise AssertionError(f"frozen selection hash mismatch {vname}/{cname}")
            if not np.array_equal(sids, np.asarray(entry["sample_ids"][: len(idx)], "U128")):
                raise AssertionError(f"sample_id/positional_idx mismatch {vname}/{cname}")
            cid = int(mapping.name_to_id[cname])
            raw = raw_all_t[torch.as_tensor(idx, device=device)]
            with torch.no_grad():
                pred = victim((raw - center) / scale).argmax(1).cpu().numpy()
            if not (np.all(y[idx] == cid) and np.all(pred == cid)):
                raise AssertionError(f"selected rows not clean-correct for {vname}/{cname}")
            rows[(vname, cname)] = {
                "idx": idx, "sids": sids, "raw": raw, "cid": cid,
                "src": {"Src IP": meta.iloc[idx]["Src IP"].astype(str).to_numpy(),
                        "Dst IP": meta.iloc[idx]["Dst IP"].astype(str).to_numpy()},
            }

    config = {
        "dataset": dataset, "mode": MODE, "methods": methods, "method_configs": cfgs,
        "eval_budget_per_flow": args.eval_budget,
        "evaluation_unit": "one victim forward pass on one flow (realized quantized flow or "
                           "continuous surrogate); gradient steps additionally cost one backward",
        "success": "victim argmax == Benign on the realized flow AND validator_v2 hybrid_valid",
        "incumbent": "success > failure; successes by normalized cost p/p_hi + delay/delay_hi "
                     "(then margin); failures by margin max(non-Benign) - Benign",
        "budgets": {b: BUDGET_LABEL[b] for b in budgets}, "classes": classes,
        "victims": {v: {k: i[k] for k in ("checkpoint", "arch", "train_seed", "attack_seeds")}
                    for v, i in victim_info.items()},
        "split": split, "selection_from": str(sel_path) if sel_path else f"fresh ({split})",
        "limit_rows": args.limit_rows,
        "calibration": str(calibration_path), "calibration_fit_split": calibration["fit_split"],
        "benign_id": BENIGN_ID, "torch": torch.__version__, "device": device,
        "python": sys.version.split()[0],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "selection.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    cells_path = out / "cells.json"
    prior = {}
    if args.resume and cells_path.exists():
        prior = {(c["victim"], c["class"], c["budget"], c["seed"], c["method"]): c
                 for c in json.loads(cells_path.read_text(encoding="utf-8"))}
    cells = []
    t_start = time.time()
    for vname in victims:
        info = victim_info[vname]
        victim = info["victim"]
        for cname in classes:
            R = rows[(vname, cname)]
            raw, cid, n = R["raw"], R["cid"], R["raw"].shape[0]
            labels = np.full(n, cid, dtype=np.int64)
            caps = model.infer_capabilities(raw)
            for budget in budgets:
                ccfg = (unbounded_calibration(calibration, cname) if budget == "unbounded"
                        else class_calibration(calibration, cname, budget))
                bounds = _apply_primitive_mode(
                    model.per_flow_bounds(raw, ccfg.bounds_config(), capabilities=caps), MODE)
                movable = ((bounds["p"] >= 1.0) | (bounds["delay"] >= 1.0)).cpu().numpy()
                for seed in info["attack_seeds"]:
                    for method in methods:
                        key = (vname, cname, budget, seed, method)
                        npz = art / f"{vname}__{cname}__{BUDGET_LABEL[budget]}__{method}__seed{seed}.npz"
                        if key in prior and npz.exists():
                            cells.append(prior[key])
                            continue
                        deterministic_runtime(seed)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        res = run_method(method, cfgs[method],
                                         (model, victim, raw, center, scale, bounds, caps),
                                         seed=seed, gate=gate, budget=args.eval_budget)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - t0

                        adv = res.adversarial_raw
                        if not bool(torch.isfinite(adv).all()):
                            raise FloatingPointError(f"non-finite adversarial flow {key}")
                        masks, _, _, adv_pred = evaluate_cell(
                            model, realizability, victim, raw, adv, center, scale, cid,
                            groups_idx)
                        semantic = semantic_validator.evaluate(
                            raw, adv, res.requested, res.projected, bounds, class_name=cname,
                            budget=ccfg.budget, original_labels=labels,
                            adversarial_labels=labels.copy(), original_metadata=R["src"],
                            adversarial_metadata=R["src"])
                        feat_cost = _decompose_cost(adv, raw, scale, groups_idx)
                        np_ = lambda t: t.detach().cpu().numpy()  # noqa: E731
                        targeted = np_(adv_pred == BENIGN_ID)
                        valid = np_(masks["domain_valid"]).astype(bool)
                        realizable = np_(masks["primitive_transform_consistent"]).astype(bool)
                        prim_feasible = semantic.primitive_feasible & realizable
                        sem_pass = semantic.semantic_status == SemanticStatus.PASS.value
                        success = np_(res.success).astype(bool)
                        final_success = targeted & valid
                        consistent = final_success == success
                        first_t = np_(res.first_targeted_evaluation)
                        failure = np.where(
                            final_success, "success",
                            np.where(~movable, "no_headroom",
                                     np.where(first_t > 0, "invalid", "exhausted")))
                        p = np_(res.projected["p"]); d = np_(res.projected["delay"])
                        shape = np_(res.projected["shape"])
                        total = np_(res.total_evaluations)
                        np.savez_compressed(
                            npz, sample_id=R["sids"], positional_idx=R["idx"],
                            success=success, final_success=final_success,
                            final_consistent=consistent, targeted=targeted, valid=valid,
                            realizable=realizable, primitive_feasible=prim_feasible,
                            semantic_pass=sem_pass, movable=movable, failure=failure,
                            p=p, delay=d, shape=shape,
                            p_hi=np_(bounds["p"]), delay_hi=np_(bounds["delay"]),
                            normalized_cost=np_(res.normalized_cost),
                            target_margin=np_(res.target_margin),
                            candidate_source=np.asarray(
                                [CANDIDATE_NAMES[int(v)] for v in np_(res.candidate_source)]),
                            realized_evaluations=np_(res.realized_evaluations),
                            surrogate_evaluations=np_(res.surrogate_evaluations),
                            backward_evaluations=np_(res.backward_evaluations),
                            total_evaluations=total,
                            first_success_evaluation=np_(res.first_success_evaluation),
                            first_targeted_evaluation=first_t,
                            first_success_phase=np_(res.first_success_phase),
                            relative_duration_change=semantic.costs.relative_duration_change,
                            added_bytes=semantic.costs.added_byte_quantity,
                            padding_percent_fwd_mean=semantic.costs.padding_percent_of_forward_mean,
                            feature_cost_total=np_(feat_cost["total"]),
                            elapsed_seconds=np.float64(elapsed), iterations=res.iterations,
                            restarts=res.restarts, method=method, victim=vname,
                            attack_class=cname, budget=budget, seed=seed, dataset=dataset,
                        )
                        s = final_success
                        cost_mean, cost_med = _stats(np_(res.normalized_cost)[s])
                        first_s = np_(res.first_success_evaluation)
                        cell = {
                            "dataset": dataset, "victim": vname, "arch": info["arch"],
                            "class": cname, "budget": budget, "budget_label": BUDGET_LABEL[budget],
                            "seed": seed, "method": method, "n": n,
                            "n_movable": int(movable.sum()),
                            "successes": int(s.sum()), "asr_valid": float(s.mean()),
                            "raw_targeted": int(targeted.sum()),
                            "asr_raw": float(targeted.mean()),
                            "asr_prim_feasible": float((s & prim_feasible).mean()),
                            "sp_asr": float((s & prim_feasible & sem_pass).mean()),
                            "fail_invalid": int((failure == "invalid").sum()),
                            "fail_exhausted": int((failure == "exhausted").sum()),
                            "fail_no_headroom": int((failure == "no_headroom").sum()),
                            "incumbent_final_mismatch": int((~consistent).sum()),
                            "cost_mean": cost_mean, "cost_median": cost_med,
                            "p_mean": _stats(p[s])[0], "p_median": _stats(p[s])[1],
                            "delay_mean": _stats(d[s])[0], "delay_median": _stats(d[s])[1],
                            "shape_mean": _stats(shape[s])[0],
                            "shape_median": _stats(shape[s])[1],
                            "rel_duration_median": _stats(
                                semantic.costs.relative_duration_change[s])[1],
                            "frac_success_padding": float((p[s] > 0).mean()) if s.any() else float("nan"),
                            "frac_success_timing": float((d[s] > 0).mean()) if s.any() else float("nan"),
                            "evals_mean": float(total.mean()), "evals_median": float(np.median(total)),
                            "evals_max": int(total.max()),
                            "realized_mean": float(np_(res.realized_evaluations).mean()),
                            "surrogate_mean": float(np_(res.surrogate_evaluations).mean()),
                            "backward_mean": float(np_(res.backward_evaluations).mean()),
                            "first_success_median": _stats(first_s[first_s > 0])[1],
                            "iterations": res.iterations, "restarts": res.restarts,
                            "elapsed_seconds": elapsed,
                        }
                        cells.append(cell)
                        print(f"[{time.time() - t_start:7.0f}s] {vname}/{cname}/"
                              f"{BUDGET_LABEL[budget]}/s{seed}/{method}: valid ASR "
                              f"{cell['asr_valid']:.4f} raw {cell['asr_raw']:.4f} "
                              f"evals {cell['evals_mean']:.1f} t {elapsed:.1f}s "
                              f"mismatch {cell['incumbent_final_mismatch']}", flush=True)
                        cells_path.write_text(json.dumps(cells, indent=2), encoding="utf-8")
    cells_path.write_text(json.dumps(cells, indent=2), encoding="utf-8")
    print(f"[done] {len(cells)} cells in {time.time() - t_start:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
