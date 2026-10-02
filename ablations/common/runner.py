"""Shared cell runner for the PrimAttack ablations.

Protocol (identical to the FINAL suite's Exp A Hybrid cells, amendment A5, unless an ablation
changes it): both datasets; one victim per architecture (2017 canonical checkpoints, 2018
``*-s42``); the canonical frozen 800 clean-correct test flows per (dataset, victim, class) from
``FINAL_OUTPUTS/runs/<dataset>/baselines_untargeted/selection.json`` (sha256-verified); attack
seeds 42/2024/2026; untargeted objective (``objective="targeted"``: realized flow classified
Benign, the Exp B / A5 targeted arm); joint mode; capability-aware primitives; train-fit
budget calibration at p75 (``maximum-evaluated``) and envelope-only ``unbounded``; per-flow
evaluation budget 256; Hybrid Search with ``PRIM_ARGS``; validator_v2 ``hybrid_valid`` in the
search success predicate.

Whatever a condition changes, every realized flow is scored afterwards by the same full
validator_v2 (all four layers, source-conditioned), and the per-row npz records each layer.

``recompute_mode="direct_only"`` (ablation P2): the search sees only φ's direct writes
(``phi_mapping.DirectOnlyPrimitiveModel``); the primitives it returns are realized again through
canonical φ, and that full-φ flow is the one scored, stored as ``adv_raw`` and counted. The
reduced flow the search believed in is stored next to it (``reduced_*``).

Layout written per experiment: ``<exp>/results/<dataset>/{config.json, cells.json,
artifacts/<victim>__<class>__<budget>__<condition>__seed<s>.npz}``. Finished cells are skipped
on re-runs (resume by default).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ablations.common.extra_rules import EXTRA_RULES, extra_rule_masks  # noqa: E402
from ablations.common.hybrid import HybridConfig, optimize_hybrid_ablation  # noqa: E402
from ablations.common.phi_mapping import (  # noqa: E402
    DirectOnlyPrimitiveModel, derive_phi_mapping, realize_full_phi,
)
from attack.flow_semantics import FlowSemanticValidator, SemanticStatus  # noqa: E402
from attack.primattack_budget import (  # noqa: E402
    class_calibration, load_calibration, unbounded_calibration,
)
from attack.primitive_optimizer import (  # noqa: E402
    CANDIDATE_NAMES, AttackObjective, row_primitive_modes,
)
from attack.realizability.base import PrimitiveCapabilities  # noqa: E402
from attack.realizability.cicids2017 import (  # noqa: E402
    CICIDS2017PrimitiveModel, primattack_joint_feature_mask,
)
from attack.realizability.validator import RealizabilityValidator  # noqa: E402
from datasets import get_adapter  # noqa: E402
from experiments.provenance import deterministic_runtime  # noqa: E402
from run_full_adversarial_eval import BUDGET_LABEL, DATASET_DEFAULTS, file_sha256, victim_checkpoint  # noqa: E402
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402
from validation.attack_interface import structural_masks  # noqa: E402

FINAL_RUNS = REPO_ROOT / "FINAL_OUTPUTS" / "runs"
DATASETS = {
    "cicids2017_distrinet": {"cli": "cicids2017", "victims": ("mlp", "cnn", "ft_transformer")},
    "cicids2018_distrinet": {"cli": "cicids2018",
                             "victims": ("mlp-s42", "cnn-s42", "ft_transformer-s42")},
}
CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")
SEEDS = (42, 2024, 2026)
BUDGETS = ("maximum-evaluated", "unbounded")
LAYERS = ("schema", "extractor", "protocol", "mined")
ABLATED_CAPABILITY = "CAPABILITY_INFERENCE_ABLATED"
RECOMPUTE_MODES = ("full_phi", "direct_only")


@dataclass(frozen=True)
class Condition:
    """One ablation arm. Defaults = the reference Hybrid configuration."""

    name: str
    description: str
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    gate_layers: tuple[str, ...] = LAYERS     # validator_v2 layers ANDed in the search gate
    capability_aware: bool = True             # source-dependent capability mask M(x)
    extra_rules: tuple[str, ...] = ()         # candidate rules (extra_rules.py) toggled on
    recompute_mode: str = "full_phi"          # "direct_only" (P2): search sees φ's direct writes only

    def __post_init__(self) -> None:
        if not self.name or "__" in self.name:
            raise ValueError("condition names must be non-empty and must not contain '__'")
        if set(self.gate_layers) - set(LAYERS) or not self.gate_layers:
            raise ValueError(f"gate_layers must be a non-empty subset of {LAYERS}")
        if set(self.extra_rules) - set(EXTRA_RULES):
            raise ValueError(f"extra_rules must be a subset of {sorted(EXTRA_RULES)}")
        if self.recompute_mode not in RECOMPUTE_MODES:
            raise ValueError(f"recompute_mode must be one of {RECOMPUTE_MODES}")
        if self.recompute_mode == "direct_only" and self.hybrid.validity_in_search:
            # validator_v2 would judge the reduced flow, which breaks φ's identities by design.
            raise ValueError("direct_only needs HybridConfig(validity_in_search=False)")

    def to_dict(self) -> dict:
        d = {"name": self.name, "description": self.description,
             "hybrid": self.hybrid.to_dict(), "gate_layers": list(self.gate_layers),
             "capability_aware": self.capability_aware}
        # Non-default options only: keeps earlier result configs comparable.
        if self.extra_rules:
            d["extra_rules"] = list(self.extra_rules)
        if self.recompute_mode != "full_phi":
            d["recompute_mode"] = self.recompute_mode
        return d


REFERENCE = Condition("reference", "FINAL-suite Hybrid Search (all components, full "
                      "validator_v2 gate, capability-aware)")
REFERENCE_DIR = REPO_ROOT / "ablations" / "reference"


def artifact_name(victim: str, cname: str, budget: str, condition: str, seed: int) -> str:
    return f"{victim}__{cname}__{BUDGET_LABEL[budget]}__{condition}__seed{seed}.npz"


def _sha_ids(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, ids.tolist())).encode("utf-8")).hexdigest()


def layer_masks(adv_raw: np.ndarray, source_raw: np.ndarray, dataset: str) -> dict[str, np.ndarray]:
    m = structural_masks(adv_raw, dataset=dataset, source_raw=source_raw)
    return {layer: np.asarray(m[f"{layer}_valid"], dtype=bool) for layer in LAYERS}


def layer_gate(dataset: str, layers: tuple[str, ...], extra_rules: tuple[str, ...] = (),
               index: dict[str, int] | None = None):
    """validator_v2 gate restricted to ``layers`` (all four = ``hybrid_valid``), ANDed with the
    toggled-on candidate ``extra_rules``."""

    def gate(adv_raw: torch.Tensor, source_raw: torch.Tensor) -> torch.Tensor:
        adv = adv_raw.detach().cpu().numpy()
        m = layer_masks(adv, source_raw.detach().cpu().numpy(), dataset)
        valid = np.logical_and.reduce([m[layer] for layer in layers])
        for rule in extra_rules:
            valid &= EXTRA_RULES[rule](adv, index)
        return torch.as_tensor(valid, dtype=torch.bool, device=adv_raw.device)

    return gate


def all_capable(n: int, device) -> PrimitiveCapabilities:
    """``M(x) = 1``: every flow is treated as admitting padding and timing."""
    ones = torch.ones(n, dtype=torch.bool, device=device)
    return PrimitiveCapabilities(pad_allowed=ones, timing_allowed=ones.clone(),
                                 pad_reason=[ABLATED_CAPABILITY] * n,
                                 timing_reason=[ABLATED_CAPABILITY] * n)


class BenignAnchors:
    """Nearest Benign TRAIN flow of each source flow in ``asinh((x - center) / scale)`` space."""

    def __init__(self, processed: Path, center: torch.Tensor, scale: torch.Tensor, device):
        X = np.load(processed / "X_train_pristine.npy", mmap_mode="r")
        y = np.load(processed / "y_train_cat.npy")
        benign = np.flatnonzero(y == 0)
        raw = torch.tensor(np.ascontiguousarray(X[benign]), dtype=torch.float32, device=device)
        self.center, self.scale = center, scale
        self.benign = torch.asinh((raw - center) / scale)
        self.benign_sq = self.benign.pow(2).sum(1)
        self.n_benign = int(len(benign))

    @torch.no_grad()
    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        z = torch.asinh((raw - self.center) / self.scale)
        best_d = torch.full((z.shape[0],), torch.inf, device=z.device)
        best_i = torch.zeros(z.shape[0], dtype=torch.int64, device=z.device)
        z_sq = z.pow(2).sum(1, keepdim=True)
        for start in range(0, self.benign.shape[0], 65536):
            b = self.benign[start:start + 65536]
            d = z_sq - 2.0 * z @ b.T + self.benign_sq[start:start + 65536][None, :]
            v, i = d.min(1)
            better = v < best_d
            best_d = torch.where(better, v, best_d)
            best_i = torch.where(better, i + start, best_i)
        return self.benign[best_i]


def _stats(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return float("nan"), float("nan")
    return float(values.mean()), float(np.median(values))


OBJECTIVES = ("untargeted", "targeted")


def run_conditions(
    results_dir: Path,
    conditions: list[Condition],
    *,
    datasets: list[str] | None = None,
    budgets: tuple[str, ...] = BUDGETS,
    seeds: tuple[int, ...] = SEEDS,
    victims: list[str] | None = None,
    classes: tuple[str, ...] = CLASSES,
    device: str = "cuda",
    limit_rows: int | None = None,
    objective: str = "untargeted",
) -> None:
    """Run every (dataset, victim, class, budget, condition, seed) cell into ``results_dir``.

    ``objective``: ``"untargeted"`` (leave the source class) or ``"targeted"`` (-> Benign).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    names = [c.name for c in conditions]
    if len(set(names)) != len(names):
        raise ValueError("duplicate condition names")
    for dataset in datasets or list(DATASETS):
        _run_dataset(results_dir / dataset, dataset, conditions, budgets=budgets,
                     seeds=seeds, victims=victims, classes=classes, device=device,
                     limit_rows=limit_rows, objective_kind=objective)


def _run_dataset(out: Path, dataset: str, conditions, *, budgets, seeds, victims, classes,
                 device, limit_rows, objective_kind) -> None:
    spec = DATASETS[dataset]
    adapter = get_adapter(spec["cli"])
    if adapter.name != dataset:
        raise AssertionError(f"adapter {adapter.name} != {dataset}")
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)
    model = CICIDS2017PrimitiveModel(manifest)
    support_mask = primattack_joint_feature_mask(manifest).to(device)
    reduced_model = (DirectOnlyPrimitiveModel(model, derive_phi_mapping(manifest))
                     if any(c.recompute_mode == "direct_only" for c in conditions) else None)
    realizability = RealizabilityValidator(model)
    calibration_path = DATASET_DEFAULTS[dataset]["calibration"]
    calibration = load_calibration(calibration_path)
    if calibration.get("dataset") != dataset or calibration.get("fit_split") != "train":
        raise ValueError("calibration must be the train-fit artifact of this dataset")
    semantic_validator = FlowSemanticValidator(model, calibration)
    anchors = (BenignAnchors(adapter._processed, center, scale, device)
               if any(c.hybrid.loss == "mimicry" for c in conditions) else None)

    processed = adapter._processed
    raw_all = np.load(processed / "X_test_pristine.npy", mmap_mode="r")
    y = np.load(processed / "y_test_cat.npy").astype(np.int64)
    meta = pd.read_parquet(processed / "test.parquet", columns=["sample_id", "Src IP", "Dst IP"])
    sample_ids_all = meta["sample_id"].astype(str).to_numpy(dtype="U128")
    sel_path = FINAL_RUNS / dataset / "baselines_untargeted" / "selection.json"
    frozen = json.loads(sel_path.read_text(encoding="utf-8"))

    victim_names = list(victims or spec["victims"])
    config_path = out / "config.json"
    known = {}
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("limit_rows") != limit_rows:
            raise ValueError(f"{out}: existing results use limit_rows="
                             f"{previous.get('limit_rows')}; refusing to mix")
        if previous.get("objective") != objective_kind:
            raise ValueError(f"{out}: existing results use objective="
                             f"{previous.get('objective')}; refusing to mix")
        known = {c["name"]: c for c in previous["conditions"]}
    for cond in conditions:
        if cond.name in known and known[cond.name] != cond.to_dict():
            raise ValueError(f"{out}: condition {cond.name!r} was run with a different "
                             "definition; refusing to mix")
        known[cond.name] = cond.to_dict()
    target_text = ("victim argmax == Benign" if objective_kind == "targeted"
                   else "victim argmax != true source class")
    config = {
        "dataset": dataset, "objective": objective_kind, "mode": "joint",
        "success": f"{target_text} on the realized flow AND the condition's validator_v2 "
                   "search gate",
        "final_validity": "validator_v2 hybrid_valid = schema & extractor & protocol & mined, "
                          "given the source flow (identical for every condition)",
        "conditions": list(known.values()),
        "budgets": {b: BUDGET_LABEL[b] for b in budgets}, "classes": list(classes),
        "seeds": list(seeds), "victims": victim_names,
        "selection_from": sel_path.relative_to(REPO_ROOT).as_posix(),
        "selection_sha256": file_sha256(sel_path), "limit_rows": limit_rows,
        "calibration": Path(calibration_path).relative_to(REPO_ROOT).as_posix(),
        "calibration_sha256": file_sha256(Path(calibration_path)),
        "benign_anchor_pool": (f"all {anchors.n_benign} Benign TRAIN flows" if anchors else None),
        "torch": torch.__version__, "device": device, "python": sys.version.split()[0],
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cells_path = out / "cells.json"
    prior = {}
    if cells_path.exists():
        prior = {(c["victim"], c["class"], c["budget"], c["condition"], c["seed"]): c
                 for c in json.loads(cells_path.read_text(encoding="utf-8"))}
    t_start = time.time()
    for vname in victim_names:
        ckpt, arch, _ = victim_checkpoint(dataset, vname)
        victim = load_category_victim(ckpt, adapter=adapter, expected_model_type=arch,
                                      device=device)
        ckpt_sha = file_sha256(ckpt)
        for cname in classes:
            entry = frozen[vname][cname]
            idx = np.asarray(entry["positional_idx"], dtype=np.int64)
            if limit_rows:
                idx = idx[:limit_rows]
            sids = sample_ids_all[idx]
            if not limit_rows and _sha_ids(sids) != entry["sha256_sample_ids"]:
                raise AssertionError(f"frozen selection hash mismatch {vname}/{cname}")
            if not np.array_equal(sids, np.asarray(entry["sample_ids"][: len(idx)], "U128")):
                raise AssertionError(f"sample_id/positional_idx mismatch {vname}/{cname}")
            cid = int(mapping.name_to_id[cname])
            raw = torch.tensor(np.ascontiguousarray(raw_all[idx]), dtype=torch.float32,
                               device=device)
            with torch.no_grad():
                clean_pred = victim((raw - center) / scale).argmax(1).cpu().numpy()
            if not (np.all(y[idx] == cid) and np.all(clean_pred == cid)):
                raise AssertionError(f"selected rows not clean-correct for {vname}/{cname}")
            n = raw.shape[0]
            labels = np.full(n, cid, dtype=np.int64)
            src_meta = {"Src IP": meta.iloc[idx]["Src IP"].astype(str).to_numpy(),
                        "Dst IP": meta.iloc[idx]["Dst IP"].astype(str).to_numpy()}
            caps_true = model.infer_capabilities(raw)
            pad_cap = caps_true.pad_allowed.cpu().numpy()
            timing_cap = caps_true.timing_allowed.cpu().numpy()
            fwd_min_zero = (raw[:, model.i["Fwd Packet Length Min"]] == 0).cpu().numpy()
            raw_np = raw.cpu().numpy()
            objective = (AttackObjective("targeted", int(mapping.name_to_id["Benign"]))
                         if objective_kind == "targeted" else AttackObjective("untargeted", cid))
            anchor = anchors(raw) if anchors is not None else None
            for budget in budgets:
                ccfg = (unbounded_calibration(calibration, cname) if budget == "unbounded"
                        else class_calibration(calibration, cname, budget))
                for cond in conditions:
                    if cond.capability_aware:
                        caps = caps_true
                    else:
                        caps = all_capable(n, device)
                    bounds = model.per_flow_bounds(raw, ccfg.bounds_config(), capabilities=caps)
                    movable = ((bounds["p"] >= 1.0) | (bounds["delay"] >= 1.0)).cpu().numpy()
                    row_modes = np.asarray(row_primitive_modes(bounds))
                    gate = layer_gate(dataset, cond.gate_layers, cond.extra_rules, model.i)
                    for seed in seeds:
                        key = (vname, cname, budget, cond.name, seed)
                        npz = art / artifact_name(vname, cname, budget, cond.name, seed)
                        if key in prior and npz.exists():
                            continue
                        deterministic_runtime(seed)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        direct_only = cond.recompute_mode == "direct_only"
                        res, diag = optimize_hybrid_ablation(
                            reduced_model if direct_only else model, victim, raw, center, scale,
                            bounds, caps, config=cond.hybrid, seed=seed, validity_fn=gate,
                            objective=objective, anchor=anchor,
                            nonfinite_gradients="raise" if cond.capability_aware else "zero")
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - t0
                        if direct_only:
                            # Outcome measurement through canonical φ (not a search query).
                            res, reduced = realize_full_phi(
                                reduced_model, res, victim=victim, raw=raw, center=center,
                                scale=scale, bounds=bounds, caps=caps, objective=objective)
                            diag = dataclasses.replace(diag, extra={**diag.extra, **reduced})
                        cell = _evaluate_and_save(
                            npz, cond, res, diag, elapsed, model=model, victim=victim,
                            realizability=realizability, semantic_validator=semantic_validator,
                            raw=raw, raw_np=raw_np, center=center, scale=scale,
                            objective=objective,
                            labels=labels, src_meta=src_meta, bounds=bounds, ccfg=ccfg,
                            support_mask=support_mask, pad_cap=pad_cap, timing_cap=timing_cap,
                            caps_true=caps_true, fwd_min_zero=fwd_min_zero, movable=movable,
                            row_modes=row_modes, dataset=dataset, vname=vname, arch=arch,
                            cname=cname, budget=budget, seed=seed, sids=sids, idx=idx,
                            ckpt_sha=ckpt_sha)
                        prior[key] = cell
                        cells_path.write_text(json.dumps(list(prior.values()), indent=2),
                                              encoding="utf-8")
                        print(f"[{time.time() - t_start:7.0f}s] {dataset}/{vname}/{cname}/"
                              f"{BUDGET_LABEL[budget]}/{cond.name}/s{seed}: valid "
                              f"{cell['asr_valid']:.4f} gate {cell['asr_gate']:.4f} raw "
                              f"{cell['asr_raw']:.4f} t {elapsed:.1f}s", flush=True)
    cells_path.write_text(json.dumps(list(prior.values()), indent=2), encoding="utf-8")


def _evaluate_and_save(npz, cond, res, diag, elapsed, *, model, victim, realizability,
                       semantic_validator, raw, raw_np, center, scale, objective, labels,
                       src_meta,
                       bounds, ccfg, support_mask, pad_cap, timing_cap, caps_true,
                       fwd_min_zero, movable, row_modes, dataset, vname, arch, cname, budget,
                       seed, sids, idx, ckpt_sha) -> dict:
    np_ = lambda t: t.detach().cpu().numpy()  # noqa: E731
    adv = res.adversarial_raw
    if not bool(torch.isfinite(adv).all()):
        raise FloatingPointError(f"non-finite adversarial flow {npz.name}")
    changed = adv != raw
    outside = np_(changed[:, ~support_mask].sum(1))
    if (outside != 0).any():
        raise AssertionError(f"PrimAttack changed a feature outside its joint support {npz.name}")
    with torch.no_grad():
        adv_logits = victim((adv - center) / scale)
    adv_pred = np_(adv_logits.argmax(1)).astype(np.int64)
    hit = np_(objective.hit(adv_logits)).astype(bool)
    adv_np = np_(adv)
    layers = layer_masks(adv_np, raw_np, dataset)
    full_valid = np.logical_and.reduce([layers[k] for k in LAYERS])
    extra = extra_rule_masks(adv_np, model.i)
    extended_valid = np.logical_and.reduce([full_valid, *extra.values()])
    gate_valid = np.logical_and.reduce(
        [layers[k] for k in cond.gate_layers] + [extra[r] for r in cond.extra_rules])
    realizable = torch.as_tensor(realizability.validate(adv, raw).valid).cpu().numpy().astype(bool)
    semantic = semantic_validator.evaluate(
        raw, adv, res.requested, res.projected, bounds, class_name=cname, budget=ccfg.budget,
        original_labels=labels, adversarial_labels=labels.copy(),
        original_metadata=src_meta, adversarial_metadata=src_meta)
    prim_feasible = semantic.primitive_feasible & realizable
    sem_pass = semantic.semantic_status == SemanticStatus.PASS.value

    valid_success = hit & full_valid
    gate_success = hit & gate_valid
    search_success = np_(res.success).astype(bool)
    expected = hit & (gate_valid if cond.hybrid.validity_in_search else True)
    mismatch = int((search_success != expected).sum())
    p, d, shape = (np_(res.projected[k]) for k in ("p", "delay", "shape"))
    pad_violation = (p > 0) & ~pad_cap
    timing_violation = (d > 0) & ~timing_cap
    if cond.capability_aware and (pad_violation.any() or timing_violation.any()):
        raise AssertionError(f"capability-aware PrimAttack used a forbidden primitive {npz.name}")
    filled = fwd_min_zero & (adv_np[:, model.i["Fwd Packet Length Min"]] > 0)
    first_s = np_(res.first_success_evaluation)
    total = np_(res.total_evaluations)
    grad_steps, zero_steps = np_(diag.gradient_steps), np_(diag.zero_gradient_steps)
    nonfinite_steps = np_(diag.nonfinite_gradient_steps)
    arm_extra = {k: np_(v) for k, v in diag.extra.items()}
    cont_cell = {}
    if "continuous_adv_raw" in arm_extra:
        # P1: the same validator_v2 on the unrealized continuous flow (diagnostic only).
        cont_layers = layer_masks(arm_extra["continuous_adv_raw"], raw_np, dataset)
        cont_valid = np.logical_and.reduce([cont_layers[k] for k in LAYERS])
        arm_extra.update({f"continuous_{k}_valid": v for k, v in cont_layers.items()})
        arm_extra["continuous_validator_pass"] = cont_valid
        cont_hit = arm_extra["continuous_success"].astype(bool)
        cont_cell = {
            "asr_continuous": float(cont_hit.mean()),
            "continuous_hit_realized_miss": int((cont_hit & ~hit).sum()),
            "continuous_miss_realized_hit": int((~cont_hit & hit).sum()),
            "prediction_changed_by_realization": int(
                (arm_extra["continuous_pred"] != adv_pred).sum()),
            "continuous_validator_pass_rate": float(cont_valid.mean()),
        }
    reduced_cell = {}
    if "reduced_adv_raw" in arm_extra:
        # P2: validator_v2 on the direct-only flow the search believed in (diagnostic only).
        red_layers = layer_masks(arm_extra["reduced_adv_raw"], raw_np, dataset)
        red_valid = np.logical_and.reduce([red_layers[k] for k in LAYERS])
        arm_extra.update({f"reduced_{k}_valid": v for k, v in red_layers.items()})
        arm_extra["reduced_validator_pass"] = red_valid
        red_hit = arm_extra["reduced_success"].astype(bool)
        pred_changed = arm_extra["reduced_pred"] != adv_pred
        reduced_cell = {
            "asr_reduced": float(red_hit.mean()), "reduced_successes": int(red_hit.sum()),
            "reduced_hit_full_miss": int((red_hit & ~hit).sum()),
            "reduced_hit_full_invalid": int((red_hit & hit & ~full_valid).sum()),
            "reduced_hit_not_valid_success": int((red_hit & ~valid_success).sum()),
            "reduced_miss_full_hit": int((~red_hit & hit).sum()),
            "prediction_changed_by_full_phi": int(pred_changed.sum()),
            "prediction_changed_among_reduced_hits": int((pred_changed & red_hit).sum()),
            "reduced_validator_pass_rate": float(red_valid.mean()),
        }

    np.savez_compressed(
        npz, sample_id=sids, positional_idx=idx, true_class=labels, adv_pred=adv_pred,
        raw_success=hit, valid_success=valid_success, gate_success=gate_success,
        search_success=search_success, validator_pass=full_valid, gate_pass=gate_valid,
        extended_valid_success=hit & extended_valid, extended_validator_pass=extended_valid,
        **{f"rule_{k}_valid": v for k, v in extra.items()},
        schema_valid=layers["schema"], extractor_valid=layers["extractor"],
        protocol_valid=layers["protocol"], mined_valid=layers["mined"],
        realizable=realizable, primitive_feasible=prim_feasible, semantic_pass=sem_pass,
        movable=movable, row_primitive_mode=row_modes,
        p=p, delay=d, shape=shape, p_hi=np_(bounds["p"]), delay_hi=np_(bounds["delay"]),
        requested_p=np_(res.requested["p"]), requested_delay=np_(res.requested["delay"]),
        requested_shape=np_(res.requested["shape"]),
        normalized_cost=np_(res.normalized_cost), objective_margin=np_(res.objective_margin),
        adv_raw=adv_np.astype(np.float32),
        candidate_source=np.asarray([CANDIDATE_NAMES[int(v)] for v in np_(res.candidate_source)]),
        realized_evaluations=np_(res.realized_evaluations),
        surrogate_evaluations=np_(res.surrogate_evaluations),
        backward_evaluations=np_(res.backward_evaluations), total_evaluations=total,
        first_success_evaluation=first_s,
        first_objective_hit_evaluation=np_(res.first_objective_hit_evaluation),
        first_success_phase=np_(res.first_success_phase),
        gradient_steps=grad_steps, zero_gradient_steps=zero_steps,
        nonfinite_gradient_steps=nonfinite_steps,
        relative_duration_change=semantic.costs.relative_duration_change,
        added_bytes=semantic.costs.added_byte_quantity,
        pad_allowed=pad_cap, timing_allowed=timing_cap,
        pad_reason=np.asarray(caps_true.pad_reason),
        timing_reason=np.asarray(caps_true.timing_reason),
        pad_capability_violation=pad_violation, timing_capability_violation=timing_violation,
        source_fwd_min_zero=fwd_min_zero, empty_fwd_packet_filled=filled,
        checkpoint_sha256=ckpt_sha, condition=cond.name, victim=vname, attack_class=cname,
        budget=budget, seed=seed, dataset=dataset, elapsed_seconds=np.float64(elapsed),
        **arm_extra,
    )
    s = valid_success
    n = len(hit)
    return {
        "dataset": dataset, "victim": vname, "arch": arch, "class": cname, "budget": budget,
        "budget_label": BUDGET_LABEL[budget], "seed": seed, "condition": cond.name, "n": n,
        "asr_valid": float(s.mean()), "valid_successes": int(s.sum()),
        "asr_gate": float(gate_success.mean()), "gate_successes": int(gate_success.sum()),
        "asr_extended_valid": float((hit & extended_valid).mean()),
        **{f"valid_success_failing_{k}": int((s & ~v).sum()) for k, v in extra.items()},
        "asr_raw": float(hit.mean()), "raw_successes": int(hit.sum()),
        "validator_pass_rate": float(full_valid.mean()),
        **{f"{k}_pass_rate": float(v.mean()) for k, v in layers.items()},
        "asr_prim_feasible": float((s & prim_feasible).mean()),
        "sp_asr": float((s & prim_feasible & sem_pass).mean()),
        "n_movable": int(movable.sum()),
        "n_pad_capable": int(pad_cap.sum()), "n_timing_capable": int(timing_cap.sum()),
        "valid_success_pad_violation": int((s & pad_violation).sum()),
        "valid_success_timing_violation": int((s & timing_violation).sum()),
        "valid_success_empty_packet_filled": int((s & filled).sum()),
        "search_final_mismatch": mismatch,
        "cost_median": _stats(np_(res.normalized_cost)[s])[1],
        "p_median": _stats(p[s])[1], "delay_median": _stats(d[s])[1],
        "shape_median": _stats(shape[s])[1],
        "rel_duration_median": _stats(semantic.costs.relative_duration_change[s])[1],
        "frac_success_padding": float((p[s] > 0).mean()) if s.any() else float("nan"),
        "frac_success_timing": float((d[s] > 0).mean()) if s.any() else float("nan"),
        "evals_mean": float(total.mean()), "evals_max": int(total.max()),
        "first_success_median": _stats(first_s[first_s > 0])[1],
        "gradient_steps": int(grad_steps.sum()), "zero_gradient_steps": int(zero_steps.sum()),
        "nonfinite_gradient_steps": int(nonfinite_steps.sum()),
        "rows_with_nonfinite_gradient": int((nonfinite_steps > 0).sum()),
        "iterations": res.iterations, "restarts": res.restarts, "elapsed_seconds": elapsed,
        **cont_cell,
        **reduced_cell,
    }
