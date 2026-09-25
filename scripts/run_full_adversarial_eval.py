"""Full paired adversarial evaluation (all attack families) on a CICFlowMeter DistriNet dataset.

Datasets: ``--dataset cicids2017`` (CICIDS2017-DistriNet; victims = the single checkpoint
per architecture) or ``--dataset cicids2018`` (CSE-CIC-IDS-2018-DistriNet; victims = the
three training-seed replicates per architecture, ``<arch>-s<seed>``).

Pairing / eligibility / denominator by construction:

* ELIGIBILITY = clean-correct (victim predicts the true malicious class on the CLEAN flow).
  Selected ONCE per (victim, class) from the FULL test split, then at most N rows: by
  default a seeded uniform sample (``--selection random``; the test split is ordered by
  source label/time, so a head slice would cover a single attack variant), or the legacy
  head slice in test order (``--selection head``). Row IDs (sample_id) + positional indices
  + source-label mix + a sha256 are saved to selection.json. EVERY attack/seed attacks
  EXACTLY these rows in this order.
* DENOMINATOR = that clean-correct eligible set (identical across attacks within a victim).

Attack families (all on the SAME eligible rows, per-row outcomes in artifacts/*.npz):
  1. Unconstrained input-space PGD/C&W, untargeted and targeted->Benign.
  2. PrimAttack {search, random-feasible} x {joint, timing-only, padding-only} x budgets
     (targeted->Benign).
  3. CAPGD (TabularBench, untargeted): native feature-space (config mask, L2 eps=0.5) and
     restricted to PrimAttack's p75 primitive-control box.
  4. FAB (AutoAttack ``FABAttack_PT``, untargeted, unconstrained minimum-norm) in the
     train-fitted min-max box (default L2 eps=0.5, same attack space as CAPGD native).

Every cell is gated by the SAME validator_v2 profile of the dataset (hybrid_valid) and the
primitive-transform realizability check. No VAE is loaded: no VAE latent attack is run and
no VAE in-distribution (IDR) / True-IDSR metric is computed. ``--resume`` reuses finished
per-row artifacts and their cells.json entries (same frozen selection asserted),
recomputing only missing cells.

    python scripts/run_full_adversarial_eval.py --dataset cicids2018 --device cuda
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
import torch.nn.functional as F
from joblib import parallel_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from attack.input_baselines import input_cw_attack, input_pgd_attack  # noqa: E402
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel  # noqa: E402
from attack.realizability.validator import RealizabilityValidator  # noqa: E402
from attack.flow_semantics import FlowSemanticValidator, SemanticStatus  # noqa: E402
from attack.primattack_budget import class_calibration, load_calibration, unbounded_calibration  # noqa: E402
from attack.run_cicids2017_primitive_attack import (  # noqa: E402
    evaluate_cell, random_feasible_primitives,
    _apply_primitive_mode, _decompose_cost, _LENGTH_COLS, _TIMING_COLS, _RATE_COLS,
)
from attack.primitive_optimizer import (  # noqa: E402
    CANDIDATE_NAMES, hybrid_valid_gate, optimize_primitive_candidates,
)
from comparisons.capgd_cicids2017 import (  # noqa: E402
    RawCICIDSVictim, build_capgd_resources, evaluate_capgd_output, finalize_capgd_output,
    fit_train_minmax, make_capgd,
)
from comparisons.primitive_capgd import run_primitive_capgd  # noqa: E402
from comparisons.fab_autoattack import AUTOATTACK_COMMIT, FABBox, load_fab_class, run_fab  # noqa: E402
from datasets import get_adapter  # noqa: E402
from experiments.provenance import deterministic_runtime  # noqa: E402
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402
from validation.attack_interface import structural_masks  # noqa: E402
from vae.cicids2017_stage_a import ATTACK_CLASSES  # noqa: E402

BUDGET_LABEL = {"intermediate": "p50", "maximum-evaluated": "p75", "restricted": "p25",
                "unbounded": "unb"}
BENIGN_ID = 0
FAMILIES = ("input", "primattack", "capgd", "fab")

DATASET_DEFAULTS = {
    "cicids2017_distrinet": {
        "victims": "mlp,cnn,ft_transformer",
        "calibration": REPO_ROOT / "artifacts/primattack/budget_calibration.json",
    },
    "cicids2018_distrinet": {
        "victims": ",".join(f"{a}-s{s}" for a in ("mlp", "cnn", "ft_transformer")
                            for s in (42, 123, 2024)),
        "calibration": REPO_ROOT / "artifacts/primattack/budget_calibration_cicids2018.json",
    },
}


def victim_checkpoint(dataset: str, victim: str) -> tuple[Path, str, int | None]:
    """Resolve a victim id to (checkpoint, architecture, training seed)."""
    if dataset == "cicids2017_distrinet":
        ckpt = {
            "mlp": REPO_ROOT / "outputs/cicids2017distrinet/models/mlp_category.pt",
            "cnn": REPO_ROOT / "outputs/cicids2017distrinet/models/cnn_category.pt",
            "ft_transformer": REPO_ROOT / "outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt",
        }
        if victim not in ckpt:
            raise KeyError(f"unknown CICIDS2017 victim {victim!r}")
        return ckpt[victim], victim, None
    if dataset == "cicids2018_distrinet":
        arch, sep, seed = victim.rpartition("-s")
        if not sep or not seed.isdigit():
            raise KeyError(f"CICIDS2018 victim must be '<arch>-s<seed>', got {victim!r}")
        path = (REPO_ROOT / "outputs/cicids2018distrinet/classifiers_multiseed/runs"
                / f"seed_{seed}" / "models" / f"{arch}_category.pt")
        return path, arch, int(seed)
    raise KeyError(f"no victim registry for dataset {dataset!r}")


# ----------------------------- targeted baselines ---------------------------------
def targeted_pgd_benign(victim, x0, *, epsilon, steps, alpha, device):
    x0 = x0.to(device)
    x_adv = (x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)).detach()
    x_adv = torch.min(torch.max(x_adv, x0 - epsilon), x0 + epsilon)
    target = torch.full((x0.shape[0],), BENIGN_ID, dtype=torch.long, device=device)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(victim(x_adv), target)
        (grad,) = torch.autograd.grad(loss, x_adv)
        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x0 - epsilon), x0 + epsilon)
        x_adv = x_adv.detach()
    return x_adv


def targeted_cw_benign(victim, x0, *, lambda_conf, kappa, iters, lr, device):
    x0 = x0.to(device)
    n = x0.shape[0]
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    best = x0.clone()
    best_l2 = torch.full((n,), float("inf"), device=device)
    best_succ = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        logits = victim(x0 + delta)
        tgt = logits[:, BENIGN_ID]
        other = logits.clone()
        other[:, BENIGN_ID] = float("-inf")
        other = other.max(dim=1).values
        conf = torch.clamp(other - tgt + kappa, min=0.0)  # push benign logit above the rest
        l2sq = delta.reshape(n, -1).pow(2).sum(1)
        (lambda_conf * conf + l2sq).mean().backward()
        opt.step()
        with torch.no_grad():
            logits2 = victim(x0 + delta)
            succ = logits2.argmax(1) == BENIGN_ID
            l2 = delta.reshape(n, -1).norm(dim=1)
            improved = succ & (l2 < best_l2)
            best[improved] = (x0 + delta).detach()[improved]
            best_l2[improved] = l2[improved]
            best_succ[improved] = True
    with torch.no_grad():
        x_adv = torch.where(best_succ.unsqueeze(1), best, (x0 + delta).detach())
    return x_adv.detach()


# ----------------------------- selection ------------------------------------------
def _sha_ids(ids: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update("\n".join(map(str, ids.tolist())).encode("utf-8"))
    return h.hexdigest()


def build_attack_roster(families, budgets, modes, optimizers):
    roster = []
    if "input" in families:
        roster += [
            {"name": "pgd_untargeted", "family": "input", "kind": "input_pgd", "goal": "untargeted"},
            {"name": "cw_untargeted", "family": "input", "kind": "input_cw", "goal": "untargeted"},
            {"name": "pgd_tb", "family": "input", "kind": "tpgd", "goal": "targeted_benign"},
            {"name": "cw_tb", "family": "input", "kind": "tcw", "goal": "targeted_benign"},
        ]
    if "primattack" in families:
        for budget in budgets:
            for opt_name in optimizers:
                for mode in modes:
                    short = {"search": "search", "random-feasible": "rand"}[opt_name]
                    mshort = {"joint": "joint", "timing-only": "timing",
                              "padding-only": "padding"}[mode]
                    roster.append({
                        "name": f"prim_{short}_{mshort}_{BUDGET_LABEL[budget]}",
                        "family": "primattack", "kind": "primattack", "goal": "targeted_benign",
                        "optimizer": opt_name, "mode": mode, "budget": budget,
                    })
    if "capgd" in families:
        roster += [
            {"name": "capgd_native", "family": "capgd", "kind": "capgd", "goal": "untargeted"},
            {"name": "capgd_prim_p75", "family": "capgd", "kind": "prim_capgd",
             "goal": "untargeted", "budget": "maximum-evaluated", "mode": "joint"},
        ]
    if "fab" in families:
        roster.append({"name": "fab_untargeted", "family": "fab", "kind": "fab",
                       "goal": "untargeted"})
    return roster


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cicids2017", help="cicids2017 | cicids2018")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", default="42,123,2024",
                    help="attack seeds (ignored for victims with a training seed when "
                         "--match-victim-seed is set)")
    ap.add_argument("--match-victim-seed", action="store_true",
                    help="attack each '<arch>-s<seed>' victim with its own training seed only")
    ap.add_argument("--n-per-class", type=int, default=800)
    ap.add_argument("--selection", choices=("random", "head"), default="random",
                    help="random: seeded uniform sample of the clean-correct rows (covers every "
                         "source label); head: first N in test order (legacy canonical)")
    ap.add_argument("--selection-seed", type=int, default=42)
    ap.add_argument("--victims", default=None, help="default: every victim of the dataset")
    ap.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--budgets", default="intermediate,maximum-evaluated,unbounded")
    ap.add_argument("--modes", default="joint,timing-only,padding-only")
    ap.add_argument("--optimizers", default="search,random-feasible")
    # input baselines
    ap.add_argument("--pgd-epsilon", type=float, default=0.5)
    ap.add_argument("--pgd-alpha", type=float, default=0.05)
    ap.add_argument("--pgd-steps", type=int, default=40)
    ap.add_argument("--cw-lambda", type=float, default=1.0)
    ap.add_argument("--cw-kappa", type=float, default=0.0)
    ap.add_argument("--cw-iters", type=int, default=60)
    ap.add_argument("--cw-lr", type=float, default=0.01)
    ap.add_argument("--cw-conv", type=float, default=1e-5)
    # primattack
    ap.add_argument("--prim-steps", type=int, default=40)
    ap.add_argument("--prim-lr", type=float, default=0.1)
    ap.add_argument("--prim-restarts", type=int, default=2)
    # capgd
    ap.add_argument("--capgd-epsilon", type=float, default=0.5)
    ap.add_argument("--capgd-steps", type=int, default=10)
    ap.add_argument("--capgd-batch-size", type=int, default=64)
    ap.add_argument("--prim-capgd-steps", type=int, default=40)
    # fab (AutoAttack)
    ap.add_argument("--fab-norm", choices=("Linf", "L2", "L1"), default="L2")
    ap.add_argument("--fab-epsilon", type=float, default=0.5,
                    help="acceptance radius in the train min-max [0,1] box")
    ap.add_argument("--fab-iter", type=int, default=100)
    ap.add_argument("--fab-restarts", type=int, default=1)
    ap.add_argument("--fab-batch-size", type=int, default=1024)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: outputs/adv_campaign/<dataset>")
    ap.add_argument("--resume", action="store_true",
                    help="skip (victim, class, attack, seed) cells whose npz AND cells.json "
                         "entry already exist in --output-dir")
    args = ap.parse_args()

    adapter = get_adapter(args.dataset)
    dataset = adapter.name
    defaults = DATASET_DEFAULTS[dataset]
    calibration_path = args.calibration or defaults["calibration"]
    out = args.output_dir or (REPO_ROOT / "outputs/adv_campaign" / dataset)

    device = args.device
    base_seeds = [int(s) for s in _csv(args.seeds)]
    victims = _csv(args.victims or defaults["victims"])
    classes = _csv(args.classes)
    families = _csv(args.families)
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown families {unknown}")
    budgets, modes, optimizers = _csv(args.budgets), _csv(args.modes), _csv(args.optimizers)
    roster = build_attack_roster(families, budgets, modes, optimizers)

    victim_info = {}
    for vname in victims:
        ckpt, arch, train_seed = victim_checkpoint(dataset, vname)
        seeds = [train_seed] if (args.match_victim_seed and train_seed is not None) else base_seeds
        victim_info[vname] = {"checkpoint": ckpt, "arch": arch, "train_seed": train_seed,
                              "attack_seeds": seeds}

    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)

    model = CICIDS2017PrimitiveModel(manifest)
    realizability = RealizabilityValidator(model)
    calibration = load_calibration(calibration_path)
    if calibration.get("dataset") != dataset:
        raise ValueError(f"calibration {calibration_path} is for {calibration.get('dataset')!r}, "
                         f"not {dataset!r}")
    semantic_validator = FlowSemanticValidator(model, calibration)
    # Search success = targeted->Benign AND validator_v2 hybrid_valid on the realized flow,
    # i.e. the same predicate the valid-ASR columns report.
    success_gate = hybrid_valid_gate(dataset)
    groups_idx = {
        "padding": torch.tensor([model.i[n] for n in _LENGTH_COLS], device=device),
        "timing": torch.tensor([model.i[n] for n in _TIMING_COLS], device=device),
        "rate": torch.tensor([model.i[n] for n in _RATE_COLS], device=device),
    }
    # ---- data + alignment assertions ----
    processed = adapter._processed
    raw_all = np.load(processed / "X_test_pristine.npy", mmap_mode="r")
    x_scaled_disk = np.load(processed / "X_test.npy", mmap_mode="r")
    y = np.load(processed / "y_test_cat.npy").astype(np.int64)
    y_split = np.asarray(adapter.load_split("test").y).astype(np.int64)
    meta = pd.read_parquet(processed / "test.parquet",
                           columns=["sample_id", "Src IP", "Dst IP", "source_label"])
    sample_ids_all = meta["sample_id"].astype(str).to_numpy(dtype="U128")
    n_rows = len(y)
    assert raw_all.shape[0] == n_rows == x_scaled_disk.shape[0] == len(sample_ids_all) == len(y_split), \
        "row-order/length mismatch across y/X_test/X_test_pristine/test.parquet"
    assert np.array_equal(y, y_split), "y_test_cat.npy disagrees with load_split('test').y"
    probe = np.asarray(raw_all[:2048], dtype=np.float64)
    recon = (probe - transform.center) / transform.scale
    assert np.allclose(recon, np.asarray(x_scaled_disk[:2048], dtype=np.float64), atol=1e-3), \
        "X_test.npy != (X_test_pristine-center)/scale"

    raw_all_t = torch.tensor(np.ascontiguousarray(raw_all), dtype=torch.float32, device=device)
    scaled_all_t = (raw_all_t - center) / scale  # single source of truth for victim input

    capgd_resources = build_capgd_resources(REPO_ROOT, adapter=adapter) if "capgd" in families else None
    fab_cls = fab_box = None
    if "fab" in families:
        fab_cls = load_fab_class(REPO_ROOT)
        fab_box = FABBox.from_minmax(*fit_train_minmax(processed / "X_train_pristine.npy"),
                                     device=device)

    config = {
        "dataset": dataset,
        "victims": {v: {"checkpoint": str(i["checkpoint"]), "arch": i["arch"],
                        "train_seed": i["train_seed"], "attack_seeds": i["attack_seeds"]}
                    for v, i in victim_info.items()},
        "classes": classes, "families": families,
        "budgets": {b: BUDGET_LABEL[b] for b in budgets}, "modes": modes,
        "optimizers": optimizers, "n_per_class_cap": args.n_per_class,
        "eligibility": "clean-correct (victim predicts true malicious class on clean flow)",
        "denominator": "clean-correct eligible set (identical across attacks within a victim)",
        "benign_id": BENIGN_ID,
        "validator_v2_profile": dataset,
        "pgd": {"epsilon": args.pgd_epsilon, "alpha": args.pgd_alpha, "steps": args.pgd_steps},
        "cw": {"lambda": args.cw_lambda, "kappa": args.cw_kappa, "iters": args.cw_iters,
               "lr": args.cw_lr, "conv": args.cw_conv},
        "primattack": {"steps": args.prim_steps, "learning_rate": args.prim_lr,
                       "restarts": args.prim_restarts, "calibration": str(calibration_path),
                       "calibration_fit_split": calibration["fit_split"]},
        "capgd": {"native": {"norm": "L2", "epsilon": args.capgd_epsilon,
                             "steps": args.capgd_steps, "batch_size": args.capgd_batch_size},
                  "primitive": {"norm": "Linf", "epsilon": 1.0, "box": "PrimAttack p75 joint",
                                "steps": args.prim_capgd_steps}},
        "fab": {"implementation": "autoattack.fab_pt.FABAttack_PT (external/auto-attack)",
                "autoattack_commit": AUTOATTACK_COMMIT, "goal": "untargeted",
                "norm": args.fab_norm, "epsilon": args.fab_epsilon, "n_iter": args.fab_iter,
                "n_restarts": args.fab_restarts, "batch_size": args.fab_batch_size,
                "attack_space": "train-fitted min-max box of pristine features, [0,1]^d; "
                                "train-constant features pinned"},
        "attack_roster": [{k: v for k, v in a.items()} for a in roster],
        "n_features": int(manifest.n_features), "n_test": int(n_rows),
        "class_names": list(mapping.names),
        "selection": {"method": args.selection, "seed": args.selection_seed,
                      "rule": ("per class, one seeded permutation of ALL class test rows; each "
                               "victim takes the first n_per_class that it classifies correctly "
                               "(max overlap across victims), re-sorted to test order"
                               if args.selection == "random" else
                               "first n_per_class clean-correct rows in test order")},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ---- selection: clean-correct eligible set per (victim, class) ----
    selection, eligibility, victim_cache = {}, {}, {}
    for vname in victims:
        info = victim_info[vname]
        victim = load_category_victim(info["checkpoint"], adapter=adapter,
                                      expected_model_type=info["arch"], device=device)
        victim_cache[vname] = victim
        with torch.no_grad():
            pred = np.concatenate([victim(scaled_all_t[s:s + 16384]).argmax(1).cpu().numpy()
                                   for s in range(0, n_rows, 16384)])
        selection[vname] = {}
        for cname in classes:
            cid = int(mapping.name_to_id[cname])
            correct = (pred == cid) & (y == cid)
            elig = np.flatnonzero(correct)  # test-order, clean-correct
            if args.selection == "random":
                order = np.random.default_rng(args.selection_seed + cid).permutation(
                    np.flatnonzero(y == cid))
                idx = np.sort(order[correct[order]][: args.n_per_class])
            else:
                idx = elig[: args.n_per_class]
            sids = sample_ids_all[idx]
            raw_t = raw_all_t[torch.tensor(idx, device=device)]
            clean_valid = structural_masks(raw_t.cpu().numpy(), dataset=dataset)["hybrid_valid"]
            selection[vname][cname] = {
                "class_id": cid,
                "n_class_test": int((y == cid).sum()),
                "n_eligible_total": int(len(elig)),
                "n_used": int(len(idx)),
                "clean_hybrid_valid_rate": float(np.mean(clean_valid)) if len(idx) else float("nan"),
                "source_label_counts": {str(k): int(v) for k, v in
                                        meta.iloc[idx]["source_label"].value_counts().items()},
                "sha256_sample_ids": _sha_ids(sids),
                "positional_idx": idx.tolist(),
                "sample_ids": sids.tolist(),
            }
            eligibility[(vname, cname)] = {
                "idx": idx, "sids": sids, "raw": raw_t, "x0": (raw_t - center) / scale,
                "src": {"Src IP": meta.iloc[idx]["Src IP"].astype(str).to_numpy(),
                        "Dst IP": meta.iloc[idx]["Dst IP"].astype(str).to_numpy()},
            }
    prior_sel = out / "selection.json"
    if args.resume and prior_sel.exists():
        old = json.loads(prior_sel.read_text(encoding="utf-8"))
        for vname in victims:
            for cname in classes:
                if old.get(vname, {}).get(cname, {}).get("sha256_sample_ids") != \
                        selection[vname][cname]["sha256_sample_ids"]:
                    raise ValueError(f"--resume: selection changed for {vname}/{cname}")
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # ---- run attacks ----
    prior_cells = {}
    if args.resume and (out / "cells.json").exists():
        prior_cells = {(c["victim"], c["class"], c["attack"], c["seed"]): c for c in
                       json.loads((out / "cells.json").read_text(encoding="utf-8"))}
    cells, failures, n_resumed = [], [], 0
    t0 = time.time()
    for vname in victims:
        victim = victim_cache[vname]
        seeds = victim_info[vname]["attack_seeds"]
        raw_victim = RawCICIDSVictim(victim, transform.center, transform.scale).to(device).eval()
        for cname in classes:
            cid = int(mapping.name_to_id[cname])
            E = eligibility[(vname, cname)]
            raw, x0, sids = E["raw"], E["x0"], E["sids"]
            n = raw.shape[0]
            labels = np.full(n, cid, dtype=np.int64)
            labels_t = torch.full((n,), cid, dtype=torch.long, device=device)
            caps = model.infer_capabilities(raw)
            for atk in roster:
                for seed in seeds:
                    key = (vname, cname, atk["name"], seed)
                    if key in prior_cells and (
                            art / f"{vname}__{cname}__{atk['name']}__seed{seed}.npz").exists():
                        cells.append(prior_cells[key])
                        n_resumed += 1
                        continue
                    deterministic_runtime(seed)
                    semantic = None
                    requested = projected = optimization = bounds = ccfg = None
                    extra: dict[str, np.ndarray] = {}
                    attack_t0 = time.perf_counter()
                    kind = atk["kind"]
                    if kind == "input_pgd":
                        x_adv, _ = input_pgd_attack(
                            classifier=victim, x_original=x0, y_true=labels_t,
                            epsilon=args.pgd_epsilon, alpha=args.pgd_alpha,
                            num_steps=args.pgd_steps, random_start=True, device=device)
                    elif kind == "input_cw":
                        x_adv, _ = input_cw_attack(
                            classifier=victim, x_original=x0, y_true=labels_t,
                            lambda_conf=args.cw_lambda, kappa=args.cw_kappa,
                            num_iterations=args.cw_iters, learning_rate=args.cw_lr,
                            convergence_threshold=args.cw_conv, device=device)
                    elif kind == "tpgd":
                        x_adv = targeted_pgd_benign(
                            victim, x0, epsilon=args.pgd_epsilon, steps=args.pgd_steps,
                            alpha=args.pgd_alpha, device=device)
                    elif kind == "tcw":
                        x_adv = targeted_cw_benign(
                            victim, x0, lambda_conf=args.cw_lambda, kappa=args.cw_kappa,
                            iters=args.cw_iters, lr=args.cw_lr, device=device)
                    elif kind == "capgd":
                        attack = make_capgd(capgd_resources, raw_victim, device=device, seed=seed,
                                            norm="L2", eps=args.capgd_epsilon,
                                            steps=args.capgd_steps)
                        chunks = []
                        for s in range(0, n, args.capgd_batch_size):
                            batch = raw[s:s + args.capgd_batch_size]
                            with parallel_backend("threading"):
                                cand = attack(batch, labels_t[s:s + args.capgd_batch_size])
                            chunks.append(finalize_capgd_output(capgd_resources, batch, cand).detach())
                        adv_raw = torch.cat(chunks, dim=0).float()
                        checked = evaluate_capgd_output(
                            capgd_resources, raw.cpu().numpy(), adv_raw.cpu().numpy(),
                            norm="L2", eps=args.capgd_epsilon)
                        extra["capgd_internal_valid"] = np.asarray(checked["internal_constraint_valid"], bool)
                        extra["capgd_distance_ok"] = np.asarray(checked["distance_ok"], bool)
                    elif kind == "fab":
                        adv_raw = run_fab(
                            fab_cls, victim, raw, labels_t, box=fab_box, center=center,
                            scale=scale, norm=args.fab_norm, eps=args.fab_epsilon,
                            n_iter=args.fab_iter, n_restarts=args.fab_restarts, seed=seed,
                            batch_size=args.fab_batch_size, device=device)
                    else:  # primattack / prim_capgd
                        ccfg = (unbounded_calibration(calibration, cname)
                                if atk["budget"] == "unbounded"
                                else class_calibration(calibration, cname, atk["budget"]))
                        bounds = _apply_primitive_mode(
                            model.per_flow_bounds(raw, ccfg.bounds_config(), capabilities=caps),
                            atk["mode"])
                        if kind == "prim_capgd":
                            with parallel_backend("threading"):
                                result = run_primitive_capgd(
                                    repo_root=REPO_ROOT, primitive_model=model, victim=victim,
                                    raw=raw, bounds=bounds, capabilities=caps, center=center,
                                    scale=scale, true_labels=labels_t, seed=seed,
                                    steps=args.prim_capgd_steps)
                            requested, projected = result.requested, result.projected
                            adv_raw = result.adversarial_raw
                        elif atk["optimizer"] == "search":
                            optimization = optimize_primitive_candidates(
                                model, victim, raw, center, scale, bounds, caps,
                                steps=args.prim_steps, learning_rate=args.prim_lr,
                                restarts=args.prim_restarts, seed=seed,
                                validity_fn=success_gate,
                            )
                            requested = optimization.requested
                            projected = optimization.projected
                            adv_raw = optimization.adversarial_raw
                        else:
                            requested = random_feasible_primitives(bounds, seed)
                            projected = model.project_controls(raw, requested, bounds, capabilities=caps)
                            adv_raw = model.generate(raw, projected, quantize=True,
                                                     capabilities=caps).detach()
                        semantic = semantic_validator.evaluate(
                            raw, adv_raw, requested, projected, bounds,
                            class_name=cname, budget=ccfg.budget,
                            original_labels=labels, adversarial_labels=labels.copy(),
                            original_metadata=E["src"], adversarial_metadata=E["src"])

                    if kind in ("input_pgd", "input_cw", "tpgd", "tcw"):
                        adv_raw = (x_adv * scale + center).detach()
                    x_adv = (adv_raw - center) / scale
                    elapsed_seconds = time.perf_counter() - attack_t0

                    if not bool(torch.isfinite(adv_raw).all()):
                        failures.append({"attack": atk["name"], "victim": vname,
                                         "class": cname, "seed": seed, "reason": "nonfinite_adv"})
                        continue

                    masks, _cost_unused, clean_pred, adv_pred = evaluate_cell(
                        model, realizability, victim, raw, adv_raw,
                        center, scale, cid, groups_idx)
                    cost = _decompose_cost(adv_raw, raw, scale, groups_idx)

                    cc = masks["clean_correct"]
                    if int(cc.sum().item()) != n:
                        failures.append({"attack": atk["name"], "victim": vname, "class": cname,
                                         "seed": seed, "reason": "eligible_not_all_clean_correct",
                                         "n_clean_correct": int(cc.sum().item()), "n": n})

                    evasion = adv_pred != cid
                    targeted = adv_pred == BENIGN_ID
                    domain_valid = masks["domain_valid"]
                    realizable = masks["primitive_transform_consistent"]
                    if semantic is not None:
                        sem_pass = torch.as_tensor(
                            semantic.semantic_status == SemanticStatus.PASS.value, device=device)
                        prim_feasible = torch.as_tensor(
                            semantic.primitive_feasible, device=device) & realizable
                    else:
                        sem_pass = prim_feasible = None
                    np_ = lambda t: t.detach().cpu().numpy()  # noqa: E731
                    nan = np.full(n, np.nan, np.float32)
                    l2 = (x_adv - x0).reshape(n, -1).norm(dim=1)

                    npz = art / f"{vname}__{cname}__{atk['name']}__seed{seed}.npz"
                    np.savez_compressed(
                        npz,
                        sample_id=sids, positional_idx=E["idx"], true_class=labels,
                        clean_pred=np_(clean_pred).astype(np.int64),
                        adv_pred=np_(adv_pred).astype(np.int64),
                        clean_correct=np_(cc), evasion=np_(evasion), targeted_success=np_(targeted),
                        domain_valid=np_(domain_valid),
                        hard_structural_valid=np_(masks["hard_structural_valid"]),
                        realizable=np_(realizable),
                        semantic_pass=(np_(sem_pass) if sem_pass is not None
                                       else np.full(n, -1, dtype=np.int8)),
                        primitive_feasible=(np_(prim_feasible) if prim_feasible is not None
                                            else np.full(n, -1, dtype=np.int8)),
                        cost_total=np_(cost["total"]).astype(np.float32),
                        cost_padding=np_(cost["padding"]).astype(np.float32),
                        cost_timing=np_(cost["timing"]).astype(np.float32),
                        l2_scaled=np_(l2).astype(np.float32),
                        primitive_p=(np_(projected["p"]).astype(np.float32) if projected is not None else nan),
                        primitive_delay=(np_(projected["delay"]).astype(np.float32) if projected is not None else nan),
                        primitive_shape=(np_(projected["shape"]).astype(np.float32) if projected is not None else nan),
                        primitive_p_hi=(np_(bounds["p"]).astype(np.float32) if bounds is not None else nan),
                        primitive_delay_hi=(np_(bounds["delay"]).astype(np.float32) if bounds is not None else nan),
                        optimizer_target_margin=(
                            np_(optimization.target_margin).astype(np.float32)
                            if optimization is not None else nan),
                        optimizer_candidate_source=(
                            np.asarray([CANDIDATE_NAMES[int(v)] for v in np_(optimization.candidate_source)])
                            if optimization is not None else np.full(n, "not-applicable")),
                        **extra,
                        elapsed_seconds=np.asarray(elapsed_seconds, np.float64),
                        attack=atk["name"], family=atk["family"], victim=vname,
                        victim_arch=victim_info[vname]["arch"], attack_class=cname, seed=seed,
                        goal=atk["goal"], dataset=dataset,
                    )

                    rate = lambda m: float((m & cc).sum().item()) / n  # noqa: E731
                    cell = {
                        "dataset": dataset, "victim": vname, "arch": victim_info[vname]["arch"],
                        "class": cname, "attack": atk["name"], "family": atk["family"],
                        "goal": atk["goal"], "seed": seed, "n_eligible": n,
                        "n_eligible_total": selection[vname][cname]["n_eligible_total"],
                        "sha256_sample_ids": selection[vname][cname]["sha256_sample_ids"],
                        "raw_asr_untargeted": rate(evasion),
                        "valid_asr_untargeted": rate(evasion & domain_valid),
                        "targeted_benign": rate(targeted),
                        "valid_targeted_benign": rate(targeted & domain_valid),
                        "domain_validity_rate": rate(domain_valid),
                        "realizable_rate": rate(realizable),
                        "mean_cost_total": float(cost["total"].mean().item()),
                        "mean_l2_scaled": float(l2.mean().item()),
                        "elapsed_seconds": elapsed_seconds,
                    }
                    if sem_pass is not None:
                        cell["semantic_pass_rate"] = rate(sem_pass)
                        cell["primitive_feasible_rate"] = rate(prim_feasible)
                        cell["sp_asr"] = rate(targeted & domain_valid & prim_feasible & sem_pass)
                    cells.append(cell)
            print(f"[{time.time()-t0:6.0f}s] {vname}/{cname}: {len(roster)} attacks x "
                  f"{len(seeds)} seeds done (n={n})", flush=True)
            (out / "cells.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")

    (out / "cells.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")
    (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    assert_pairing(out, victim_info, classes, roster, selection)
    print(f"[done] {len(cells)} cells ({n_resumed} resumed), {len(failures)} failures; "
          f"pairing assertions PASSED; artifacts in {art}", flush=True)


def assert_pairing(out: Path, victim_info, classes, roster, selection):
    """Fail loudly if any two attacks on the same (victim,class,seed) used different rows,
    order, labels, clean predictions, or counts."""
    art = out / "artifacts"
    for vname, info in victim_info.items():
        for cname in classes:
            ref_sids = np.asarray(selection[vname][cname]["sample_ids"], dtype="U128")
            n_ref = len(ref_sids)
            ref_clean = None
            for seed in info["attack_seeds"]:
                for atk in roster:
                    npz = art / f"{vname}__{cname}__{atk['name']}__seed{seed}.npz"
                    if not npz.exists():
                        continue
                    d = np.load(npz, allow_pickle=True)
                    sids = d["sample_id"].astype("U128")
                    assert len(sids) == n_ref, (
                        f"count mismatch {npz.name}: {len(sids)} != {n_ref}")
                    assert np.array_equal(sids, ref_sids), (
                        f"row-id/order mismatch in {npz.name} vs selection")
                    assert np.array_equal(d["true_class"], np.full(n_ref, selection[vname][cname]["class_id"])), (
                        f"label mismatch in {npz.name}")
                    assert bool((d["clean_pred"] == selection[vname][cname]["class_id"]).all()), (
                        f"eligible rows not all clean-correct in {npz.name}")
                    if ref_clean is None:
                        ref_clean = d["clean_pred"]
                    else:
                        assert np.array_equal(d["clean_pred"], ref_clean), (
                            f"clean predictions differ across attacks in {npz.name} "
                            "(victim/eligibility desynchronized)")


if __name__ == "__main__":
    main()
