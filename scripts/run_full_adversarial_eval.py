"""Full paired adversarial evaluation on CICIDS2017-DistriNet.

Fixes the pairing/eligibility/denominator/cap defects found in the legacy runners (see
FULL_ADVERSARIAL_EVALUATION_CICIDS2017.md audit section) by construction:

* ELIGIBILITY = clean-correct (victim predicts the true malicious class on the CLEAN flow).
  Selected ONCE per (victim, class) from the FULL test split (no pre-eligibility cap), then a
  fixed max-N head slice in test order. Row IDs (sample_id) + positional indices + a sha256 are
  saved to selection.json. EVERY attack/goal/seed attacks EXACTLY these rows in this order.
* DENOMINATOR = that clean-correct eligible set (identical across every compared attack within a
  victim). Rates are numerator/eligible with one consistent denominator.
* No `_class_rows` true-label selection, no `--test-limit` cap, no broken `mined_valid`/`benign`
  mask keys (uses the current evaluate_cell contract).

Attacks (all on the SAME eligible rows):
  Unconstrained input-space baselines: PGD/C&W, untargeted AND targeted->Benign.
  PrimAttack: {optimized, random-feasible} x {joint, timing-only, padding-only} x
              {intermediate(p50), maximum-evaluated(p75)} budgets (targeted->Benign).

Per-row outcomes for every cell are written to artifacts/*.npz so the analysis step can run
sample-level PAIRED McNemar tests. Run under the CUDA thesis env:
    python scripts/run_full_adversarial_eval.py --device cuda
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from attack.input_baselines import input_cw_attack, input_pgd_attack  # noqa: E402
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel, SCALER_ATOL  # noqa: E402
from attack.realizability.validator import RealizabilityValidator  # noqa: E402
from attack.flow_semantics import FlowSemanticValidator, SemanticStatus  # noqa: E402
from attack.primattack_budget import class_calibration, load_calibration, unbounded_calibration  # noqa: E402
from attack.run_cicids2017_primitive_attack import (  # noqa: E402
    evaluate_cell, optimize_primitives, random_feasible_primitives,
    _apply_primitive_mode, _decompose_cost, _LENGTH_COLS, _TIMING_COLS, _RATE_COLS,
)
from datasets.cicids2017 import CICIDS2017Adapter  # noqa: E402
from experiments.provenance import deterministic_runtime  # noqa: E402
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402
from vae.cicids2017_stage_a import ATTACK_CLASSES, load_stage_a  # noqa: E402

VICTIM_CKPT = {
    "mlp": REPO_ROOT / "outputs/cicids2017distrinet/models/mlp_category.pt",
    "cnn": REPO_ROOT / "outputs/cicids2017distrinet/models/cnn_category.pt",
    "ft_transformer": REPO_ROOT / "outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt",
}
BUDGET_LABEL = {"intermediate": "p50", "maximum-evaluated": "p75", "restricted": "p25",
                "unbounded": "unb"}
BENIGN_ID = 0


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


def build_attack_roster(budgets, modes, optimizers):
    roster = [
        {"name": "pgd_untargeted", "kind": "input_pgd", "goal": "untargeted"},
        {"name": "cw_untargeted", "kind": "input_cw", "goal": "untargeted"},
        {"name": "pgd_tb", "kind": "tpgd", "goal": "targeted_benign"},
        {"name": "cw_tb", "kind": "tcw", "goal": "targeted_benign"},
    ]
    for budget in budgets:
        for opt_name in optimizers:
            for mode in modes:
                short = {"optimized": "opt", "random-feasible": "rand"}[opt_name]
                mshort = {"joint": "joint", "timing-only": "timing", "padding-only": "padding"}[mode]
                roster.append({
                    "name": f"prim_{short}_{mshort}_{BUDGET_LABEL[budget]}",
                    "kind": "primattack", "goal": "targeted_benign",
                    "optimizer": opt_name, "mode": mode, "budget": budget,
                })
    return roster


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", default="42,123,2024")
    ap.add_argument("--n-per-class", type=int, default=800)
    ap.add_argument("--victims", default="mlp,cnn,ft_transformer")
    ap.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    ap.add_argument("--budgets", default="intermediate,maximum-evaluated,unbounded")
    ap.add_argument("--modes", default="joint,timing-only,padding-only")
    ap.add_argument("--optimizers", default="optimized,random-feasible")
    # baseline hyperparams
    ap.add_argument("--pgd-epsilon", type=float, default=0.5)
    ap.add_argument("--pgd-alpha", type=float, default=0.05)
    ap.add_argument("--pgd-steps", type=int, default=40)
    ap.add_argument("--cw-lambda", type=float, default=1.0)
    ap.add_argument("--cw-kappa", type=float, default=0.0)
    ap.add_argument("--cw-iters", type=int, default=60)
    ap.add_argument("--cw-lr", type=float, default=0.01)
    ap.add_argument("--cw-conv", type=float, default=1e-5)
    # primattack hyperparams
    ap.add_argument("--prim-steps", type=int, default=40)
    ap.add_argument("--prim-lr", type=float, default=0.1)
    ap.add_argument("--prim-cost-weight", type=float, default=0.01)
    ap.add_argument("--prim-init-noise", type=float, default=0.5)
    ap.add_argument("--calibration", type=Path,
                    default=REPO_ROOT / "artifacts/primattack/budget_calibration.json")
    ap.add_argument("--stage-a-dir", type=Path,
                    default=REPO_ROOT / "outputs/cicids2017_vae_stage_a")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/full_adv_eval")
    args = ap.parse_args()

    device = args.device
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    victims = [v.strip() for v in args.victims.split(",") if v.strip()]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    budgets = [b.strip() for b in args.budgets.split(",") if b.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    optimizers = [o.strip() for o in args.optimizers.split(",") if o.strip()]
    roster = build_attack_roster(budgets, modes, optimizers)

    out = args.output_dir
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    adapter = CICIDS2017Adapter()
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)

    model = CICIDS2017PrimitiveModel(manifest)
    realizability = RealizabilityValidator(model)
    calibration = load_calibration(args.calibration)
    semantic_validator = FlowSemanticValidator(model, calibration)
    groups_idx = {
        "padding": torch.tensor([model.i[n] for n in _LENGTH_COLS], device=device),
        "timing": torch.tensor([model.i[n] for n in _TIMING_COLS], device=device),
        "rate": torch.tensor([model.i[n] for n in _RATE_COLS], device=device),
    }

    # ---- data + alignment assertions (Hazard 5) ----
    raw_all = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    x_scaled_disk = np.load(adapter._processed / "X_test.npy", mmap_mode="r")
    y = np.load(adapter._processed / "y_test_cat.npy").astype(np.int64)
    y_split = np.asarray(adapter.load_split("test").y).astype(np.int64)
    meta = pd.read_parquet(adapter._processed / "test.parquet",
                           columns=["sample_id", "Src IP", "Dst IP"])
    sample_ids_all = meta["sample_id"].astype(str).to_numpy(dtype="U128")
    n_rows = len(y)
    assert raw_all.shape[0] == n_rows == x_scaled_disk.shape[0] == len(sample_ids_all) == len(y_split), \
        "row-order/length mismatch across y/X_test/X_test_pristine/test.parquet"
    assert np.array_equal(y, y_split), "y_test_cat.npy disagrees with load_split('test').y"
    # verify scaled == (pristine-center)/scale on a probe (preprocessing invariant)
    probe = np.asarray(raw_all[:2048], dtype=np.float64)
    recon = (probe - transform.center) / transform.scale
    assert np.allclose(recon, np.asarray(x_scaled_disk[:2048], dtype=np.float64), atol=1e-3), \
        "X_test.npy != (X_test_pristine-center)/scale"

    raw_all_t = torch.tensor(np.ascontiguousarray(raw_all), dtype=torch.float32, device=device)
    scaled_all_t = (raw_all_t - center) / scale  # single source of truth for victim input

    # per-class Stage-A VAE + IDR
    base_vae, idr_path = {}, {}
    for cname in classes:
        vae, _ = load_stage_a(adapter, args.stage_a_dir / f"vae_{cname}.pt",
                              expected_class_name=cname, device=device)
        base_vae[cname] = vae
        idr_path[cname] = args.stage_a_dir / f"idr_{cname}.npz"

    config = {
        "seeds": seeds, "victims": victims, "classes": classes,
        "budgets": {b: BUDGET_LABEL[b] for b in budgets}, "modes": modes,
        "optimizers": optimizers, "n_per_class_cap": args.n_per_class,
        "eligibility": "clean-correct (victim predicts true malicious class on clean flow)",
        "denominator": "clean-correct eligible set (identical across attacks within a victim)",
        "benign_id": BENIGN_ID,
        "pgd": {"epsilon": args.pgd_epsilon, "alpha": args.pgd_alpha, "steps": args.pgd_steps},
        "cw": {"lambda": args.cw_lambda, "kappa": args.cw_kappa, "iters": args.cw_iters,
               "lr": args.cw_lr, "conv": args.cw_conv},
        "primattack": {"steps": args.prim_steps, "lr": args.prim_lr,
                       "cost_weight": args.prim_cost_weight, "init_noise": args.prim_init_noise,
                       "calibration": str(args.calibration),
                       "calibration_fit_split": calibration["fit_split"]},
        "attack_roster": [a["name"] for a in roster],
        "n_features": int(manifest.n_features), "n_test": int(n_rows),
        "class_names": list(mapping.names),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ---- selection: clean-correct eligible set per (victim, class) ----
    selection = {}
    eligibility = {}  # (victim, class) -> dict(idx, sample_ids, raw_t, x0_t, meta)
    victim_cache = {}
    for vname in victims:
        victim = load_category_victim(VICTIM_CKPT[vname], adapter=adapter,
                                      expected_model_type=vname, device=device)
        victim_cache[vname] = victim
        with torch.no_grad():
            preds = []
            for s in range(0, n_rows, 16384):
                preds.append(victim(scaled_all_t[s:s + 16384]).argmax(1).cpu().numpy())
        pred = np.concatenate(preds)
        selection[vname] = {}
        for cname in classes:
            cid = int(mapping.name_to_id[cname])
            elig = np.flatnonzero((pred == cid) & (y == cid))  # test-order, clean-correct
            n_elig_total = int(len(elig))
            idx = elig[: args.n_per_class]  # head slice in test order (deterministic)
            sids = sample_ids_all[idx]
            selection[vname][cname] = {
                "class_id": cid,
                "n_eligible_total": n_elig_total,
                "n_used": int(len(idx)),
                "sha256_sample_ids": _sha_ids(sids),
                "positional_idx": idx.tolist(),
                "sample_ids": sids.tolist(),
            }
            raw_t = raw_all_t[torch.tensor(idx, device=device)]
            eligibility[(vname, cname)] = {
                "idx": idx, "sids": sids, "raw": raw_t,
                "x0": (raw_t - center) / scale,
                "src": {"Src IP": meta.iloc[idx]["Src IP"].astype(str).to_numpy(),
                        "Dst IP": meta.iloc[idx]["Dst IP"].astype(str).to_numpy()},
            }
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # ---- run attacks ----
    cells = []
    failures = []
    t0 = time.time()
    for vname in victims:
        victim = victim_cache[vname]
        for cname in classes:
            cid = int(mapping.name_to_id[cname])
            E = eligibility[(vname, cname)]
            raw, x0, sids = E["raw"], E["x0"], E["sids"]
            n = raw.shape[0]
            labels = np.full(n, cid, dtype=np.int64)
            # primitive scaffolding (per victim,class; reused across seeds/modes)
            caps = model.infer_capabilities(raw)
            for atk in roster:
                for seed in seeds:
                    deterministic_runtime(seed)
                    semantic = None
                    if atk["kind"] == "input_pgd":
                        x_adv, _ = input_pgd_attack(
                            classifier=victim, x_original=x0,
                            y_true=torch.full((n,), cid, dtype=torch.long, device=device),
                            epsilon=args.pgd_epsilon, alpha=args.pgd_alpha,
                            num_steps=args.pgd_steps, random_start=True, device=device)
                    elif atk["kind"] == "input_cw":
                        x_adv, _ = input_cw_attack(
                            classifier=victim, x_original=x0,
                            y_true=torch.full((n,), cid, dtype=torch.long, device=device),
                            lambda_conf=args.cw_lambda, kappa=args.cw_kappa,
                            num_iterations=args.cw_iters, learning_rate=args.cw_lr,
                            convergence_threshold=args.cw_conv, device=device)
                    elif atk["kind"] == "tpgd":
                        x_adv = targeted_pgd_benign(
                            victim, x0, epsilon=args.pgd_epsilon, steps=args.pgd_steps,
                            alpha=args.pgd_alpha, device=device)
                    elif atk["kind"] == "tcw":
                        x_adv = targeted_cw_benign(
                            victim, x0, lambda_conf=args.cw_lambda, kappa=args.cw_kappa,
                            iters=args.cw_iters, lr=args.cw_lr, device=device)
                    else:  # primattack
                        ccfg = (unbounded_calibration(calibration, cname)
                                if atk["budget"] == "unbounded"
                                else class_calibration(calibration, cname, atk["budget"]))
                        bounds = _apply_primitive_mode(
                            model.per_flow_bounds(raw, ccfg.bounds_config(), capabilities=caps),
                            atk["mode"])
                        if atk["optimizer"] == "optimized":
                            requested = optimize_primitives(
                                model, victim, raw, center, scale, bounds, caps,
                                steps=args.prim_steps, lr=args.prim_lr,
                                cost_weight=args.prim_cost_weight,
                                init_noise=args.prim_init_noise, seed=seed)
                        else:
                            requested = random_feasible_primitives(bounds, seed)
                        projected = model.project_controls(raw, requested, bounds)
                        adv_raw = model.generate(raw, projected, quantize=True).detach()
                        semantic = semantic_validator.evaluate(
                            raw, adv_raw, requested, projected, bounds,
                            class_name=cname, budget=ccfg.budget,
                            original_labels=labels, adversarial_labels=labels.copy(),
                            original_metadata=E["src"], adversarial_metadata=E["src"])
                        x_adv = (adv_raw - center) / scale

                    if atk["kind"] in ("input_pgd", "input_cw", "tpgd", "tcw"):
                        adv_raw = (x_adv * scale + center).detach()

                    if not bool(np.isfinite(adv_raw.detach().cpu().numpy()).all()):
                        failures.append({"attack": atk["name"], "victim": vname,
                                         "class": cname, "seed": seed, "reason": "nonfinite_adv"})
                        continue

                    masks, _cost_unused, clean_pred, adv_pred = evaluate_cell(
                        model, realizability, victim, base_vae[cname], raw, adv_raw,
                        center, scale, cid, idr_path[cname], groups_idx)
                    cost = _decompose_cost(adv_raw, raw, scale, groups_idx)

                    cc = masks["clean_correct"]
                    if int(cc.sum().item()) != n:
                        failures.append({"attack": atk["name"], "victim": vname, "class": cname,
                                         "seed": seed, "reason": "eligible_not_all_clean_correct",
                                         "n_clean_correct": int(cc.sum().item()), "n": n})

                    evasion = (adv_pred != cid)
                    targeted = (adv_pred == BENIGN_ID)
                    domain_valid = masks["domain_valid"]
                    realizable = masks["primitive_transform_consistent"]
                    in_dist = masks["in_dist"]
                    if semantic is not None:
                        sem_pass = torch.as_tensor(
                            semantic.semantic_status == SemanticStatus.PASS.value, device=device)
                        prim_feasible = torch.as_tensor(
                            semantic.primitive_feasible, device=device) & realizable
                    else:
                        sem_pass = None
                        prim_feasible = None

                    npz = art / f"{vname}__{cname}__{atk['name']}__seed{seed}.npz"
                    np.savez_compressed(
                        npz,
                        sample_id=sids,
                        positional_idx=E["idx"],
                        true_class=labels,
                        clean_pred=clean_pred.cpu().numpy().astype(np.int64),
                        adv_pred=adv_pred.cpu().numpy().astype(np.int64),
                        clean_correct=cc.cpu().numpy(),
                        evasion=evasion.cpu().numpy(),
                        targeted_success=targeted.cpu().numpy(),
                        domain_valid=domain_valid.cpu().numpy(),
                        realizable=realizable.cpu().numpy(),
                        in_dist=in_dist.cpu().numpy(),
                        semantic_pass=(sem_pass.cpu().numpy() if sem_pass is not None
                                       else np.full(n, -1, dtype=np.int8)),
                        primitive_feasible=(prim_feasible.cpu().numpy() if prim_feasible is not None
                                            else np.full(n, -1, dtype=np.int8)),
                        cost_total=cost["total"].cpu().numpy().astype(np.float32),
                        cost_padding=cost["padding"].cpu().numpy().astype(np.float32),
                        cost_timing=cost["timing"].cpu().numpy().astype(np.float32),
                        l2_scaled=(x_adv - x0).reshape(n, -1).norm(dim=1).cpu().numpy().astype(np.float32),
                        attack=atk["name"], victim=vname, attack_class=cname, seed=seed,
                        goal=atk["goal"],
                    )

                    rate = lambda m: float((m & cc).sum().item()) / n
                    cell = {
                        "victim": vname, "class": cname, "attack": atk["name"],
                        "goal": atk["goal"], "seed": seed, "n_eligible": n,
                        "n_eligible_total": selection[vname][cname]["n_eligible_total"],
                        "sha256_sample_ids": selection[vname][cname]["sha256_sample_ids"],
                        "raw_asr_untargeted": rate(evasion),
                        "valid_asr_untargeted": rate(evasion & domain_valid),
                        "targeted_benign": rate(targeted),
                        "valid_targeted_benign": rate(targeted & domain_valid),
                        "domain_validity_rate": rate(domain_valid),
                        "realizable_rate": rate(realizable),
                        "idr": rate(in_dist),
                        "mean_cost_total": float(cost["total"].mean().item()),
                        "mean_l2_scaled": float((x_adv - x0).reshape(n, -1).norm(dim=1).mean().item()),
                    }
                    if sem_pass is not None:
                        cell["semantic_pass_rate"] = rate(sem_pass)
                        cell["primitive_feasible_rate"] = rate(prim_feasible)
                        cell["sp_asr"] = rate(targeted & domain_valid & prim_feasible & sem_pass)
                    cells.append(cell)
            print(f"[{time.time()-t0:6.0f}s] {vname}/{cname}: {len(roster)} attacks x "
                  f"{len(seeds)} seeds done (n={n})", flush=True)

    (out / "cells.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")
    (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    # ---- runtime PAIRING ASSERTIONS (requirement 11) ----
    assert_pairing(out, victims, classes, seeds, roster, selection)
    print(f"[done] {len(cells)} cells, {len(failures)} failures; "
          f"pairing assertions PASSED; artifacts in {art}", flush=True)


def assert_pairing(out: Path, victims, classes, seeds, roster, selection):
    """Fail loudly if any two attacks on the same (victim,class,seed) used different rows,
    order, labels, clean predictions, or counts."""
    art = out / "artifacts"
    for vname in victims:
        for cname in classes:
            ref_sids = np.asarray(selection[vname][cname]["sample_ids"], dtype="U128")
            n_ref = len(ref_sids)
            ref_clean = None
            for seed in seeds:
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
