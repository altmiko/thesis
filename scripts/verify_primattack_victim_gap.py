"""Independent verification of the FINAL Exp A PrimAttack victim gap (MLP vs CNN).

Re-runs the capability-aware untargeted Prim-PGD attack (joint mode, p75 + unbounded budgets,
same hyperparameters as ``run_final_suite.py``) from scratch and re-derives every audit claim:

1. **Reproduction.** Re-attack each Exp A victim on its frozen ``selection.json`` rows for every
   attack seed and compare with the stored FINAL artifacts (success set, adversarial prediction,
   adversarial flow).
2. **Statistics.** Success-set identity across attack seeds, Wilson 95% CIs over flows, ASR
   conditional on primitive headroom, where the successes land (Benign vs another attack class),
   paired exact McNemar MLP vs CNN on shared flows.
3. **Training-seed replication (CICIDS2018).** Re-attack mlp/cnn x training seeds 42/123/2024 on
   the shared rows that are clean-correct for all six victims.
4. **Mechanism.** On shared flows with timing headroom: an optimizer-free delay x shape grid,
   the effect of the delay-allocation shape, leave-one-out restoration of each written feature,
   the fraction of PrimAttack's own successes that revert when only ``Fwd IAT Min`` is restored,
   ``Fwd IAT Min`` displacement vs the train distribution, and a 1-D ``Fwd IAT Min`` response
   curve (all other features clean).

Writes ``report.json`` and ``report.md`` under ``--output-dir``. Does not touch FINAL_OUTPUTS.

    PYTHONPATH=src python scripts/verify_primattack_victim_gap.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from attack.primattack_budget import (  # noqa: E402
    class_calibration, load_calibration, unbounded_calibration,
)
from attack.primitive_optimizer import (  # noqa: E402
    AttackObjective, hybrid_valid_gate, optimize_primitive_pgd,
)
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel  # noqa: E402
from datasets import get_adapter  # noqa: E402
from experiments.provenance import deterministic_runtime, ensure_fresh_output_dir  # noqa: E402
from run_full_adversarial_eval import DATASET_DEFAULTS, victim_checkpoint  # noqa: E402
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402

FINAL_RUNS = REPO_ROOT / "FINAL_OUTPUTS" / "runs"
CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")
CLASS_NAMES = ("Benign", "DoS", "DDoS", "Recon", "BruteForce")
BENIGN_ID = 0
BUDGETS = {"maximum-evaluated": "p75", "unbounded": "unb"}
# run_final_suite.PRIM_ARGS: eval budget 256, Prim-PGD 3 restarts, step 0.05, momentum 0.75.
EVAL_BUDGET, PGD_RESTARTS, PGD_STEP, PGD_MOMENTUM = 256, 3, 0.05, 0.75
PGD_STEPS = (EVAL_BUDGET - 1) // (2 * PGD_RESTARTS)
GRID_FRACS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
GRID_SHAPES = (0.0, 0.5, 1.0)
CURVE_VALUES = (1e1, 1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6)
KEY_FEATURE = "Fwd IAT Min"
EXP_A = {
    "cicids2017": ("mlp", "cnn", "ft_transformer"),
    "cicids2018": ("mlp-s42", "cnn-s42", "ft_transformer-s42"),
}
PAIR = {"cicids2017": ("mlp", "cnn"), "cicids2018": ("mlp-s42", "cnn-s42")}
SEED_REPLICATION = ("mlp-s42", "mlp-s123", "mlp-s2024", "cnn-s42", "cnn-s123", "cnn-s2024")


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    ci = binomtest(k, n).proportion_ci(method="wilson")
    return [float(ci.low), float(ci.high)]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    return float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0


class Dataset:
    """Frozen test rows, transform, primitive model, validator gate and victims of one dataset."""

    def __init__(self, cli: str, device: str):
        self.cli = cli
        self.adapter = get_adapter(cli)
        self.name = self.adapter.name
        self.device = device
        transform = self.adapter.feature_transform()
        self.center = torch.tensor(transform.center, dtype=torch.float32, device=device)
        self.scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)
        self.model = CICIDS2017PrimitiveModel(self.adapter.feature_manifest())
        self.names = list(self.model.feature_names)
        self.key = self.model.i[KEY_FEATURE]
        self.gate = hybrid_valid_gate(self.name)
        self.calibration = load_calibration(DATASET_DEFAULTS[self.name]["calibration"])
        processed = self.adapter._processed
        self.X = torch.tensor(np.load(processed / "X_test_pristine.npy"), dtype=torch.float32,
                              device=device)
        self.y = np.load(processed / "y_test_cat.npy").astype(np.int64)
        self._train = (np.load(processed / "X_train_pristine.npy", mmap_mode="r"),
                       np.load(processed / "y_train_cat.npy").astype(np.int64))
        self.selection = json.loads(
            (FINAL_RUNS / self.name / "baselines_untargeted" / "selection.json").read_text("utf-8"))
        self.class_id = {c: int(self.adapter.class_mapping().name_to_id[c]) for c in CLASSES}
        self._victims: dict[str, torch.nn.Module] = {}

    def victim(self, vname: str) -> torch.nn.Module:
        if vname not in self._victims:
            ckpt, arch, _ = victim_checkpoint(self.name, vname)
            self._victims[vname] = load_category_victim(
                ckpt, adapter=self.adapter, expected_model_type=arch, device=self.device)
        return self._victims[vname]

    def rows(self, idx: np.ndarray) -> torch.Tensor:
        return self.X[torch.as_tensor(idx, device=self.device)]

    def logits(self, vname: str, raw: torch.Tensor) -> torch.Tensor:
        victim = self.victim(vname)
        with torch.no_grad():
            return torch.cat([victim((raw[s:s + 32768] - self.center) / self.scale)
                              for s in range(0, len(raw), 32768)])

    def margin(self, vname: str, raw: torch.Tensor, cid: int) -> torch.Tensor:
        """Source logit - max other logit; negative iff the untargeted objective is met."""
        z = self.logits(vname, raw)
        other = z.clone()
        other[:, cid] = -torch.inf
        return z[:, cid] - other.amax(1)

    def bounds(self, raw: torch.Tensor, cname: str, budget: str, caps):
        cfg = (unbounded_calibration(self.calibration, cname) if budget == "unbounded"
               else class_calibration(self.calibration, cname, budget))
        return self.model.per_flow_bounds(raw, cfg.bounds_config(), capabilities=caps)

    def train_quantiles(self, class_id: int, j: int, qs) -> list[float]:
        X, y = self._train
        rows = np.flatnonzero(y == class_id)
        return np.quantile(np.asarray(X[rows, j], dtype=np.float64), qs).tolist()


def attack(ds: Dataset, vname: str, idx: np.ndarray, cname: str, budget: str, seed: int) -> dict:
    """One Prim-PGD cell exactly as the FINAL untargeted joint run configures it."""
    raw, cid = ds.rows(idx), ds.class_id[cname]
    caps = ds.model.infer_capabilities(raw)
    bounds = ds.bounds(raw, cname, budget, caps)
    deterministic_runtime(seed)
    res = optimize_primitive_pgd(
        ds.model, ds.victim(vname), raw, ds.center, ds.scale, bounds, caps,
        steps=PGD_STEPS, step_size=PGD_STEP, restarts=PGD_RESTARTS, momentum=PGD_MOMENTUM,
        seed=seed, validity_fn=ds.gate, eval_budget=EVAL_BUDGET,
        objective=AttackObjective("untargeted", cid))
    adv = res.adversarial_raw
    adv_pred = ds.logits(vname, adv).argmax(1).cpu().numpy()
    valid = ds.gate(adv, raw).cpu().numpy().astype(bool)
    valid_success = (adv_pred != cid) & valid
    return {
        "idx": idx, "adv_raw": adv.cpu().numpy(), "adv_pred": adv_pred, "valid": valid,
        "valid_success": valid_success,
        "optimizer_success_consistent": bool(np.array_equal(
            res.success.cpu().numpy().astype(bool), valid_success)),
        "movable": ((bounds["p"] >= 1) | (bounds["delay"] >= 1)).cpu().numpy(),
        "shape": res.projected["shape"].cpu().numpy(),
    }


def final_artifact(ds: Dataset, vname: str, cname: str, budget: str, seed: int):
    path = (FINAL_RUNS / ds.name / "primattack_untargeted" / "artifacts"
            / f"{vname}__{cname}__{BUDGETS[budget]}__pgd__seed{seed}.npz")
    return np.load(path) if path.exists() else None


# --------------------------------------------------------------------------------------------
def stage_reproduction(ds: Dataset, seeds, log) -> tuple[dict, dict]:
    """Stage 1: re-attack Exp A victims on the frozen rows; compare with FINAL artifacts."""
    results, checks = {}, []
    for vname in EXP_A[ds.cli]:
        for cname in CLASSES:
            idx = np.asarray(ds.selection[vname][cname]["positional_idx"], dtype=np.int64)
            for budget in BUDGETS:
                for seed in seeds:
                    t0 = time.perf_counter()
                    cell = attack(ds, vname, idx, cname, budget, seed)
                    results[(vname, cname, budget, seed)] = cell
                    ref = final_artifact(ds, vname, cname, budget, seed)
                    check = {"victim": vname, "class": cname, "budget": BUDGETS[budget],
                             "seed": seed, "n": int(len(idx)),
                             "rerun_valid_successes": int(cell["valid_success"].sum()),
                             "optimizer_success_consistent": cell["optimizer_success_consistent"]}
                    if ref is None:
                        check["final_artifact"] = "missing"
                    else:
                        check.update({
                            "final_valid_successes": int(ref["valid_success"].sum()),
                            "same_rows": bool(np.array_equal(ref["positional_idx"], idx)),
                            "same_success_set": bool(np.array_equal(
                                ref["valid_success"], cell["valid_success"])),
                            "same_adv_pred": bool(np.array_equal(ref["adv_pred"], cell["adv_pred"])),
                            "max_abs_adv_raw_diff": float(np.abs(
                                ref["adv_raw"].astype(np.float64) - cell["adv_raw"]).max()),
                        })
                    checks.append(check)
                    log(f"[repro] {ds.name} {vname}/{cname}/{BUDGETS[budget]}/s{seed}: rerun "
                        f"{check['rerun_valid_successes']} final {check.get('final_valid_successes')} "
                        f"same_set={check.get('same_success_set')} ({time.perf_counter() - t0:.1f}s)")
    return results, {"cells": checks,
                     "all_success_sets_match": all(c.get("same_success_set", False) for c in checks),
                     "all_adv_pred_match": all(c.get("same_adv_pred", False) for c in checks)}


def stage_statistics(ds: Dataset, results: dict, seeds) -> dict:
    """Stage 2: seed identity, Wilson CIs, headroom-conditional ASR, landing class, McNemar."""
    key = ds.cli
    out = {"victims": {}, "pair": {}}
    for vname in EXP_A[key]:
        for budget in BUDGETS:
            s0 = seeds[0]
            cells = [results[(vname, c, budget, s0)] for c in CLASSES]
            succ = np.concatenate([c["valid_success"] for c in cells])
            movable = np.concatenate([c["movable"] for c in cells])
            pred = np.concatenate([c["adv_pred"] for c in cells])
            n, k = int(len(succ)), int(succ.sum())
            landing = {CLASS_NAMES[int(p)]: int(m) for p, m in
                       zip(*np.unique(pred[succ], return_counts=True))}
            seed_identical = all(
                np.array_equal(results[(vname, c, budget, s0)]["valid_success"],
                               results[(vname, c, budget, s)]["valid_success"])
                for c in CLASSES for s in seeds[1:])
            benign = int(landing.get("Benign", 0))
            out["victims"][f"{vname}/{BUDGETS[budget]}"] = {
                "n": n, "valid_successes": k, "valid_asr": k / n, "wilson95": wilson(k, n),
                "success_sets_identical_across_seeds": seed_identical,
                "n_movable": int(movable.sum()),
                "valid_asr_given_headroom": k / max(int(movable.sum()), 1),
                "landing_class": landing,
                "benign_evasions": benign, "benign_evasion_rate": benign / n,
                "benign_wilson95": wilson(benign, n),
                "per_class": {c: int(results[(vname, c, budget, s0)]["valid_success"].sum())
                              for c in CLASSES},
            }
    a, b = PAIR[key]
    for budget in BUDGETS:
        only_a = only_b = both = shared_n = 0
        per_class = {}
        for cname in CLASSES:
            ra, rb = results[(a, cname, budget, seeds[0])], results[(b, cname, budget, seeds[0])]
            ia = {int(i): n for n, i in enumerate(ra["idx"])}
            ib = {int(i): n for n, i in enumerate(rb["idx"])}
            shared = sorted(set(ia) & set(ib))
            sa = ra["valid_success"][[ia[i] for i in shared]]
            sb = rb["valid_success"][[ib[i] for i in shared]]
            per_class[cname] = {"shared": len(shared), f"{a}_only": int((sa & ~sb).sum()),
                                f"{b}_only": int((~sa & sb).sum()), "both": int((sa & sb).sum())}
            only_a += int((sa & ~sb).sum()); only_b += int((~sa & sb).sum())
            both += int((sa & sb).sum()); shared_n += len(shared)
        out["pair"][BUDGETS[budget]] = {
            "shared_flows": shared_n, f"{a}_only": only_a, f"{b}_only": only_b, "both": both,
            "mcnemar_exact_p": mcnemar_exact(only_a, only_b), "per_class": per_class}
    return out


def stage_seed_replication(ds: Dataset, seed: int, log) -> dict:
    """Stage 3 (CICIDS2018): all six mlp/cnn training seeds on common clean-correct rows."""
    out = {}
    for cname in CLASSES:
        cid = ds.class_id[cname]
        base = [np.asarray(ds.selection[v][cname]["positional_idx"], np.int64)
                for v in ("mlp-s42", "cnn-s42")]
        idx = np.intersect1d(base[0], base[1])
        raw = ds.rows(idx)
        keep = np.ones(len(idx), bool)
        for v in SEED_REPLICATION:
            keep &= ds.margin(v, raw, cid).cpu().numpy() > 0
        idx = idx[keep]
        out[cname] = {"n_common_clean_correct": int(len(idx))}
        for v in SEED_REPLICATION:
            for budget in BUDGETS:
                cell = attack(ds, v, idx, cname, budget, seed)
                k = int(cell["valid_success"].sum())
                benign = int((cell["valid_success"] & (cell["adv_pred"] == BENIGN_ID)).sum())
                out[cname][f"{v}/{BUDGETS[budget]}"] = {
                    "valid_successes": k, "benign_evasions": benign,
                    "n_movable": int(cell["movable"].sum()),
                    "valid_asr_given_headroom": k / max(int(cell["movable"].sum()), 1)}
                log(f"[train-seed] {cname} {v}/{BUDGETS[budget]}: {k}/{len(idx)} "
                    f"(benign {benign})")
    return out


def stage_mechanism(ds: Dataset, victims, results: dict | None, seed: int, classes, log) -> dict:
    """Stage 4: optimizer-free grid, shape effect, leave-one-out, Fwd IAT Min extrapolation."""
    out = {}
    for cname in classes:
        cid = ds.class_id[cname]
        sets = [np.asarray(ds.selection[v][cname]["positional_idx"], np.int64)
                for v in victims if v in ds.selection]
        idx = sets[0]
        for s in sets[1:]:
            idx = np.intersect1d(idx, s)
        raw = ds.rows(idx)
        keep = torch.ones(len(idx), dtype=torch.bool, device=ds.device)
        for v in victims:
            keep &= ds.margin(v, raw, cid) > 0
        caps = ds.model.infer_capabilities(raw)
        b = ds.bounds(raw, cname, "maximum-evaluated", caps)
        use = keep & caps.timing_allowed & (b["delay"] >= 1)
        r, dhi, idx_u = raw[use], b["delay"][use], idx[use.cpu().numpy()]
        n = len(r)
        row = {"n_shared_timing_headroom": int(n)}
        if n == 0:
            out[cname] = row
            continue
        sub_caps = ds.model.infer_capabilities(r)

        def realize(frac: float, shape: float) -> torch.Tensor:
            ctrl = {"p": torch.zeros(n, device=ds.device), "delay": torch.floor(frac * dhi),
                    "shape": torch.full((n,), shape, device=ds.device)}
            return ds.model.generate(r, ctrl, quantize=True, capabilities=sub_caps)

        grid = [realize(f, s) for f in GRID_FRACS for s in GRID_SHAPES]
        stacked = torch.cat(grid)
        valid = ds.gate(stacked, r.repeat(len(grid), 1)).reshape(len(grid), n)
        full = {s: grid[(len(GRID_FRACS) - 1) * len(GRID_SHAPES) + k]
                for k, s in enumerate(GRID_SHAPES)}
        changed = [j for j in range(r.shape[1]) if bool((full[1.0][:, j] != r[:, j]).any())]
        own_p99 = ds.train_quantiles(cid, ds.key, [0.99])[0]
        ben_q = ds.train_quantiles(BENIGN_ID, ds.key, [0.5, 0.99])
        row["grid_valid_rate"] = float(valid.float().mean())
        row["fwd_iat_min"] = {
            "clean_median": float(r[:, ds.key].median()),
            "adv_median_full_p75_shape1": float(full[1.0][:, ds.key].median()),
            "train_own_class_p99": own_p99, "train_benign_p50": ben_q[0],
            "train_benign_p99": ben_q[1],
        }
        disp = (torch.asinh((full[1.0] - ds.center) / ds.scale)
                - torch.asinh((r - ds.center) / ds.scale)).abs().median(0).values
        row["median_abs_displacement_asinh_scaled"] = {ds.names[j]: float(disp[j]) for j in changed}
        for v in victims:
            m0 = ds.margin(v, r, cid)
            mg = ds.margin(v, stacked, cid).reshape(len(grid), n)
            flips = (mg < 0) & valid
            vr = {
                "clean_margin_median": float(m0.median()),
                "grid_valid_flip_rate": float(flips.any(0).float().mean()),
                "shape_full_p75": {
                    str(s): {"flip_rate": float((ds.margin(v, full[s], cid) < 0).float().mean()),
                             "median_margin_drop": float((m0 - ds.margin(v, full[s], cid)).median())}
                    for s in GRID_SHAPES},
            }
            a1 = full[1.0]
            m1 = ds.margin(v, a1, cid)
            loo = {}
            for j in changed:
                x = a1.clone(); x[:, j] = r[:, j]
                loo[ds.names[j]] = float((ds.margin(v, x, cid) - m1).median())
            vr["leave_one_out_margin_recovery_shape1"] = loo
            vr["median_margin_drop_shape1"] = float((m0 - m1).median())
            curve = {}
            for val in CURVE_VALUES:
                x = r.clone()
                x[:, ds.key] = torch.clamp(x[:, ds.key], min=val)
                mm = ds.margin(v, x, cid)
                curve[f"{val:.0e}"] = {"median_margin": float(mm.median()),
                                       "flip_rate": float((mm < 0).float().mean())}
            vr["fwd_iat_min_only_response"] = curve
            if results is not None and (v, cname, "maximum-evaluated", seed) in results:
                cell = results[(v, cname, "maximum-evaluated", seed)]
                pos = {int(i): k for k, i in enumerate(cell["idx"])}
                rows = [pos[int(i)] for i in idx_u]
                succ = cell["valid_success"][rows]
                vr["primattack_on_these_rows"] = {
                    "valid_successes": int(succ.sum()), "rate": float(succ.mean())}
                all_succ = np.flatnonzero(cell["valid_success"])
                if len(all_succ):
                    adv = torch.tensor(cell["adv_raw"][all_succ], device=ds.device)
                    src = ds.rows(cell["idx"][all_succ])
                    restored = adv.clone(); restored[:, ds.key] = src[:, ds.key]
                    only_key = src.clone(); only_key[:, ds.key] = adv[:, ds.key]
                    vr["primattack_successes"] = {
                        "n": int(len(all_succ)),
                        "revert_when_only_fwd_iat_min_restored": float(
                            (ds.margin(v, restored, cid) > 0).float().mean()),
                        "still_evade_with_only_fwd_iat_min_changed": float(
                            (ds.margin(v, only_key, cid) < 0).float().mean()),
                        "success_shape_quantiles_p10_p50_p90": np.quantile(
                            cell["shape"][all_succ], [0.1, 0.5, 0.9]).tolist(),
                    }
            row[v] = vr
        out[cname] = row
        log(f"[mechanism] {ds.name} {cname}: n={n} " + " ".join(
            f"{v}: grid={row[v]['grid_valid_flip_rate']:.3f}"
            + (f"/prim={row[v]['primattack_on_these_rows']['rate']:.3f}"
               if "primattack_on_these_rows" in row[v] else "") for v in victims))
    return out


# --------------------------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def render_markdown(report: dict) -> str:
    L = ["# PrimAttack victim-gap verification", "",
         f"Generated by `scripts/verify_primattack_victim_gap.py` ({report['device']}, "
         f"{report['elapsed_seconds']:.0f}s). Attack: untargeted capability-aware Prim-PGD, joint, "
         f"{EVAL_BUDGET} evals/flow, {PGD_RESTARTS} restarts x {PGD_STEPS} steps.", ""]
    for ds, block in report["datasets"].items():
        L += [f"## {ds}", "", "### 1. Reproduction of FINAL artifacts", "",
              f"- success sets match in every cell: **{block['reproduction']['all_success_sets_match']}**",
              f"- adversarial predictions match in every cell: **{block['reproduction']['all_adv_pred_match']}**",
              "", "| victim | class | budget | seed | rerun | FINAL | same set | max abs Δ adv_raw |",
              "|---|---|---|---|---|---|---|---|"]
        for c in block["reproduction"]["cells"]:
            L.append(f"| {c['victim']} | {c['class']} | {c['budget']} | {c['seed']} | "
                     f"{c['rerun_valid_successes']} | {c.get('final_valid_successes', '—')} | "
                     f"{c.get('same_success_set', '—')} | {c.get('max_abs_adv_raw_diff', float('nan')):.3g} |")
        L += ["", "### 2. Statistics (first attack seed)", "",
              "| victim/budget | Valid ASR | Wilson 95% | identical across seeds | ASR given headroom "
              "| landing class | Benign evasion (Wilson 95%) |", "|---|---|---|---|---|---|---|"]
        for k, s in block["statistics"]["victims"].items():
            L.append(f"| {k} | {s['valid_successes']}/{s['n']} = {pct(s['valid_asr'])} | "
                     f"[{pct(s['wilson95'][0])}, {pct(s['wilson95'][1])}] | "
                     f"{s['success_sets_identical_across_seeds']} | {pct(s['valid_asr_given_headroom'])} "
                     f"({s['n_movable']} movable) | {s['landing_class']} | {pct(s['benign_evasion_rate'])} "
                     f"[{pct(s['benign_wilson95'][0])}, {pct(s['benign_wilson95'][1])}] |")
        L += ["", "Paired MLP vs CNN on shared flows (exact McNemar):", ""]
        for bud, p in block["statistics"]["pair"].items():
            L.append(f"- {bud}: {json.dumps({k: v for k, v in p.items() if k != 'per_class'})}; "
                     f"per class {json.dumps(p['per_class'])}")
        if "training_seed_replication" in block:
            L += ["", "### 3. Training-seed replication (common clean-correct rows, attack seed "
                  f"{report['seeds'][0]})", "",
                  "| class | n | " + " | ".join(SEED_REPLICATION) + " |",
                  "|---|---|" + "---|" * len(SEED_REPLICATION)]
            for cname, row in block["training_seed_replication"].items():
                for bud in BUDGETS.values():
                    L.append(f"| {cname} {bud} | {row['n_common_clean_correct']} | " + " | ".join(
                        f"{row[f'{v}/{bud}']['valid_successes']} (B {row[f'{v}/{bud}']['benign_evasions']}; "
                        f"{pct(row[f'{v}/{bud}']['valid_asr_given_headroom'])} of movable)"
                        for v in SEED_REPLICATION) + " |")
        L += ["", "### 4. Mechanism (p75 box, shared flows with timing headroom)", ""]
        for cname, row in block["mechanism"].items():
            if row["n_shared_timing_headroom"] == 0:
                continue
            f = row["fwd_iat_min"]
            L += [f"**{cname}** (n={row['n_shared_timing_headroom']}; grid validity "
                  f"{pct(row['grid_valid_rate'])}). `{KEY_FEATURE}` median {f['clean_median']:.4g} -> "
                  f"{f['adv_median_full_p75_shape1']:.4g} µs; train own-class p99 "
                  f"{f['train_own_class_p99']:.4g}, Benign p50/p99 {f['train_benign_p50']:.4g}/"
                  f"{f['train_benign_p99']:.4g}. Median |Δ asinh(scaled)|: " + ", ".join(
                      f"{k}={v:.2f}" for k, v in row["median_abs_displacement_asinh_scaled"].items()),
                  "", "| victim | clean margin | grid flip | PrimAttack same rows | flip s=0/0.5/1 | "
                  "drop s=1 | LOO recovery Fwd IAT Min | successes revert (only Fwd IAT Min restored) "
                  "| still evade (only Fwd IAT Min changed) | success shape p10/p50/p90 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            victims = [k for k in row if isinstance(row[k], dict) and "clean_margin_median" in row[k]]
            for v in victims:
                x = row[v]
                pa = x.get("primattack_on_these_rows")
                ps = x.get("primattack_successes")
                L.append(
                    f"| {v} | {x['clean_margin_median']:.1f} | {pct(x['grid_valid_flip_rate'])} | "
                    f"{pct(pa['rate']) if pa else '—'} | " + "/".join(
                        pct(x['shape_full_p75'][str(s)]['flip_rate']) for s in GRID_SHAPES) +
                    f" | {x['median_margin_drop_shape1']:.2f} | "
                    f"{x['leave_one_out_margin_recovery_shape1'].get(KEY_FEATURE, float('nan')):.2f} | "
                    + (f"{pct(ps['revert_when_only_fwd_iat_min_restored'])} of {ps['n']} | "
                       f"{pct(ps['still_evade_with_only_fwd_iat_min_changed'])} | "
                       + "/".join(f"{q:.2f}" for q in ps["success_shape_quantiles_p10_p50_p90"])
                       if ps else "— | — | —") + " |")
            L += ["", f"`{KEY_FEATURE}`-only response (median margin, flip rate):", "",
                  "| victim | " + " | ".join(f"{v:.0e}" for v in CURVE_VALUES) + " |",
                  "|---|" + "---|" * len(CURVE_VALUES)]
            for v in victims:
                cur = row[v]["fwd_iat_min_only_response"]
                L.append(f"| {v} | " + " | ".join(
                    f"{cur[f'{c:.0e}']['median_margin']:.1f} ({cur[f'{c:.0e}']['flip_rate']:.2f})"
                    for c in CURVE_VALUES) + " |")
            L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--datasets", default="cicids2017,cicids2018")
    ap.add_argument("--seeds", default="42,2024,2026", help="attack seeds (FINAL protocol)")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "audit" / "primattack_victim_gap")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = ensure_fresh_output_dir(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    report = {"device": args.device, "seeds": seeds, "torch": torch.__version__, "datasets": {}}
    for cli in args.datasets.split(","):
        ds = Dataset(cli, args.device)
        results, repro = stage_reproduction(ds, seeds, log)
        block = {"reproduction": repro, "statistics": stage_statistics(ds, results, seeds)}
        if cli == "cicids2018":
            block["training_seed_replication"] = stage_seed_replication(ds, seeds[0], log)
            mech_victims = SEED_REPLICATION
        else:
            mech_victims = EXP_A[cli]
        block["mechanism"] = stage_mechanism(ds, mech_victims, results, seeds[0], CLASSES, log)
        report["datasets"][ds.name] = block
        del results
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    report["elapsed_seconds"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    log(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
