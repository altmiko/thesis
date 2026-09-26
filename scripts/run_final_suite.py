"""Run the FINAL thesis experiment suite (master_experiments.md) into ``FINAL_OUTPUTS/runs``.

Locked design (see ``FINAL_OUTPUTS/00_PROTOCOL.md``):

* datasets cicids2017 + cicids2018; one frozen category victim per architecture
  (2017: the canonical checkpoints; 2018: the training-seed-42 replicates);
* attack seeds 42, 2024, 2026 for every run; classes DoS, DDoS, Recon, BruteForce;
* one canonical clean-correct source-sample list per (dataset, victim, class): 800 rows,
  seeded uniform rule (selection seed 42), produced ONCE by the baseline stage and reused
  (sha256-verified) by every PrimAttack stage.

Stages (each resumable; a finished cell is never recomputed):

1. ``baselines_untargeted``  (Exp A): PGD, C&W, CAPGD-PrimSupport, C-PGD-PrimSupport.
2. ``primattack_targeted_optimizers`` (Exp B): Hybrid / Prim-PGD / Prim-C&W, targeted->Benign,
   p75 budget.
3. optimizer selection (pre-registered rule, both datasets pooled) -> ``optimizer_selection.json``.
4. ``primattack_targeted_budgets`` (Exp C): the two top-ranked optimizers at p50 and unbounded
   (their p75 cells are the stage-2 cells: same rows, seeds, configuration).
5. ``primattack_untargeted`` (Exp A PrimAttack row + Exp D untargeted arm): selected optimizer,
   untargeted, p75 (+ unbounded, descriptive p75-vs-unbounded comparison).

    python scripts/run_final_suite.py            # everything
    python scripts/run_final_suite.py --stages baselines,optimizers
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL = REPO_ROOT / "FINAL_OUTPUTS"
RUNS = FINAL / "runs"

SEEDS = "42,2024,2026"
CLASSES = "DoS,DDoS,Recon,BruteForce"
N_PER_CLASS = 800
SELECTION_SEED = 42
DATASETS = {
    "cicids2017_distrinet": {"cli": "cicids2017", "victims": "mlp,cnn,ft_transformer"},
    "cicids2018_distrinet": {"cli": "cicids2018",
                             "victims": "mlp-s42,cnn-s42,ft_transformer-s42"},
}
BASELINE_ATTACKS = "pgd_untargeted,cw_untargeted,capgd_prim_support,cpgd_prim_support"
OPTIMIZERS = ("hybrid", "pgd", "cw")
EVAL_BUDGET = 256
# Locked hyperparameters (identical to the frozen, validation-tuned optimizer ablation).
BASELINE_ARGS = [
    "--pgd-epsilon", "0.5", "--pgd-alpha", "0.05", "--pgd-steps", "40",
    "--cw-lambda", "1.0", "--cw-kappa", "0.0", "--cw-iters", "60", "--cw-lr", "0.01",
    "--cw-conv", "1e-5",
    "--capgd-norm", "L2", "--capgd-epsilon", "0.5", "--capgd-steps", "10",
    "--capgd-batch-size", "64",
    "--cpgd-epsilon", "0.5", "--cpgd-norm", "L2", "--cpgd-step-size", "0.05",
    "--cpgd-iterations", "40", "--cpgd-constraint-weight", "1.0", "--cpgd-batch-size", "64",
]
PRIM_ARGS = [
    "--eval-budget", str(EVAL_BUDGET), "--hybrid-steps", "40", "--hybrid-lr", "0.1",
    "--pgd-restarts", "3", "--pgd-step", "0.05", "--pgd-momentum", "0.75",
    "--cw-stages", "3", "--cw-lr", "0.5", "--cw-c", "1.0", "--cw-kappa", "0.0",
]


def selection_path(dataset: str) -> Path:
    return RUNS / dataset / "baselines_untargeted" / "selection.json"


def _run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"[run] {' '.join(cmd)}\n      log -> {log}", flush=True)
    t0 = time.time()
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"stage failed (exit {proc.returncode}); see {log}")
    print(f"[ok] {time.time() - t0:.0f}s", flush=True)


def run_baselines(dataset: str, device: str) -> None:
    spec = DATASETS[dataset]
    out = RUNS / dataset / "baselines_untargeted"
    _run([sys.executable, "scripts/run_full_adversarial_eval.py", "--dataset", spec["cli"],
          "--device", device, "--seeds", SEEDS, "--n-per-class", str(N_PER_CLASS),
          "--selection", "random", "--selection-seed", str(SELECTION_SEED),
          "--victims", spec["victims"], "--classes", CLASSES,
          "--families", "input,capgd,cpgd", "--attacks", BASELINE_ATTACKS,
          *BASELINE_ARGS, "--output-dir", str(out), "--resume"],
         RUNS / dataset / "logs" / "baselines_untargeted.log")


def run_primattack(dataset: str, device: str, stage: str, *, objective: str,
                   methods: list[str], budgets: list[str]) -> None:
    spec = DATASETS[dataset]
    sel = selection_path(dataset)
    if not sel.exists():
        raise FileNotFoundError(f"{sel} missing: run the baselines stage first (it freezes the "
                                "canonical source-sample list)")
    _run([sys.executable, "scripts/run_primattack_optimizer_ablation.py",
          "--dataset", spec["cli"], "--device", device, "--selection-from", str(sel),
          "--split", "test", "--seeds", SEEDS, "--objective", objective,
          "--victims", spec["victims"], "--classes", CLASSES,
          "--budgets", ",".join(budgets), "--methods", ",".join(methods), *PRIM_ARGS,
          "--output-dir", str(RUNS / dataset / stage), "--resume"],
         RUNS / dataset / "logs" / f"{stage}.log")


def select_optimizer() -> dict:
    """Pre-registered rule: highest aggregate Valid Targeted ASR at p75, pooled over both
    datasets, all victims, classes and seeds (total valid successes / total attempts).
    Ties: fewer mean victim evaluations per flow, then the fixed order hybrid, pgd, cw."""
    totals = {m: {"successes": 0, "attempts": 0, "evals": 0.0} for m in OPTIMIZERS}
    expected = None
    for dataset in DATASETS:
        cells = json.loads((RUNS / dataset / "primattack_targeted_optimizers" / "cells.json")
                           .read_text(encoding="utf-8"))
        keys = {}
        for c in cells:
            if c["budget"] != "maximum-evaluated" or c["objective"] != "targeted":
                raise ValueError(f"unexpected optimizer-selection cell {c}")
            keys.setdefault(c["method"], set()).add((c["victim"], c["class"], c["seed"]))
            t = totals[c["method"]]
            t["successes"] += int(c["successes"])
            t["attempts"] += int(c["n"])
            t["evals"] += float(c["evals_mean"]) * int(c["n"])
        n_expected = (len(DATASETS[dataset]["victims"].split(",")) * len(CLASSES.split(","))
                      * len(SEEDS.split(",")))
        for m in OPTIMIZERS:
            if len(keys.get(m, ())) != n_expected:
                raise ValueError(f"{dataset}: optimizer {m} has {len(keys.get(m, ()))} of "
                                 f"{n_expected} cells; selection needs complete runs")
        if expected is None:
            expected = {m: keys[m] for m in OPTIMIZERS}
    attempts = {totals[m]["attempts"] for m in OPTIMIZERS}
    if len(attempts) != 1:
        raise ValueError(f"optimizers have different denominators: {attempts}")
    ranking = sorted(
        OPTIMIZERS,
        key=lambda m: (-totals[m]["successes"] / totals[m]["attempts"],
                       totals[m]["evals"] / totals[m]["attempts"], OPTIMIZERS.index(m)))
    result = {
        "rule": "highest aggregate Valid Targeted ASR at p75 pooled over both datasets, all "
                "victims, classes and seeds (sum successes / sum attempts); ties -> fewer mean "
                "victim evaluations per flow -> order hybrid, pgd, cw",
        "aggregate": {m: {"valid_targeted_successes": totals[m]["successes"],
                          "attempts": totals[m]["attempts"],
                          "aggregate_valid_targeted_asr":
                              totals[m]["successes"] / totals[m]["attempts"],
                          "mean_evaluations_per_flow": totals[m]["evals"] / totals[m]["attempts"]}
                      for m in OPTIMIZERS},
        "ranking": ranking,
        "selected": ranking[0],
        "budget_sensitivity_optimizers": ranking[:2],
    }
    (RUNS / "optimizer_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[selection] ranking {ranking}; selected {ranking[0]}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--stages", default="baselines,optimizers,select,budgets,untargeted")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "final_suite_config.json").write_text(json.dumps({
        "seeds": SEEDS, "classes": CLASSES, "n_per_class": N_PER_CLASS,
        "selection": {"rule": "random", "seed": SELECTION_SEED,
                      "owner": "baselines_untargeted/selection.json"},
        "datasets": DATASETS, "baseline_attacks": BASELINE_ATTACKS,
        "baseline_args": BASELINE_ARGS, "primattack_args": PRIM_ARGS,
        "primattack_mode": "joint", "python": sys.version.split()[0],
    }, indent=2), encoding="utf-8")

    if "baselines" in stages:
        for d in datasets:
            run_baselines(d, args.device)
    if "optimizers" in stages:
        for d in datasets:
            run_primattack(d, args.device, "primattack_targeted_optimizers",
                           objective="targeted", methods=list(OPTIMIZERS),
                           budgets=["maximum-evaluated"])
    selection = None
    if "select" in stages:
        selection = select_optimizer()
    if {"budgets", "untargeted"} & set(stages):
        selection = selection or json.loads(
            (RUNS / "optimizer_selection.json").read_text(encoding="utf-8"))
    if "budgets" in stages:
        for d in datasets:
            run_primattack(d, args.device, "primattack_targeted_budgets", objective="targeted",
                           methods=selection["budget_sensitivity_optimizers"],
                           budgets=["intermediate", "unbounded"])
    if "untargeted" in stages:
        for d in datasets:
            run_primattack(d, args.device, "primattack_untargeted", objective="untargeted",
                           methods=[selection["selected"]],
                           budgets=["maximum-evaluated", "unbounded"])


if __name__ == "__main__":
    main()
