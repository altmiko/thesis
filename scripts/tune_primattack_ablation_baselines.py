"""Validation-split hyperparameter selection for the Prim-PGD / Prim-C&W ablation baselines.

Runs ``run_primattack_optimizer_ablation.py --split val`` over a small grid (Prim-PGD step
size; Prim-C&W initial trade-off ``c`` x Adam learning rate) with the same per-flow evaluation
budget, victims, classes and budgets as the test run, then selects per method the setting with
the highest pooled valid ASR (tie-break: lower median normalized cost). Hybrid is not tuned:
it keeps the canonical PrimAttack settings. Test data is never read.

Writes ``<out>/<config>/`` run directories and ``<out>/selection.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/run_primattack_optimizer_ablation.py"
GRID = {
    "pgd": [{"pgd-step": s} for s in (0.02, 0.05, 0.1, 0.2)],
    "cw": [{"cw-c": c, "cw-lr": lr} for c in (1.0, 10.0, 100.0)
           for lr in (0.02, 0.05, 0.1, 0.2, 0.5)],
}


def tag(method: str, params: dict) -> str:
    return method + "_" + "_".join(f"{k}{v:g}" for k, v in params.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="cicids2017")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-budget", type=int, default=256)
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "outputs/primattack_optimizer_ablation/val_tuning")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for method, grid in GRID.items():
        for params in grid:
            run_dir = args.output_dir / tag(method, params)
            if not (run_dir / "cells.json").exists():
                cmd = [sys.executable, str(RUNNER), "--dataset", args.dataset,
                       "--device", args.device, "--split", "val",
                       "--n-per-class", str(args.n_per_class), "--seeds", str(args.seed),
                       "--methods", method, "--eval-budget", str(args.eval_budget),
                       "--output-dir", str(run_dir), "--resume"]
                for k, v in params.items():
                    cmd += [f"--{k}", str(v)]
                subprocess.run(cmd, check=True)
            cells = pd.DataFrame(json.loads((run_dir / "cells.json").read_text(encoding="utf-8")))
            successes, n = int(cells.successes.sum()), int(cells.n.sum())
            costs = []
            for npz in (run_dir / "artifacts").glob("*.npz"):
                with np.load(npz) as d:
                    costs.append(d["normalized_cost"][d["final_success"]])
            costs = np.concatenate(costs) if costs else np.array([])
            summary.append({"method": method, "params": params, "tag": tag(method, params),
                            "pooled_valid_asr": successes / n, "successes": successes, "n": n,
                            "median_cost": float(np.median(costs)) if costs.size else float("nan")})
            print(json.dumps(summary[-1]), flush=True)

    chosen = {}
    for method in GRID:
        rows = [r for r in summary if r["method"] == method]
        rows.sort(key=lambda r: (-r["pooled_valid_asr"], r["median_cost"]))
        chosen[method] = rows[0]
    (args.output_dir / "selection.json").write_text(
        json.dumps({"criterion": "max pooled valid ASR on val (tie: lower median cost)",
                    "chosen": chosen, "grid": summary}, indent=2), encoding="utf-8")
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
