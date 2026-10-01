"""Exp A PrimAttack row re-run over more attack seeds; attack-seed variance of raw / valid ASR.

Exactly the FINAL suite's Exp A PrimAttack configuration (stage ``primattack_untargeted``):
selected optimizer from ``FINAL_OUTPUTS/runs/optimizer_selection.json``, untargeted, joint mode,
p75 (``maximum-evaluated``) budget, validator_v2 gate, locked ``PRIM_ARGS``, the frozen canonical
source flows and the same victims -- only the attack-seed list is longer. The canonical runner
``scripts/run_primattack_optimizer_ablation.py`` is invoked unchanged (subprocess, same env as
``run_final_suite.py``). Victims are fixed, so this measures attack-seed variance only, not
victim-training-seed variance.

Outputs (non-canonical, outside ``FINAL_OUTPUTS``): ``outputs/expA_primattack_multiseed/<dataset>/``
(runner artifacts + ``cells.json``) and ``outputs/expA_primattack_multiseed/variance.{json,md}``.

    python scripts/run_expA_primattack_multiseed.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_final_suite as final  # noqa: E402

OUT_ROOT = REPO_ROOT / "outputs" / "expA_primattack_multiseed"
BUDGET = "maximum-evaluated"
DEFAULT_SEEDS = "42,2024,2026,0,1,7,123,999,31337,271828"


def run_dataset(dataset: str, device: str, method: str, seeds: str) -> None:
    spec = final.DATASETS[dataset]
    out = OUT_ROOT / dataset
    final._run([sys.executable, "scripts/run_primattack_optimizer_ablation.py",
                "--dataset", spec["cli"], "--device", device,
                "--selection-from", str(final.selection_path(dataset)),
                "--split", "test", "--seeds", seeds, "--objective", "untargeted",
                "--victims", spec["victims"], "--classes", final.CLASSES,
                "--budgets", BUDGET, "--methods", method, "--modes", "joint",
                *final.PRIM_ARGS, "--output-dir", str(out), "--resume"],
               OUT_ROOT / "logs" / f"{dataset}.log")


def _spread(values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "variance": float(v.var(ddof=1)),
            "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}


def analyze(datasets: list[str], method: str, seeds: list[int]) -> dict:
    result = {}
    for dataset in datasets:
        out = OUT_ROOT / dataset
        cells = [c for c in json.loads((out / "cells.json").read_text(encoding="utf-8"))
                 if c["budget"] == BUDGET and c["method"] == method and c["mode"] == "joint"
                 and c["seed"] in seeds]
        pooled = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # victim -> seed -> n,raw,valid
        per_class = defaultdict(lambda: defaultdict(dict))  # victim -> class -> seed -> (raw,valid)
        for c in cells:
            t = pooled[c["victim"]][c["seed"]]
            t[0] += c["n"]; t[1] += c["raw_successes"]; t[2] += c["successes"]
            per_class[c["victim"]][c["class"]][c["seed"]] = (c["asr_raw"], c["asr_valid"])
        ds = {}
        for victim, by_seed in pooled.items():
            missing = set(seeds) - set(by_seed)
            if missing:
                raise KeyError(f"{dataset}/{victim}: missing seeds {sorted(missing)}")
            if len({t[0] for t in by_seed.values()}) != 1:
                raise AssertionError(f"{dataset}/{victim}: denominator differs across seeds")
            # Row-level: flows whose raw / valid outcome is not the same under every seed.
            flips_raw = flips_valid = 0
            n_rows = 0
            for cname in per_class[victim]:
                raw_m, val_m = [], []
                for seed in seeds:
                    name = f"{victim}__{cname}__p75__{method}__seed{seed}.npz"
                    with np.load(out / "artifacts" / name, allow_pickle=True) as d:
                        raw_m.append(d["raw_success"].astype(bool))
                        val_m.append(d["valid_success"].astype(bool))
                raw_m, val_m = np.stack(raw_m), np.stack(val_m)
                flips_raw += int((raw_m.any(0) & ~raw_m.all(0)).sum())
                flips_valid += int((val_m.any(0) & ~val_m.all(0)).sum())
                n_rows += raw_m.shape[1]
            ds[victim] = {
                "n_flows": n_rows, "n_seeds": len(seeds),
                "raw_asr": _spread([by_seed[s][1] / by_seed[s][0] for s in seeds]),
                "valid_asr": _spread([by_seed[s][2] / by_seed[s][0] for s in seeds]),
                "flows_raw_outcome_varies": flips_raw,
                "flows_valid_outcome_varies": flips_valid,
                "per_seed": {str(s): {"raw_asr": by_seed[s][1] / by_seed[s][0],
                                      "valid_asr": by_seed[s][2] / by_seed[s][0]}
                             for s in seeds},
                "per_class": {
                    cname: {"raw_asr": _spread([v[s][0] for s in seeds]),
                            "valid_asr": _spread([v[s][1] for s in seeds])}
                    for cname, v in per_class[victim].items()},
            }
        result[dataset] = ds
    return result


def write_report(res: dict, method: str, seeds: list[int]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "variance.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pct = lambda v: f"{100 * v:.3f}%"  # noqa: E731
    L = ["# Exp A PrimAttack: attack-seed variance of raw and valid ASR", "",
         f"FINAL Exp A PrimAttack configuration (optimizer `{method}`, untargeted, joint, p75, "
         f"validator_v2 gate, locked PRIM_ARGS, frozen canonical flows); {len(seeds)} attack seeds "
         f"({', '.join(map(str, seeds))}). ASR pooled over classes {final.CLASSES} per seed "
         "(successes / attempted flows). Variance/SD: sample (ddof=1) over attack seeds, in "
         "ASR fraction units (variance) and percentage points (SD). Victims fixed: attack-seed "
         "variance only.", "",
         "| Dataset | Victim | Raw ASR mean | Raw var | Raw SD (pp) | Raw min–max | "
         "Valid ASR mean | Valid var | Valid SD (pp) | Valid min–max | Flows with seed-dependent "
         "raw / valid outcome |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for dataset, ds in res.items():
        for victim, r in ds.items():
            a, b = r["raw_asr"], r["valid_asr"]
            L.append(f"| {dataset} | {victim} | {pct(a['mean'])} | {a['variance']:.3e} | "
                     f"{100 * a['sd']:.4f} | {pct(a['min'])}–{pct(a['max'])} | {pct(b['mean'])} | "
                     f"{b['variance']:.3e} | {100 * b['sd']:.4f} | {pct(b['min'])}–{pct(b['max'])} | "
                     f"{r['flows_raw_outcome_varies']} / {r['flows_valid_outcome_varies']} "
                     f"of {r['n_flows']} |")
    L += ["", "Non-canonical re-run; feature-space proxy only (CLAUDE.md claim boundary). "
          "Attack-seed variance is not seed robustness across victims.", ""]
    (OUT_ROOT / "variance.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default=",".join(final.DATASETS))
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        raise ValueError("variance needs at least two attack seeds")
    method = json.loads((final.RUNS / "optimizer_selection.json").read_text("utf-8"))["selected"]
    if not args.analyze_only:
        for dataset in datasets:
            run_dataset(dataset, args.device, method, args.seeds)
    write_report(analyze(datasets, method, seeds), method, seeds)


if __name__ == "__main__":
    main()
