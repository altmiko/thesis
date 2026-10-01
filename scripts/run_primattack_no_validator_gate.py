"""PrimAttack WITHOUT validator_v2 in the search success predicate (validity-gate ablation).

The FINAL suite's PrimAttack (``scripts/run_primattack_optimizer_ablation.py``) injects
validator_v2 ``hybrid_valid`` as the search's validity gate: a candidate only counts as a success
(and only wins the per-flow incumbent) if the victim objective is met AND the realized flow is
``hybrid_valid`` given its source. The baselines never query the validator. This script re-runs
the Exp A PrimAttack configuration (stage ``primattack_untargeted``: selected optimizer,
untargeted, joint mode, p75 + unbounded, same frozen flows, victims, seeds and hyperparameters)
with the gate removed (``validity_fn=None``), so the search keeps the cheapest objective-meeting
candidate and validator_v2 is applied only afterwards -- the same post-hoc treatment the
baselines get. Everything else is the canonical runner, executed in-process, unchanged.

Outputs (never inside ``FINAL_OUTPUTS``; this is a non-canonical ablation):
``outputs/primattack_no_validator_gate/<dataset>/`` (runner artifacts, ``cells.json``) and
``outputs/primattack_no_validator_gate/summary.{json,md}`` (gated vs ungated, paired per row).

    python scripts/run_primattack_no_validator_gate.py --device cuda
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_final_suite as final  # noqa: E402

OUT_ROOT = REPO_ROOT / "outputs" / "primattack_no_validator_gate"
STAGE = "primattack_untargeted"
BUDGETS = ("maximum-evaluated", "unbounded")
NO_GATE_SUCCESS = ("victim argmax != true source class on the realized flow "
                   "(validator_v2 NOT in the search; applied post hoc only)")


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_primattack_optimizer_ablation",
        REPO_ROOT / "scripts" / "run_primattack_optimizer_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The runner builds its gate via this module-level factory; None = no validity gate
    # (RealizedSearch then treats every realized candidate as valid during the search).
    mod.hybrid_valid_gate = lambda dataset: None
    return mod


def run_dataset(runner, dataset: str, device: str, method: str) -> Path:
    spec = final.DATASETS[dataset]
    out = OUT_ROOT / dataset
    sys.argv = [
        "run_primattack_optimizer_ablation.py",
        "--dataset", spec["cli"], "--device", device,
        "--selection-from", str(final.selection_path(dataset)),
        "--split", "test", "--seeds", final.SEEDS, "--objective", "untargeted",
        "--victims", spec["victims"], "--classes", final.CLASSES,
        "--budgets", ",".join(BUDGETS), "--methods", method, "--modes", "joint",
        *final.PRIM_ARGS, "--output-dir", str(out), "--resume",
    ]
    runner.main()
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["success"] = NO_GATE_SUCCESS
    cfg["validity_gate"] = "none (ablation of the FINAL suite's hybrid_valid gate)"
    cfg["gated_reference"] = str(final.RUNS / dataset / STAGE)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def _rows(art: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(art / name, allow_pickle=True) as d:
        return {k: d[k] for k in ("sample_id", "raw_success", "validator_pass",
                                  "valid_success", "primitive_feasible", "semantic_pass")}


def summarize(datasets: list[str], method: str) -> dict:
    result = {}
    for dataset in datasets:
        gated_dir = final.RUNS / dataset / STAGE
        free_dir = OUT_ROOT / dataset
        free_cells = json.loads((free_dir / "cells.json").read_text(encoding="utf-8"))
        gated_cells = {(c["victim"], c["class"], c["budget"], c["seed"], c["method"]): c
                       for c in json.loads((gated_dir / "cells.json").read_text("utf-8"))}
        # (victim, budget, seed) -> pooled-over-classes counters
        acc = defaultdict(lambda: defaultdict(int))
        for c in free_cells:
            if c["method"] != method or c["mode"] != "joint":
                continue
            key = (c["victim"], c["class"], c["budget"], c["seed"], c["method"])
            if key not in gated_cells:
                raise KeyError(f"no gated reference cell for {key}")
            name = f"{c['victim']}__{c['class']}__{c['budget_label']}__{method}__seed{c['seed']}.npz"
            g = _rows(gated_dir / "artifacts", name)
            f = _rows(free_dir / "artifacts", name)
            if not np.array_equal(g["sample_id"], f["sample_id"]):
                raise AssertionError(f"row mismatch gated vs ungated {key}")
            gv, fv = g["valid_success"].astype(bool), f["valid_success"].astype(bool)
            a = acc[(c["victim"], c["budget"], c["seed"])]
            a["n"] += len(gv)
            a["gated_raw"] += int(g["raw_success"].astype(bool).sum())
            a["gated_valid"] += int(gv.sum())
            a["free_raw"] += int(f["raw_success"].astype(bool).sum())
            a["free_valid"] += int(fv.sum())
            a["free_raw_invalid"] += int((f["raw_success"].astype(bool) & ~fv).sum())
            a["valid_gated_only"] += int((gv & ~fv).sum())
            a["valid_free_only"] += int((fv & ~gv).sum())
            a["gated_sp"] += int((gv & g["primitive_feasible"].astype(bool)
                                  & g["semantic_pass"].astype(bool)).sum())
            a["free_sp"] += int((fv & f["primitive_feasible"].astype(bool)
                                 & f["semantic_pass"].astype(bool)).sum())
        per = defaultdict(list)
        for (victim, budget, seed), a in sorted(acc.items()):
            n = a["n"]
            per[(victim, budget)].append({
                "seed": seed, "n": n,
                "gated_raw_asr": a["gated_raw"] / n, "gated_valid_asr": a["gated_valid"] / n,
                "free_raw_asr": a["free_raw"] / n, "free_valid_asr": a["free_valid"] / n,
                "free_raw_but_invalid": a["free_raw_invalid"] / n,
                "gated_sp_asr": a["gated_sp"] / n, "free_sp_asr": a["free_sp"] / n,
                "valid_gated_only": a["valid_gated_only"],
                "valid_free_only": a["valid_free_only"],
            })
        result[dataset] = {
            f"{victim}|{budget}": {
                "per_seed": seeds,
                "mean": {k: float(np.mean([s[k] for s in seeds]))
                         for k in seeds[0] if k not in ("seed",)},
            }
            for (victim, budget), seeds in per.items()
        }
    return result


def write_report(summary: dict, method: str) -> None:
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pct = lambda v: f"{100 * v:.2f}%"  # noqa: E731
    L = ["# PrimAttack with vs without validator_v2 in the search success predicate", "",
         f"Configuration: FINAL stage `{STAGE}` (optimizer `{method}`, untargeted, joint mode, "
         f"seeds {final.SEEDS}, classes {final.CLASSES}); only the search's validity gate "
         "differs. Gated = FINAL_OUTPUTS reference (validator_v2 `hybrid_valid` in the success "
         "predicate). Ungated = `validity_fn=None`; validator_v2 applied post hoc, like the "
         "baselines. Rates pooled over classes per seed (successes / attempted flows), then "
         "mean over attack seeds. Row-level columns are summed over seeds.", "",
         "| Dataset | Victim | Budget | Raw ASR gated | Raw ASR ungated | Valid ASR gated | "
         "Valid ASR ungated | Ungated raw-but-invalid | SP-ASR gated | SP-ASR ungated | "
         "Valid only gated (rows) | Valid only ungated (rows) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for dataset, cells in summary.items():
        for key, cell in cells.items():
            victim, budget = key.split("|")
            m = cell["mean"]
            L.append(f"| {dataset} | {victim} | {final_budget(budget)} | {pct(m['gated_raw_asr'])} | "
                     f"{pct(m['free_raw_asr'])} | {pct(m['gated_valid_asr'])} | "
                     f"{pct(m['free_valid_asr'])} | {pct(m['free_raw_but_invalid'])} | "
                     f"{pct(m['gated_sp_asr'])} | {pct(m['free_sp_asr'])} | "
                     f"{sum(s['valid_gated_only'] for s in cell['per_seed'])} | "
                     f"{sum(s['valid_free_only'] for s in cell['per_seed'])} |")
    L += ["", "Non-canonical ablation; feature-space proxy only (see CLAUDE.md claim boundary).", ""]
    (OUT_ROOT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def final_budget(budget: str) -> str:
    return {"maximum-evaluated": "p75", "unbounded": "unbounded"}.get(budget, budget)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default=",".join(final.DATASETS))
    ap.add_argument("--summary-only", action="store_true",
                    help="skip the runs; rebuild summary from existing artifacts")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    selection = json.loads((final.RUNS / "optimizer_selection.json").read_text("utf-8"))
    method = selection["selected"]
    if not args.summary_only:
        runner = _load_runner()
        for dataset in datasets:
            run_dataset(runner, dataset, args.device, method)
    write_report(summarize(datasets, method), method)


if __name__ == "__main__":
    main()
