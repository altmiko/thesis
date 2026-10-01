"""Shared reference arm of every PrimAttack ablation.

The canonical Hybrid Search (FINAL-suite ``PRIM_ARGS``: 40 steps, lr 0.1, restarts until the
256-evaluation budget is spent; validator_v2 ``hybrid_valid`` in the success predicate;
capability-aware primitives), untargeted, joint, p75 + unbounded, both datasets, three victims,
four classes, attack seeds 42/2024/2026, frozen 800 clean-correct flows per cell.

It is run once here and read by every experiment's analysis. The analysis below proves that the
ablation harness (``ablations/common``) reproduces the FINAL suite: every p75 cell must match
``FINAL_OUTPUTS/runs/<dataset>/primattack_hybrid_objective_untargeted`` flow-for-flow (same
realized adversarial flow, same victim prediction, same validity, same success).

    python ablations/reference/run.py --device cuda
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))

import numpy as np  # noqa: E402

from ablations.common.cli import experiment_main  # noqa: E402
from ablations.common.runner import FINAL_RUNS, REFERENCE  # noqa: E402

FINAL_STAGE = "primattack_hybrid_objective_untargeted"


def verify_against_final(results_dir: Path, _reference: Path) -> None:
    rows, lines = [], ["# Reference reproduction check", "",
                       f"Each p75 reference cell vs `FINAL_OUTPUTS/runs/<dataset>/{FINAL_STAGE}` "
                       "(same frozen flows, victim, seed and configuration).", "",
                       "| dataset | cells compared | flows | adv flow identical | prediction identical "
                       "| valid identical | success identical |", "|---|---|---|---|---|---|---|"]
    for ds_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        final_art = FINAL_RUNS / ds_dir.name / FINAL_STAGE / "artifacts"
        cells = flows = same_adv = same_pred = same_valid = same_success = 0
        for npz in sorted((ds_dir / "artifacts").glob("*__p75__reference__seed*.npz")):
            victim, cname, _, _, seed = npz.stem.split("__")
            final = final_art / f"{victim}__{cname}__p75__hybrid__{seed}.npz"
            if not final.exists():
                continue
            with np.load(npz, allow_pickle=True) as a, np.load(final, allow_pickle=True) as b:
                n = len(a["sample_id"])
                if not np.array_equal(a["sample_id"], b["sample_id"][:n]):
                    raise AssertionError(f"{npz.name}: sample order differs from FINAL")
                adv = np.all(a["adv_raw"] == b["adv_raw"][:n], axis=1)
                pred = a["adv_pred"] == b["adv_pred"][:n]
                valid = a["validator_pass"] == b["validator_pass"][:n]
                succ = a["valid_success"] == b["valid_success"][:n]
            cells += 1
            flows += n
            same_adv += int(adv.sum()); same_pred += int(pred.sum())
            same_valid += int(valid.sum()); same_success += int(succ.sum())
            rows.append({"dataset": ds_dir.name, "artifact": npz.name, "flows": n,
                         "adv_identical": int(adv.sum()), "success_identical": int(succ.sum())})
        lines.append(f"| {ds_dir.name} | {cells} | {flows} | {same_adv} | {same_pred} | "
                     f"{same_valid} | {same_success} |")
    (results_dir / "reproduction_check.json").write_text(json.dumps(rows, indent=2),
                                                          encoding="utf-8")
    (results_dir / "reproduction_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    experiment_main(EXP_DIR, [REFERENCE], verify_against_final, __doc__)
