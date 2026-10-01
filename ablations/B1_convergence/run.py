"""B1 - Convergence: more refinement steps and more restarts; ASR vs per-flow queries.

Design source: Carlini et al., On Evaluating Adversarial Robustness (2019, Sec. 4.8: doubling the
iterations must not raise success; plot success against iterations), Tramer et al. (NeurIPS
2020, Sec. 5: 40 -> 400 PGD steps exposed non-convergence), and the query-budget curves of
Amoeba (CoNEXT 2023, Fig. 7) and NetMasquerade (NDSS 2026, Fig. 10).

Arms (steps per restart, per-flow evaluation budget; restarts continue until the budget is
spent, so the restart count follows from the two):

* reference          - 40 steps, 256 evaluations (about three refinement restarts);
* ``steps80_eval512``  - 2x steps and 2x budget (same number of restarts, longer each);
* ``steps160_eval1024``- 4x steps and 4x budget;
* ``steps40_eval512``  - 2x budget spent on more 40-step restarts;
* ``steps40_eval1024`` - 4x budget spent on more 40-step restarts.

The anytime curve reads ``first_success_evaluation`` (the evaluation index of each flow's first
validator-passing success) from the 4x-budget arms: ASR(k) = share of flows with a success
within their first k victim evaluations.

    python ablations/B1_convergence/run.py --device cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ablations.common.analysis import (  # noqa: E402
    Source, _groups, _seeds, append_report, pooled_rows, write_standard_report,
)
from ablations.common.cli import experiment_main  # noqa: E402
from ablations.common.hybrid import HybridConfig  # noqa: E402
from ablations.common.runner import REFERENCE, Condition  # noqa: E402

CONDITIONS = [
    Condition("steps80_eval512", "2x steps, 2x per-flow budget",
              HybridConfig(steps=80, eval_budget=512)),
    Condition("steps160_eval1024", "4x steps, 4x per-flow budget",
              HybridConfig(steps=160, eval_budget=1024)),
    Condition("steps40_eval512", "2x per-flow budget, more restarts",
              HybridConfig(eval_budget=512)),
    Condition("steps40_eval1024", "4x per-flow budget, more restarts",
              HybridConfig(eval_budget=1024)),
]
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
CURVE_ARMS = (REFERENCE.name, "steps40_eval1024", "steps160_eval1024")


def anytime(results_dir: Path, reference_dir: Path) -> list[str]:
    rows = []
    for g in _groups(results_dir, reference_dir, [c.name for c in CONDITIONS]):
        for arm in CURVE_ARMS:
            src = Source(reference_dir if arm == REFERENCE.name else results_dir, arm)
            seeds = sorted(_seeds(src.results_dir, g, arm))
            parts = [pooled_rows(src, g.dataset, g.victim, g.budget_label, s,
                                 ("valid_success", "first_success_evaluation")) for s in seeds]
            parts = [p for p in parts if p is not None]
            if not parts:
                continue
            curve = {}
            for k in CHECKPOINTS:
                curve[k] = float(np.mean([
                    ((p["first_success_evaluation"] > 0) & (p["first_success_evaluation"] <= k)
                     & p["valid_success"].astype(bool)).mean() for p in parts]))
            final = float(np.mean([p["valid_success"].mean() for p in parts]))
            rows.append({"dataset": g.dataset, "victim": g.victim, "budget": g.budget_label,
                         "arm": arm, **{f"asr_at_{k}": v for k, v in curve.items()},
                         "final_asr": final})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "anytime_curve.csv", index=False)
    if df.empty:
        return []
    head = " | ".join(f"≤{k}" for k in CHECKPOINTS)
    lines = ["## Anytime curve: Valid ASR within the first k victim evaluations per flow "
             "(mean over seeds)", "",
             f"| dataset | victim | budget | arm | {head} | final |",
             "|---|---|---|---|" + "---|" * len(CHECKPOINTS) + "---|"]
    for r in df.to_dict("records"):
        vals = " | ".join(f"{100 * r[f'asr_at_{k}']:.2f}" for k in CHECKPOINTS)
        lines.append(f"| {r['dataset']} | {r['victim']} | {r['budget']} | {r['arm']} | {vals} | "
                     f"{100 * r['final_asr']:.2f} |")
    lines.append("")
    lines.append("Values in %. The reference spends at most 256 evaluations, so its curve is flat "
                 "after 256.")
    return lines + [""]


def analyze(results_dir: Path, reference_dir: Path) -> None:
    write_standard_report(results_dir, reference_dir, [c.name for c in CONDITIONS],
                          title="B1 - Convergence (steps x restarts x query budget)")
    append_report(results_dir, anytime(results_dir, reference_dir))


if __name__ == "__main__":
    experiment_main(EXP_DIR, CONDITIONS, analyze, __doc__)
