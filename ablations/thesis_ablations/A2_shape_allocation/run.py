"""A2 - Delay allocation: fixed ``shape`` vs learned ``shape``.

Design source: Nasr et al. (USENIX Security 2021), which sweep the mean and the spread of the
added delay separately (Sec. 7.2, Table 3), and FRONT (USENIX Security 2020, Sec. 5.4), which
moves the same padding budget in time. Here the total added forward delay and its budget stay
as in the reference; only HOW it is allocated over the forward gaps changes:

* ``shape_fixed_0``   - proportional dilation of the existing gaps (g' = a g);
* ``shape_fixed_0p5`` - equal mixture of proportional and uniform allocation;
* ``shape_fixed_1``   - uniform additive delay on every gap (g' = g + b);
* reference           - ``shape`` optimized jointly with ``p`` and ``delay``.

A pinned shape is applied to every candidate (clean start, random restarts) and receives no
gradient step; the surrogate floor is not applied to it.

    python ablations/thesis_ablations/A2_shape_allocation/run.py --device cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[2]))

import numpy as np  # noqa: E402

from ablations.common.analysis import (  # noqa: E402
    Source, _groups, _seeds, append_report, pooled_rows, success_profile, write_standard_report,
)
from ablations.common.cli import experiment_main  # noqa: E402
from ablations.common.hybrid import HybridConfig  # noqa: E402
from ablations.common.runner import REFERENCE, Condition  # noqa: E402

CONDITIONS = [
    Condition("shape_fixed_0", "shape pinned to 0 (proportional gap dilation)",
              HybridConfig(fixed_shape=0.0)),
    Condition("shape_fixed_0p5", "shape pinned to 0.5", HybridConfig(fixed_shape=0.5)),
    Condition("shape_fixed_1", "shape pinned to 1 (uniform per-gap delay)",
              HybridConfig(fixed_shape=1.0)),
]


def learned_shape(results_dir: Path, reference_dir: Path) -> list[str]:
    """Distribution of the optimized shape among the reference's timing successes."""
    lines = ["## Learned shape among the reference's timing successes (all seeds)", "",
             "| dataset | victim | budget | timing successes | shape median | shape < 0.05 | "
             "shape > 0.95 | 0.05-0.95 |", "|---|---|---|---|---|---|---|---|"]
    for g in _groups(results_dir, reference_dir, [c.name for c in CONDITIONS]):
        src = Source(reference_dir, REFERENCE.name)
        parts = [pooled_rows(src, g.dataset, g.victim, g.budget_label, s,
                             ("valid_success", "delay", "shape"))
                 for s in sorted(_seeds(reference_dir, g, REFERENCE.name))]
        parts = [p for p in parts if p is not None]
        if not parts:
            continue
        r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        m = r["valid_success"].astype(bool) & (r["delay"] > 0)
        sh = r["shape"][m]
        if not m.any():
            lines.append(f"| {g.dataset} | {g.victim} | {g.budget_label} | 0 | n/a | n/a | n/a "
                         "| n/a |")
            continue
        lines.append(f"| {g.dataset} | {g.victim} | {g.budget_label} | {int(m.sum())} | "
                     f"{np.median(sh):.3f} | {(sh < 0.05).mean():.3f} | {(sh > 0.95).mean():.3f} | "
                     f"{((sh >= 0.05) & (sh <= 0.95)).mean():.3f} |")
    return lines + [""]


def analyze(results_dir: Path, reference_dir: Path) -> None:
    names = [c.name for c in CONDITIONS]
    write_standard_report(results_dir, reference_dir, names,
                          title="A2 - Fixed vs learned delay allocation (shape)")
    append_report(results_dir, learned_shape(results_dir, reference_dir)
                  + success_profile(results_dir, reference_dir, names))


if __name__ == "__main__":
    experiment_main(EXP_DIR, CONDITIONS, analyze, __doc__)
