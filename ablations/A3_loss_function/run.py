"""A3 - Refinement loss: logit margin vs cross-entropy vs DLR, with zero-gradient counts.

Design source: Croce & Hein, AutoAttack (ICML 2020): CE vs CW-margin vs DLR losses (Sec. 4.2,
Tables 9-11) and the fraction of exactly-zero gradients as a gradient-masking signal (Fig. 2);
Pintor et al., Indicators of Attack Failure (I1, unavailable gradients).

Only the loss that drives the refinement gradient changes. The success predicate, the
incumbent ordering (failures ranked by the objective margin), the step-size adaptation (keyed on
the realized margin) and the padding enumeration are the reference's:

* reference   - untargeted margin ``z_y - max_{i != y} z_i``;
* ``loss_ce`` - ``log softmax(z)_y`` (minimizing it maximizes the source-class cross-entropy);
* ``loss_dlr``- ``(z_y - max_{i != y} z_i) / (z_pi1 - z_pi3)`` (scale-invariant DLR).

Every arm records, per refinement step, whether the gradient was exactly zero on all free
coordinates (``zero_gradient_steps`` / ``gradient_steps``).

    python ablations/A3_loss_function/run.py --device cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))

import pandas as pd  # noqa: E402

from ablations.common.analysis import (  # noqa: E402
    BUDGET_ORDER, append_report, load_cells, success_profile, write_standard_report,
)
from ablations.common.cli import experiment_main  # noqa: E402
from ablations.common.hybrid import HybridConfig  # noqa: E402
from ablations.common.runner import REFERENCE, Condition  # noqa: E402

CONDITIONS = [
    Condition("loss_ce", "cross-entropy refinement loss", HybridConfig(loss="ce")),
    Condition("loss_dlr", "DLR refinement loss", HybridConfig(loss="dlr")),
]


def zero_gradients(results_dir: Path, reference_dir: Path) -> list[str]:
    cells = pd.concat([load_cells(reference_dir), load_cells(results_dir)], ignore_index=True)
    names = [REFERENCE.name] + [c.name for c in CONDITIONS]
    ours = cells[cells["condition"].isin([c.name for c in CONDITIONS])]
    groups = ours[["dataset", "victim", "budget_label"]].drop_duplicates()
    cells = cells.merge(groups, on=["dataset", "victim", "budget_label"])
    cells = cells[cells["condition"].isin(names)]
    agg = (cells.groupby(["dataset", "victim", "budget_label", "condition"], sort=False)
           [["gradient_steps", "zero_gradient_steps"]].sum().reset_index())
    agg["zero_gradient_fraction"] = agg["zero_gradient_steps"] / agg["gradient_steps"].where(
        agg["gradient_steps"] > 0)
    agg["_b"] = agg["budget_label"].map(BUDGET_ORDER)
    agg["_c"] = agg["condition"].map({n: i for i, n in enumerate(names)})
    agg = agg.sort_values(["dataset", "victim", "_b", "_c"]).drop(columns=["_b", "_c"])
    agg.to_csv(results_dir / "zero_gradients.csv", index=False)
    lines = ["## Zero gradients during refinement (all seeds and classes)", "",
             "A step counts as zero when the loss gradient is exactly 0 on every free control "
             "coordinate of the row.", "",
             "| dataset | victim | budget | condition | gradient steps | zero-gradient steps | "
             "fraction |", "|---|---|---|---|---|---|---|"]
    for r in agg.itertuples(index=False):
        frac = "n/a" if r.zero_gradient_fraction != r.zero_gradient_fraction else \
            f"{r.zero_gradient_fraction:.4f}"
        lines.append(f"| {r.dataset} | {r.victim} | {r.budget_label} | {r.condition} | "
                     f"{r.gradient_steps} | {r.zero_gradient_steps} | {frac} |")
    return lines + [""]


def analyze(results_dir: Path, reference_dir: Path) -> None:
    names = [c.name for c in CONDITIONS]
    write_standard_report(results_dir, reference_dir, names,
                          title="A3 - Refinement loss (margin vs CE vs DLR)")
    append_report(results_dir, zero_gradients(results_dir, reference_dir)
                  + success_profile(results_dir, reference_dir, names))


if __name__ == "__main__":
    experiment_main(EXP_DIR, CONDITIONS, analyze, __doc__)
