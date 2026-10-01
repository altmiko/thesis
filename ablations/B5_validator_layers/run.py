"""B5 - Remove one validator_v2 layer at a time and measure Valid ASR.

Design source: Carlini et al. (2019, Sec. 5.2-5.3: relax the constraints until the attack must
succeed, so that a low success rate is attributable to the constraints and not to the optimizer)
and Sheatsley et al. (JCS 2022, Fig. 4: success as features become uncontrollable).

validator_v2 ``hybrid_valid`` = SCHEMA ∧ EXTRACTOR ∧ PROTOCOL ∧ MINED. Each arm drops exactly
one layer from the validity definition, consistently in the search success predicate AND in the
measured validity:

* ``drop_schema``    - EXTRACTOR ∧ PROTOCOL ∧ MINED;
* ``drop_extractor`` - SCHEMA ∧ PROTOCOL ∧ MINED;
* ``drop_protocol``  - SCHEMA ∧ EXTRACTOR ∧ MINED (incl. source-conditioned ``PROTO_0080``);
* ``drop_mined``     - SCHEMA ∧ EXTRACTOR ∧ PROTOCOL (= ``hard_structural_valid``);
* reference          - all four layers.

Primary outcome: Valid ASR under the arm's reduced validator (``gate_success``). The same flows
are also scored by the full four-layer validator_v2 (``valid_success``), which shows how many of
the extra successes the dropped layer was blocking.

    python ablations/B5_validator_layers/run.py --device cuda
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
from ablations.common.runner import LAYERS, Condition  # noqa: E402

CONDITIONS = [
    Condition(f"drop_{layer}", f"validator_v2 without the {layer.upper()} layer",
              gate_layers=tuple(x for x in LAYERS if x != layer))
    for layer in LAYERS
]


def dropped_layer_audit(results_dir: Path, reference_dir: Path) -> list[str]:
    """Among each arm's reduced-validator successes: how many fail the dropped layer."""
    rows = []
    fields = ("gate_success", "valid_success", "raw_success") + tuple(f"{x}_valid" for x in LAYERS)
    for g in _groups(results_dir, reference_dir, [c.name for c in CONDITIONS]):
        for cond in CONDITIONS:
            layer = cond.name.removeprefix("drop_")
            src = Source(results_dir, cond.name)
            parts = [pooled_rows(src, g.dataset, g.victim, g.budget_label, s, fields)
                     for s in sorted(_seeds(results_dir, g, cond.name))]
            parts = [p for p in parts if p is not None]
            if not parts:
                continue
            r = {k: np.concatenate([p[k] for p in parts]) for k in fields}
            gs = r["gate_success"].astype(bool)
            fail_dropped = gs & ~r[f"{layer}_valid"].astype(bool)
            other_fail = {x: int((gs & ~r[f"{x}_valid"].astype(bool)).sum())
                          for x in LAYERS if x != layer}
            rows.append({"dataset": g.dataset, "victim": g.victim, "budget": g.budget_label,
                         "condition": cond.name, "flows": int(len(gs)),
                         "reduced_valid_successes": int(gs.sum()),
                         "full_valid_successes": int(r["valid_success"].sum()),
                         "fail_dropped_layer": int(fail_dropped.sum()),
                         **{f"fail_{x}": v for x, v in other_fail.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "dropped_layer_audit.csv", index=False)
    if df.empty:
        return []
    lines = ["## What the dropped layer was blocking (all seeds and classes pooled)", "",
             "`reduced` = successes valid under the arm's 3-layer validator; `full` = the same "
             "flows that also pass the full validator_v2; `fail dropped` = reduced successes "
             "rejected by the dropped layer (= reduced - full).", "",
             "| dataset | victim | budget | condition | flows | reduced | full | fail dropped |",
             "|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        lines.append(f"| {r.dataset} | {r.victim} | {r.budget} | {r.condition} | {r.flows} | "
                     f"{r.reduced_valid_successes} | {r.full_valid_successes} | "
                     f"{r.fail_dropped_layer} |")
    return lines + [""]


def analyze(results_dir: Path, reference_dir: Path) -> None:
    write_standard_report(
        results_dir, reference_dir, [c.name for c in CONDITIONS],
        title="B5 - validator_v2 layer removal",
        outcome="gate_success", outcome_label="Valid ASR (arm's validator)",
        secondary={"valid_success": "Valid ASR (full validator_v2)", "raw_success": "Raw ASR"})
    append_report(results_dir, dropped_layer_audit(results_dir, reference_dir))


if __name__ == "__main__":
    experiment_main(EXP_DIR, CONDITIONS, analyze, __doc__)
