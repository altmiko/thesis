r"""D6 - Remove source-dependent capability inference.

Full (reference): the controls are masked by the source flow's capability,

    u = M(x) ⊙ (p, D, s),

where ``M(x)`` (``CICIDS2017PrimitiveModel.infer_capabilities``) permits padding only for flows
with forward packets, forward payload and no zero-length forward packet
(``Fwd Packet Length Min > 0``), and timing only for flows with at least two forward packets and
a positive ``Fwd IAT Total``. The mask is applied in the per-flow box, the projection and the
canonical map.

Ablated (``no_capability``): ``u = (p, D, s)``. ``M(x) = 1`` for every flow, so the box is the
numeric train-envelope/budget headroom alone and the canonical map applies padding / delay to any
flow (e.g. bytes into an empty forward packet, delay on a single-packet flow).

Both arms use the same search (Hybrid, validator_v2 ``hybrid_valid`` in the success predicate)
and are then scored by the SAME final validator_v2 (all four layers, source-conditioned). The
analysis splits the ablated arm's valid successes by whether they used a primitive the source
flow's capability forbids, and reports which validator layer (if any) rejected such candidates.

    python ablations/D6_capability_inference/run.py --device cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ablations.common.analysis import (  # noqa: E402
    Source, _groups, _seeds, append_report, load_cells, pooled_rows, success_profile,
    write_standard_report,
)
from ablations.common.cli import experiment_main  # noqa: E402
from ablations.common.runner import LAYERS, Condition  # noqa: E402

CONDITIONS = [
    Condition("no_capability", "capability mask removed: M(x)=1 for every flow",
              capability_aware=False),
]
FIELDS = ("valid_success", "raw_success", "primitive_feasible", "semantic_pass",
          "pad_capability_violation", "timing_capability_violation", "empty_fwd_packet_filled",
          "pad_reason", "timing_reason", "p", "delay") + tuple(f"{x}_valid" for x in LAYERS)


def capability_breakdown(results_dir: Path, reference_dir: Path) -> list[str]:
    rows, reasons = [], []
    for g in _groups(results_dir, reference_dir, [c.name for c in CONDITIONS]):
        src = Source(results_dir, "no_capability")
        parts = [pooled_rows(src, g.dataset, g.victim, g.budget_label, s, FIELDS)
                 for s in sorted(_seeds(results_dir, g, "no_capability"))]
        parts = [p for p in parts if p is not None]
        if not parts:
            continue
        r = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
        vs = r["valid_success"].astype(bool)
        hit = r["raw_success"].astype(bool)
        pv = r["pad_capability_violation"].astype(bool)
        tv = r["timing_capability_violation"].astype(bool)
        viol = pv | tv
        feasible = r["primitive_feasible"].astype(bool)
        rows.append({
            "dataset": g.dataset, "victim": g.victim, "budget": g.budget_label,
            "flows": int(len(vs)), "valid_successes": int(vs.sum()),
            "valid_success_capability_violation": int((vs & viol).sum()),
            "valid_success_pad_violation": int((vs & pv).sum()),
            "valid_success_timing_violation": int((vs & tv).sum()),
            "valid_success_empty_packet_filled": int((vs & r["empty_fwd_packet_filled"]).sum()),
            "valid_success_within_capability": int((vs & ~viol).sum()),
            "valid_and_primitive_feasible": int((vs & feasible).sum()),
            "violating_hits": int((hit & viol).sum()),
            **{f"violating_hits_fail_{x}": int((hit & viol & ~r[f"{x}_valid"].astype(bool)).sum())
               for x in LAYERS},
        })
        for kind, mask, col in (("padding", pv, "pad_reason"), ("timing", tv, "timing_reason")):
            for reason in np.unique(r[col][vs & mask]):
                m = vs & mask & (r[col] == reason)
                reasons.append({"dataset": g.dataset, "victim": g.victim,
                                "budget": g.budget_label, "primitive": kind,
                                "source_reason": str(reason), "valid_successes": int(m.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "capability_breakdown.csv", index=False)
    pd.DataFrame(reasons).to_csv(results_dir / "capability_violation_reasons.csv", index=False)
    if df.empty:
        return []
    lines = ["## Ablated arm: valid successes that a capability-aware attack could not produce "
             "(all seeds pooled)", "",
             "`violating` = the realized controls use padding on a flow without padding capability "
             "or delay on a flow without timing capability. `feasible` = valid success that also "
             "passes the primitive-feasibility/realizability check.", "",
             "| dataset | victim | budget | flows | valid successes | violating | pad viol. | "
             "timing viol. | empty fwd packet filled | within capability | feasible |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        lines.append(f"| {r.dataset} | {r.victim} | {r.budget} | {r.flows} | {r.valid_successes} "
                     f"| {r.valid_success_capability_violation} | {r.valid_success_pad_violation} "
                     f"| {r.valid_success_timing_violation} | "
                     f"{r.valid_success_empty_packet_filled} | "
                     f"{r.valid_success_within_capability} | {r.valid_and_primitive_feasible} |")
    lines += ["", "### Which validator layer rejects capability-violating objective hits", "",
              "| dataset | victim | budget | violating hits | " + " | ".join(
                  f"fail {x}" for x in LAYERS) + " |", "|---|---|---|---|" + "---|" * len(LAYERS)]
    for r in df.to_dict("records"):
        lines.append(f"| {r['dataset']} | {r['victim']} | {r['budget']} | {r['violating_hits']} | "
                     + " | ".join(str(r[f"violating_hits_fail_{x}"]) for x in LAYERS) + " |")
    if reasons:
        lines += ["", "### Source capability reason of the violating valid successes", "",
                  "| dataset | victim | budget | primitive | source reason | valid successes |",
                  "|---|---|---|---|---|---|"]
        for r in reasons:
            lines.append(f"| {r['dataset']} | {r['victim']} | {r['budget']} | {r['primitive']} | "
                         f"{r['source_reason']} | {r['valid_successes']} |")
    return lines + [""]


def budget_allocation(results_dir: Path, reference_dir: Path) -> list[str]:
    """Did the per-flow evaluation budget still reach the timing refinement?

    Over flows whose source permits timing (true capability, delay headroom >= 1 µs): the
    share that received zero refinement gradient steps (their 256 evaluations were spent on
    the padding enumeration first) and their Valid ASR, per arm.
    """
    fields = ("valid_success", "timing_allowed", "delay_hi", "gradient_steps",
              "total_evaluations", "pad_allowed", "p_hi")
    lines = ["## Where the evaluation budget went (flows whose source permits timing)", "",
             "| dataset | victim | budget | arm | timing-capable flows | with padding headroom | "
             "never refined | Valid ASR on these flows |", "|---|---|---|---|---|---|---|---|"]
    rows = []
    for g in _groups(results_dir, reference_dir, [c.name for c in CONDITIONS]):
        for arm, base in (("reference", reference_dir), ("no_capability", results_dir)):
            src = Source(base, arm)
            parts = [pooled_rows(src, g.dataset, g.victim, g.budget_label, s, fields)
                     for s in sorted(_seeds(base, g, arm))]
            parts = [p for p in parts if p is not None]
            if not parts:
                continue
            r = {k: np.concatenate([p[k] for p in parts]) for k in fields}
            m = r["timing_allowed"].astype(bool) & (r["delay_hi"] >= 1.0)
            if not m.any():
                continue
            rows.append({"dataset": g.dataset, "victim": g.victim, "budget": g.budget_label,
                         "arm": arm, "timing_capable": int(m.sum()),
                         "with_padding_headroom": int((m & (r["p_hi"] >= 1.0)).sum()),
                         "never_refined": int((m & (r["gradient_steps"] == 0)).sum()),
                         "valid_asr": float(r["valid_success"][m].mean())})
            x = rows[-1]
            lines.append(f"| {g.dataset} | {g.victim} | {g.budget_label} | {arm} | "
                         f"{x['timing_capable']} | {x['with_padding_headroom']} | "
                         f"{x['never_refined']} | {100 * x['valid_asr']:.2f}% |")
    pd.DataFrame(rows).to_csv(results_dir / "budget_allocation.csv", index=False)
    lines += ["", "`never refined` counts flows that reached no refinement gradient step: either "
              "already solved before refinement (identity / padding enumeration) or their "
              "evaluation budget was used up by the padding enumeration. Without capability "
              "inference every flow gets padding headroom, so the enumeration runs on every flow "
              "before refinement (all seeds pooled)."]
    return lines + [""]


def nonfinite_gradients(results_dir: Path) -> list[str]:
    """Rows whose refinement hit a non-finite gradient of φ (zeroed in the ablated arm only).

    Cells finished before the zeroing was added never met one (the search raised on the first
    occurrence), so a missing count is 0.
    """
    cells = load_cells(results_dir)
    cells = cells[cells["condition"] == "no_capability"]
    if "rows_with_nonfinite_gradient" not in cells:
        cells = cells.assign(rows_with_nonfinite_gradient=0)
    agg = (cells.fillna({"rows_with_nonfinite_gradient": 0})
           .groupby(["dataset", "victim", "budget_label"], sort=True)
           .agg(flows=("n", "sum"), rows=("rows_with_nonfinite_gradient", "sum")).reset_index())
    lines = ["## Non-finite gradients of φ in the ablated arm (all seeds pooled)", "",
             "Outside the capability-admissible set φ is not differentiable everywhere (e.g. "
             "padding a flow whose packet-length variance is 0: d sqrt(var)/dp = inf · 0). The "
             "ablated arm zeroes such gradient coordinates; the reference never meets one.", "",
             "| dataset | victim | budget | flows | flows with a non-finite gradient step |",
             "|---|---|---|---|---|"]
    for r in agg.itertuples(index=False):
        lines.append(f"| {r.dataset} | {r.victim} | {r.budget_label} | {r.flows} | "
                     f"{int(r.rows)} |")
    return lines + [""]


def analyze(results_dir: Path, reference_dir: Path) -> None:
    names = [c.name for c in CONDITIONS]
    write_standard_report(
        results_dir, reference_dir, names,
        title="D6 - Capability inference removed (u = (p, D, s) vs u = M(x) ⊙ (p, D, s))",
        secondary={"raw_success": "Raw ASR"})
    append_report(results_dir, capability_breakdown(results_dir, reference_dir)
                  + budget_allocation(results_dir, reference_dir)
                  + nonfinite_gradients(results_dir)
                  + success_profile(results_dir, reference_dir, names))


if __name__ == "__main__":
    experiment_main(EXP_DIR, CONDITIONS, analyze, __doc__)
