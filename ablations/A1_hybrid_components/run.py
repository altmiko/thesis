"""A1 - Hybrid Search component leave-one-out, with a per-flow coverage matrix.

Design source: CAA's CAPGD component ablation (Simonetto et al., NeurIPS 2024, App. B.1 Table 7:
no repair / no clean start / no random start / no adaptive step; coverage |C_B|/|C_A u C_B|,
Fig. 4) and NetMasquerade's stage knock-out (NDSS 2026, App. D Table VI).

Each arm removes exactly one component of the canonical Hybrid Search; everything else (frozen
flows, victims, budgets, 256-evaluation cap, objective, final validator_v2) is the reference:

* ``no_padding_sweep``   - no exhaustive integer padding enumeration; refinement starts from the
                           clean flow and must find padding by gradient (padding-only rows refined).
* ``no_refinement``      - identity + padding enumeration only (no gradient stage).
* ``no_random_restarts`` - refinement restart 0 (clean start) only; no random restarts.
* ``fixed_step``         - no stall-triggered step halving / reset to the restart's best point.
* ``no_momentum``        - momentum 0 (sign of the normalized current gradient).
* ``no_surrogate_floor`` - no straight-through floor: the relaxation is evaluated at q itself, so
                           a zero control has an exactly-zero gradient (finding F2).
* ``validator_post_hoc`` - validator_v2 removed from the search success predicate (CAA "no repair"
                           analogue); the search keeps the cheapest objective-meeting candidate
                           and validity is judged only afterwards.
* ``last_iterate``       - the last evaluated candidate is returned instead of the success-first,
                           lowest-cost incumbent.

The coverage analysis adds the FINAL suite's Prim-PGD (Exp A PrimAttack row) as an external arm.

    python ablations/A1_hybrid_components/run.py --device cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ablations.common.analysis import (  # noqa: E402
    REF_SEED, Source, append_report, pooled_rows, success_profile, write_standard_report,
)
from ablations.common.cli import experiment_main  # noqa: E402
from ablations.common.hybrid import HybridConfig  # noqa: E402
from ablations.common.runner import CLASSES, FINAL_RUNS, REFERENCE, Condition  # noqa: E402

CONDITIONS = [
    Condition("no_padding_sweep", "no exhaustive integer padding enumeration",
              HybridConfig(padding_sweep=False)),
    Condition("no_refinement", "identity + padding enumeration only",
              HybridConfig(refinement=False)),
    Condition("no_random_restarts", "clean-start refinement only (restarts=1)",
              HybridConfig(restarts=1)),
    Condition("fixed_step", "no stall-triggered step halving / reset",
              HybridConfig(adaptive_step=False)),
    Condition("no_momentum", "momentum 0", HybridConfig(momentum=0.0)),
    Condition("no_surrogate_floor", "no straight-through surrogate floor",
              HybridConfig(surrogate_floor=0.0)),
    Condition("validator_post_hoc", "validator_v2 not in the search success predicate",
              HybridConfig(validity_in_search=False)),
    Condition("last_iterate", "return the last evaluated candidate, not the incumbent",
              HybridConfig(selection="last")),
]
PGD_ARM = "final_prim_pgd"


def _pgd_rows(dataset: str, victim: str, budget: str, seed: int,
              ref_ids: np.ndarray) -> dict | None:
    """FINAL Prim-PGD success of the reference's flows (aligned by sample id)."""
    art = FINAL_RUNS / dataset / "primattack_untargeted" / "artifacts"
    parts = []
    for cname in CLASSES:
        path = art / f"{victim}__{cname}__{budget}__pgd__seed{seed}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=True) as z:
            parts.append({"sample_id": z["sample_id"], "valid_success": z["valid_success"]})
    rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    pos = {sid: k for k, sid in enumerate(rows["sample_id"])}
    take = np.asarray([pos[sid] for sid in ref_ids])
    return {k: v[take] for k, v in rows.items()}


def coverage(results_dir: Path, reference_dir: Path, summary: pd.DataFrame) -> list[str]:
    """Seed-42 success sets: overlap of every arm with the reference and the union."""
    rows = []
    arms = [REFERENCE.name] + [c.name for c in CONDITIONS]
    for g in summary[["dataset", "victim", "budget"]].drop_duplicates().itertuples(index=False):
        sets = {}
        for arm in arms:
            src = Source(reference_dir if arm == REFERENCE.name else results_dir, arm)
            r = pooled_rows(src, g.dataset, g.victim, g.budget, REF_SEED, ("valid_success",))
            if r is not None:
                sets[arm] = r
        if REFERENCE.name not in sets:
            continue
        ids = sets[REFERENCE.name]["sample_id"]
        pgd = _pgd_rows(g.dataset, g.victim, g.budget, REF_SEED, ids)
        if pgd is not None:
            sets[PGD_ARM] = pgd
        for arm, r in sets.items():
            if not np.array_equal(r["sample_id"], ids):
                raise AssertionError(f"{g}: {arm} not aligned with the reference rows")
        ref = sets[REFERENCE.name]["valid_success"].astype(bool)
        union = np.logical_or.reduce([r["valid_success"].astype(bool) for r in sets.values()])
        for arm, r in sets.items():
            s = r["valid_success"].astype(bool)
            rows.append({
                "dataset": g.dataset, "victim": g.victim, "budget": g.budget, "arm": arm,
                "successes": int(s.sum()), "reference_successes": int(ref.sum()),
                "covers_reference": float((s & ref).sum() / ref.sum()) if ref.any() else np.nan,
                "covered_by_reference": float((s & ref).sum() / s.sum()) if s.any() else np.nan,
                "caa_coverage_of_union": float(s.sum() / union.sum()) if union.any() else np.nan,
                "union_successes": int(union.sum()),
                "successes_outside_reference": int((s & ~ref).sum()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "coverage_seed42.csv", index=False)
    if df.empty:
        return []
    lines = ["## Coverage (seed 42, valid successes)", "",
             "`covers ref` = |S_arm ∩ S_ref| / |S_ref|; `in ref` = |S_arm ∩ S_ref| / |S_arm|; "
             "`CAA cov.` = |S_arm| / |union of all arms| (CAA Fig. 4); `outside ref` = flows the "
             f"arm breaks that the reference does not. `{PGD_ARM}` = FINAL Exp A Prim-PGD.", "",
             "| dataset | victim | budget | arm | successes | covers ref | in ref | CAA cov. | "
             "outside ref | union |", "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: "n/a" if x != x else f"{x:.3f}"  # noqa: E731
    for r in df.itertuples(index=False):
        lines.append(f"| {r.dataset} | {r.victim} | {r.budget} | {r.arm} | {r.successes} | "
                     f"{fmt(r.covers_reference)} | {fmt(r.covered_by_reference)} | "
                     f"{fmt(r.caa_coverage_of_union)} | {r.successes_outside_reference} | "
                     f"{r.union_successes} |")
    return lines + [""]


def analyze(results_dir: Path, reference_dir: Path) -> None:
    summary, _ = write_standard_report(
        results_dir, reference_dir, [c.name for c in CONDITIONS],
        title="A1 - Hybrid Search component leave-one-out")
    names = [c.name for c in CONDITIONS]
    append_report(results_dir, coverage(results_dir, reference_dir, summary)
                  + success_profile(results_dir, reference_dir, names))


if __name__ == "__main__":
    experiment_main(EXP_DIR, CONDITIONS, analyze, __doc__)
