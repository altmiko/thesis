"""Clean-data sanity validation (§13).

Runs validator_v2 on UNTOUCHED genuine samples (held-out TEST split -- used here
only for pure evaluation, never for mining/threshold selection) and reports the
acceptance rate per provenance layer plus the exact rules (if any) that reject
genuine data. Hard structural rules should not routinely reject real flows.

Run:  python -m validation.evaluation.clean_acceptance
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from validation import load_validator
from validation.mining import data_access as da
from validation.metrics import clean_acceptance_rate

PKG = Path(__file__).resolve().parents[1]
N = 100000


def run(dataset: str = "cicids2017_distrinet", n: int = N, split: str = "test") -> dict:
    v = load_validator("cicids2017_distrinet")
    X = da.load_split(split, allow_test_for_final_reporting=True)
    Xs = da.sample_rows(X, n, seed=7)
    b = v.validate_batch(Xs)
    rates = b.rates()
    rejecting = b.rules_rejecting_any()
    by_id = {r.id: r for r in v.rules}
    md = _report(dataset, split, Xs.shape[0], rates, rejecting, by_id, v)
    out = PKG / "reports" / dataset / "clean_acceptance_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return {"rates": rates, "n": Xs.shape[0],
            "car_hard": rates["hard_structural_valid"],
            "car_hybrid": rates["hybrid_valid"],
            "rejecting_rules": {k: int(c) for k, c in rejecting.items()}}


def _report(dataset, split, n, rates, rejecting, by_id, v) -> str:
    L = [f"# Clean acceptance report — {dataset}", "",
         f"Validator_v2 on **{n:,}** untouched genuine `{split}`-split flows "
         f"(pure evaluation; not used for mining).", "",
         "## Acceptance rates", "",
         "| layer | acceptance rate |", "|---|---|",
         f"| SCHEMA | {rates['schema_valid']:.6f} |",
         f"| PROTOCOL | {rates['protocol_valid']:.6f} |",
         f"| EXTRACTOR | {rates['extractor_valid']:.6f} |",
         f"| MINED | {rates['mined_valid']:.6f} |",
         f"| **hard_structural** (SCHEMA+EXTRACTOR+PROTOCOL) | **{rates['hard_structural_valid']:.6f}** |",
         f"| **hybrid** (+MINED) | **{rates['hybrid_valid']:.6f}** |",
         f"| in_distribution (plausibility, separate) | {rates['in_distribution']:.6f} |",
         "",
         "## Rules rejecting genuine samples", ""]
    if not rejecting:
        L.append("None. No structural rule rejects any genuine sample in this set.")
    else:
        L.append("| rule | source | expression | rejected | rate |")
        L.append("|---|---|---|---|---|")
        for rid, c in sorted(rejecting.items(), key=lambda kv: -kv[1]):
            r = by_id[rid]
            L.append(f"| {rid} | {r.source_type} | `{r.expression()}` | {c} | {c/n:.6f} |")
        L.append("")
        L.append("> Investigate any HARD (SCHEMA/EXTRACTOR/PROTOCOL) rule here before "
                 "adjusting tolerances — the rule, dataset, or preprocessing may be at fault "
                 "(§13), not the threshold.")
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
