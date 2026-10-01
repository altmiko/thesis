"""Shared command line for every ablation's ``run.py``: run the cells, then analyze."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from ablations.common.runner import (
    BUDGETS, CLASSES, DATASETS, REFERENCE_DIR, SEEDS, Condition, run_conditions,
)


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def experiment_main(
    exp_dir: Path,
    conditions: list[Condition],
    analyze: Callable[[Path, Path], None] | None,
    doc: str,
) -> None:
    """``analyze(results_dir, reference_results_dir)`` writes the experiment's tables."""
    ap = argparse.ArgumentParser(description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--budgets", default=",".join(BUDGETS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--victims", default=None, help="default: the dataset's three victims")
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--conditions", default=None, help="subset of condition names")
    ap.add_argument("--limit-rows", type=int, default=None, help="smoke tests only")
    ap.add_argument("--results-dir", type=Path, default=exp_dir / "results")
    ap.add_argument("--reference-results", type=Path, default=REFERENCE_DIR / "results")
    ap.add_argument("--skip-run", action="store_true", help="only (re)write the analysis")
    ap.add_argument("--skip-analysis", action="store_true")
    args = ap.parse_args()

    selected = conditions
    if args.conditions:
        wanted = set(_csv(args.conditions))
        unknown = wanted - {c.name for c in conditions}
        if unknown:
            raise ValueError(f"unknown conditions {sorted(unknown)}")
        selected = [c for c in conditions if c.name in wanted]
    if not args.skip_run and selected:
        run_conditions(
            args.results_dir, selected, datasets=_csv(args.datasets),
            budgets=tuple(_csv(args.budgets)), seeds=tuple(int(s) for s in _csv(args.seeds)),
            victims=_csv(args.victims) if args.victims else None,
            classes=tuple(_csv(args.classes)), device=args.device, limit_rows=args.limit_rows)
    if analyze is not None and not args.skip_analysis:
        analyze(args.results_dir, args.reference_results)
