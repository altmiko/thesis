"""Run matched PrimAttack budget sensitivity and primitive ablations.

All configurations use the same deterministic source-row selector, victim, target, validator,
semantic rules, and train-only calibration. The script asserts identical ``sample_id`` arrays
across budgets/modes before writing results and seven requested figures.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from attack.primattack_budget import BUDGET_NAMES
from attack.run_cicids2017_primitive_attack import PRIMITIVE_MODES, VICTIMS, run
from vae.cicids2017_stage_a import ATTACK_CLASSES


def _pool_artifacts(cells: list[dict]) -> dict[str, float]:
    arrays: dict[str, list[np.ndarray]] = {}
    keys = (
        "targeted_success", "domain_valid", "primitive_feasible", "semantic_status",
        "relative_duration_change", "relative_byte_change", "rate_retention",
        "number_features_changed", "cost_total", "clean_correct",
    )
    for cell in cells:
        with np.load(cell["artifact"]) as artifact:
            for key in keys:
                arrays.setdefault(key, []).append(np.asarray(artifact[key]))
    pooled = {key: np.concatenate(values) for key, values in arrays.items()}
    eligible = pooled["clean_correct"].astype(bool)
    target = pooled["targeted_success"].astype(bool)
    domain = pooled["domain_valid"].astype(bool)
    feasible = pooled["primitive_feasible"].astype(bool)
    semantic_pass = pooled["semantic_status"] == "PASS"
    semantic_fail = pooled["semantic_status"] == "FAIL"
    semantic_nt = pooled["semantic_status"] == "NOT_FULLY_TESTABLE"
    denominator = int(eligible.sum())
    if denominator == 0:
        raise ValueError("budget experiment has no clean-correct eligible rows")
    rate = lambda mask: float((mask & eligible).sum()) / denominator
    values = lambda key: pooled[key][eligible]
    return {
        "eligible_original_samples": denominator,
        "raw_targeted_asr": rate(target),
        "valid_targeted_asr": rate(target & domain),
        "primitive_feasible_targeted_asr": rate(target & domain & feasible),
        "sp_asr": rate(target & domain & feasible & semantic_pass),
        "semantic_pass_rate": rate(semantic_pass),
        "semantic_fail_rate": rate(semantic_fail),
        "not_fully_testable_rate": rate(semantic_nt),
        "semantic_testability_rate": rate(~semantic_nt),
        "median_relative_duration_change": float(np.median(values("relative_duration_change"))),
        "median_relative_byte_change": float(np.median(values("relative_byte_change"))),
        "median_rate_retention": float(np.median(values("rate_retention"))),
        "median_number_features_changed": float(np.median(values("number_features_changed"))),
        "median_primitive_cost": float(np.median(values("cost_total"))),
    }


def _assert_same_source_ids(reference: dict[tuple[str, str, int], np.ndarray], cells: list[dict]) -> None:
    for cell in cells:
        key = (cell["class"], cell["victim"], int(cell["seed"]))
        with np.load(cell["artifact"]) as artifact:
            ids = np.asarray(artifact["sample_id"]).astype(str)
        if key in reference and not np.array_equal(reference[key], ids):
            raise AssertionError(f"source sample IDs changed across configurations for {key}")
        reference.setdefault(key, ids)


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _line_plot(rows, metric, ylabel, output, *, modes=PRIMITIVE_MODES):
    x = np.arange(len(BUDGET_NAMES))
    for mode in modes:
        subset = [row for row in rows if row["primitive_mode"] == mode]
        subset.sort(key=lambda row: BUDGET_NAMES.index(row["budget_name"]))
        plt.plot(x, [row[metric] for row in subset], marker="o", label=mode)
    plt.xticks(x, BUDGET_NAMES, rotation=15)
    plt.ylabel(ylabel)
    plt.xlabel("Train-calibrated budget")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _plots(rows: list[dict], output_dir: Path) -> None:
    _line_plot(rows, "raw_targeted_asr", "Target-Benign ASR", output_dir / "01_raw_asr_vs_budget.png")
    _line_plot(rows, "valid_targeted_asr", "Valid Target-Benign ASR", output_dir / "02_valid_asr_vs_budget.png")
    _line_plot(rows, "sp_asr", "SP-ASR", output_dir / "03_sp_asr_vs_budget.png")
    _line_plot(rows, "semantic_pass_rate", "Semantic PASS rate", output_dir / "04_semantic_pass_vs_budget.png")
    _line_plot(
        rows, "median_rate_retention", "Median rate retention",
        output_dir / "05_rate_retention_vs_timing_budget.png",
        modes=("timing-only", "joint"),
    )

    for mode in PRIMITIVE_MODES:
        subset = [row for row in rows if row["primitive_mode"] == mode]
        plt.plot(
            [row["median_primitive_cost"] for row in subset],
            [row["raw_targeted_asr"] for row in subset],
            marker="o", label=mode,
        )
    plt.xlabel("Median normalized primitive cost")
    plt.ylabel("Target-Benign ASR")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "06_primitive_cost_vs_asr.png", dpi=180)
    plt.close()

    maximum = [row for row in rows if row["budget_name"] == "maximum-evaluated"]
    plt.bar(
        [row["primitive_mode"] for row in maximum],
        [row["raw_targeted_asr"] for row in maximum],
        label="Raw ASR",
    )
    plt.scatter(
        [row["primitive_mode"] for row in maximum],
        [row["sp_asr"] for row in maximum],
        color="black", label="SP-ASR", zorder=3,
    )
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "07_timing_padding_combined.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    parser.add_argument("--victims", default=",".join(VICTIMS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--calibration", type=Path,
        default=Path("artifacts/primattack/budget_calibration.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/primattack_budget_sensitivity")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classes = [value.strip() for value in args.classes.split(",") if value.strip()]
    victims = [value.strip() for value in args.victims.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    rows: list[dict] = []
    reference_ids: dict[tuple[str, str, int], np.ndarray] = {}
    for mode in PRIMITIVE_MODES:
        for budget_name in BUDGET_NAMES:
            run_dir = args.output_dir / mode / budget_name
            result = run(
                classes=classes,
                victims=victims,
                device=args.device,
                test_limit=args.test_limit,
                steps=args.steps,
                lr=0.1,
                stage_a_dir=None,
                output_dir=run_dir,
                seeds=seeds,
                calibration_path=args.calibration,
                budget_name=budget_name,
                primitive_mode=mode,
                optimizer_name="search",
            )
            _assert_same_source_ids(reference_ids, result["cells"])
            rows.append({
                "primitive_mode": mode,
                "budget_name": budget_name,
                **_pool_artifacts(result["cells"]),
            })
            print(json.dumps(rows[-1]), flush=True)

    _write_rows(args.output_dir / "budget_sensitivity.csv", rows)
    (args.output_dir / "budget_sensitivity.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "source_id_consistency.json").write_text(
        json.dumps({
            "identical_across_all_configurations": True,
            "cells_checked": len(reference_ids),
            "source_selection": "fixed class rows, victim, seed",
        }, indent=2), encoding="utf-8"
    )
    _plots(rows, args.output_dir)


if __name__ == "__main__":
    main()
