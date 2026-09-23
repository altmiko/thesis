"""Independently recompute CICIDS2017 attack metrics from per-sample artifacts.

This module deliberately imports no attack runner or metric helper.  It treats each saved
NPZ as the source of truth, reconstructs the clean-correct denominator and validity gates,
and optionally checks the recomputed cell values against ``attack_results.json``.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


RATE_FIELDS = (
    "untargeted_asr",
    "targeted_benign_asr",
    "targeted_strict_valid_asr",
    "strict_validity",
)


def _artifact_path(output_dir: Path, recorded: str) -> Path:
    path = Path(recorded)
    if path.exists():
        return path
    fallback = output_dir / "attack_artifacts" / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"missing attack artifact: {recorded} (also tried {fallback})")


def _strict_mask(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, tuple[str, ...]]:
    keys = set(data.files)
    if "strict_valid" in keys:
        names = ("strict_valid",)
    elif "mined_valid" in keys:
        # validator_v2 hybrid_valid is the canonical strict mask.
        names = ("mined_valid",)
    else:
        raise KeyError(
            "artifact has no recognized independent strict-validity masks; "
            f"available keys={sorted(keys)}"
        )
    mask = np.ones(len(data["clean_correct"]), dtype=bool)
    for name in names:
        mask &= np.asarray(data[name], dtype=bool)
    return mask, names


def _require_finite(data: np.lib.npyio.NpzFile, artifact: Path) -> None:
    for name in data.files:
        value = np.asarray(data[name])
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            count = int((~np.isfinite(value)).sum())
            raise ValueError(f"{artifact}:{name} contains {count} NaN/Inf values")


def _counts(data: np.lib.npyio.NpzFile) -> dict[str, float | int | tuple[str, ...]]:
    _require_equal_lengths(data)
    cc = np.asarray(data["clean_correct"], dtype=bool)
    evasion = np.asarray(data["evasion"], dtype=bool)
    benign = np.asarray(data["benign"], dtype=bool)
    strict, strict_keys = _strict_mask(data)
    denominator = int(cc.sum())
    if denominator == 0:
        raise ValueError("artifact has zero clean-correct rows")
    cost = np.asarray(data["cost_total"], dtype=np.float64)
    return {
        "n": int(len(cc)),
        "denominator": denominator,
        "untargeted_success": int((cc & evasion).sum()),
        "targeted_success": int((cc & benign).sum()),
        "strict_valid": int((cc & strict).sum()),
        "targeted_strict_valid_success": int((cc & benign & strict).sum()),
        "cost_sum_clean_correct": float(cost[cc].sum()),
        "strict_keys": strict_keys,
    }


def _require_equal_lengths(data: np.lib.npyio.NpzFile) -> None:
    expected = len(data["clean_correct"])
    for name in ("evasion", "benign", "cost_total", "y_true", "y_pred_clean", "y_pred_adv"):
        if name not in data.files:
            raise KeyError(f"artifact lacks required field {name!r}")
        if len(data[name]) != expected:
            raise ValueError(f"field {name!r} length {len(data[name])} != {expected}")


def _rates(counts: dict[str, float | int | tuple[str, ...]]) -> dict[str, float]:
    d = int(counts["denominator"])
    return {
        "untargeted_asr": int(counts["untargeted_success"]) / d,
        "targeted_benign_asr": int(counts["targeted_success"]) / d,
        "targeted_strict_valid_asr": int(counts["targeted_strict_valid_success"]) / d,
        "strict_validity": int(counts["strict_valid"]) / d,
        "cost_total_mean": float(counts["cost_sum_clean_correct"]) / d,
    }


def _merge(rows: Iterable[dict[str, float | int | tuple[str, ...]]]) -> dict[str, float | int]:
    rows = list(rows)
    return {
        "n": sum(int(row["n"]) for row in rows),
        "denominator": sum(int(row["denominator"]) for row in rows),
        "untargeted_success": sum(int(row["untargeted_success"]) for row in rows),
        "targeted_success": sum(int(row["targeted_success"]) for row in rows),
        "strict_valid": sum(int(row["strict_valid"]) for row in rows),
        "targeted_strict_valid_success": sum(
            int(row["targeted_strict_valid_success"]) for row in rows
        ),
        "cost_sum_clean_correct": sum(float(row["cost_sum_clean_correct"]) for row in rows),
    }


def audit_output(output_dir: Path, *, tolerance: float = 1e-6) -> dict:
    summary_path = output_dir / "attack_results.json"
    recorded = json.loads(summary_path.read_text(encoding="utf-8"))
    cells: list[dict] = []
    mismatches: list[dict] = []

    for cell in recorded.get("cells", []):
        artifact = _artifact_path(output_dir, cell["artifact"])
        with np.load(artifact, allow_pickle=False) as data:
            _require_finite(data, artifact)
            counts = _counts(data)
        rates = _rates(counts)
        audited = {
            "class": cell["class"],
            "victim": cell["victim"],
            "seed": int(cell["seed"]),
            "artifact": str(artifact),
            **counts,
            **rates,
        }
        audited["strict_keys"] = list(audited["strict_keys"])
        cells.append(audited)

        expected_counts = {
            "denominator": cell.get("n_clean_correct"),
            "targeted_success": cell.get("n_targeted_benign_success"),
            "targeted_strict_valid_success": cell.get("n_targeted_strict_valid"),
        }
        for field, expected in expected_counts.items():
            if expected is not None and int(expected) != int(audited[field]):
                mismatches.append(
                    {"artifact": str(artifact), "field": field, "recorded": expected,
                     "audited": audited[field]}
                )
        for field in (*RATE_FIELDS, "cost_total_mean"):
            expected = cell.get(field)
            if expected is not None and not math.isclose(
                float(expected), float(audited[field]), rel_tol=tolerance, abs_tol=tolerance
            ):
                mismatches.append(
                    {"artifact": str(artifact), "field": field, "recorded": expected,
                     "audited": audited[field]}
                )

    by_seed: dict[int, list[dict]] = defaultdict(list)
    by_class_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    by_victim_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for cell in cells:
        by_seed[cell["seed"]].append(cell)
        by_class_seed[(cell["class"], cell["seed"])].append(cell)
        by_victim_seed[(cell["victim"], cell["seed"])].append(cell)

    seed_rows = []
    for seed, rows in sorted(by_seed.items()):
        merged = _merge(rows)
        cell_macro = {
            field: float(np.mean([float(row[field]) for row in rows])) for field in RATE_FIELDS
        }
        seed_rows.append({"seed": seed, "micro": {**merged, **_rates(merged)},
                          "macro_over_cells": cell_macro})

    per_class = []
    for (class_name, seed), rows in sorted(by_class_seed.items()):
        merged = _merge(rows)
        per_class.append({"class": class_name, "seed": seed, **merged, **_rates(merged)})
    per_victim = []
    for (victim, seed), rows in sorted(by_victim_seed.items()):
        merged = _merge(rows)
        per_victim.append({"victim": victim, "seed": seed, **merged, **_rates(merged)})

    seed_tsv = [row["micro"]["targeted_strict_valid_asr"] for row in seed_rows]
    provenance_required = {
        "row_id", "dataset", "class_name", "victim", "method", "seed", "true_label",
        "clean_prediction", "final_adversarial_prediction", "clean_correct",
        "target_success_flag", "strict_valid", "cost_total", "git_commit", "git_dirty",
        "source_tree_sha256", "run_id", "config_json", "checkpoint_identifiers_json",
    }
    artifact_keys: set[str] = set()
    if cells:
        with np.load(cells[0]["artifact"], allow_pickle=False) as first:
            artifact_keys = set(first.files)

    return {
        "output_dir": str(output_dir),
        "dataset": recorded.get("dataset"),
        "method_id": recorded.get("method_id"),
        "config": recorded.get("config"),
        "cells": cells,
        "per_seed": seed_rows,
        "per_class": per_class,
        "per_victim": per_victim,
        "three_seed_targeted_strict_valid": {
            "values": seed_tsv,
            "mean": float(np.mean(seed_tsv)) if seed_tsv else None,
            "std_population": float(np.std(seed_tsv)) if seed_tsv else None,
        },
        "runner_disagreements": mismatches,
        "artifact_provenance": {
            "first_artifact_keys": sorted(artifact_keys),
            "missing_required_concepts": sorted(provenance_required - artifact_keys),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", nargs="+", type=Path, help="attack output directories")
    parser.add_argument("--json", type=Path, help="write full audit JSON")
    args = parser.parse_args()
    reports = [audit_output(path) for path in args.outputs]
    for report in reports:
        three = report["three_seed_targeted_strict_valid"]
        print(f"{report['output_dir']}: method={report['method_id']}")
        for seed in report["per_seed"]:
            micro = seed["micro"]
            print(
                f"  seed={seed['seed']} N={micro['n']} clean_correct={micro['denominator']} "
                f"targeted_ASR={100*micro['targeted_benign_asr']:.4f}% "
                f"targeted_strict_valid_ASR={100*micro['targeted_strict_valid_asr']:.4f}% "
                f"cost={micro['cost_total_mean']:.6g}"
            )
        print(
            f"  3-seed TSV-ASR mean={100*three['mean']:.4f}% "
            f"std={100*three['std_population']:.4f}% "
            f"runner_disagreements={len(report['runner_disagreements'])}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
