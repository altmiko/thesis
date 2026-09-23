"""Paired tests for matched PrimAttack budgets and primitive ablations.

Reuses the repository's McNemar and Holm implementations. Cochran's Q/Friedman omnibus tests
are followed by Holm-corrected McNemar/Wilcoxon comparisons. Pairing is enforced by the exact
sample ID, source class, victim, and optimization seed.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, friedmanchisquare, wilcoxon

from evaluation.paired_validity_gap import holm_adjust, mcnemar_test

_KEY = ["sample_id", "attack_class", "victim_model", "seed"]


def load_rows(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/*/attack_artifacts/*.npz")):
        with np.load(path) as artifact:
            n = len(artifact["sample_id"])
            frames.append(pd.DataFrame({
                "sample_id": artifact["sample_id"].astype(str),
                "attack_class": artifact["attack_class"].astype(str),
                "victim_model": artifact["victim_model"].astype(str),
                "seed": np.full(n, int(artifact["seed"])),
                "budget_name": artifact["budget_name"].astype(str),
                "primitive_mode": artifact["primitive_mode"].astype(str),
                "eligible": artifact["clean_correct"].astype(bool),
                "targeted_success": artifact["targeted_success"].astype(bool),
                "sp_success": (
                    artifact["targeted_success"].astype(bool)
                    & artifact["domain_valid"].astype(bool)
                    & artifact["primitive_feasible"].astype(bool)
                    & (artifact["semantic_status"].astype(str) == "PASS")
                ),
                "relative_duration_change": artifact["relative_duration_change"],
                "relative_byte_change": artifact["relative_byte_change"],
                "rate_retention": artifact["rate_retention"],
            }))
    if not frames:
        raise FileNotFoundError(f"no PrimAttack artifacts under {root}")
    return pd.concat(frames, ignore_index=True)


def _paired_matrix(frame: pd.DataFrame, condition: str, outcome: str):
    pivot = frame.pivot(index=_KEY, columns=condition, values=outcome).dropna()
    if pivot.empty:
        raise ValueError(f"no complete pairing for {condition}/{outcome}")
    return pivot
def _cochrans_q(values: np.ndarray) -> tuple[float, float]:
    """Cochran's Q for an N×K matched binary matrix."""
    matrix = np.asarray(values, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("Cochran's Q requires at least two matched conditions")
    k = matrix.shape[1]
    column_totals = matrix.sum(axis=0)
    row_totals = matrix.sum(axis=1)
    numerator = (k - 1) * (
        k * np.square(column_totals).sum() - column_totals.sum() ** 2
    )
    denominator = k * row_totals.sum() - np.square(row_totals).sum()
    if denominator == 0:
        return 0.0, 1.0
    statistic = float(numerator / denominator)
    return statistic, float(chi2.sf(statistic, k - 1))




def _binary_family(frame, condition, outcome, fixed_name, fixed_value):
    pivot = _paired_matrix(frame, condition, outcome).astype(bool)
    conditions = list(pivot.columns)
    omnibus_statistic, omnibus_p = _cochrans_q(pivot.to_numpy(dtype=np.int64))
    pairs = []
    raw_p = []
    for left, right in itertools.combinations(conditions, 2):
        x = pivot[left].to_numpy()
        y = pivot[right].to_numpy()
        b = int((x & ~y).sum())
        c = int((~x & y).sum())
        result = mcnemar_test(b, c)
        raw_p.append(float(result["p_value"]))
        pairs.append({
            "fixed_axis": fixed_name,
            "fixed_value": fixed_value,
            "outcome": outcome,
            "left": left,
            "right": right,
            "n": len(pivot),
            "b_left_only": b,
            "c_right_only": c,
            **result,
        })
    for row, adjusted in zip(pairs, holm_adjust(raw_p)):
        row["holm_p"] = adjusted
    return {
        "omnibus": {
            "test": "Cochran's Q",
            "fixed_axis": fixed_name,
            "fixed_value": fixed_value,
            "outcome": outcome,
            "n": len(pivot),
            "conditions": conditions,
            "statistic": omnibus_statistic,
            "p_value": omnibus_p,
        },
        "pairwise": pairs,
    }


def _continuous_family(frame, condition, outcome, fixed_name, fixed_value):
    pivot = _paired_matrix(frame, condition, outcome)
    conditions = list(pivot.columns)
    samples = [pivot[name].to_numpy(dtype=float) for name in conditions]
    if all(np.array_equal(samples[0], sample) for sample in samples[1:]):
        omnibus_statistic, omnibus_p = 0.0, 1.0
    else:
        omnibus = friedmanchisquare(*samples)
        omnibus_statistic = float(omnibus.statistic)
        omnibus_p = float(omnibus.pvalue)
    pairs = []
    raw_p = []
    for left, right in itertools.combinations(conditions, 2):
        x = pivot[left].to_numpy(dtype=float)
        y = pivot[right].to_numpy(dtype=float)
        if np.array_equal(x, y):
            statistic, p_value = 0.0, 1.0
        else:
            try:
                result = wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
                statistic, p_value = float(result.statistic), float(result.pvalue)
                if not np.isfinite(p_value):
                    statistic, p_value = 0.0, 1.0
            except ValueError:
                statistic, p_value = 0.0, 1.0
        raw_p.append(p_value)
        pairs.append({
            "fixed_axis": fixed_name,
            "fixed_value": fixed_value,
            "outcome": outcome,
            "left": left,
            "right": right,
            "n": len(pivot),
            "test": "Wilcoxon signed-rank (Pratt zeros)",
            "statistic": statistic,
            "p_value": p_value,
        })
    for row, adjusted in zip(pairs, holm_adjust(raw_p)):
        row["holm_p"] = adjusted
    return {
        "omnibus": {
            "test": "Friedman",
            "fixed_axis": fixed_name,
            "fixed_value": fixed_value,
            "outcome": outcome,
            "n": len(pivot),
            "conditions": conditions,
            "statistic": omnibus_statistic,
            "p_value": omnibus_p,
        },
        "pairwise": pairs,
    }


def analyze(frame: pd.DataFrame) -> dict:
    frame = frame[frame["eligible"]].copy()
    output = {"binary": [], "continuous": []}
    for budget_name, subset in frame.groupby("budget_name", sort=True):
        for outcome in ("targeted_success", "sp_success"):
            output["binary"].append(_binary_family(
                subset, "primitive_mode", outcome, "budget_name", budget_name
            ))
        for outcome in (
            "relative_duration_change", "relative_byte_change", "rate_retention"
        ):
            output["continuous"].append(_continuous_family(
                subset, "primitive_mode", outcome, "budget_name", budget_name
            ))
    for mode, subset in frame.groupby("primitive_mode", sort=True):
        for outcome in ("targeted_success", "sp_success"):
            output["binary"].append(_binary_family(
                subset, "budget_name", outcome, "primitive_mode", mode
            ))
        for outcome in (
            "relative_duration_change", "relative_byte_change", "rate_retention"
        ):
            output["continuous"].append(_continuous_family(
                subset, "budget_name", outcome, "primitive_mode", mode
            ))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("outputs/primattack_budget_sensitivity")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input_dir / "paired_statistics.json"
    results = analyze(load_rows(args.input_dir))
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "binary_families": len(results["binary"]),
        "continuous_families": len(results["continuous"]),
    }))


if __name__ == "__main__":
    main()
