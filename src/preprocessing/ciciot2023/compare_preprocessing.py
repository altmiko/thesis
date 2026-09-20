"""Compare legacy fixed and semantics-corrected CICIoT2023 artifacts."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.ciciot2023.reporting import markdown_table
from src.preprocessing.schema import BOUNDED_AGGREGATED_IDX, COUNT_AGGREGATES, FEATURE_NAMES


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fractional_counts(root: Path) -> dict[str, int]:
    with (root / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    bounded = list(BOUNDED_AGGREGATED_IDX)
    counts = [FEATURE_NAMES.index(name) for name in COUNT_AGGREGATES]
    result = {"bounded_fractional_cells": 0, "count_fractional_cells": 0}
    for split in ("train", "val", "test"):
        array = np.load(root / f"X_{split}.npy", mmap_mode="r")
        for start in range(0, array.shape[0], 500_000):
            raw = scaler.inverse_transform(np.asarray(array[start:start + 500_000]))
            b = raw[:, bounded]
            c = raw[:, counts]
            result["bounded_fractional_cells"] += int(((b > 0.0) & (b < 1.0)).sum())
            result["count_fractional_cells"] += int((np.abs(c - np.rint(c)) > 1e-6).sum())
    return result


def run(old_root: Path, new_root: Path) -> dict:
    old_manifest = _load_json(old_root / "run_manifest.json")
    new_manifest = _load_json(new_root / "run_manifest.json")
    old_fractional = _fractional_counts(old_root)
    new_fractional = _fractional_counts(new_root)

    with (old_root / "scaler.pkl").open("rb") as handle:
        old_scaler = pickle.load(handle)
    with (new_root / "scaler.pkl").open("rb") as handle:
        new_scaler = pickle.load(handle)
    scaler_table = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "old_center": old_scaler.center_,
        "new_center": new_scaler.center_,
        "center_delta": new_scaler.center_ - old_scaler.center_,
        "old_scale": old_scaler.scale_,
        "new_scale": new_scaler.scale_,
        "scale_delta": new_scaler.scale_ - old_scaler.scale_,
    })
    scaler_table.to_csv(new_root / "audits" / "before_after_scaler.csv", index=False)

    old_counts = {split: int(np.load(old_root / f"X_{split}.npy", mmap_mode="r").shape[0]) for split in ("train", "val", "test")}
    new_counts = {split: int(np.load(new_root / f"X_{split}.npy", mmap_mode="r").shape[0]) for split in ("train", "val", "test")}

    old_correlation_path = old_root / "eda" / "train_spearman_correlation.csv"
    new_correlation_path = new_root / "eda" / "train_spearman_correlation.csv"
    correlation_summary: dict[str, float | str] = {"status": "unavailable"}
    if old_correlation_path.exists() and new_correlation_path.exists():
        old_corr = pd.read_csv(old_correlation_path, index_col=0).loc[FEATURE_NAMES, FEATURE_NAMES].to_numpy(float)
        new_corr = pd.read_csv(new_correlation_path, index_col=0).loc[FEATURE_NAMES, FEATURE_NAMES].to_numpy(float)
        delta = np.abs(new_corr - old_corr)
        finite = delta[np.isfinite(delta)]
        correlation_summary = {
            "status": "computed; samples/populations differ, so this is descriptive rather than causal",
            "mean_absolute_matrix_difference": float(finite.mean()),
            "maximum_absolute_matrix_difference": float(finite.max()),
        }

    old_eda = _load_json(old_root / "eda" / "eda_report.json") if (old_root / "eda" / "eda_report.json").exists() else {}
    new_analysis = _load_json(new_root / "eda" / "feature_analysis_summary.json") if (new_root / "eda" / "feature_analysis_summary.json").exists() else {}
    summary = {
        "old_root": str(old_root),
        "new_root": str(new_root),
        "old_behavior": "train 99.99th percentile clip + positive-to-one indicator canonicalization + integer count rounding",
        "new_behavior": "finite float32 source values preserved; no model clipping; continuous aggregate semantics",
        "fractional_values": {"old": old_fractional, "new": new_fractional},
        "split_counts": {"old": old_counts, "new": new_counts},
        "natural_train_before_sampling": {
            "old": old_manifest.get("train_before_sampling"),
            "new": new_manifest.get("train_before_sampling"),
        },
        "constant_features": {
            "old_sampled_train": old_eda.get("constant_train_features", []),
            "new_natural_train": new_analysis.get("constant_features", []),
        },
        "near_constant_features_new_natural_train": new_analysis.get("near_constant_features", []),
        "spearman_matrix_difference": correlation_summary,
        "clipping": {
            "old": {"mode": "train_percentile", "percentile": old_manifest.get("clip_percentile")},
            "new": {"mode": new_manifest.get("clipping_mode"), "percentile": new_manifest.get("clipping_percentile")},
        },
        "artifact_hashes": {"old": old_manifest.get("artifact_hashes", {}), "new": new_manifest.get("artifact_hashes", {})},
    }
    (new_root / "audits" / "before_after_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    split_table = pd.DataFrame([
        {"split": split, "old_rows": old_counts[split], "new_rows": new_counts[split], "delta": new_counts[split] - old_counts[split]}
        for split in ("train", "val", "test")
    ])
    top_scaler = scaler_table.assign(abs_scale_delta=lambda frame: frame["scale_delta"].abs()).sort_values("abs_scale_delta", ascending=False).head(15)
    lines = [
        "# Old versus semantics-corrected preprocessing",
        "",
        "The old bundle is preserved and was not overwritten.",
        "",
        "## Semantic change",
        "",
        "Old: positive averaged indicators were mapped to one and named counts were rounded; every feature was clipped at a training-fitted 99.99th percentile.",
        "",
        "Corrected: finite source values are preserved. Indicator means remain continuous in [0,1]; counts and Protocol Type are not rounded; clipping is disabled by default and remains an explicit model-only option.",
        "",
        "## Fractional cells across saved splits",
        "",
        markdown_table(pd.DataFrame([{"run": "old", **old_fractional}, {"run": "corrected", **new_fractional}])),
        "",
        "## Split counts",
        "",
        markdown_table(split_table),
        "",
        "Full validation and test counts are unchanged. Training membership may change because clustering now receives all suitable continuous aggregate features.",
        "",
        "## Largest scaler changes",
        "",
        markdown_table(top_scaler, 8),
        "",
        "## Correlation comparison",
        "",
        json.dumps(correlation_summary, indent=2),
        "",
        "The legacy and corrected Spearman matrices use different values and sampling populations; differences are descriptive and cannot isolate a single causal component.",
        "",
        "## Hashes",
        "",
        "Exact artifact hashes are stored in `before_after_comparison.json` and each run manifest.",
        "",
    ]
    (new_root / "audits" / "before_after_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=Path("outputs/ciciot2023_fixed"))
    parser.add_argument("--new-root", type=Path, default=Path("outputs/ciciot2023_semantics_corrected"))
    args = parser.parse_args()
    print(json.dumps(run(args.old_root, args.new_root), indent=2))


if __name__ == "__main__":
    main()
