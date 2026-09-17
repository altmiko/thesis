"""Leakage-safe exploratory analysis for the processed CICIoT2023 dataset.

The script consumes the canonical artifacts produced by
``src.preprocessing.ciciot2023.pipeline``. Descriptive counts cover every saved
row. Feature summaries cover the sampled training matrix only. Correlation and
PCA are fitted on a deterministic, category-stratified training sample; neither
validation nor test data is used to fit an EDA transform.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from config import paths  # noqa: E402
from src.preprocessing.schema import (  # noqa: E402
    BINARY_FEATURES,
    FEATURE_NAMES,
    INTEGER_FEATURES,
)

SELECTED_FEATURES = (
    "Header_Length",
    "Time_To_Live",
    "Rate",
    "Tot sum",
    "IAT",
    "Variance",
)
CATEGORY_COLORS = {
    "Benign": "#2ca02c",
    "BruteForce": "#8c564b",
    "DDoS": "#d62728",
    "DoS": "#ff7f0e",
    "Mirai": "#9467bd",
    "Recon": "#bcbd22",
    "Spoofing": "#1f77b4",
    "Web": "#e377c2",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _require(processed_dir: Path) -> None:
    required = [
        "run_manifest.json",
        "ciciot2023_labeled_full_manifest.json",
        "scaler.pkl",
        "label_encoder.pkl",
        "category_encoder.pkl",
    ]
    required.extend(
        f"{stem}_{split}{suffix}.npy"
        for split in ("train", "val", "test")
        for stem, suffix in (("X", ""), ("y", ""), ("y", "_cat"), ("y", "_bin"))
    )
    missing = [name for name in required if not (processed_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessing artifacts: {missing}")


def _load_arrays(processed_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        arrays[split] = {
            "X": np.load(processed_dir / f"X_{split}.npy", mmap_mode="r"),
            "fine": np.load(processed_dir / f"y_{split}.npy", mmap_mode="r"),
            "category": np.load(processed_dir / f"y_{split}_cat.npy", mmap_mode="r"),
            "binary": np.load(processed_dir / f"y_{split}_bin.npy", mmap_mode="r"),
        }
        n = arrays[split]["X"].shape[0]
        if arrays[split]["X"].shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"{split} feature width is not {len(FEATURE_NAMES)}")
        if any(arrays[split][key].shape != (n,) for key in ("fine", "category", "binary")):
            raise ValueError(f"{split} feature and label lengths disagree")
    return arrays


def _distribution_table(
    arrays: dict[str, dict[str, np.ndarray]], key: str, names: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        labels = np.asarray(arrays[split][key])
        counts = np.bincount(labels, minlength=len(names))
        if counts.size != len(names):
            raise ValueError(f"{split} {key} contains an out-of-range label code")
        total = int(labels.size)
        for code, name in enumerate(names):
            rows.append(
                {
                    "split": split,
                    "split_population": "sampled_train" if split == "train" else "natural_holdout",
                    "code": code,
                    "name": name,
                    "count": int(counts[code]),
                    "share": float(counts[code] / total),
                }
            )
    return pd.DataFrame(rows)


def _stratified_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    selected: list[np.ndarray] = []
    for code in np.unique(labels_array):
        candidates = np.flatnonzero(labels_array == code)
        take = min(per_class, candidates.size)
        selected.append(rng.choice(candidates, size=take, replace=False))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def _feature_type(feature: str) -> str:
    if feature == "Protocol Type":
        return "protocol"
    if feature in BINARY_FEATURES:
        return "binary"
    if feature in INTEGER_FEATURES:
        return "integer"
    return "continuous"


def _feature_summary(
    x_train: np.ndarray, scaler: Any
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    center = np.asarray(scaler.center_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    if center.shape != (len(FEATURE_NAMES),) or scale.shape != center.shape:
        raise ValueError("Scaler does not match the frozen 39-feature schema")

    rows: list[dict[str, Any]] = []
    protocol_values: tuple[np.ndarray, np.ndarray] | None = None
    invalid_binary: dict[str, int] = {}
    invalid_integer: dict[str, int] = {}

    for column, feature in enumerate(FEATURE_NAMES):
        raw = np.asarray(x_train[:, column], dtype=np.float64) * scale[column] + center[column]
        finite = np.isfinite(raw)
        values = raw[finite]
        if values.size == 0:
            raise ValueError(f"Training feature {feature!r} has no finite values")
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
        rows.append(
            {
                "feature": feature,
                "type": _feature_type(feature),
                "count": int(raw.size),
                "nonfinite_count": int((~finite).sum()),
                "min": float(values.min()),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "zero_share": float(np.isclose(values, 0.0, atol=1e-6).mean()),
            }
        )
        if feature in BINARY_FEATURES:
            valid = np.isclose(values, 0.0, atol=1e-5) | np.isclose(values, 1.0, atol=1e-5)
            invalid_binary[feature] = int((~valid).sum())
        if feature in INTEGER_FEATURES:
            invalid_integer[feature] = int((np.abs(values - np.rint(values)) > 1e-5).sum())
        if feature == "Protocol Type":
            protocol_values = np.unique(values, return_counts=True)

    if protocol_values is None:
        raise AssertionError("Protocol Type missing from FEATURE_NAMES")
    protocol, counts = protocol_values
    protocol_table = pd.DataFrame(
        {
            "protocol_value": protocol,
            "count": counts,
            "share": counts / counts.sum(),
        }
    ).sort_values("protocol_value", ignore_index=True)
    checks = {
        "nonfinite_train_values": int(sum(row["nonfinite_count"] for row in rows)),
        "invalid_binary_train_values": int(sum(invalid_binary.values())),
        "invalid_integer_train_values": int(sum(invalid_integer.values())),
    }
    return pd.DataFrame(rows), protocol_table, checks


def _plot_distributions(category_table: pd.DataFrame, output: Path) -> None:
    names = list(category_table["name"].drop_duplicates())
    splits = ("train", "val", "test")
    x = np.arange(len(names))
    width = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    for offset, split in enumerate(splits):
        rows = category_table[category_table["split"] == split].set_index("name").loc[names]
        axes[0].bar(x + (offset - 1) * width, rows["count"], width, label=split)
        axes[1].bar(x + (offset - 1) * width, rows["share"], width, label=split)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Rows (log scale)")
    axes[0].set_title("CICIoT2023 category counts: sampled train vs natural holdouts")
    axes[1].set_ylabel("Within-split share")
    axes[1].set_xticks(x, names, rotation=30, ha="right")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation(correlation: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = matplotlib.colormaps["coolwarm"].copy()
    cmap.set_bad("#bdbdbd")
    image = ax.imshow(
        np.ma.masked_invalid(correlation.to_numpy()), cmap=cmap, vmin=-1, vmax=1
    )
    positions = np.arange(len(FEATURE_NAMES))
    ax.set_xticks(positions, FEATURE_NAMES, rotation=90, fontsize=6)
    ax.set_yticks(positions, FEATURE_NAMES, fontsize=6)
    ax.set_title("Spearman correlation: category-stratified training sample")
    ax.text(
        0.0,
        -0.14,
        "Gray cells: undefined correlation because at least one feature is constant.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_pca(
    coordinates: np.ndarray,
    categories: np.ndarray,
    category_names: list[str],
    explained: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    for code, name in enumerate(category_names):
        mask = categories == code
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=4,
            alpha=0.28,
            color=CATEGORY_COLORS.get(name, "#777777"),
            label=name,
            rasterized=True,
        )
    ax.set_xlabel(f"PC1 ({explained[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.2%} variance)")
    ax.set_title("PCA fitted on category-stratified training sample")
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_histograms(
    raw_sample: np.ndarray,
    categories: np.ndarray,
    category_names: list[str],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feature in zip(axes.flat, SELECTED_FEATURES):
        column = FEATURE_NAMES.index(feature)
        transformed = np.log1p(np.clip(raw_sample[:, column], 0.0, None))
        lo, hi = np.quantile(transformed, (0.005, 0.995))
        bins = np.linspace(lo, hi if hi > lo else lo + 1.0, 55)
        for code, name in enumerate(category_names):
            values = transformed[categories == code]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.1,
                color=CATEGORY_COLORS.get(name, "#777777"),
                label=name,
            )
        ax.set_title(feature)
        ax.set_xlabel("log1p(raw value)")
        ax.set_ylabel("Density")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Training feature distributions by category (stratified sample)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(processed_dir: Path, output_dir: Path, sample_per_category: int, seed: int) -> dict[str, Any]:
    _require(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = _json(processed_dir / "run_manifest.json")
    build_manifest = _json(processed_dir / "ciciot2023_labeled_full_manifest.json")
    if build_manifest.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Labelled build manifest feature order differs from schema.py")

    arrays = _load_arrays(processed_dir)
    scaler = _pickle(processed_dir / "scaler.pkl")
    fine_encoder = _pickle(processed_dir / "label_encoder.pkl")
    category_encoder = _pickle(processed_dir / "category_encoder.pkl")
    fine_names = [str(name) for name in fine_encoder.classes_]
    category_names = [str(name) for name in category_encoder.classes_]

    fine_table = _distribution_table(arrays, "fine", fine_names)
    category_table = _distribution_table(arrays, "category", category_names)
    fine_table.to_csv(output_dir / "fine_class_distribution_by_split.csv", index=False)
    category_table.to_csv(output_dir / "category_distribution_by_split.csv", index=False)

    feature_table, protocol_table, value_checks = _feature_summary(arrays["train"]["X"], scaler)
    feature_table.to_csv(output_dir / "train_feature_summary_raw.csv", index=False)
    protocol_table.to_csv(output_dir / "train_protocol_distribution.csv", index=False)

    sample_idx = _stratified_indices(
        arrays["train"]["category"], sample_per_category, seed
    )
    sample_scaled = np.asarray(arrays["train"]["X"][sample_idx], dtype=np.float64)
    sample_categories = np.asarray(arrays["train"]["category"][sample_idx])
    center = np.asarray(scaler.center_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    sample_raw = sample_scaled * scale + center

    correlation = pd.DataFrame(sample_raw, columns=FEATURE_NAMES).corr(method="spearman")
    correlation.to_csv(output_dir / "train_spearman_correlation.csv")

    pca_input = np.clip(sample_scaled, -10.0, 10.0)
    pca = PCA(n_components=2, random_state=seed)
    coordinates = pca.fit_transform(pca_input)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURE_NAMES,
        columns=("PC1", "PC2"),
    )
    loadings.index.name = "feature"
    loadings.to_csv(output_dir / "train_pca_loadings.csv")
    np.savez_compressed(
        output_dir / "train_pca_coordinates.npz",
        coordinates=coordinates.astype(np.float32),
        category_codes=sample_categories.astype(np.int32),
        source_train_indices=sample_idx,
    )

    _plot_distributions(category_table, output_dir / "category_distribution_by_split.png")
    _plot_correlation(correlation, output_dir / "train_spearman_correlation.png")
    _plot_pca(
        coordinates,
        sample_categories,
        category_names,
        pca.explained_variance_ratio_,
        output_dir / "train_pca_by_category.png",
    )
    _plot_feature_histograms(
        sample_raw,
        sample_categories,
        category_names,
        output_dir / "train_feature_distributions.png",
    )

    shapes = {
        split: {key: list(value.shape) for key, value in split_arrays.items()}
        for split, split_arrays in arrays.items()
    }
    expected_counts = run_manifest.get("split_row_counts", {})
    count_match = all(
        int(arrays[split]["X"].shape[0]) == int(expected_counts.get(split, -1))
        for split in ("train", "val", "test")
    )
    fine_coverage = (
        fine_table.groupby("split")["count"].apply(lambda values: bool((values > 0).all())).to_dict()
    )
    category_coverage = (
        category_table.groupby("split")["count"].apply(lambda values: bool((values > 0).all())).to_dict()
    )
    report: dict[str, Any] = {
        "dataset": "CICIoT2023",
        "schema_features": len(FEATURE_NAMES),
        "seed": seed,
        "source_build": {
            "files": int(build_manifest["n_files"]),
            "rows_read": int(build_manifest["total_rows_read"]),
            "rows_kept": int(build_manifest["kept_rows"]),
            "rows_dropped_nan_inf": int(build_manifest["dropped_nan_inf"]),
        },
        "preprocessing": {
            "run_timestamp": run_manifest.get("timestamp"),
            "git_hash": run_manifest.get("git_hash"),
            "split_protocol": run_manifest.get("split_protocol"),
            "train_before_sampling": run_manifest.get("train_before_sampling"),
            "train_after_sampling": run_manifest.get("train_after_sampling"),
            "split_row_counts": expected_counts,
        },
        "artifact_shapes": shapes,
        "fit_provenance": {
            "feature_summary": "all sampled training rows, inverse-transformed with the train-fitted scaler",
            "correlation": "category-stratified training sample only",
            "pca": "category-stratified training sample only; scaled values clipped to [-10, 10] for visualization",
            "validation_and_test": "counts only; never used to fit transforms",
        },
        "stratified_training_sample": {
            "per_category_cap": sample_per_category,
            "rows": int(sample_idx.size),
        },
        "pca_explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "constant_train_features": feature_table.loc[
            np.isclose(feature_table["std"], 0.0), "feature"
        ].tolist(),
        "undefined_spearman_cells": int(correlation.isna().to_numpy().sum()),
        "checks": {
            "manifest_feature_order_matches_schema": True,
            "array_counts_match_run_manifest": count_match,
            "all_fine_classes_present": fine_coverage,
            "all_categories_present": category_coverage,
            **value_checks,
        },
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.name != "eda_report.json"),
    }
    if not count_match or not all(fine_coverage.values()) or not all(category_coverage.values()):
        raise AssertionError(f"EDA contract check failed: {report['checks']}")
    (output_dir / "eda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=paths.PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-per-category", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=paths.SEED)
    args = parser.parse_args()
    if args.sample_per_category <= 0:
        parser.error("--sample-per-category must be positive")
    output_dir = args.output_dir or args.processed_dir / "ciciot2023_eda"
    report = run(args.processed_dir, output_dir, args.sample_per_category, args.seed)
    print(f"EDA complete: {output_dir}")
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
