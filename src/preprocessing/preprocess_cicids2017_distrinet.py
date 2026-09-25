#!/usr/bin/env python3
"""Preprocess the corrected DistriNet CIC-IDS-2017 release for two targets.

Only rows mapping to Benign, DoS, DDoS, Recon, or BruteForce are retained.
Filtering and category mapping happen before chronological 70/15/15 splits
within each source attack label. All fitted statistics use training rows only.

Run from the repository root:
    python src/preprocessing/preprocess_cicids2017_distrinet.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.preprocessing import RobustScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.paths import SEED  # noqa: E402

EXPECTED_FILES: dict[str, tuple[str, int, str]] = {
    "Monday-WorkingHours.csv": ("Monday", 0, "2017-07-03"),
    "Tuesday-WorkingHours.csv": ("Tuesday", 1, "2017-07-04"),
    "Wednesday-WorkingHours.csv": ("Wednesday", 2, "2017-07-05"),
    "Thursday-WorkingHours.csv": ("Thursday", 3, "2017-07-06"),
    "Friday-WorkingHours.csv": ("Friday", 4, "2017-07-07"),
}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = np.asarray((0.70, 0.15, 0.15), dtype=np.float64)
IDENTIFIER_COLUMNS = ("Flow ID", "Src IP", "Dst IP")
TIMESTAMP_COLUMN = "Timestamp"
LABEL_COLUMN = "Label"
NON_FEATURE_COLUMNS = set(IDENTIFIER_COLUMNS) | {TIMESTAMP_COLUMN, LABEL_COLUMN}
TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"
ATTEMPTED_SUFFIX = " - Attempted"
DEFAULT_MIN_PER_SPLIT_WARNING = 10
CATEGORY_NAMES = ("Benign", "DoS", "DDoS", "Recon", "BruteForce")
CATEGORY_TO_ID = {name: index for index, name in enumerate(CATEGORY_NAMES)}
SOURCE_TO_CATEGORY = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "Recon",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
}




@dataclass
class CleanedPart:
    features: pd.DataFrame
    metadata: pd.DataFrame
    category_labels: np.ndarray
    report: dict[str, Any]


@dataclass
class CleanedDataset:
    features: pd.DataFrame
    metadata: pd.DataFrame
    category_labels: np.ndarray
    duplicate_audit: dict[str, Any]
    file_reports: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "CICIDS_2017_Distrinet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "CICIDS_2017_Distrinet",
    )
    parser.add_argument(
        "--attempted-policy",
        choices=("benign", "parent"),
        default="benign",
        help=(
            "DistriNet forbids Attempted as a separate target. 'benign' is their "
            "recommended default; 'parent' maps it to the parent attack label."
        ),
    )
    parser.add_argument(
        "--min-per-split-warning",
        type=int,
        default=DEFAULT_MIN_PER_SPLIT_WARNING,
        help="Warn when any class partition contains fewer rows than this value.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Debug/smoke-test limit. Limited runs are marked non-production.",
    )
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="Skip SHA-256 input hashes only for a quick smoke test.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_column_name(value: Any) -> str:
    return str(value).lstrip("\ufeff").strip()


def normalize_label(value: Any) -> str:
    label = (
        str(value)
        .strip()
        .replace("\x96", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return "BENIGN" if label.upper() == "BENIGN" else label


def source_label_for_mapping(original_label: str, attempted_policy: str) -> str:
    if not original_label.endswith(ATTEMPTED_SUFFIX):
        return original_label
    if attempted_policy == "benign":
        return "BENIGN"
    return original_label[: -len(ATTEMPTED_SUFFIX)]


def normalized_header(path: Path) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    return [normalize_column_name(column) for column in columns]


def validate_inventory(input_dir: Path) -> tuple[list[Path], list[str], list[str]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    present = sorted(path.name for path in input_dir.glob("*.csv"))
    expected = sorted(EXPECTED_FILES)
    if present != expected:
        missing = sorted(set(expected) - set(present))
        extra = sorted(set(present) - set(expected))
        raise ValueError(f"CSV inventory mismatch; missing={missing}, extra={extra}")

    paths = [input_dir / name for name in EXPECTED_FILES]
    headers = [normalized_header(path) for path in paths]
    if any(header != headers[0] for header in headers[1:]):
        raise ValueError("CSV schemas differ across corrected DistriNet capture days")
    if len(headers[0]) != len(set(headers[0])):
        raise ValueError("duplicate normalized column names are not supported")

    required = set(IDENTIFIER_COLUMNS) | {TIMESTAMP_COLUMN, LABEL_COLUMN}
    missing_required = sorted(required - set(headers[0]))
    if missing_required:
        raise ValueError(f"required columns absent: {missing_required}")

    index_columns = [column for column in headers[0] if column.lower().startswith("unnamed:")]
    modelling_columns = [
        column
        for column in headers[0]
        if column not in NON_FEATURE_COLUMNS and column not in index_columns
    ]
    if not modelling_columns:
        raise ValueError("no modelling feature columns found")
    return paths, headers[0], modelling_columns


def parse_timestamps(raw: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    missing_input = raw.isna() | raw.astype(str).str.strip().eq("")
    strict = pd.to_datetime(raw, format=TIMESTAMP_FORMAT, errors="coerce", utc=True)
    fallback = pd.to_datetime(raw, format="mixed", dayfirst=True, errors="coerce", utc=True)
    fallback_used = strict.isna() & fallback.notna()
    parsed = strict.where(strict.notna(), fallback)
    report = {
        "missing_input": int(missing_input.sum()),
        "strict_format_failures": int(strict.isna().sum()),
        "mixed_format_recoveries": int(fallback_used.sum()),
        "unparseable": int(parsed.isna().sum()),
    }
    return parsed, report


def load_clean_file(
    path: Path,
    raw_columns: list[str],
    modelling_columns: list[str],
    attempted_policy: str,
    max_rows: int | None,
) -> CleanedPart:
    day_name, day_order, expected_date = EXPECTED_FILES[path.name]
    raw = pd.read_csv(path, nrows=max_rows, low_memory=False)
    raw.columns = [normalize_column_name(column) for column in raw.columns]
    if raw.columns.tolist() != raw_columns:
        raise ValueError(f"schema changed while reading {path.name}")

    n_raw = len(raw)
    source_row = np.arange(2, n_raw + 2, dtype=np.int64)
    original_labels = raw[LABEL_COLUMN].map(normalize_label)
    source_labels = np.asarray(
        [source_label_for_mapping(label, attempted_policy) for label in original_labels],
        dtype=str,
    )
    category_labels = np.asarray(
        [SOURCE_TO_CATEGORY.get(label, "") for label in source_labels],
        dtype=str,
    )
    timestamps, timestamp_report = parse_timestamps(raw[TIMESTAMP_COLUMN])
    numeric = raw[modelling_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=False)

    finite_numeric = np.isfinite(values).all(axis=1)
    valid_timestamp = timestamps.notna().to_numpy()
    negative_physical = (values < 0).any(axis=1)
    supported_category = category_labels != ""
    clean = finite_numeric & valid_timestamp & ~negative_physical
    valid = clean & supported_category
    keep = np.flatnonzero(valid)

    attempted = original_labels.str.endswith(ATTEMPTED_SUFFIX).to_numpy()[keep]
    metadata = pd.DataFrame(
        {
            "sample_id": [f"{path.name}:{row}" for row in source_row[keep]],
            "source_file": path.name,
            "source_day": day_name,
            "source_day_order": day_order,
            "source_row": source_row[keep],
            "Flow ID": raw.iloc[keep]["Flow ID"].astype(str).to_numpy(),
            "Src IP": raw.iloc[keep]["Src IP"].astype(str).to_numpy(),
            "Dst IP": raw.iloc[keep]["Dst IP"].astype(str).to_numpy(),
            "Timestamp": timestamps.iloc[keep].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
            "timestamp_epoch_seconds": timestamps.iloc[keep]
            .dt.as_unit("s")
            .astype("int64")
            .to_numpy(dtype=np.int64),
            "original_label": original_labels.iloc[keep].to_numpy(dtype=str),
            "source_label": source_labels[keep],
            "category_label": category_labels[keep],
            "is_attempted": attempted.astype(np.uint8),
        }
    )
    # Canonicalize to the exact float32 representation consumed by the models
    # before duplicate detection.
    features = numeric.iloc[keep].astype(np.float32).reset_index(drop=True)
    kept_labels = category_labels[keep]

    observed_dates = sorted(
        pd.to_datetime(metadata["Timestamp"], utc=True).dt.strftime("%Y-%m-%d").unique()
    )
    if observed_dates != [expected_date]:
        raise ValueError(f"{path.name}: expected date {expected_date}, found {observed_dates}")

    report = {
        "day": day_name,
        "raw_rows": n_raw,
        "timestamp": timestamp_report,
        "dropped_nonfinite_numeric": int((~finite_numeric).sum()),
        "dropped_bad_timestamp_additional": int((~valid_timestamp & finite_numeric).sum()),
        "dropped_negative_physical_value_additional": int(
            (negative_physical & finite_numeric & valid_timestamp).sum()
        ),
        "dropped_unsupported_category_after_cleaning": int((clean & ~supported_category).sum()),
        "kept_before_global_dedup": len(features),
        "original_label_counts_after_filtering": dict(
            sorted(Counter(metadata["original_label"]).items())
        ),
        "source_label_counts_after_filtering": dict(
            sorted(Counter(metadata["source_label"]).items())
        ),
        "category_label_counts_after_filtering": dict(sorted(Counter(kept_labels).items())),
    }
    return CleanedPart(features, metadata, kept_labels, report)


def merge_clean_and_deduplicate(
    parts: list[CleanedPart],
    modelling_columns: list[str],
) -> CleanedDataset:
    features = pd.concat([part.features for part in parts], ignore_index=True)
    metadata = pd.concat([part.metadata for part in parts], ignore_index=True)
    labels = np.concatenate([part.category_labels for part in parts])

    # Preserve deterministic chronology before choosing which duplicate survives.
    order = np.lexsort(
        (
            metadata["source_row"].to_numpy(),
            metadata["source_day_order"].to_numpy(),
            metadata["timestamp_epoch_seconds"].to_numpy(),
        )
    )
    features = features.iloc[order].reset_index(drop=True)
    metadata = metadata.iloc[order].reset_index(drop=True)
    labels = labels[order]

    rows_before = len(features)
    # Deduplicate on the exact target consumed by both classifier heads.
    features["__category_label_for_dedup__"] = labels
    duplicate_mask = features.duplicated(keep="first").to_numpy()
    features.drop(columns="__category_label_for_dedup__", inplace=True)
    duplicate_count = int(duplicate_mask.sum())
    keep = ~duplicate_mask
    features = features.loc[keep, modelling_columns].reset_index(drop=True)
    metadata = metadata.loc[keep].reset_index(drop=True)
    labels = labels[keep]
    metadata["record_id"] = np.arange(len(metadata), dtype=np.int64)

    if not metadata["sample_id"].is_unique or not metadata["record_id"].is_unique:
        raise AssertionError("canonical sample identifiers are not globally unique")

    duplicate_audit = {
        "definition": "exact equality of every float32 modelling feature plus category_label",
        "rows_before_duplicate_removal": rows_before,
        "exact_duplicates_removed": duplicate_count,
        "duplicate_percentage": 100.0 * duplicate_count / rows_before,
        "rows_after_duplicate_removal": len(features),
    }
    file_reports = {part.metadata["source_file"].iloc[0]: part.report for part in parts}
    return CleanedDataset(features, metadata, labels, duplicate_audit, file_reports)


def allocate_class_counts(n_rows: int) -> np.ndarray:
    """Largest-remainder 70/15/15 allocation with one row per split."""
    if n_rows < len(SPLIT_NAMES):
        raise ValueError(
            f"class has only {n_rows} rows; at least {len(SPLIT_NAMES)} are required "
            "to represent it in train, validation, and test"
        )
    counts = np.ones(len(SPLIT_NAMES), dtype=np.int64)
    remaining = n_rows - len(SPLIT_NAMES)
    desired_extra = np.maximum(SPLIT_RATIOS * n_rows - 1.0, 0.0)
    base_extra = np.floor(desired_extra).astype(np.int64)
    if base_extra.sum() > remaining:
        # Defensive only; positive ratios summing to one should not reach this.
        base_extra[:] = 0
    counts += base_extra
    remainder = remaining - int(base_extra.sum())
    fractions = desired_extra - base_extra
    priority = np.argsort(-fractions, kind="stable")
    for i in range(remainder):
        counts[priority[i % len(priority)]] += 1
    if counts.sum() != n_rows or (counts < 1).any():
        raise AssertionError(f"invalid class allocation for n={n_rows}: {counts.tolist()}")
    return counts


def chronological_within_source_label_split(
    dataset: CleanedDataset,
    warning_floor: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, list[str]]:
    source_labels = dataset.metadata["source_label"].to_numpy(dtype=str)
    observed = set(source_labels)
    expected = set(SOURCE_TO_CATEGORY)
    if observed != expected:
        raise ValueError(f"expected source labels {sorted(expected)}, found {sorted(observed)}")

    split_indices: dict[str, list[np.ndarray]] = {name: [] for name in SPLIT_NAMES}
    source_rows: list[dict[str, Any]] = []
    warnings_list: list[str] = []

    for source_label, category_label in SOURCE_TO_CATEGORY.items():
        indices = np.flatnonzero(source_labels == source_label)
        # The merged dataset is globally ordered by timestamp/day/source row, so
        # filtering retains deterministic chronology for this attack subtype.
        counts = allocate_class_counts(len(indices))
        boundaries = np.cumsum(counts)
        source_slices = np.split(indices, boundaries[:-1])
        for name, selected in zip(SPLIT_NAMES, source_slices):
            split_indices[name].append(selected)

        row: dict[str, Any] = {
            "source_label": source_label,
            "category_label": category_label,
            "total": len(indices),
        }
        for name, count in zip(SPLIT_NAMES, counts):
            row[name] = int(count)
            row[f"{name}_pct_of_source"] = 100.0 * int(count) / len(indices)
        too_small = [
            f"{name}={int(count)}"
            for name, count in zip(SPLIT_NAMES, counts)
            if count < warning_floor
        ]
        row["small_partition_warning"] = "; ".join(too_small)
        if too_small:
            warning = (
                f"{source_label}: small partition(s) {', '.join(too_small)}; estimates for "
                "this source label will be unstable"
            )
            warnings_list.append(warning)
            warnings.warn(warning, RuntimeWarning, stacklevel=2)
        source_rows.append(row)

    final_indices: dict[str, np.ndarray] = {}
    for name in SPLIT_NAMES:
        combined = np.concatenate(split_indices[name]).astype(np.int64)
        final_indices[name] = np.sort(combined)

    all_indices = np.concatenate([final_indices[name] for name in SPLIT_NAMES])
    if len(np.unique(all_indices)) != len(dataset.features) or len(all_indices) != len(dataset.features):
        raise AssertionError("split membership is not a disjoint exhaustive partition")

    source_report = pd.DataFrame(source_rows)
    category_rows: list[dict[str, Any]] = []
    for category_label in CATEGORY_NAMES:
        category_mask = dataset.category_labels == category_label
        row = {"category_label": category_label, "total": int(category_mask.sum())}
        for name in SPLIT_NAMES:
            count = int(category_mask[final_indices[name]].sum())
            row[name] = count
            row[f"{name}_pct_of_class"] = 100.0 * count / row["total"]
        row["small_partition_warning"] = ""
        category_rows.append(row)
    category_report = pd.DataFrame(category_rows)

    for report in (category_report, source_report):
        for name in SPLIT_NAMES:
            split_total = int(report[name].sum())
            report[f"{name}_pct_of_split"] = 100.0 * report[name] / split_total
    return final_indices, category_report, source_report, warnings_list


def fingerprint(values: pd.DataFrame) -> np.ndarray:
    return hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)


def overlap_record(left: np.ndarray, right: np.ndarray) -> dict[str, int]:
    left_unique = np.unique(left)
    right_unique = np.unique(right)
    shared = np.intersect1d(left_unique, right_unique, assume_unique=True)
    return {
        "shared_unique_fingerprints": int(len(shared)),
        "left_rows_with_shared_fingerprint": int(np.isin(left, shared).sum()),
        "right_rows_with_shared_fingerprint": int(np.isin(right, shared).sum()),
    }


def build_leakage_audit(
    dataset: CleanedDataset,
    split_indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    feature_hash = fingerprint(dataset.features)
    labelled_frame = dataset.features.copy(deep=False)
    labelled_frame = labelled_frame.assign(__category_label__=dataset.category_labels)
    labelled_hash = fingerprint(labelled_frame)
    sample_ids = dataset.metadata["sample_id"].to_numpy(dtype=str)

    pairwise: dict[str, Any] = {}
    for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
        left_idx = split_indices[left_name]
        right_idx = split_indices[right_name]
        key = f"{left_name}_vs_{right_name}"
        shared_ids = np.intersect1d(sample_ids[left_idx], sample_ids[right_idx]).size
        labelled_overlap = overlap_record(labelled_hash[left_idx], labelled_hash[right_idx])
        feature_overlap = overlap_record(feature_hash[left_idx], feature_hash[right_idx])
        shared_feature_hashes = np.intersect1d(
            np.unique(feature_hash[left_idx]),
            np.unique(feature_hash[right_idx]),
            assume_unique=True,
        )
        feature_overlap_examples: list[dict[str, Any]] = []
        for shared_hash in shared_feature_hashes[:20]:
            left_global = int(left_idx[np.flatnonzero(feature_hash[left_idx] == shared_hash)[0]])
            right_global = int(right_idx[np.flatnonzero(feature_hash[right_idx] == shared_hash)[0]])
            left_meta = dataset.metadata.iloc[left_global]
            right_meta = dataset.metadata.iloc[right_global]
            feature_overlap_examples.append(
                {
                    "fingerprint_u64": int(shared_hash),
                    left_name: {
                        "sample_id": str(left_meta["sample_id"]),
                        "source_file": str(left_meta["source_file"]),
                        "source_row": int(left_meta["source_row"]),
                        "timestamp": str(left_meta["Timestamp"]),
                        "category_label": str(dataset.category_labels[left_global]),
                    },
                    right_name: {
                        "sample_id": str(right_meta["sample_id"]),
                        "source_file": str(right_meta["source_file"]),
                        "source_row": int(right_meta["source_row"]),
                        "timestamp": str(right_meta["Timestamp"]),
                        "category_label": str(dataset.category_labels[right_global]),
                    },
                }
            )
        feature_overlap["examples_first_20"] = feature_overlap_examples
        pairwise[key] = {
            "shared_sample_ids": int(shared_ids),
            "feature_plus_label_overlap": labelled_overlap,
            "feature_only_overlap": feature_overlap,
        }
        if shared_ids:
            raise AssertionError(f"{key}: shared canonical sample IDs")
        if labelled_overlap["shared_unique_fingerprints"]:
            raise AssertionError(f"{key}: exact feature+label duplicates survived deduplication")

    coverage: dict[str, dict[str, int]] = {}
    source_chronology: dict[str, dict[str, dict[str, int]]] = {}
    source_labels = dataset.metadata["source_label"].to_numpy(dtype=str)
    timestamps = dataset.metadata["timestamp_epoch_seconds"].to_numpy(dtype=np.int64)
    for source_label in SOURCE_TO_CATEGORY:
        label_indices = {
            name: indices[source_labels[indices] == source_label]
            for name, indices in split_indices.items()
        }
        coverage[source_label] = {name: int(len(indices)) for name, indices in label_indices.items()}
        if any(count == 0 for count in coverage[source_label].values()):
            raise AssertionError(
                f"source-label coverage failure for {source_label}: {coverage[source_label]}"
            )
        source_chronology[source_label] = {
            name: {
                "min_epoch_seconds": int(timestamps[indices].min()),
                "max_epoch_seconds": int(timestamps[indices].max()),
            }
            for name, indices in label_indices.items()
        }
        boundaries = source_chronology[source_label]
        if not (
            boundaries["train"]["max_epoch_seconds"]
            <= boundaries["val"]["min_epoch_seconds"]
            <= boundaries["test"]["min_epoch_seconds"]
        ):
            raise AssertionError(
                f"within-source chronology failure for {source_label}: {boundaries}"
            )

    return {
        "membership_is_disjoint_and_exhaustive": True,
        "source_label_coverage_asserted_all_splits": True,
        "pairwise": pairwise,
        "source_label_coverage": coverage,
        "source_label_chronology_asserted": True,
        "source_label_chronology_epoch_seconds": source_chronology,
        "feature_only_overlap_note": (
            "Feature-only overlap can remain when an identical numeric vector has different "
            "category labels. Feature+category duplicates were removed globally before splitting."
        ),
    }


def encoded_targets(category_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    category = np.asarray([CATEGORY_TO_ID[label] for label in category_labels], dtype=np.int8)
    binary = (category != CATEGORY_TO_ID["Benign"]).astype(np.int8)
    return category, binary


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"cannot compute class weights with empty classes: {counts.tolist()}")
    return (len(labels) / (num_classes * counts)).astype(np.float32)


def save_split(
    output_dir: Path,
    name: str,
    dataset: CleanedDataset,
    indices: np.ndarray,
    modelling_columns: list[str],
    scaler: RobustScaler,
) -> dict[str, Any]:
    pristine = np.ascontiguousarray(
        dataset.features.iloc[indices][modelling_columns].to_numpy(dtype=np.float32)
    )
    scaled = scaler.transform(pristine).astype(np.float32)
    category_labels = dataset.category_labels[indices]
    y_category, y_binary = encoded_targets(category_labels)
    metadata = dataset.metadata.iloc[indices].reset_index(drop=True).copy()
    metadata["category_label"] = category_labels
    metadata["binary_label"] = y_binary

    if not np.isfinite(pristine).all() or not np.isfinite(scaled).all():
        raise AssertionError(f"{name}: output matrix contains NaN/Inf")
    if (pristine < 0).any():
        raise AssertionError(f"{name}: pristine matrix contains a negative physical value")
    if len({len(pristine), len(metadata), len(y_category), len(y_binary)}) != 1:
        raise AssertionError(f"{name}: output rows are not aligned")

    sample_n = min(10_000, len(pristine))
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(pristine), size=sample_n, replace=False)
    roundtrip = scaler.inverse_transform(scaled[sample_idx])
    absolute_error = np.abs(roundtrip - pristine[sample_idx])
    tolerance = 1e-3 + 1e-5 * np.maximum(
        np.abs(pristine[sample_idx]), np.asarray(scaler.scale_, dtype=np.float32)
    )
    if np.any(absolute_error > tolerance):
        raise AssertionError(f"{name}: scaler round-trip failed")

    np.save(output_dir / f"X_{name}.npy", scaled)
    np.save(output_dir / f"X_{name}_pristine.npy", pristine)
    np.save(output_dir / f"y_{name}_cat.npy", y_category)
    np.save(output_dir / f"y_{name}_bin.npy", y_binary)
    np.save(
        output_dir / f"timestamp_epoch_seconds_{name}.npy",
        metadata["timestamp_epoch_seconds"].to_numpy(dtype=np.int64),
    )

    # The project already uses Parquet. This table is the reproducible, inspectable
    # split dataset; metadata remains outside the model NumPy feature matrix.
    parquet = metadata.copy()
    for column_index, column in enumerate(modelling_columns):
        parquet[column] = pristine[:, column_index]
    parquet.to_parquet(output_dir / f"{name}.parquet", index=False)

    return {
        "rows": len(pristine),
        "shape": list(scaled.shape),
        "timestamp_min": metadata["Timestamp"].min(),
        "timestamp_max": metadata["Timestamp"].max(),
        "category_label_counts": dict(sorted(Counter(category_labels).items())),
        "binary_counts": {
            "benign": int((y_binary == 0).sum()),
            "attack": int((y_binary == 1).sum()),
        },
        "max_scaler_roundtrip_abs_error": float(absolute_error.max()),
    }


def print_reports(
    duplicate_audit: dict[str, Any],
    class_report: pd.DataFrame,
    source_report: pd.DataFrame,
    leakage_audit: dict[str, Any],
) -> None:
    print("\nDuplicate audit (modelling features + category label)")
    print(json.dumps(duplicate_audit, indent=2))
    class_columns = [
        "category_label",
        "total",
        "train",
        "train_pct_of_class",
        "val",
        "val_pct_of_class",
        "test",
        "test_pct_of_class",
    ]
    print("\nPer-category split")
    print(class_report[class_columns].to_string(index=False))
    source_columns = [
        "source_label",
        "category_label",
        "total",
        "train",
        "val",
        "test",
        "small_partition_warning",
    ]
    print("\nChronological split within each source attack label")
    print(source_report[source_columns].to_string(index=False))
    print("\nPairwise leakage audit")
    for pair, record in leakage_audit["pairwise"].items():
        print(
            f"  {pair}: sample_ids={record['shared_sample_ids']}, "
            f"feature+label={record['feature_plus_label_overlap']['shared_unique_fingerprints']}, "
            f"feature_only={record['feature_only_overlap']['shared_unique_fingerprints']}"
        )


def main() -> None:
    args = parse_args()
    if args.min_per_split_warning < 1:
        raise ValueError("--min-per-split-warning must be >= 1")
    started = time.time()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        legacy = output_dir / f"y_{split}_fine.npy"
        if legacy.exists():
            legacy.unlink()

    paths, raw_columns, modelling_columns = validate_inventory(input_dir)
    print(
        f"Validated {len(paths)} corrected DistriNet files with "
        f"{len(raw_columns)} columns ({len(modelling_columns)} modelling features)"
    )

    parts: list[CleanedPart] = []
    for path in paths:
        part = load_clean_file(
            path,
            raw_columns=raw_columns,
            modelling_columns=modelling_columns,
            attempted_policy=args.attempted_policy,
            max_rows=args.max_rows_per_file,
        )
        parts.append(part)
        print(
            f"{path.name}: {part.report['raw_rows']:,} -> "
            f"{part.report['kept_before_global_dedup']:,} after cleaning/category filtering"
        )

    dataset = merge_clean_and_deduplicate(parts, modelling_columns)
    split_indices, class_report, source_report, small_class_warnings = (
        chronological_within_source_label_split(
            dataset,
            warning_floor=args.min_per_split_warning,
        )
    )

    # All fitted preprocessing begins here, after immutable split membership exists.
    train_matrix = np.ascontiguousarray(
        dataset.features.iloc[split_indices["train"]][modelling_columns].to_numpy(
            dtype=np.float32
        )
    )
    train_constant_columns = [
        column
        for column_index, column in enumerate(modelling_columns)
        if np.min(train_matrix[:, column_index]) == np.max(train_matrix[:, column_index])
    ]
    # Constants are reported, not removed: removing a train-constant field can
    # erase a value that appears only in val/test and create new cross-split duplicates.
    scaler = RobustScaler(copy=True)
    scaler.fit(train_matrix)
    with (output_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    leakage_audit = build_leakage_audit(dataset, split_indices)
    split_reports = {
        name: save_split(
            output_dir,
            name,
            dataset,
            split_indices[name],
            modelling_columns,
            scaler,
        )
        for name in SPLIT_NAMES
    }

    y_train_category, y_train_binary = encoded_targets(
        dataset.category_labels[split_indices["train"]]
    )
    category_weights = balanced_class_weights(y_train_category, len(CATEGORY_NAMES))
    binary_weights = balanced_class_weights(y_train_binary, 2)
    np.save(output_dir / "class_weights_5.npy", category_weights)
    np.save(output_dir / "class_weights_2.npy", binary_weights)
    label_encoders = {
        "binary": {"Benign": 0, "Attack": 1},
        "category": CATEGORY_TO_ID,
    }
    with (output_dir / "label_encoders.json").open("w", encoding="utf-8") as handle:
        json.dump(label_encoders, handle, indent=2)
        handle.write("\n")

    class_report.to_csv(output_dir / "class_distribution.csv", index=False)
    source_report.to_csv(output_dir / "source_label_distribution.csv", index=False)
    with (output_dir / "duplicate_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset.duplicate_audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "leakage_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(leakage_audit, handle, indent=2, sort_keys=True)
        handle.write("\n")

    input_hashes = None
    if not args.skip_input_hashes:
        input_hashes = {path.name: sha256_file(path) for path in paths}

    manifest = {
        "dataset": "corrected/relabelled DistriNet CIC-IDS-2017 five-file release",
        "methodological_description": (
            "Rows are mapped to five supported categories and filtered before leakage-controlled "
            "chronological splits within each retained source attack label. This preserves source-"
            "label coverage in train, validation, and test while exact duplicates are removed "
            "before splitting and preprocessing statistics are train-fitted."
        ),
        "not_claimed": "global forward-time or independent attack-campaign generalization",
        "research_target": "binary and five-category closed-set classification",
        "seed": SEED,
        "non_production": args.max_rows_per_file is not None,
        "max_rows_per_file": args.max_rows_per_file,
        "input_dir": str(input_dir),
        "input_sha256": input_hashes,
        "input_files": [path.name for path in paths],
        "raw_columns": raw_columns,
        "raw_column_count": len(raw_columns),
        "modelling_feature_names": modelling_columns,
        "modelling_feature_count": len(modelling_columns),
        "metadata_excluded_from_X": [
            "sample_id",
            "record_id",
            "source_file",
            "source_day",
            "source_day_order",
            "source_row",
            "Flow ID",
            "Src IP",
            "Dst IP",
            "Timestamp",
            "timestamp_epoch_seconds",
            "original_label",
            "source_label",
            "category_label",
            "is_attempted",
            "binary_label",
        ],
        "timestamp_policy": {
            "primary_format": TIMESTAMP_FORMAT,
            "fallback": "mixed day-first parsing",
            "timezone": "UTC per DistriNet documentation; local strings are timezone-naive",
            "tie_breakers": ["source_day_order", "source_row"],
            "classifier_input": False,
        },
        "label_policy": {
            "heads": ["binary", "category"],
            "source_to_category": SOURCE_TO_CATEGORY,
            "retained_categories": list(CATEGORY_NAMES),
            "category_to_id": CATEGORY_TO_ID,
            "attempted_policy": args.attempted_policy,
            "binary_to_id": {"Benign": 0, "Attack": 1},
            "mapping_and_filter_stage": "before split membership assignment",
        },
        "cleaning_policy": {
            "fitted_statistics_used_before_split": False,
            "drop_unparseable_or_nonfinite": True,
            "drop_negative_physical_values": True,
            "imputation": None,
            "winsorization": None,
            "balancing": None,
        },
        "duplicate_audit": dataset.duplicate_audit,
        "split_policy": {
            "protocol": "chronological within each retained source attack label",
            "rationale": (
                "Closed-set category classification requires every retained source attack label "
                "to be represented in training; splitting only after category aggregation placed "
                "DoS GoldenEye exclusively in test."
            ),
            "ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS.tolist())),
            "allocation": "largest remainder with at least one row per split",
            "shuffle_before_assignment": False,
            "retained_classes": list(CATEGORY_NAMES),
            "retained_source_labels": list(SOURCE_TO_CATEGORY),
            "small_partition_warning_floor": args.min_per_split_warning,
            "small_source_label_warnings": small_class_warnings,
        },
        "file_reports": dataset.file_reports,
        "class_distribution": class_report.to_dict(orient="records"),
        "source_label_distribution": source_report.to_dict(orient="records"),
        "leakage_audit": leakage_audit,
        "train_only_preprocessing": {
            "scaler": "sklearn.preprocessing.RobustScaler",
            "scaler_fit_rows": len(train_matrix),
            "train_constant_columns_reported_not_removed": train_constant_columns,
            "VAE_training_allowed_split": "train only",
            "constraint_mining_allowed_split": "train only",
            "validation_role": "classifier/attack hyperparameter selection",
            "test_role": "final reporting only",
            "class_weights": {
                "binary": binary_weights.tolist(),
                "category": category_weights.tolist(),
                "formula": "n_samples / (n_classes * class_count), train labels only",
            },
        },
        "split_reports": split_reports,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    with (output_dir / "preprocessing_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print_reports(dataset.duplicate_audit, class_report, source_report, leakage_audit)
    print(f"\nWrote reproducible artifacts to {output_dir}")
    for name in SPLIT_NAMES:
        print(f"  {name}: {split_reports[name]['shape']}")


if __name__ == "__main__":
    main()
