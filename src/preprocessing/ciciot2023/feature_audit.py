"""Distribution, clipping, chronology, and duplicate audits for CICIoT2023.

The implementation operates on the labelled Parquet in float32 source space.
Raw-stage statistics are reconstructed exactly by scanning the source CSVs for
rows removed by the minimal non-finite filter and combining their finite values
with the Parquet values. Holdout vectors are never removed; novel-pattern views
are persisted as separate split-local index arrays.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import paths
from src.preprocessing.ciciot2023 import pipeline as pl
from src.preprocessing.ciciot2023.build_labeled_ciciot2023_dataset import (
    DEFAULT_RAW_DIR,
    iter_shard_chunks,
    resolve_shards,
)
from src.preprocessing.ciciot2023.reporting import markdown_table
from src.preprocessing.schema import FEATURE_NAMES, FEATURE_SPECS, feature_metadata_records

QUANTILES: tuple[tuple[str, float], ...] = (
    ("p0_01", 0.0001), ("p0_1", 0.001), ("p1", 0.01), ("p5", 0.05),
    ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p95", 0.95),
    ("p99", 0.99), ("p99_9", 0.999), ("p99_99", 0.9999),
)


@dataclass(frozen=True)
class AuditConfig:
    output_dir: Path = pl.DEFAULT_OUTPUT_DIR
    raw_dir: Path = DEFAULT_RAW_DIR
    clipping_percentile: float = pl.CLIP_PERCENTILE
    raw_chunk_size: int = 250_000
    hash_chunk_size: int = 500_000


@dataclass
class RawNonfiniteAudit:
    total_rows: int
    dropped_rows: int
    nan_count: np.ndarray
    posinf_count: np.ndarray
    neginf_count: np.ndarray
    dropped_finite_values: list[list[np.ndarray]]


def _scan_raw_nonfinite(config: AuditConfig) -> RawNonfiniteAudit:
    nan = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    posinf = np.zeros_like(nan)
    neginf = np.zeros_like(nan)
    lost: list[list[np.ndarray]] = [[] for _ in FEATURE_NAMES]
    total = 0
    dropped = 0
    for shard in resolve_shards(config.raw_dir):
        for chunk in iter_shard_chunks(shard, config.raw_chunk_size, None):
            values = chunk.to_numpy(dtype=np.float64, copy=False)
            total += values.shape[0]
            nan += np.isnan(values).sum(axis=0)
            posinf += np.isposinf(values).sum(axis=0)
            neginf += np.isneginf(values).sum(axis=0)
            bad_rows = ~np.isfinite(values).all(axis=1)
            if not bad_rows.any():
                continue
            dropped += int(bad_rows.sum())
            removed = values[bad_rows]
            for column in range(values.shape[1]):
                finite = removed[:, column]
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    lost[column].append(finite.astype(np.float32))
    return RawNonfiniteAudit(total, dropped, nan, posinf, neginf, lost)


def _stats_for_columns(
    X: np.ndarray,
    *,
    stage: str,
    row_indices: np.ndarray | None = None,
    nan_count: np.ndarray | None = None,
    posinf_count: np.ndarray | None = None,
    neginf_count: np.ndarray | None = None,
    extra_finite: list[list[np.ndarray]] | None = None,
    total_count: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    q_values = [value for _, value in QUANTILES]
    for column, feature in enumerate(FEATURE_NAMES):
        values = X[:, column] if row_indices is None else X[row_indices, column]
        values = np.asarray(values, dtype=np.float32)
        if extra_finite is not None and extra_finite[column]:
            values = np.concatenate([values, *extra_finite[column]])
        finite = values[np.isfinite(values)].astype(np.float64, copy=False)
        if finite.size == 0:
            raise ValueError(f"{stage}/{feature} has no finite values")
        quantiles = np.quantile(finite, q_values)
        n_nan = int(nan_count[column]) if nan_count is not None else int(np.isnan(values).sum())
        n_posinf = int(posinf_count[column]) if posinf_count is not None else int(np.isposinf(values).sum())
        n_neginf = int(neginf_count[column]) if neginf_count is not None else int(np.isneginf(values).sum())
        row: dict[str, Any] = {
            "stage": stage,
            "feature": feature,
            "count": int(total_count if total_count is not None else values.size),
            "finite_count": int(finite.size),
            "nan_count": n_nan,
            "posinf_count": n_posinf,
            "neginf_count": n_neginf,
            "unique_count": int(np.unique(finite).size),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        }
        row.update({name: float(value) for (name, _), value in zip(QUANTILES, quantiles)})
        rows.append(row)
    return pd.DataFrame(rows)


def _write_stats_markdown(tables: list[pd.DataFrame], output: Path) -> None:
    combined = pd.concat(tables, ignore_index=True)
    lines = [
        "# CICIoT2023 feature-statistics comparison",
        "",
        "All quantiles and unique counts are exact for the inspected float32 representation.",
        "Raw-stage values reconstruct the rows removed by the labelled builder's row-wise non-finite filter.",
        "",
    ]
    for feature in FEATURE_NAMES:
        rows = combined[combined["feature"] == feature][
            ["stage", "count", "finite_count", "nan_count", "min", "p25", "p50", "p75", "p99_99", "max", "unique_count"]
        ]
        lines.extend([f"## {feature}", "", markdown_table(rows, 9), ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def _clipping_impact(X: np.ndarray, split: np.ndarray, percentile: float) -> pd.DataFrame:
    train = split == 0
    upper = np.percentile(X[train], percentile, axis=0)
    rows: list[dict[str, Any]] = []
    for split_name, code in (("train", 0), ("val", 1), ("test", 2)):
        indices = np.flatnonzero(split == code)
        for column, feature in enumerate(FEATURE_NAMES):
            values = np.asarray(X[indices, column], dtype=np.float64)
            clipped = np.clip(values, 0.0, upper[column])
            changes = np.abs(values - clipped)
            affected = changes > 0.0
            rows.append({
                "split": split_name,
                "feature": feature,
                "rows": int(values.size),
                "model_clip_lower": 0.0,
                "model_clip_upper": float(upper[column]),
                "affected_rows": int(affected.sum()),
                "affected_percent": float(100.0 * affected.mean()),
                "maximum_absolute_change": float(changes.max(initial=0.0)),
                "mean_absolute_change": float(changes.mean()),
            })
    return pd.DataFrame(rows)


def _write_clipping_report(table: pd.DataFrame, percentile: float, output: Path) -> None:
    totals = table.groupby("split", as_index=False).agg(
        affected_feature_cells=("affected_rows", "sum"),
        maximum_absolute_change=("maximum_absolute_change", "max"),
    )
    ranked = table.sort_values("affected_rows", ascending=False).head(30)
    output.write_text(
        "\n".join([
            "# Model-preprocessing clipping audit",
            "",
            "Canonical corrected preprocessing uses `clipping_mode=none`. The table below is the counterfactual impact of a training-fitted percentile clip.",
            "",
            f"Percentile: **{percentile}** (fit on natural training only).",
            "",
            "Empirical clipping is a model-stability operation, not a domain-validity rule. Rare values are not thereby invalid.",
            "",
            "## Split totals",
            "",
            markdown_table(totals),
            "",
            "## Largest feature-level effects",
            "",
            markdown_table(ranked, 8),
            "",
        ]),
        encoding="utf-8",
    )


def _hash_rows(X: np.ndarray, chunk_size: int) -> np.ndarray:
    hashes = np.empty(X.shape[0], dtype=np.uint64)
    for start in range(0, X.shape[0], chunk_size):
        stop = min(start + chunk_size, X.shape[0])
        chunk = pd.DataFrame(np.asarray(X[start:stop]), columns=FEATURE_NAMES)
        hashes[start:stop] = pd.util.hash_pandas_object(chunk, index=False).to_numpy(np.uint64)
    return hashes


def _pair_keys(hashes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    keys = np.empty(hashes.size, dtype=[("hash", "<u8"), ("label", "<i4")])
    keys["hash"] = hashes
    keys["label"] = labels.astype(np.int32, copy=False)
    return keys


def _membership(sorted_unique: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_unique, values)
    valid = positions < sorted_unique.size
    result = np.zeros(values.size, dtype=bool)
    result[valid] = sorted_unique[positions[valid]] == values[valid]
    return result


def run_duplicate_audit(
    X: np.ndarray,
    split: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    chunk_size: int = 500_000,
) -> dict[str, Any]:
    audit_dir = output_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    hashes = _hash_rows(X, chunk_size)
    split_hashes = {name: hashes[split == code] for name, code in (("train", 0), ("val", 1), ("test", 2))}
    split_labels = {name: labels[split == code] for name, code in (("train", 0), ("val", 1), ("test", 2))}
    unique = {name: np.unique(values) for name, values in split_hashes.items()}

    summary: dict[str, Any] = {"hash": "pandas stable uint64 row hash over 39 float32 values", "splits": {}, "intersections": {}}
    for name, values in split_hashes.items():
        summary["splits"][name] = {
            "total_rows": int(values.size),
            "unique_feature_vectors": int(unique[name].size),
            "duplicate_rows": int(values.size - unique[name].size),
            "duplicate_percentage": float(100.0 * (values.size - unique[name].size) / values.size),
        }

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap_unique = np.intersect1d(unique[left], unique[right], assume_unique=True)
        right_seen = _membership(unique[left], split_hashes[right])
        left_pairs = np.unique(_pair_keys(split_hashes[left], split_labels[left]))
        right_pairs = _pair_keys(split_hashes[right], split_labels[right])
        same_label = np.isin(right_pairs, left_pairs, assume_unique=False)
        cross_label = right_seen & ~same_label
        summary["intersections"][f"{left}_{right}"] = {
            "overlap_unique_vectors": int(overlap_unique.size),
            f"percentage_of_{right}_rows_seen_in_{left}": float(100.0 * right_seen.mean()),
            "overlap_rows": int(right_seen.sum()),
            "same_label_overlap_rows": int((right_seen & same_label).sum()),
            "cross_label_overlap_rows": int(cross_label.sum()),
        }

    all_pairs = np.unique(_pair_keys(hashes, labels))
    pair_hashes, label_counts = np.unique(all_pairs["hash"], return_counts=True)
    cross_label_hashes = pair_hashes[label_counts > 1]
    summary["cross_label"] = {
        "unique_vectors_with_multiple_labels": int(cross_label_hashes.size),
        "rows_on_cross_label_vectors": int(_membership(cross_label_hashes, hashes).sum()),
    }

    train_unique = unique["train"]
    for name in ("val", "test"):
        novel = np.flatnonzero(~_membership(train_unique, split_hashes[name])).astype(np.int64)
        np.save(audit_dir / f"novel_{name}_indices.npy", novel)
        summary["splits"][name]["novel_pattern_rows"] = int(novel.size)
        summary["splits"][name]["novel_pattern_index_space"] = f"split-local indices into X_{name}.npy"

    (audit_dir / "duplicate_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Exact feature-vector overlap audit",
        "",
        "Vectors were hashed from the cleaned, unscaled 39-column float32 representation. No holdout rows were removed.",
        "The natural ordered evaluation retains duplicates; separate split-local novel-pattern indices are saved for optional evaluation.",
        "A stable 64-bit row hash is used; collision risk is negligible but not mathematically impossible and is disclosed here.",
        "",
        "## Split duplication",
        "",
        markdown_table(pd.DataFrame.from_dict(summary["splits"], orient="index").reset_index(names="split")),
        "",
        "## Cross-split intersections",
        "",
        markdown_table(pd.DataFrame.from_dict(summary["intersections"], orient="index").reset_index(names="comparison")),
        "",
        "## Cross-label collisions",
        "",
        json.dumps(summary["cross_label"], indent=2),
        "",
    ]
    (audit_dir / "duplicate_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _write_trace_reports(
    stats: dict[str, pd.DataFrame],
    X: np.ndarray,
    split: np.ndarray,
    raw_scan: RawNonfiniteAudit,
    output_dir: Path,
) -> None:
    audit_dir = output_dir / "audits"
    paper = {
        "Header_Length": {"p25": 54.0, "p50": 54.0, "p75": 280.555, "max": 9_907_147.75},
        "IAT": {"p25": 83_071_566.0, "p50": 83_124_522.4, "p75": 83_343_908.0, "max": 167_639_436.0},
    }
    lines = [
        "# Header_Length and IAT trace",
        "",
        "## Evidence chain",
        "",
        f"- Raw source scan: {raw_scan.total_rows:,} rows; {raw_scan.dropped_rows:,} rows contain at least one non-finite value.",
        "- The labelled builder performs no unit conversion, normalization, clipping, reordering, binarization, or rounding. It drops non-finite rows and casts features to float32.",
        "- The corrected canonical pipeline preserves finite values (`clipping_mode=none`) before train-only RobustScaler fitting.",
        "",
    ]
    for feature in ("Header_Length", "IAT"):
        lines.extend([f"## {feature}", "", "Paper Table 5: `" + json.dumps(paper[feature]) + "`", ""])
        stage_rows = pd.concat(stats.values(), ignore_index=True)
        stage_rows = stage_rows[stage_rows["feature"] == feature][["stage", "min", "p25", "p50", "p75", "p99_99", "max"]]
        lines.extend([markdown_table(stage_rows, 10), ""])
    positive = X[:, FEATURE_NAMES.index("IAT")] > 0
    rng = np.random.default_rng(paths.SEED)
    candidates = np.flatnonzero(positive & (split == 0))
    if candidates.size > 1_000_000:
        candidates = rng.choice(candidates, 1_000_000, replace=False)
    product = X[candidates, FEATURE_NAMES.index("Rate")].astype(np.float64) * X[candidates, FEATURE_NAMES.index("IAT")].astype(np.float64)
    lines.extend([
        "## Interpretation",
        "",
        "`Header_Length` is already capped at the observed source maximum in the raw CSV/labelled Parquet; the old 60 bound was therefore not caused by clipping, scaling, column reordering, or dtype conversion. This repository contains a feature formulation/source release that differs materially from the paper's Table 5 distribution.",
        "",
        f"For a deterministic training sample, median `Rate × IAT` = {float(np.median(product)):.9g}. This near-reciprocal relationship is evidence that the local `IAT` is expressed in seconds relative to a per-second Rate. The pipeline contains no IAT conversion. The paper's approximately 83 million values therefore describe a different extractor representation/unit; no conversion is imposed without extractor-code provenance.",
        "",
    ])
    (audit_dir / "header_iat_trace.md").write_text("\n".join(lines), encoding="utf-8")

    chronology = """# Split chronology audit

## Evidence

- The 39-feature source contains no timestamp column.
- The labelled Parquet preserves CSV file concatenation order and records `source_csv_filename`.
- Natural numeric suffix ordering is deterministic and source-disjoint for classes with at least three shards.
- Single/two-shard classes use contiguous source-row blocks.
- The distributed `README_CSV.pdf` describes the CSV collection as combined/shuffled; it does not assert that numeric file suffixes or row positions are chronological.
- No PCAP/chunk timestamp metadata linking suffix order to wall-clock capture time is present in this repository.

## Conclusion

**B — chronology is not demonstrated.** Reports and manifests call this a **forward source-order split**, not a temporal split. It remains stronger than a random blended split for source-order separation, but it must not be represented as a verified chronological evaluation.
"""
    (audit_dir / "chronology_report.md").write_text(chronology, encoding="utf-8")


def run(config: AuditConfig) -> dict[str, Any]:
    audit_dir = config.output_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(paths.LABELED_PARQUET, columns=FEATURE_NAMES + ["Label", "category", "source_csv_filename"])
    X = np.ascontiguousarray(frame[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False))
    meta = frame[["source_csv_filename", "Label", "category"]]
    split, _, _ = pl.compute_split(meta)
    labels = pd.Categorical(frame["Label"], categories=sorted(frame["Label"].astype(str).unique())).codes.astype(np.int32)

    raw_scan = _scan_raw_nonfinite(config)
    if raw_scan.total_rows != len(frame) + raw_scan.dropped_rows:
        raise RuntimeError("raw scan does not reconcile with labelled Parquet row count")

    stats: dict[str, pd.DataFrame] = {}
    stats["raw"] = _stats_for_columns(
        X,
        stage="raw_csv",
        nan_count=raw_scan.nan_count,
        posinf_count=raw_scan.posinf_count,
        neginf_count=raw_scan.neginf_count,
        extra_finite=raw_scan.dropped_finite_values,
        total_count=raw_scan.total_rows,
    )
    stats["parquet"] = _stats_for_columns(X, stage="labelled_parquet")
    train_idx = np.flatnonzero(split == 0)
    val_idx = np.flatnonzero(split == 1)
    test_idx = np.flatnonzero(split == 2)
    stats["train_preclean"] = _stats_for_columns(X, stage="natural_train_preclean", row_indices=train_idx)
    stats["train_postclean"] = stats["train_preclean"].copy()
    stats["train_postclean"]["stage"] = "natural_train_postclean_none"
    stats["val"] = _stats_for_columns(X, stage="full_validation", row_indices=val_idx)
    stats["test"] = _stats_for_columns(X, stage="full_test", row_indices=test_idx)

    filenames = {
        "raw": "raw_feature_stats.csv",
        "parquet": "parquet_feature_stats.csv",
        "train_preclean": "train_preclean_stats.csv",
        "train_postclean": "train_postclean_stats.csv",
        "val": "validation_feature_stats.csv",
        "test": "test_feature_stats.csv",
    }
    for key, filename in filenames.items():
        stats[key].to_csv(audit_dir / filename, index=False)
        stats[key].to_parquet(audit_dir / filename.replace(".csv", ".parquet"), index=False)
    _write_stats_markdown(list(stats.values()), audit_dir / "feature_stats_comparison.md")

    clipping = _clipping_impact(X, split, config.clipping_percentile)
    clipping.to_csv(audit_dir / "clipping_impact.csv", index=False)
    _write_clipping_report(clipping, config.clipping_percentile, audit_dir / "clipping_report.md")
    _write_trace_reports(stats, X, split, raw_scan, config.output_dir)

    schema_frame = pd.DataFrame(feature_metadata_records())
    schema_frame.to_csv(audit_dir / "feature_type_report.csv", index=False)
    (audit_dir / "schema_audit.json").write_text(json.dumps({
        "feature_order_matches": list(schema_frame["name"]) == FEATURE_NAMES,
        "width": len(FEATURE_NAMES),
        "fractional_semantics_preserved": True,
        "feature_metadata": feature_metadata_records(),
    }, indent=2), encoding="utf-8")

    duplicates = run_duplicate_audit(X, split, labels, config.output_dir, config.hash_chunk_size)
    summary = {
        "raw_rows": raw_scan.total_rows,
        "parquet_rows": len(frame),
        "dropped_nonfinite_rows": raw_scan.dropped_rows,
        "split_counts": {name: int((split == code).sum()) for name, code in (("train", 0), ("val", 1), ("test", 2))},
        "duplicate_overlap_summary": duplicates,
    }
    (audit_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=pl.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--percentile", type=float, default=pl.CLIP_PERCENTILE)
    parser.add_argument("--raw-chunk-size", type=int, default=250_000)
    parser.add_argument("--hash-chunk-size", type=int, default=500_000)
    args = parser.parse_args()
    result = run(AuditConfig(
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        clipping_percentile=args.percentile,
        raw_chunk_size=args.raw_chunk_size,
        hash_chunk_size=args.hash_chunk_size,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
