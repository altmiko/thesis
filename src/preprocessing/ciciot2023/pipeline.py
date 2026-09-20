"""Leakage-safe CICIoT2023 preprocessing with CSV-level aggregate semantics.

Order: split source rows first; optionally fit model-clipping bounds on natural
training only; fit RobustScaler on cleaned natural training only; sample only the
training partition; transform sampled train and complete validation/test holdouts.

No feature is binarized or integer-rounded. The released CSV rows summarize
packet windows, so fractional indicators, counts, and protocol-code averages are
meaningful source values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.preprocessing import LabelEncoder, RobustScaler  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

from config import paths  # noqa: E402
from src.preprocessing.ciciot2023.sampler import cluster_proportional_floor_sample
from src.preprocessing.ciciot2023.splitter import (
    ClassSplitPlan,
    assert_forward_chaining,
    assign_row_splits,
    build_shard_runs,
    plan_all_classes,
)
from src.preprocessing.schema import (
    CATEGORY_MAP,
    FEATURE_NAMES,
    FEATURE_SPECS,
    feature_metadata_records,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("pipeline").info

CAP_PER_CATEGORY: dict[str, int] = {
    "DDoS": 200_000, "DoS": 200_000, "Mirai": 200_000,
    "Recon": 200_000, "Spoofing": 200_000, "Benign": 200_000,
}
RARE_KEPT_WHOLE = ("BruteForce", "Web")
K_PER_CATEGORY: dict[str, int] = {
    "DDoS": 20, "DoS": 20, "Mirai": 10, "Recon": 15,
    "Spoofing": 15, "Benign": 10,
}
FLOOR = 500
SELECTION_MODE = "random_within"
VAL_FRAC = 0.10
TEST_FRAC = 0.20
CLIP_PERCENTILE = 99.99
DEFAULT_OUTPUT_DIR = paths.REPO_ROOT / "outputs" / "ciciot2023_semantics_corrected"

# All 39 CSV columns are continuous at the released window-aggregate level.
# Protocol Type is retained: it is an averaged code-like value, not a category.
CLUSTER_FEATURES = list(FEATURE_NAMES)
CLUSTER_IDX = list(range(len(FEATURE_NAMES)))
# Compatibility names for report callers. They no longer exclude indicators.
CONTINUOUS_FEATURES = CLUSTER_FEATURES
CONTINUOUS_IDX = CLUSTER_IDX

ClippingMode = Literal["none", "train_percentile"]


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    clipping_mode: ClippingMode = "none"
    percentile: float = CLIP_PERCENTILE
    seed: int = paths.SEED

    def __post_init__(self) -> None:
        if self.clipping_mode not in ("none", "train_percentile"):
            raise ValueError(f"unsupported clipping_mode={self.clipping_mode!r}")
        if not 0.0 < self.percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100]")


def load_metadata(parquet: Path = paths.LABELED_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(parquet, columns=["source_csv_filename", "Label", "category"])


def compute_split(
    meta: pd.DataFrame,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> tuple[np.ndarray, dict[str, ClassSplitPlan], list]:
    runs = build_shard_runs(
        meta["source_csv_filename"].to_numpy(),
        meta["Label"].astype(str).to_numpy(),
    )
    plans = plan_all_classes(runs, val_frac=val_frac, test_frac=test_frac)
    split_codes = assign_row_splits(runs, plans, n_rows=len(meta))
    if (split_codes < 0).any():
        raise RuntimeError(f"{int((split_codes < 0).sum())} rows were not assigned")
    return split_codes, plans, runs


def load_features(
    parquet: Path = paths.LABELED_PARQUET,
    columns: list[str] | None = None,
) -> np.ndarray:
    cols = columns or FEATURE_NAMES
    frame = pd.read_parquet(parquet, columns=cols)
    return np.ascontiguousarray(frame[cols].to_numpy(dtype=np.float32))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_clip_upper(upper: np.ndarray) -> np.ndarray:
    """Return a defensive copy; no semantic floors or discrete coercion."""

    values = np.asarray(upper, dtype=np.float32)
    if values.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"expected {len(FEATURE_NAMES)} clip bounds, got {values.shape}")
    return values.copy()


def clip_round(X: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Apply model-preprocessing clipping only; preserve fractional semantics.

    The historical function name remains for callers, but this function performs
    no rounding and no presence canonicalization. It clips to train-fitted upper
    bounds and zero only when explicitly invoked by percentile-clipping mode.
    """

    bounds = effective_clip_upper(upper)
    np.clip(X, 0.0, bounds, out=X)
    return X


def clean_features(
    X: np.ndarray,
    train_mask: np.ndarray,
    clipping_mode: ClippingMode = "none",
    percentile: float = CLIP_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Preserve finite CSV values or apply explicitly configured train-only clip."""

    values = np.asarray(X)
    mask = np.asarray(train_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"X must have shape (N, {len(FEATURE_NAMES)})")
    if mask.shape != (values.shape[0],) or not mask.any():
        raise ValueError("train_mask must align with X and contain training rows")
    if not np.isfinite(values).all():
        raise ValueError("labelled source contains non-finite feature values")
    if clipping_mode == "none":
        log("=== Clean: preserve finite source values (no model clipping) ===")
        return values, None
    if clipping_mode != "train_percentile":
        raise ValueError(f"unsupported clipping_mode={clipping_mode!r}")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    log("=== Clean: train-only %.5gth-percentile model clip ===", percentile)
    upper = np.percentile(values[mask], percentile, axis=0).astype(np.float32)
    return clip_round(values, upper), upper

def fit_scaler(X: np.ndarray, train_mask: np.ndarray) -> RobustScaler:
    """Fit exclusively on the complete natural training partition."""

    values = np.asarray(X)
    mask = np.asarray(train_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != (values.shape[0],) or not mask.any():
        raise ValueError("train_mask must select aligned natural training rows")
    return RobustScaler().fit(values[mask])



def sample_train(
    X: np.ndarray,
    scaler: RobustScaler,
    split: np.ndarray,
    category: np.ndarray,
    seed: int = paths.SEED,
) -> tuple[np.ndarray, dict]:
    """Sample majority categories using all non-degenerate aggregate features."""

    log("=== Sample: cluster-proportional-floor (training only) ===")
    center = np.asarray(scaler.center_, dtype=np.float32)[CLUSTER_IDX]
    scale = np.asarray(scaler.scale_, dtype=np.float32)[CLUSTER_IDX]
    scale = np.where(scale == 0.0, 1.0, scale)
    train_mask = split == 0
    kept_global: list[np.ndarray] = []
    record: dict[str, dict] = {}
    for cat in sorted(set(category[train_mask])):
        cat_rows = np.flatnonzero(train_mask & (category == cat))
        count = cat_rows.size
        if cat in RARE_KEPT_WHOLE:
            kept_global.append(cat_rows)
            record[cat] = {"policy": "kept_whole", "n_train": int(count), "n_kept": int(count)}
            continue
        cap = CAP_PER_CATEGORY.get(cat)
        if cap is None or count <= cap:
            kept_global.append(cat_rows)
            record[cat] = {"policy": "uncapped_or_below_cap", "n_train": int(count), "n_kept": int(count)}
            continue
        cluster_matrix = (X[cat_rows][:, CLUSTER_IDX] - center) / scale
        result = cluster_proportional_floor_sample(
            cluster_matrix,
            target_n=cap,
            k=K_PER_CATEGORY[cat],
            floor=FLOOR,
            seed=seed,
            selection_mode=SELECTION_MODE,
        )
        kept_global.append(cat_rows[result.kept_indices])
        record[cat] = {
            "policy": "cluster_proportional_floor",
            "n_train": int(count),
            "n_kept": int(result.kept_indices.size),
            "cap": cap,
            "k": K_PER_CATEGORY[cat],
            "floor": FLOOR,
            "selection_mode": SELECTION_MODE,
            "cap_not_binding": result.cap_not_binding,
            "cluster_sizes": result.cluster_sizes,
            "allocations": result.allocations,
            "small_pool": result.small_pool,
            "large_pool": result.large_pool,
        }
        del cluster_matrix
    train_idx = np.concatenate(kept_global).astype(np.int64)
    train_idx.sort()
    return train_idx, record


def _fixed_encoders() -> tuple[LabelEncoder, LabelEncoder]:
    fine = LabelEncoder().fit(np.asarray(sorted(CATEGORY_MAP), dtype=object))
    category = LabelEncoder().fit(np.asarray(sorted(set(CATEGORY_MAP.values())), dtype=object))
    return fine, category


def run(config: PipelineConfig) -> dict:
    started = time.time()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    log("=== Load labelled parquet: %s ===", paths.LABELED_PARQUET)
    frame = pd.read_parquet(paths.LABELED_PARQUET)
    expected = FEATURE_NAMES + ["Label", "category", "source_csv_filename", "source_folder"]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"labelled parquet missing columns: {missing}")
    meta = frame[["source_csv_filename", "Label", "category"]]
    split, plans, _ = compute_split(meta, VAL_FRAC, TEST_FRAC)
    for plan in plans.values():
        assert_forward_chaining(plan)

    labels = frame["Label"].astype(str).to_numpy()
    categories = frame["category"].astype(str).to_numpy()
    X = np.ascontiguousarray(frame[FEATURE_NAMES].to_numpy(dtype=np.float32))
    del frame, meta

    train_mask = split == 0
    val_idx = np.flatnonzero(split == 1)
    test_idx = np.flatnonzero(split == 2)
    X, clip_upper = clean_features(
        X,
        train_mask,
        clipping_mode=config.clipping_mode,
        percentile=config.percentile,
    )

    log("=== Fit RobustScaler on complete natural training partition: %d rows ===", int(train_mask.sum()))
    scaler = fit_scaler(X, train_mask)
    train_idx, sampling = sample_train(X, scaler, split, categories, seed=config.seed)
    np.save(output / "train_kept_indices.npy", train_idx)

    fine_encoder, category_encoder = _fixed_encoders()
    benign_labels = [label for label, category in CATEGORY_MAP.items() if category == "Benign"]

    def labels_for(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_fine = fine_encoder.transform(labels[indices]).astype(np.int32)
        y_category = category_encoder.transform(categories[indices]).astype(np.int32)
        y_binary = (~np.isin(labels[indices], benign_labels)).astype(np.int32)
        return y_fine, y_category, y_binary

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    split_counts: dict[str, int] = {}
    for name, indices in split_indices.items():
        transformed = scaler.transform(X[indices]).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise RuntimeError(f"non-finite scaled values in {name}")
        np.save(output / f"X_{name}.npy", transformed)
        y_fine, y_category, y_binary = labels_for(indices)
        np.save(output / f"y_{name}.npy", y_fine)
        np.save(output / f"y_{name}_cat.npy", y_category)
        np.save(output / f"y_{name}_bin.npy", y_binary)
        split_counts[name] = int(indices.size)
        del transformed

    y_train, y_train_category, y_train_binary = labels_for(train_idx)
    weights = {
        "34": compute_class_weight("balanced", classes=np.unique(y_train), y=y_train).astype(np.float32),
        "8": compute_class_weight("balanced", classes=np.unique(y_train_category), y=y_train_category).astype(np.float32),
        "2": compute_class_weight("balanced", classes=np.unique(y_train_binary), y=y_train_binary).astype(np.float32),
    }
    for key, value in weights.items():
        np.save(output / f"class_weights_{key}.npy", value)

    with (output / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    with (output / "label_encoder.pkl").open("wb") as handle:
        pickle.dump(fine_encoder, handle)
    with (output / "category_encoder.pkl").open("wb") as handle:
        pickle.dump(category_encoder, handle)
    (output / "class_names.json").write_text(json.dumps(list(fine_encoder.classes_), indent=2))
    (output / "category_names.json").write_text(json.dumps(list(category_encoder.classes_), indent=2))
    (output / "class_to_category.json").write_text(json.dumps(CATEGORY_MAP, indent=2))
    (output / "feature_schema.json").write_text(json.dumps(feature_metadata_records(), indent=2))

    # Preserve labelled-source provenance beside the run without modifying it.
    build_manifest_target = output / "ciciot2023_labeled_full_manifest.json"
    build_manifest_target.write_bytes(paths.LABELED_MANIFEST.read_bytes())

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=paths.REPO_ROOT
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    artifact_names = [
        *(f"X_{name}.npy" for name in split_indices),
        *(f"y_{name}.npy" for name in split_indices),
        *(f"y_{name}_cat.npy" for name in split_indices),
        *(f"y_{name}_bin.npy" for name in split_indices),
        "scaler.pkl", "train_kept_indices.npy", "feature_schema.json",
    ]
    artifact_hashes = {name: _sha256_file(output / name) for name in artifact_names}
    per_class_split = {
        label: {
            "protocol": plan.protocol,
            "n_shards": plan.n_shards,
            "n_train": plan.n_train,
            "n_val": plan.n_val,
            "n_test": plan.n_test,
            "train_shards": plan.train_shards,
            "val_shards": plan.val_shards,
            "test_shards": plan.test_shards,
            "block_ranges": plan.block_ranges,
        }
        for label, plan in plans.items()
    }
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "seed": config.seed,
        "source_paths": {
            "labelled_parquet": str(paths.LABELED_PARQUET),
            "labelled_manifest": str(paths.LABELED_MANIFEST),
        },
        "source_hashes": {"labelled_parquet_sha256": _sha256_file(paths.LABELED_PARQUET)},
        "feature_order": FEATURE_NAMES,
        "feature_semantic_types": feature_metadata_records(),
        "full_domain_schema": FEATURE_NAMES,
        "model_schema": FEATURE_NAMES,
        "feature_selection_mode": "none",
        "feature_selection_threshold": None,
        "split_procedure": "forward source-order by natural shard name; contiguous source-row blocks for classes with fewer than three shards",
        "chronology_claim": "not established; source order is retained but is not described as temporal",
        "val_fraction": VAL_FRAC,
        "test_fraction": TEST_FRAC,
        "split_protocol_counts": {
            protocol: sum(plan.protocol == protocol for plan in plans.values())
            for protocol in ("forward_chain", "two_shard_hybrid", "block")
        },
        "per_class_split": per_class_split,
        "split_row_counts": split_counts,
        "train_before_sampling": int(train_mask.sum()),
        "train_after_sampling": int(train_idx.size),
        "clipping_mode": config.clipping_mode,
        "clipping_percentile": config.percentile if config.clipping_mode == "train_percentile" else None,
        "model_clip_upper_train": (
            {name: float(value) for name, value in zip(FEATURE_NAMES, clip_upper)}
            if clip_upper is not None else None
        ),
        "domain_validity_bounds": "not inferred from model clipping",
        "scaler": {"type": "RobustScaler", "fit_population": "complete cleaned natural training partition"},
        "sampler": {
            "method": "intra-class cluster-proportional-with-floor selection",
            "fit_population": "training only",
            "caps": CAP_PER_CATEGORY,
            "rare_kept_whole": list(RARE_KEPT_WHOLE),
            "k_per_category": K_PER_CATEGORY,
            "floor": FLOOR,
            "selection_mode": SELECTION_MODE,
            "cluster_features": CLUSTER_FEATURES,
        },
        "sampling_per_category": sampling,
        "duplicate_overlap_summary": None,
        "artifact_hashes": artifact_hashes,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(output_dir: Path | None = None) -> None:
    config = PipelineConfig(output_dir=output_dir or DEFAULT_OUTPUT_DIR)
    run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clipping-mode", choices=("none", "train_percentile"), default="none")
    parser.add_argument("--percentile", type=float, default=CLIP_PERCENTILE)
    arguments = parser.parse_args()
    result = run(PipelineConfig(
        output_dir=arguments.output_dir,
        clipping_mode=arguments.clipping_mode,
        percentile=arguments.percentile,
    ))
    print(json.dumps({key: result[key] for key in ("split_row_counts", "clipping_mode", "runtime_seconds")}, indent=2))
