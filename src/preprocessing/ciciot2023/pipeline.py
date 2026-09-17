"""End-to-end preprocessing pipeline for CICIoT2023 (downsampling_strategy.md §4.1).

Order (leakage-critical, spec §4.1)::

    labelled parquet (46,775,660 rows, 309 shards)
      -> clean (int-floor, zero-floor, binary-round; train-only 99.99% clip)
      -> forward-chaining temporal split (split BEFORE scaling/sampling)   §4.2/§4.3
      -> fit RobustScaler on TRAIN ONLY, transform all splits
      -> cluster + sample majority categories, TRAIN ONLY                  §5
      -> class weights from the subsampled train set
      -> encoders / masks / manifest (downstream artifact contract)

Rationale for the order (spec §4.1): split before scaling (else the scaler sees
test), split before sampling (sampling reads class counts), cluster on TRAIN
only, and val/test are **never** sampled so headline metrics face a realistic
class mix.

Run: ``python -m src.preprocessing.ciciot2023.pipeline``. Use ``--output-dir``
to redirect generated artifacts; the labelled Parquet input remains canonical.
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.preprocessing import LabelEncoder, RobustScaler  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

from config import paths  # noqa: E402
from src.preprocessing.schema import (  # noqa: E402
    BINARY_FEATURES,
    CATEGORY_MAP,
    FEATURE_NAMES,
    INTEGER_FEATURES,
)
from src.preprocessing.ciciot2023.sampler import cluster_proportional_floor_sample, random_floor_sample
from src.preprocessing.ciciot2023.splitter import (  # noqa: E402
    ClassSplitPlan,
    assert_forward_chaining,
    assign_row_splits,
    build_shard_runs,
    plan_all_classes,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("pipeline").info

# ── Downsampling parameters (chosen from the §6 diagnostics; see reports.py and
# docs/downsampling_implementation.md §3 for the evidence table). ─────────────
# All six majority categories are multi-modal → all use cluster-proportional-
# floor; k per category = grid value maximising cluster→subtype purity (spec §6.4).
CAP_PER_CATEGORY: dict[str, int] = {
    "DDoS": 200_000, "DoS": 200_000, "Mirai": 200_000,
    "Recon": 200_000, "Spoofing": 200_000, "Benign": 200_000,
}
RARE_KEPT_WHOLE: tuple[str, ...] = ("BruteForce", "Web")   # never clustered/capped (§5.3)
K_PER_CATEGORY: dict[str, int] = {
    "DDoS": 20, "DoS": 20, "Mirai": 10, "Recon": 15, "Spoofing": 15, "Benign": 10,
}
FLOOR: int = 500              # ≫ latent dim 16 & 39-dim Σ rank; k·floor ≪ cap (§6.4)
SELECTION_MODE: str = "random_within"   # seeded uniform within-cluster pick (§5.2)
VAL_FRAC: float = 0.10
TEST_FRAC: float = 0.20
CLIP_PERCENTILE = 99.99
NETDIFFUSER_SAMPLE = 200_000

# The 23 continuous features clustered by the sampler (spec §5.3): all except the
# 15 binary protocol indicators and the categorical Protocol Type.
CONTINUOUS_FEATURES: list[str] = [
    f for f in FEATURE_NAMES if f not in set(BINARY_FEATURES) and f != "Protocol Type"
]
assert len(CONTINUOUS_FEATURES) == 23, len(CONTINUOUS_FEATURES)
CONTINUOUS_IDX: list[int] = [FEATURE_NAMES.index(f) for f in CONTINUOUS_FEATURES]


# ── Shared parquet IO (previously src/preprocessing/dataset.py) ──────────────
def load_metadata(parquet: Path = paths.LABELED_PARQUET) -> "pd.DataFrame":
    """Load only the columns needed to compute the split: shard id + labels."""
    return pd.read_parquet(parquet, columns=["source_csv_filename", "Label", "category"])


def compute_split(
    meta: "pd.DataFrame", val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC,
) -> tuple[np.ndarray, dict[str, ClassSplitPlan], list]:
    """Return ``(split_codes int8 [0=train,1=val,2=test], plans, shard_runs)``
    aligned to parquet row order (spec §4.2/§4.3)."""
    shard_col = meta["source_csv_filename"].to_numpy()
    label_col = meta["Label"].astype(str).to_numpy()
    runs = build_shard_runs(shard_col, label_col)
    plans = plan_all_classes(runs, val_frac=val_frac, test_frac=test_frac)
    split_codes = assign_row_splits(runs, plans, n_rows=len(meta))
    if (split_codes < 0).any():
        raise RuntimeError(f"{int((split_codes < 0).sum())} rows left unassigned by the split")
    return split_codes, plans, runs


def load_features(
    parquet: Path = paths.LABELED_PARQUET, columns: list[str] | None = None
) -> np.ndarray:
    """Load the 39 feature columns (or a subset) as a writable float32 array."""
    cols = columns if columns is not None else FEATURE_NAMES
    df = pd.read_parquet(parquet, columns=cols)
    return np.ascontiguousarray(df[cols].to_numpy(dtype=np.float32))  # writable copy


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clip_round(X: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Apply the cleaning transforms in place given precomputed clip bounds:
    lower-clip 0, upper-clip ``upper`` per feature, round integer + binary
    features to their lattice. Shared by the pipeline and the evidence figures
    so both reproduce the exact same cleaned feature space."""
    np.clip(X, 0.0, upper, out=X)  # lower 0 (validator G1), upper = train-99.99
    int_idx = [FEATURE_NAMES.index(f) for f in INTEGER_FEATURES]
    bin_idx = [FEATURE_NAMES.index(f) for f in BINARY_FEATURES]
    X[:, int_idx] = np.clip(np.round(X[:, int_idx]), 0, None)
    X[:, bin_idx] = np.clip(np.round(X[:, bin_idx]), 0, 1)
    return X


def clean_features(X: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-level cleaning (spec §4.1). NaN/inf were already dropped at build time.

    The 99.99th-percentile upper clip is computed **on train only** (leakage-safe;
    the spec sketches this under "clean" but computing the bound on the whole
    dataset would leak test extremes — we take train-only bounds instead).

    Returns ``(X_cleaned, upper_clip)``.
    """
    log("=== Clean: train-only 99.99%% clip + int/binary rounding ===")
    q = np.percentile(X[train_mask], CLIP_PERCENTILE, axis=0).astype(np.float32)
    return clip_round(X, q), q


def sample_train(
    X: np.ndarray,
    scaler: RobustScaler,
    split: np.ndarray,
    category: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Cluster-proportional-floor undersample majority categories, TRAIN ONLY.

    Rare categories (spec §5.3) pass through whole. Returns the sorted global
    train indices kept and a per-category record for the manifest.
    """
    log("=== Sample: intra-class cluster-proportional-floor (train only) ===")
    cont_idx = CONTINUOUS_IDX
    center = scaler.center_[cont_idx]
    scale = scaler.scale_[cont_idx]
    train_mask = split == 0

    kept_global: list[np.ndarray] = []
    record: dict = {}
    for cat in sorted(set(category[train_mask])):
        cat_rows = np.flatnonzero(train_mask & (category == cat))
        n = cat_rows.size
        if cat in RARE_KEPT_WHOLE:
            kept_global.append(cat_rows)
            record[cat] = {"policy": "kept_whole", "n_train": int(n), "n_kept": int(n)}
            log("  %-10s kept whole: %d rows", cat, n)
            continue

        cap = CAP_PER_CATEGORY.get(cat)
        if cap is None or n <= cap:
            kept_global.append(cat_rows)
            record[cat] = {"policy": "uncapped_or_below_cap", "n_train": int(n), "n_kept": int(n)}
            log("  %-10s <= cap, kept whole: %d rows", cat, n)
            continue

        # Scaled continuous features for clustering (spec §5.3).
        Xc = (X[cat_rows][:, cont_idx] - center) / scale
        k = K_PER_CATEGORY[cat]
        res = cluster_proportional_floor_sample(
            Xc, target_n=cap, k=k, floor=FLOOR, seed=paths.SEED,
            selection_mode=SELECTION_MODE,
        )
        kept_global.append(cat_rows[res.kept_indices])
        record[cat] = {
            "policy": "cluster_proportional_floor",
            "n_train": int(n),
            "n_kept": int(res.kept_indices.size),
            "cap": cap,
            "k": k,
            "floor": FLOOR,
            "selection_mode": SELECTION_MODE,
            "cap_not_binding": res.cap_not_binding,
            "cluster_sizes": res.cluster_sizes,
            "allocations": res.allocations,
            "small_pool": res.small_pool,
            "large_pool": res.large_pool,
        }
        log("  %-10s clustered k=%d: %d -> %d rows (small_pool=%d)",
            cat, k, n, res.kept_indices.size, len(res.small_pool))

    train_idx = np.concatenate(kept_global)
    train_idx.sort()
    return train_idx.astype(np.int64), record



def main(output_dir: Path | None = None) -> None:
    t0 = time.time()
    paths.ensure_dirs()
    P = output_dir or paths.PROCESSED_DIR
    P.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────
    log("=== Load labelled parquet (%s) ===", paths.LABELED_PARQUET.name)
    df = pd.read_parquet(paths.LABELED_PARQUET)
    n = len(df)
    log("  %d rows x %d cols", n, df.shape[1])

    meta = df[["source_csv_filename", "Label", "category"]]
    split, plans, runs = compute_split(meta, VAL_FRAC, TEST_FRAC)
    for plan in plans.values():          # leakage guard (spec §6.5)
        assert_forward_chaining(plan)

    labels_str = df["Label"].astype(str).to_numpy()
    category_str = df["category"].astype(str).to_numpy()
    X = np.ascontiguousarray(df[FEATURE_NAMES].to_numpy(dtype=np.float32))  # writable copy (to_numpy may return a read-only block view)
    del df, meta

    train_mask = split == 0
    val_idx = np.flatnonzero(split == 1)
    test_idx = np.flatnonzero(split == 2)

    # ── Clean (train-only clip) ───────────────────────────────────────────────
    X, upper_clip = clean_features(X, train_mask)

    # ── Scale (fit on train only) ─────────────────────────────────────────────
    log("=== Fit RobustScaler on train only (%d rows) ===", int(train_mask.sum()))
    scaler = RobustScaler()
    scaler.fit(X[train_mask])


    # ── Sample (train only) ───────────────────────────────────────────────────
    train_idx, sample_record = sample_train(X, scaler, split, category_str)
    # Persist the kept global row indices — lets evidence_figures reproduce the
    # exact selection without re-running the sampler, and is an audit artefact.
    np.save(P / "train_kept_indices.npy", train_idx)

    # ── Encoders ──────────────────────────────────────────────────────────────
    le_34 = LabelEncoder().fit(labels_str)
    le_cat = LabelEncoder().fit(category_str)
    benign_labels = {k for k, v in CATEGORY_MAP.items() if v == "Benign"}

    def y_all(idx: np.ndarray):
        y34 = le_34.transform(labels_str[idx]).astype(np.int32)
        y8 = le_cat.transform(category_str[idx]).astype(np.int32)
        ybin = (~np.isin(labels_str[idx], list(benign_labels))).astype(np.int32)
        return y34, y8, ybin

    # ── Build + save arrays ───────────────────────────────────────────────────
    log("=== Build + save arrays ===")
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    split_counts = {}
    for name, idx in splits.items():
        Xs = scaler.transform(X[idx]).astype(np.float32)
        np.save(P / f"X_{name}.npy", Xs)
        y34, y8, ybin = y_all(idx)
        np.save(P / f"y_{name}.npy", y34)
        np.save(P / f"y_{name}_cat.npy", y8)
        np.save(P / f"y_{name}_bin.npy", ybin)
        split_counts[name] = int(idx.size)
        log("  %-5s X=%s  y34/y8/ybin saved", name, Xs.shape)
        del Xs

    # ── Class weights (from subsampled train) ─────────────────────────────────
    y_tr34, y_tr8, y_trbin = y_all(train_idx)
    w34 = compute_class_weight("balanced", classes=np.unique(y_tr34), y=y_tr34).astype(np.float32)
    w8 = compute_class_weight("balanced", classes=np.unique(y_tr8), y=y_tr8).astype(np.float32)
    w2 = compute_class_weight("balanced", classes=np.unique(y_trbin), y=y_trbin).astype(np.float32)
    np.save(P / "class_weights_34.npy", w34)
    np.save(P / "class_weights_8.npy", w8)
    np.save(P / "class_weights_2.npy", w2)

    # ── Scaler + encoders + name maps ─────────────────────────────────────────
    with open(P / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(P / "label_encoder.pkl", "wb") as f:
        pickle.dump(le_34, f)
    with open(P / "category_encoder.pkl", "wb") as f:
        pickle.dump(le_cat, f)
    (P / "class_names.json").write_text(json.dumps(list(le_34.classes_), indent=2))
    (P / "category_names.json").write_text(json.dumps(list(le_cat.classes_), indent=2))
    (P / "class_to_category.json").write_text(json.dumps(CATEGORY_MAP, indent=2))


    # ── Run manifest (spec §7.3 / §9 tracked items) ───────────────────────────
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=paths.REPO_ROOT
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    per_class_split = {
        lbl: {
            "protocol": p.protocol, "n_shards": p.n_shards,
            "n_train": p.n_train, "n_val": p.n_val, "n_test": p.n_test,
            "train_shards": p.train_shards, "val_shards": p.val_shards,
            "test_shards": p.test_shards,
        }
        for lbl, p in plans.items()
    }
    hashes = {f"X_{s}": _sha256_file(P / f"X_{s}.npy") for s in splits}
    hashes["scaler"] = hashlib.sha256(
        scaler.center_.tobytes() + scaler.scale_.tobytes()
    ).hexdigest()

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "seed": paths.SEED,
        "method": "intra-class clustering-based undersampling (selection variant), "
                  "cluster-proportional-with-floor",
        "split_protocol": "forward-chaining temporal (shard) / contiguous-block (single-shard)",
        "val_frac": VAL_FRAC, "test_frac": TEST_FRAC,
        "caps": CAP_PER_CATEGORY,
        "rare_kept_whole": list(RARE_KEPT_WHOLE),
        "k_per_category": K_PER_CATEGORY,
        "floor": FLOOR,
        "selection_mode": SELECTION_MODE,
        "continuous_features": CONTINUOUS_FEATURES,
        "clip_percentile": CLIP_PERCENTILE,
        "clip_upper_train": {f: float(v) for f, v in zip(FEATURE_NAMES, upper_clip)},
        "split_protocol_counts": {
            proto: sum(1 for p in plans.values() if p.protocol == proto)
            for proto in ("forward_chain", "two_shard_hybrid", "block")
        },
        "per_class_split": per_class_split,
        "split_row_counts": split_counts,
        "train_before_sampling": int(train_mask.sum()),
        "train_after_sampling": int(train_idx.size),
        "sampling_per_category": sample_record,
        "class_weight_ranges": {
            "34class": [float(w34.min()), float(w34.max())],
            "8class": [float(w8.min()), float(w8.max())],
            "binary": [float(w2.min()), float(w2.max())],
        },
        "artifact_hashes": hashes,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    run_manifest = P / "run_manifest.json"
    run_manifest.write_text(json.dumps(manifest, indent=2))
    log("=== DONE in %.1fs. Manifest: %s ===", time.time() - t0, run_manifest)
    log("  train %d->%d (sampled), val %d, test %d",
        manifest["train_before_sampling"], manifest["train_after_sampling"],
        split_counts["val"], split_counts["test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated arrays and metadata (default: data/processed)",
    )
    main(parser.parse_args().output_dir)
