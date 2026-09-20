"""End-to-end verification for a semantics-corrected CICIoT2023 artifact root."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import paths
from src.preprocessing.ciciot2023 import pipeline as pl
from src.preprocessing.schema import BOUNDED_AGGREGATED_IDX, CATEGORY_MAP, COUNT_AGGREGATES, FEATURE_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_finite(array: np.ndarray, chunk_size: int = 500_000) -> None:
    for start in range(0, array.shape[0], chunk_size):
        if not np.isfinite(array[start:start + chunk_size]).all():
            raise AssertionError(f"non-finite values at rows {start}:{start + chunk_size}")


def run(output_dir: Path) -> list[str]:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["feature_order"] != FEATURE_NAMES:
        raise AssertionError("manifest feature order differs from frozen schema")
    if manifest["clipping_mode"] != "none":
        raise AssertionError("canonical corrected run must preserve finite source values")

    with (output_dir / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (output_dir / "label_encoder.pkl").open("rb") as handle:
        fine_encoder = pickle.load(handle)
    with (output_dir / "category_encoder.pkl").open("rb") as handle:
        category_encoder = pickle.load(handle)

    frame = pd.read_parquet(paths.LABELED_PARQUET, columns=FEATURE_NAMES + ["Label", "category", "source_csv_filename"])
    X_source = np.ascontiguousarray(frame[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False))
    split, plans, _ = pl.compute_split(frame[["source_csv_filename", "Label", "category"]])
    train_natural = split == 0
    expected_scaler = RobustScaler().fit(X_source[train_natural])
    np.testing.assert_allclose(scaler.center_, expected_scaler.center_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scaler.scale_, expected_scaler.scale_, rtol=0.0, atol=0.0)

    kept_train = np.load(output_dir / "train_kept_indices.npy", mmap_mode="r")
    indices_by_split = {
        "train": np.asarray(kept_train),
        "val": np.flatnonzero(split == 1),
        "test": np.flatnonzero(split == 2),
    }
    labels = frame["Label"].astype(str).to_numpy()
    categories = frame["category"].astype(str).to_numpy()
    benign = [label for label, category in CATEGORY_MAP.items() if category == "Benign"]
    checks: list[str] = []

    for name, indices in indices_by_split.items():
        X_saved = np.load(output_dir / f"X_{name}.npy", mmap_mode="r")
        y_fine = np.load(output_dir / f"y_{name}.npy", mmap_mode="r")
        y_category = np.load(output_dir / f"y_{name}_cat.npy", mmap_mode="r")
        y_binary = np.load(output_dir / f"y_{name}_bin.npy", mmap_mode="r")
        if X_saved.shape != (indices.size, len(FEATURE_NAMES)):
            raise AssertionError(f"{name} width/count mismatch: {X_saved.shape}")
        if any(array.shape != (indices.size,) for array in (y_fine, y_category, y_binary)):
            raise AssertionError(f"{name} feature/label alignment failure")
        _assert_finite(X_saved)
        np.testing.assert_array_equal(y_fine, fine_encoder.transform(labels[indices]).astype(np.int32))
        np.testing.assert_array_equal(y_category, category_encoder.transform(categories[indices]).astype(np.int32))
        np.testing.assert_array_equal(y_binary, (~np.isin(labels[indices], benign)).astype(np.int32))

        # Verify saved arrays are exactly the train-fitted affine transform of
        # the intended unmodified source rows. Chunked to bound temporary RAM.
        for start in range(0, indices.size, 500_000):
            stop = min(start + 500_000, indices.size)
            expected = scaler.transform(X_source[indices[start:stop]]).astype(np.float32)
            np.testing.assert_allclose(X_saved[start:stop], expected, rtol=0.0, atol=0.0)
        checks.append(f"{name}: shape, finiteness, labels, and source-row transform verified ({indices.size:,})")

    expected_val = sum(plan.n_val for plan in plans.values())
    expected_test = sum(plan.n_test for plan in plans.values())
    if indices_by_split["val"].size != expected_val or indices_by_split["test"].size != expected_test:
        raise AssertionError("validation/test holdouts were altered")
    checks.append("complete validation/test holdouts retained without sampling")

    bounded = X_source[:, list(BOUNDED_AGGREGATED_IDX)]
    if bounded.min() < 0.0 or bounded.max() > 1.0:
        raise AssertionError("source bounded aggregate indicators exceed [0,1]")
    fractional_bounded = ((bounded > 0.0) & (bounded < 1.0)).any()
    if not fractional_bounded:
        raise AssertionError("source audit unexpectedly found no fractional bounded aggregates")
    checks.append("fractional bounded aggregates preserved and source range [0,1] verified")

    count_indices = [FEATURE_NAMES.index(name) for name in COUNT_AGGREGATES]
    fractional_count_values = int(((X_source[:, count_indices] % 1.0) != 0.0).sum())
    checks.append(
        f"source fractional count cells observed: {fractional_count_values}; synthetic regression protects fractional-count preservation"
    )

    for filename, expected_hash in manifest["artifact_hashes"].items():
        actual = _sha256(output_dir / filename)
        if actual != expected_hash:
            raise AssertionError(f"artifact hash mismatch: {filename}")
    checks.append("all manifest artifact hashes verified")

    verification = "SEMANTICS-CORRECTED PREPROCESSING VERIFICATION\n" + "=" * 52 + "\n" + "\n".join(f"[OK] {check}" for check in checks) + "\n"
    (output_dir / "preprocessing_verification.txt").write_text(verification, encoding="utf-8")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=pl.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for check in run(args.output_dir):
        print("[OK]", check)


if __name__ == "__main__":
    main()
