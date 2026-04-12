"""Audit all project data, models, and results for validity."""
import os
import sys
import pickle
import numpy as np

ROOT = r"D:\thesis"
PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
MODELS_VAE = os.path.join(ROOT, "models", "vae")
MODELS_NIDS = os.path.join(ROOT, "models", "nids")

CONTINUOUS_IDX = list(range(0, 23)) + [36]
BINARY_IDX = list(range(23, 36))

print("=" * 70)
print("PROJECT AUDIT")
print("=" * 70)

# 1. Processed data
print("\n--- Processed Data ---")
for name in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]:
    path = os.path.join(PROCESSED, name)
    if os.path.exists(path):
        arr = np.load(path, allow_pickle=True)
        nans = np.isnan(arr).sum() if arr.dtype.kind == 'f' else 0
        infs = np.isinf(arr).sum() if arr.dtype.kind == 'f' else 0
        print(f"  {name:25s} shape={str(arr.shape):20s} dtype={arr.dtype}  nan={nans}  inf={infs}")
    else:
        print(f"  {name:25s} MISSING")

# 2. Label encoder
print("\n--- Label Encoder ---")
le_path = os.path.join(PROCESSED, "label_encoder.pkl")
with open(le_path, "rb") as f:
    le = pickle.load(f)
print(f"  Classes: {list(le.classes_)}")
class_to_idx = {c: i for i, c in enumerate(le.classes_)}

# 7. Attack results — the critical audit
print("\n--- Attack Results (per class) ---")
issues = []
for cls in ["DDoS", "DoS", "Mirai", "Recon"]:
    cdir = os.path.join(RESULTS, cls.lower())
    print(f"\n  [{cls}]")
    if not os.path.isdir(cdir):
        print(f"    MISSING DIRECTORY")
        issues.append(f"{cls}: missing results directory")
        continue

    for fname in sorted(os.listdir(cdir)):
        fpath = os.path.join(cdir, fname)
        arr = np.load(fpath, allow_pickle=True)
        n_nan = np.isnan(arr).sum() if arr.dtype.kind == 'f' else 0
        n_inf = np.isinf(arr).sum() if arr.dtype.kind == 'f' else 0
        nan_rows = np.isnan(arr).any(axis=1).sum() if arr.ndim == 2 and arr.dtype.kind == 'f' else 0
        clean_rows = arr.shape[0] - nan_rows if arr.ndim == 2 else "n/a"

        status = "OK" if n_nan == 0 and n_inf == 0 else "PROBLEM"
        print(f"    {fname:30s} shape={str(arr.shape):12s} dtype={arr.dtype}  "
              f"nan={n_nan:>5d}  inf={n_inf:>5d}  clean_rows={clean_rows}  [{status}]")

        if n_nan > 0:
            issues.append(f"{cls}/{fname}: {n_nan} NaN values, {nan_rows} bad rows out of {arr.shape[0]}")

# 8. Summary
print(f"\n{'='*70}")
print(f"AUDIT SUMMARY")
print(f"{'='*70}")
if issues:
    print(f"  ISSUES FOUND: {len(issues)}")
    for iss in issues:
        print(f"    [FAIL] {iss}")
else:
    print("  [OK] No issues found.")
