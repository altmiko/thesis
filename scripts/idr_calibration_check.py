"""Calibrate-check the IDR gate: does it pass CLEAN malicious test rows?

For each attack class computes:
  - val_in_distribution_rate  (stored anchor; ~0.95 by construction, fit on VAL)
  - IDR_clean on the SAME 1024-row eval sample the attack used (all + clean-correct subset)
  - IDR_clean on the FULL test split of that class
All as P(IDR passes | x_clean_malicious).
"""
from __future__ import annotations

import numpy as np
import torch

from attack.run_cicids2017_primitive_attack import _class_rows
from attack.run_cicids2017_vae_attacks import _idr_mask
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel  # noqa: F401 (parity import)
from datasets.cicids2017 import CICIDS2017Adapter
from vae.cicids2017_stage_a import ATTACK_CLASSES, load_stage_a
from src.classifiers.cicids2017d_victims import load_category_victim  # for clean-correct subset

TEST_LIMIT = 1024
DEVICE = "cpu"


def main() -> None:
    adapter = CICIDS2017Adapter()
    repo = adapter.repo_root
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    center = torch.tensor(transform.center, dtype=torch.float32, device=DEVICE)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=DEVICE)

    test = adapter.load_split("test")
    raw_test = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
    stage_a_dir = repo / "outputs" / "cicids2017_vae_stage_a"
    victim_dir = repo / "outputs" / "cicids2017distrinet" / "models"
    victim = load_category_victim(victim_dir / "mlp_category.pt", adapter=adapter,
                                  expected_model_type="mlp", device=DEVICE)

    print(f"{'class':>11} | {'val_p95':>8} | {'IDR_clean(sample)':>17} | "
          f"{'IDR_clean(cc)':>13} | {'IDR_clean(full)':>15} | n_sample n_full")
    print("-" * 96)
    for class_name in ATTACK_CLASSES:
        cid = mapping.name_to_id[class_name]
        idr_path = stage_a_dir / f"idr_{class_name}.npz"
        val_anchor = float(np.load(idr_path)["val_in_distribution_rate"])
        base_vae, _ = load_stage_a(adapter, stage_a_dir / f"vae_{class_name}.pt",
                                   expected_class_name=class_name, device=DEVICE)

        # same eval sample the attack used
        idx = _class_rows(test.y, cid, TEST_LIMIT, 42 + cid)
        raw = torch.tensor(np.ascontiguousarray(np.asarray(raw_test[idx]), np.float32), device=DEVICE)
        x_clean = (raw - center) / scale
        in_dist = _idr_mask(base_vae, x_clean, idr_path).cpu().numpy()
        clean_correct = (victim(x_clean).argmax(1).cpu().numpy() == cid)
        idr_sample = float(in_dist.mean())
        idr_cc = float(in_dist[clean_correct].mean()) if clean_correct.any() else float("nan")

        # full test split of this class
        full_idx = np.flatnonzero(np.asarray(test.y) == cid)
        di = np.zeros(len(full_idx), dtype=bool)
        for s in range(0, len(full_idx), 8192):
            chunk = full_idx[s:s + 8192]
            rc = torch.tensor(np.ascontiguousarray(np.asarray(raw_test[chunk]), np.float32), device=DEVICE)
            di[s:s + len(chunk)] = _idr_mask(base_vae, (rc - center) / scale, idr_path).cpu().numpy()
        idr_full = float(di.mean())

        print(f"{class_name:>11} | {val_anchor*100:7.1f}% | {idr_sample*100:16.1f}% | "
              f"{idr_cc*100:12.1f}% | {idr_full*100:14.1f}% | {len(idx):8d} {len(full_idx):7d}")


if __name__ == "__main__":
    main()
