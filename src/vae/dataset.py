"""Per-class dataset for the continuous window-aggregate β-VAE."""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PerClassDataset(Dataset):
    """Hold scaled 39-feature rows for one coarse category.

    No Bernoulli or integer protocol targets are constructed. Empty compatibility
    tensors keep older training-call signatures explicit without coercing values.
    """

    def __init__(self, X_split, y_split, class_id: int, scaler, partition: dict, n_pseudo_binary: int = 0) -> None:
        super().__init__()
        mask = np.asarray(y_split) == class_id
        if not mask.any():
            raise ValueError(f"No samples found for class_id={class_id}")
        selected = np.asarray(X_split[mask], dtype=np.float32)
        if selected.ndim != 2 or selected.shape[1] != 39 or not np.isfinite(selected).all():
            raise ValueError("VAE input must be finite with shape (N,39)")
        self.x_scaled = torch.from_numpy(selected)
        self.target_independent_binary = torch.empty((selected.shape[0], 0), dtype=torch.float32)
        self.target_protocol_index = torch.zeros(selected.shape[0], dtype=torch.int64)
        self.class_id = class_id
        self.n_samples = selected.shape[0]
        logger.info("Class %d: %d continuous aggregate samples loaded", class_id, self.n_samples)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "x_scaled": self.x_scaled[idx],
            "target_ind_binary": self.target_independent_binary[idx],
            "target_proto_idx": self.target_protocol_index[idx],
        }
