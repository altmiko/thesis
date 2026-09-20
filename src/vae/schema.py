"""CICIoT2023 VAE feature partition for window-aggregated CSV semantics.

All 39 released features are continuous at CSV-row level. Flag/service/protocol
indicators are bounded continuous aggregates, not Bernoulli variables. Protocol
Type is an averaged code-like field, not a categorical packet protocol.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from sklearn.preprocessing import RobustScaler

from src.preprocessing.schema import BOUNDED_AGGREGATED_IDX, FEATURE_NAMES

# Retained only as historical audit evidence for old checkpoints. It is not used
# to encode or validate the averaged Protocol Type field in corrected models.
PROTOCOL_ALLOWLIST: list[int] = [0, 1, 2, 6, 17, 47]
PROTOCOL_TO_BINARY: dict[int, str] = {}
DERIVED_BINARY_ORDER: list[str] = []


def get_partition(scaler: "RobustScaler | None" = None) -> dict[str, list[int]]:
    """Return the centralized continuous partition in frozen feature order."""

    bounded = list(BOUNDED_AGGREGATED_IDX)
    bounded_set = set(bounded)
    return {
        "continuous_idx": list(range(len(FEATURE_NAMES))),
        "bounded_continuous_idx": bounded,
        "unbounded_continuous_idx": [i for i in range(len(FEATURE_NAMES)) if i not in bounded_set],
        "protocol_idx": [FEATURE_NAMES.index("Protocol Type")],
        # Compatibility keys are deliberately empty: no CSV feature is binary.
        "independent_binary_idx": [],
        "derived_binary_idx": [],
        "pseudo_binary_idx": [],
    }


def apply_scaler_to_columns(
    values: np.ndarray,
    scaler: "RobustScaler",
    col_indices: list[int],
    n_features: int = 39,
) -> np.ndarray:
    center = np.asarray(scaler.center_)[col_indices]
    scale = np.asarray(scaler.scale_)[col_indices]
    return (np.asarray(values, dtype=np.float64) - center) / scale


def inverse_transform_columns(
    x_scaled: np.ndarray,
    scaler: "RobustScaler",
    col_indices: list[int],
    n_features: int = 39,
) -> np.ndarray:
    values = np.asarray(x_scaled)
    subset = values[:, col_indices] if values.shape[1] == n_features else values
    center = np.asarray(scaler.center_)[col_indices]
    scale = np.asarray(scaler.scale_)[col_indices]
    return subset * scale + center

def scaled_to_raw_protocol(
    x_scaled: np.ndarray,
    scaler: "RobustScaler",
    protocol_idx: int,
) -> np.ndarray:
    """Inverse-transform averaged Protocol Type without rounding."""

    return inverse_transform_columns(x_scaled, scaler, [protocol_idx])[:, 0]


def raw_protocol_to_scaled(
    raw_protocol_values: np.ndarray,
    scaler: "RobustScaler",
    protocol_idx: int,
    n_features: int = 39,
) -> np.ndarray:
    return apply_scaler_to_columns(
        np.asarray(raw_protocol_values).reshape(-1, 1),
        scaler,
        [protocol_idx],
        n_features,
    )[:, 0]


def raw_to_protocol_index(raw_value: int) -> int:
    raise RuntimeError("averaged Protocol Type is continuous and has no categorical index")


def protocol_index_to_raw(idx: int) -> int:
    raise RuntimeError("averaged Protocol Type is continuous and has no categorical index")


def derive_binaries_from_protocol_index(protocol_idx_batch: torch.Tensor) -> torch.Tensor:
    """Return no derived binary columns under aggregate semantics."""

    return torch.empty(
        (protocol_idx_batch.shape[0], 0),
        dtype=torch.float32,
        device=protocol_idx_batch.device,
    )


def raw_postprocess(x_raw: np.ndarray) -> np.ndarray:
    """Apply only structural continuous bounds; never round aggregate values."""

    x = np.asarray(x_raw, dtype=np.float32).copy()
    bounded = list(BOUNDED_AGGREGATED_IDX)
    x[:, bounded] = np.clip(x[:, bounded], 0.0, 1.0)
    return x
