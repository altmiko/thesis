"""CICIDS2017-DistriNet perturbation mask (79 CICFlowMeter features).

Attacker degrees of freedom are a small forward-direction / timing set; every other
feature is frozen unless it is an *exact* CICFlowMeter identity of a perturbed parent.

Formula verification is empirical against the pristine (unscaled) TRAIN array
``data/processed/CICIDS_2017_Distrinet/X_train_pristine.npy`` (1,456,265 rows) and
cross-checked on the TEST array (312,056 rows). Each identity below holds with
``exact_fraction == 1.0`` at ``rtol = 1e-4`` and worst-case relative error ~1e-7 once
the zero-duration convention is applied (both splits contain zero rows with
Flow Duration == 0). This mirrors the repo's own audit,
``outputs/cicids2017distrinet/feature_audit/structural_relationship_checks.csv``.

Rate features are enabled here because the check passes; note the CICFlowMeter
zero-duration convention (rate := 0 when Flow Duration == 0) is reproduced so an
attack that drives duration to zero stays extractor-consistent.
"""
from __future__ import annotations

import torch

from attack.masks.base import DatasetMask, DerivedFeature

_US_PER_S = 1.0e6  # CICFlowMeter emits Flow Duration / IAT totals in microseconds.
# Genuine Flow Duration floor is 1 us (0% of train/test rows fall below it). A flow
# below this is effectively instantaneous -> CICFlowMeter-style rate 0. Guarding here
# (rather than at duration==0) also removes the float32 rate blow-up when an attack
# drives duration to ~0, where the huge RobustScaler scale (~5e6) makes the scaled
# duration indistinguishable from zero.
_DUR_FLOOR_US = 0.5

PERTURBABLE = (
    "Flow Duration",
    "Total Length of Fwd Packet",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Std",
    "Fwd IAT Total",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
)
# 1-indexed positions in the frozen 79-feature contract (verified in resolve()).
EXPECTED_PERTURBABLE_INDEX1 = (4, 7, 9, 10, 12, 23, 25, 26, 27)


def _col(x: torch.Tensor, i: dict[str, int], name: str) -> torch.Tensor:
    return x[:, i[name]]


def _fwd_packet_length_mean(x: torch.Tensor, i: dict[str, int]) -> torch.Tensor:
    tlf = _col(x, i, "Total Length of Fwd Packet")
    count = _col(x, i, "Total Fwd Packet").clamp(min=1.0)
    return tlf / count


def _fwd_segment_size_avg(x: torch.Tensor, i: dict[str, int]) -> torch.Tensor:
    # CICFlowMeter identity: Fwd Segment Size Avg == Fwd Packet Length Mean (exact).
    return _col(x, i, "Fwd Packet Length Mean")


def _fwd_iat_mean(x: torch.Tensor, i: dict[str, int]) -> torch.Tensor:
    # Mean of (count-1) forward inter-arrival intervals; single-packet flows -> 0.
    total = _col(x, i, "Fwd IAT Total")
    intervals = (_col(x, i, "Total Fwd Packet") - 1.0).clamp(min=1.0)
    return total / intervals


def _rate(numerator):
    def formula(x: torch.Tensor, i: dict[str, int]) -> torch.Tensor:
        dur = _col(x, i, "Flow Duration")
        num = numerator(x, i)
        dur_s = (dur / _US_PER_S).clamp(min=1e-12)
        return torch.where(dur > _DUR_FLOOR_US, num / dur_s, torch.zeros_like(num))

    return formula


DERIVED: tuple[DerivedFeature, ...] = (
    DerivedFeature(
        name="Fwd Packet Length Mean",
        parents=("Total Length of Fwd Packet", "Total Fwd Packet"),
        formula=_fwd_packet_length_mean,
        expression="Total Length of Fwd Packet / max(Total Fwd Packet, 1)",
        source="CICFlowMeter mean; empirical exact_fraction=1.0 (train+test, rtol 1e-4)",
    ),
    DerivedFeature(
        name="Fwd Segment Size Avg",
        parents=("Fwd Packet Length Mean",),
        formula=_fwd_segment_size_avg,
        expression="Fwd Packet Length Mean",
        source="structural_relationship_checks.csv: identity, satisfaction 1.0, max_rel_err 0",
    ),
    DerivedFeature(
        name="Fwd IAT Mean",
        parents=("Fwd IAT Total", "Total Fwd Packet"),
        formula=_fwd_iat_mean,
        expression="Fwd IAT Total / max(Total Fwd Packet - 1, 1)",
        source="structural_relationship_checks.csv: 'Fwd IAT Total from mean and interval count' TRUE",
    ),
    DerivedFeature(
        name="Fwd Packets/s",
        parents=("Total Fwd Packet", "Flow Duration"),
        formula=_rate(lambda x, i: _col(x, i, "Total Fwd Packet")),
        expression="Total Fwd Packet / (Flow Duration / 1e6)   [0 if duration==0]",
        source="empirical exact_fraction=1.0 (train+test, rtol 1e-4) with zero-duration guard",
    ),
    DerivedFeature(
        name="Bwd Packets/s",
        parents=("Total Bwd packets", "Flow Duration"),
        formula=_rate(lambda x, i: _col(x, i, "Total Bwd packets")),
        expression="Total Bwd packets / (Flow Duration / 1e6)   [0 if duration==0]",
        source="empirical exact_fraction=1.0 (train+test, rtol 1e-4) with zero-duration guard",
    ),
    DerivedFeature(
        name="Flow Packets/s",
        parents=("Total Fwd Packet", "Total Bwd packets", "Flow Duration"),
        formula=_rate(lambda x, i: _col(x, i, "Total Fwd Packet") + _col(x, i, "Total Bwd packets")),
        expression="(Total Fwd Packet + Total Bwd packets) / (Flow Duration / 1e6)   [0 if duration==0]",
        source="empirical exact_fraction=1.0 (train+test, rtol 1e-4) with zero-duration guard",
    ),
    DerivedFeature(
        name="Flow Bytes/s",
        parents=("Total Length of Fwd Packet", "Total Length of Bwd Packet", "Flow Duration"),
        formula=_rate(
            lambda x, i: _col(x, i, "Total Length of Fwd Packet") + _col(x, i, "Total Length of Bwd Packet")
        ),
        expression="(Total Length of Fwd Packet + Total Length of Bwd Packet) / (Flow Duration / 1e6)   [0 if duration==0]",
        source="empirical exact_fraction=1.0 (train+test, rtol 1e-4) with zero-duration guard",
    ),
)


def build_mask() -> DatasetMask:
    return DatasetMask(
        dataset_name="cicids2017_distrinet",
        n_features=79,
        perturbable=PERTURBABLE,
        derived=DERIVED,
        expected_perturbable_index1=EXPECTED_PERTURBABLE_INDEX1,
        notes=(
            "Backward-direction stats, packet counts, ports/protocol, TCP flags, "
            "window/header fields, packet-length/IAT std/min/max aggregates, bulk, "
            "subflow and active/idle statistics are frozen: they cannot be uniquely "
            "reconstructed from the perturbable aggregates."
        ),
    )
