"""CICIoT2023 perturbation mask (39 window-aggregate features, Modified Schema A).

Attacker degrees of freedom are the packet-size / timing aggregates the task pins;
derived features are the exact window statistics that follow from them.

Formula verification is empirical against the TRAIN split. CICIoT2023 stores only the
RobustScaler-scaled ``X_train.npy``; the checks below invert the fitted scaler
(``data/processed/scaler.pkl``: ``raw = scaled * scale_ + center_``) on 300k rows and
compare with the recorded feature:

* ``Variance == Std**2``            -> exact_fraction 1.00000 (rtol 1e-3, max_rel_err 3e-7)
* ``Tot size == AVG``               -> exact_fraction 1.00000 (max_rel_err 0) -- in this
  release "Tot size" is NOT a total; it equals the mean packet size AVG.
* ``AVG == Tot sum / Number``       -> exact_fraction 0.99885; the 0.115% exceptions are
  rows where AVG saturates at a preprocessing cap (~3117.36) while Tot sum keeps rising.
  Enabled as DERIVED_EXACT with that documented caveat (see notes).

``Rate`` is deliberately FROZEN: ``Rate == 1/IAT`` holds for only 41% of rows
(max_rel_err ~1.8e4), so no exact dependency on the perturbable set exists.

CICIoT2023 is provided for parity/portability; only CICIDS2017 attacks are rerun.
"""
from __future__ import annotations

import torch

from attack.masks.base import DatasetMask, DerivedFeature

PERTURBABLE = (
    "Tot sum",
    "Min",
    "Max",
    "Std",
    "IAT",
)
# 1-indexed positions in the frozen 39-feature schema (verified in resolve()).
EXPECTED_PERTURBABLE_INDEX1 = (31, 32, 33, 35, 37)


def _col(x: torch.Tensor, i: dict[str, int], name: str) -> torch.Tensor:
    return x[:, i[name]]


def _avg(x: torch.Tensor, i: dict[str, int]) -> torch.Tensor:
    return _col(x, i, "Tot sum") / _col(x, i, "Number").clamp(min=1.0)


def _tot_size(x: torch.Tensor, i: dict[str, int]) -> torch.Tensor:
    # This release's "Tot size" equals the mean packet size AVG (verified exact).
    return _col(x, i, "AVG")


def _variance(x: torch.Tensor, i: dict[str, int]) -> torch.Tensor:
    return _col(x, i, "Std") * _col(x, i, "Std")


DERIVED: tuple[DerivedFeature, ...] = (
    DerivedFeature(
        name="AVG",
        parents=("Tot sum", "Number"),
        formula=_avg,
        expression="Tot sum / max(Number, 1)",
        source="empirical exact_fraction=0.99885 (train, cap saturation tail; see notes)",
    ),
    DerivedFeature(
        name="Tot size",
        parents=("AVG",),
        formula=_tot_size,
        expression="AVG",
        source="empirical exact_fraction=1.0 (train, max_rel_err 0)",
    ),
    DerivedFeature(
        name="Variance",
        parents=("Std",),
        formula=_variance,
        expression="Std**2",
        source="empirical exact_fraction=1.0 (train, rtol 1e-3, max_rel_err 3e-7)",
    ),
)


def build_mask() -> DatasetMask:
    return DatasetMask(
        dataset_name="ciciot2023",
        n_features=39,
        perturbable=PERTURBABLE,
        derived=DERIVED,
        expected_perturbable_index1=EXPECTED_PERTURBABLE_INDEX1,
        notes=(
            "Rate is frozen (not reconstructible: Rate==1/IAT holds for only ~41% of "
            "rows). AVG=Tot sum/Number is exact except a ~0.115% preprocessing-cap "
            "saturation tail. Header/protocol/TTL, flag and service/protocol window "
            "means, packet counts, Number, Min/Max are frozen unless perturbable."
        ),
    )
