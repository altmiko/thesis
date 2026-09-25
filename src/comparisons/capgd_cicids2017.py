"""CICIDS2017 / CSE-CIC-IDS-2018 DistriNet adapter for TabularBench's upstream CAPGD.

The attack implementation is imported from the frozen repository at
``external/tabularbench``.  This module supplies only the dataset-specific pieces
that upstream CAPGD requires: a train-fitted min-max attack space, feature types,
mutability, exact relations, and a raw-space victim wrapper.

The primary comparison mask is the repository's existing nine-feature CICIDS mask.
Its seven exact derived features are repairable outputs, not additional independent
attacker controls.  Validator v2 remains the independent final validity authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import yaml

from attack.masks import get_dataset_mask
from datasets.cicids2017 import CICIDS2017Adapter
from validation import load_validator

CAPGD_METHOD_ID = "capgd_config_mask_l2_eps0.5"
TABULARBENCH_COMMIT = "bfb75415a6a31a41ddfeef34478eea1da227d19c"


@dataclass(frozen=True)
class CAPGDResources:
    """Frozen objects and metadata shared across CAPGD cells."""

    api: SimpleNamespace
    scaler: Any
    constraints: Any
    resolved_mask: Any
    train_min: np.ndarray
    train_max: np.ndarray
    feature_types: np.ndarray
    mutable_features: np.ndarray
    validator: Any
    manifest_payload: dict[str, Any]


class RawCICIDSVictim(nn.Module):
    """Expose a RobustScaler-space victim as a raw/pristine-space module."""

    def __init__(
        self,
        victim: nn.Module,
        center: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        super().__init__()
        self.victim = victim
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.victim((raw - self.center) / self.scale)


def load_tabularbench_api(repo_root: str | Path) -> SimpleNamespace:
    """Import CAPGD from the frozen clone without modifying upstream source.

    TabularBench 0.1 references ``np.float_``, removed by NumPy 2.0.  Restoring the
    historical alias before import is sufficient for the CAPGD-only path and leaves
    the frozen external checkout untouched.
    """

    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]

    root = Path(repo_root).resolve()
    package_root = root / "external" / "tabularbench"

    # CAPGD passes CUDA index tensors to NumPy's setdiff1d while filtering
    # successful restarts. NumPy cannot materialize CUDA tensors directly.
    # Preserve the upstream operation and normalize only those index arguments.
    if not getattr(np.setdiff1d, "_capgd_torch_safe", False):
        original_setdiff1d = np.setdiff1d

        def torch_safe_setdiff1d(ar1, ar2, assume_unique=False):
            if isinstance(ar1, torch.Tensor):
                ar1 = ar1.detach().cpu().numpy()
            if isinstance(ar2, torch.Tensor):
                ar2 = ar2.detach().cpu().numpy()
            return original_setdiff1d(ar1, ar2, assume_unique=assume_unique)

        torch_safe_setdiff1d._capgd_torch_safe = True  # type: ignore[attr-defined]
        np.setdiff1d = torch_safe_setdiff1d
    if not (package_root / "tabularbench" / "attacks" / "capgd" / "capgd.py").exists():
        raise FileNotFoundError(f"missing frozen TabularBench clone: {package_root}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    from tabularbench.attacks.capgd.capgd import CAPGD
    from tabularbench.attacks.utils import compute_distance
    from tabularbench.constraints.constraints import Constraints
    from tabularbench.constraints.constraints_checker import ConstraintChecker
    from tabularbench.constraints.relation_constraint import (
        Constant,
        EqualConstraint,
        Feature,
        LessEqualConstraint,
        SafeDivision,
    )
    from tabularbench.models.tab_scaler import ScalerData, TabScaler

    return SimpleNamespace(
        CAPGD=CAPGD,
        compute_distance=compute_distance,
        Constraints=Constraints,
        ConstraintChecker=ConstraintChecker,
        Constant=Constant,
        EqualConstraint=EqualConstraint,
        Feature=Feature,
        LessEqualConstraint=LessEqualConstraint,
        SafeDivision=SafeDivision,
        ScalerData=ScalerData,
        TabScaler=TabScaler,
    )


def fit_train_minmax(
    pristine_train_path: str | Path,
    *,
    chunk_rows: int = 131_072,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit finite feature bounds from pristine TRAIN only, in bounded memory."""

    x = np.load(pristine_train_path, mmap_mode="r")
    if x.ndim != 2 or x.shape[1] != 79:
        raise ValueError(f"expected train matrix (N,79), got {x.shape}")
    low = np.full(x.shape[1], np.inf, dtype=np.float64)
    high = np.full(x.shape[1], -np.inf, dtype=np.float64)
    for start in range(0, x.shape[0], chunk_rows):
        batch = np.asarray(x[start : start + chunk_rows], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError("nonfinite value in pristine training data")
        low = np.minimum(low, batch.min(axis=0))
        high = np.maximum(high, batch.max(axis=0))
    return low.astype(np.float32), high.astype(np.float32)


def _feature_types(repo_root: Path, feature_names: list[str], dataset: str) -> np.ndarray:
    profile_path = repo_root / "validation" / "schema" / f"{dataset}.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if list(profile["feature_order"]) != feature_names:
        raise ValueError("validator schema feature order differs from dataset manifest")
    result = []
    for name in feature_names:
        value_type = profile["features"][name]["value_type"]
        result.append("real" if value_type == "numeric" else "int")
    return np.asarray(result, dtype="U8")


def _relations(api: SimpleNamespace) -> list[Any]:
    """Relations repaired/penalized by CAPGD for the config-mask dependency map."""

    F = api.Feature
    C = api.Constant
    Eq = api.EqualConstraint
    Div = api.SafeDivision
    Le = api.LessEqualConstraint

    count_fwd = F("Total Fwd Packet")
    count_bwd = F("Total Bwd packets")
    duration = F("Flow Duration")
    total_fwd = F("Total Length of Fwd Packet")
    total_bwd = F("Total Length of Bwd Packet")

    # Integral constants preserve the float32 feature dtype in TabularBench's
    # PyTorch backend. Float constants promote divisors to float64 and make its
    # in-place SafeDivision assignment fail.
    relations = [
        Eq(F("Fwd Packet Length Mean"), Div(total_fwd, count_fwd, C(0)), C(1e-3)),
        Eq(F("Fwd Segment Size Avg"), F("Fwd Packet Length Mean"), C(1e-3)),
        Eq(
            F("Fwd IAT Mean"),
            Div(F("Fwd IAT Total"), count_fwd - C(1), C(0)),
            C(1e-3),
        ),
        Eq(F("Fwd Packets/s"), Div(count_fwd * C(1_000_000), duration, C(0)), C(0.1)),
        Eq(F("Bwd Packets/s"), Div(count_bwd * C(1_000_000), duration, C(0)), C(0.1)),
        Eq(
            F("Flow Packets/s"),
            Div((count_fwd + count_bwd) * C(1_000_000), duration, C(0)),
            C(0.1),
        ),
        Eq(
            F("Flow Bytes/s"),
            Div((total_fwd + total_bwd) * C(1_000_000), duration, C(0)),
            C(0.1),
        ),
        Le(F("Fwd Packet Length Min"), F("Fwd Packet Length Mean")),
        Le(F("Fwd Packet Length Mean"), F("Fwd Packet Length Max")),
        Le(F("Fwd IAT Min"), F("Fwd IAT Mean")),
        Le(F("Fwd IAT Mean"), F("Fwd IAT Max")),
    ]
    return relations


def _make_device_aware_scaler(api: SimpleNamespace) -> Any:
    """Keep upstream scaler outputs on CAPGD's attack device.

    Upstream CAPGD creates a fresh CPU tensor while reinserting successful
    repaired candidates. Its next assignment targets a CUDA tensor. Moving only
    tensor transform results to the configured device fixes that compatibility
    defect without changing the frozen CAPGD source or any NumPy call.
    """

    class DeviceAwareTabScaler(api.TabScaler):
        output_device: torch.device | None = None

        def transform(self, x_in, cat_encode_method: str = "elu"):
            out = super().transform(x_in, cat_encode_method)
            if isinstance(out, torch.Tensor) and self.output_device is not None:
                out = out.to(self.output_device)
            return out

    return DeviceAwareTabScaler(num_scaler="min_max", one_hot_encode=False)


def build_capgd_resources(
    repo_root: str | Path,
    *,
    adapter: CICIDS2017Adapter | None = None,
) -> CAPGDResources:
    """Build train-only attack metadata and upstream TabularBench objects."""

    repo = Path(repo_root).resolve()
    adapter = adapter or CICIDS2017Adapter(repo)
    manifest = adapter.feature_manifest()
    feature_names = manifest.names
    api = load_tabularbench_api(repo)

    train_path = adapter._processed / "X_train_pristine.npy"
    train_min, train_max = fit_train_minmax(train_path)
    dataset = manifest.dataset_name
    feature_types = _feature_types(repo, feature_names, dataset)

    resolved = get_dataset_mask(dataset).resolve(manifest)
    direct = np.zeros(manifest.n_features, dtype=bool)
    direct[list(resolved.perturbable_idx)] = True
    repairable = np.zeros(manifest.n_features, dtype=bool)
    repairable[list(resolved.derived_idx)] = True
    mutable = direct | repairable

    constraints = api.Constraints(
        feature_types=feature_types,
        mutable_features=mutable,
        lower_bounds=train_min,
        upper_bounds=train_max,
        relation_constraints=_relations(api),
        feature_names=np.asarray(feature_names),
    )

    scaler = _make_device_aware_scaler(api)
    scaler.fit_scaler_data(
        api.ScalerData(
            x_min=torch.as_tensor(train_min, dtype=torch.float32),
            x_max=torch.as_tensor(train_max, dtype=torch.float32),
            categories=[],
            cat_idx=[],
            num_idx=list(range(manifest.n_features)),
        )
    )

    constants = np.flatnonzero(train_min == train_max).tolist()
    payload = {
        "dataset": dataset,
        "fit_split": "train",
        "fit_array": str(train_path),
        "n_features": manifest.n_features,
        "feature_order": feature_names,
        "feature_types": feature_types.tolist(),
        "direct_mutable_features": list(resolved.mask.perturbable),
        "repairable_derived_features": [d.name for d in resolved.mask.derived],
        "frozen_features": [feature_names[i] for i in resolved.frozen_idx],
        "train_min": train_min.astype(float).tolist(),
        "train_max": train_max.astype(float).tolist(),
        "train_constant_features": [feature_names[i] for i in constants],
        "relations": [
            {"target": d.name, "expression": d.expression, "source": d.source}
            for d in resolved.mask.derived
        ]
        + [
            {"expression": "Fwd Packet Length Min <= Fwd Packet Length Mean <= Fwd Packet Length Max"},
            {"expression": "Fwd IAT Min <= Fwd IAT Mean <= Fwd IAT Max"},
        ],
        "tabularbench_commit": TABULARBENCH_COMMIT,
        "numpy_compatibility": "np.float_ alias restored to np.float64 before importing frozen TabularBench",
    }

    return CAPGDResources(
        api=api,
        scaler=scaler,
        constraints=constraints,
        resolved_mask=resolved,
        train_min=train_min,
        train_max=train_max,
        feature_types=feature_types,
        mutable_features=mutable,
        validator=load_validator(dataset),
        manifest_payload=payload,
    )


def numpy_predict_proba(
    model: nn.Module,
    *,
    device: str | torch.device,
    batch_size: int = 4096,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the NumPy probability API expected by TabularBench."""

    target = torch.device(device)

    def predict(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                raw = torch.as_tensor(values[start : start + batch_size], device=target)
                outputs.append(torch.softmax(model(raw), dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 5), np.float32)

    return predict


def make_capgd(
    resources: CAPGDResources,
    raw_victim: RawCICIDSVictim,
    *,
    device: str | torch.device,
    seed: int,
    norm: str = "L2",
    eps: float = 0.5,
    steps: int = 10,
) -> Any:
    """Construct the exact CAPGD component configuration used by standard CAA."""

    resources.scaler.output_device = torch.device(device)

    model_objective = numpy_predict_proba(raw_victim, device=device)
    attack = resources.api.CAPGD(
        constraints=resources.constraints,
        scaler=resources.scaler,
        model=raw_victim,
        model_objective=model_objective,
        norm=norm,
        eps=eps,
        steps=steps,
        n_restarts=2,
        seed=seed,
        loss="ce",
        eot_iter=1,
        rho=0.75,
        fix_equality_constraints_end=True,
        fix_equality_constraints_iter=True,
        adaptive_eps=True,
        random_start=True,
        init_start=True,
        best_restart=False,
        eps_margin=0.01,
        verbose=False,
    )
    attack.set_device(str(device))
    return attack


def finalize_capgd_output(
    resources: CAPGDResources,
    clean_raw: torch.Tensor,
    candidate_raw: torch.Tensor,
) -> torch.Tensor:
    """Apply the repository-native immutable/derived contract after upstream repair."""

    return resources.resolved_mask.apply(candidate_raw, clean_raw)


def evaluate_capgd_output(
    resources: CAPGDResources,
    clean_raw: np.ndarray,
    adversarial_raw: np.ndarray,
    *,
    norm: str,
    eps: float,
) -> dict[str, np.ndarray | dict[str, int]]:
    """Independently evaluate CAPGD constraints, distance, and validator-v2 layers."""

    clean = np.asarray(clean_raw, dtype=np.float32)
    adv = np.asarray(adversarial_raw, dtype=np.float32)
    if clean.shape != adv.shape or clean.ndim != 2:
        raise ValueError(f"clean/adv shape mismatch: {clean.shape} vs {adv.shape}")
    if not np.isfinite(adv).all():
        raise FloatingPointError("CAPGD produced NaN or Inf")

    checker = resources.api.ConstraintChecker(resources.constraints, tolerance=0.0)
    internal = checker.check_constraints(clean, adv).astype(bool)
    clean_scaled = resources.scaler.transform(clean)
    adv_scaled = resources.scaler.transform(adv)
    distance = np.asarray(
        resources.api.compute_distance(clean_scaled, adv_scaled, norm),
        dtype=np.float64,
    )
    distance_ok = distance <= float(eps) + 1e-6

    validated = resources.validator.validate_batch(adv)
    rejecting = validated.rules_rejecting_any()
    return {
        "internal_constraint_valid": internal,
        "distance": distance,
        "distance_ok": distance_ok,
        "schema_valid": validated.schema_valid,
        "extractor_valid": validated.extractor_valid,
        "protocol_valid": validated.protocol_valid,
        "mined_valid": validated.mined_valid,
        "hard_structural_valid": validated.hard_structural_valid,
        "hybrid_valid": validated.hybrid_valid,
        "in_distribution": validated.in_distribution,
        "plausibility_score": validated.plausibility_score,
        "rules_rejecting": rejecting,
    }


def save_constraint_manifest(resources: CAPGDResources, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(resources.manifest_payload, indent=2), encoding="utf-8"
    )
