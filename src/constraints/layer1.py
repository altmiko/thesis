"""Layer 1 — generic soft constraints.

The *algorithm* of each constraint is dataset-independent; only its numeric
parameters (medians, IQRs, tolerances) or which features it references are supplied
externally. Parameters that come from data MUST be fit on TRAIN only.

Constraints are deliberately few and strongly justified (per the spec's "do not
over-engineer Layer 1"): a robust per-feature tail bound, plus three generic
cross-feature relations that Layer 2 instantiates for a given dataset.
"""
from __future__ import annotations

import numpy as np
import torch

from constraints.base import Constraint, _as_tensor
from datasets.feature_manifest import FeatureManifest

_EPS = 1e-8


class RobustTailBound(Constraint):
    """Per-feature robust tail: penalize |x - median| / scale beyond ``tau``.

    Scale is TRAIN-fit IQR, with standard deviation fallback for sparse columns whose
    IQR is zero, and 1 only for truly constant columns. This avoids treating every
    non-median value in a sparse-but-variable feature as an outlier.
    """

    layer = 1

    def __init__(
        self,
        manifest: FeatureManifest,
        feature_names: list[str],
        median: np.ndarray,
        iqr: np.ndarray,
        tau: float = 4.0,
        name: str = "robust_tail",
    ) -> None:
        super().__init__(name, manifest)
        self.feature_names = list(feature_names)
        self.idx = torch.tensor([self.col(n) for n in self.feature_names], dtype=torch.long)
        self.median = torch.tensor(np.asarray(median), dtype=torch.float32)
        self.iqr = torch.tensor(np.asarray(iqr), dtype=torch.float32)
        self.tau = float(tau)

    @classmethod
    def fit(
        cls,
        manifest: FeatureManifest,
        x_train_raw: np.ndarray,
        *,
        feature_names: list[str] | None = None,
        tau: float | None = 4.0,
        coverage: float = 0.99,
        name: str = "robust_tail",
    ) -> "RobustTailBound":
        x = np.asarray(x_train_raw, dtype=np.float64)
        manifest.assert_matches_array(x)
        names = feature_names or manifest.names
        cols = [manifest.index_by_name(n) for n in names]
        med = np.median(x[:, cols], axis=0)
        q75, q25 = np.percentile(x[:, cols], [75, 25], axis=0)
        iqr = q75 - q25
        std = x[:, cols].std(axis=0)
        scale = np.where(iqr > _EPS, iqr, std)
        scale = np.where(scale > _EPS, scale, 1.0)
        if tau is None:
            if not 0.0 < coverage < 1.0:
                raise ValueError("coverage must be between 0 and 1")
            max_z = (np.abs(x[:, cols] - med) / scale).max(axis=1)
            tau = float(np.quantile(max_z, coverage, method="higher"))
        return cls(manifest, names, med, scale, tau=tau, name=name)

    def _z(self, x_raw: torch.Tensor) -> torch.Tensor:
        sub = x_raw[:, self.idx.to(x_raw.device)]
        med = self.median.to(x_raw.device).unsqueeze(0)
        iqr = self.iqr.to(x_raw.device).unsqueeze(0)
        return (sub - med).abs() / (iqr + _EPS)

    def penalty(self, x_raw: torch.Tensor, ctx: dict | None = None) -> torch.Tensor:
        return torch.relu(self._z(x_raw) - self.tau).mean()

    def validate(self, x_raw: torch.Tensor, ctx: dict | None = None) -> torch.Tensor:
        return (self._z(x_raw) <= self.tau).all(dim=1)

    def to_config(self) -> dict:
        return {
            "type": "RobustTailBound",
            "name": self.name,
            "params": {
                "feature_names": self.feature_names,
                "median": self.median.tolist(),
                "iqr": self.iqr.tolist(),
                "tau": self.tau,
            },
        }

    @classmethod
    def from_config(cls, manifest: FeatureManifest, cfg: dict) -> "RobustTailBound":
        p = cfg["params"]
        return cls(
            manifest,
            p["feature_names"],
            np.asarray(p["median"]),
            np.asarray(p["iqr"]),
            tau=float(p.get("tau", 4.0)),
            name=cfg.get("name", "robust_tail"),
        )


class ProductEquality(Constraint):
    """Relative equality ``target ~= prod(factors)`` within ``rtol`` (e.g. Tot=N*AVG)."""

    layer = 1

    def __init__(self, manifest, target, factors, rtol=0.05, name="product_equality"):
        super().__init__(name, manifest)
        self.target = target
        self.factors = list(factors)
        self.rtol = float(rtol)
        self.t_idx = self.col(target)
        self.f_idx = torch.tensor([self.col(f) for f in self.factors], dtype=torch.long)

    def _rel(self, x_raw: torch.Tensor) -> torch.Tensor:
        prod = x_raw[:, self.f_idx.to(x_raw.device)].prod(dim=1)
        t = x_raw[:, self.t_idx]
        return (t - prod).abs() / (t.abs() + 1.0)

    def penalty(self, x_raw, ctx=None):
        return self._rel(x_raw).mean()

    def validate(self, x_raw, ctx=None):
        return self._rel(x_raw) < self.rtol

    def to_config(self):
        return {"type": "ProductEquality", "name": self.name,
                "params": {"target": self.target, "factors": self.factors, "rtol": self.rtol}}

    @classmethod
    def from_config(cls, manifest, cfg):
        p = cfg["params"]
        return cls(manifest, p["target"], p["factors"], rtol=float(p.get("rtol", 0.05)),
                   name=cfg.get("name", "product_equality"))


class MonotoneNondecreasing(Constraint):
    """Ordered features must be non-decreasing, e.g. Min <= AVG <= Max."""

    layer = 1

    def __init__(self, manifest, features, tol=1e-6, name="monotone"):
        super().__init__(name, manifest)
        self.features = list(features)
        self.tol = float(tol)
        self.idx = torch.tensor([self.col(f) for f in self.features], dtype=torch.long)

    def penalty(self, x_raw, ctx=None):
        sub = x_raw[:, self.idx.to(x_raw.device)]
        diffs = sub[:, :-1] - sub[:, 1:]  # prev - next; >0 is a violation
        scale = sub[:, :-1].abs() + sub[:, 1:].abs() + 1.0
        return (torch.relu(diffs) / scale).mean()

    def validate(self, x_raw, ctx=None):
        sub = x_raw[:, self.idx.to(x_raw.device)]
        return (sub[:, 1:] - sub[:, :-1] >= -self.tol).all(dim=1)

    def to_config(self):
        return {"type": "MonotoneNondecreasing", "name": self.name,
                "params": {"features": self.features, "tol": self.tol}}

    @classmethod
    def from_config(cls, manifest, cfg):
        p = cfg["params"]
        return cls(manifest, p["features"], tol=float(p.get("tol", 1e-6)),
                   name=cfg.get("name", "monotone"))


class HalfRangeBound(Constraint):
    """Spread bound: ``value <= 0.5*(hi - lo)*(1+rtol)`` (e.g. Std vs Min/Max)."""

    layer = 1

    def __init__(self, manifest, value, lo, hi, rtol=0.05, name="half_range"):
        super().__init__(name, manifest)
        self.value, self.lo, self.hi = value, lo, hi
        self.rtol = float(rtol)
        self.v_idx, self.lo_idx, self.hi_idx = self.col(value), self.col(lo), self.col(hi)

    def _excess(self, x_raw):
        bound = 0.5 * (x_raw[:, self.hi_idx] - x_raw[:, self.lo_idx]) * (1.0 + self.rtol)
        return x_raw[:, self.v_idx] - bound

    def penalty(self, x_raw, ctx=None):
        excess = self._excess(x_raw)
        value = x_raw[:, self.v_idx].abs()
        return (torch.relu(excess) / (value + 1.0)).mean()

    def validate(self, x_raw, ctx=None):
        return self._excess(x_raw) <= 0.0

    def to_config(self):
        return {"type": "HalfRangeBound", "name": self.name,
                "params": {"value": self.value, "lo": self.lo, "hi": self.hi, "rtol": self.rtol}}

    @classmethod
    def from_config(cls, manifest, cfg):
        p = cfg["params"]
        return cls(manifest, p["value"], p["lo"], p["hi"], rtol=float(p.get("rtol", 0.05)),
                   name=cfg.get("name", "half_range"))
