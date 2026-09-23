"""Distributional plausibility -- kept STRICTLY separate from structural validity.

A plausibility profile stores, per feature, robust train quantiles (a low/high
band) plus median/IQR for a robust z-score. It answers "is this sample near the
training distribution?" and NEVER contributes to ``structurally_valid``.

Observed train min/max are deliberately NOT used as hard validity constraints
(see methodology §9): they live here, as plausibility, not there.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class PlausibilityProfile:
    dataset: str
    feature_order: list[str]
    low: np.ndarray          # (F,) low quantile per feature
    high: np.ndarray         # (F,) high quantile per feature
    median: np.ndarray       # (F,)
    iqr: np.ndarray          # (F,)
    quantiles: tuple[float, float]
    max_feature_extremeness: float = 0.0  # z at which a feature counts as OOD
    fit_split: str = "train"

    def evaluate(self, X: np.ndarray) -> dict:
        """Return per-sample in_distribution mask, score, and violations list."""
        X = np.asarray(X, np.float64)
        n = X.shape[0]
        lo = self.low[None, :]
        hi = self.high[None, :]
        within = (X >= lo) & (X <= hi)
        score = within.mean(axis=1)
        # robust z on features outside the band, for reporting extremeness
        z = np.abs(X - self.median[None, :]) / (self.iqr[None, :] + 1e-6)
        in_dist = within.all(axis=1)
        violations: list[list[dict]] = []
        for i in range(n):
            bad = np.where(~within[i])[0]
            v = []
            for j in bad:
                side = "above" if X[i, j] > hi[0, j] else "below"
                v.append({
                    "feature": self.feature_order[j],
                    "reason": f"{side} {self.quantiles[1]*100:.1f}th/{self.quantiles[0]*100:.1f}th "
                              f"percentile band of training distribution",
                    "value": float(X[i, j]),
                    "band": [float(lo[0, j]), float(hi[0, j])],
                    "robust_z": float(z[i, j]),
                })
            violations.append(v)
        return {"in_distribution": in_dist, "score": score, "violations": violations}

    # ---- io ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "fit_split": self.fit_split,
            "quantiles": {"low": self.quantiles[0], "high": self.quantiles[1]},
            "feature_order": self.feature_order,
            "features": {
                f: {"low": float(self.low[i]), "high": float(self.high[i]),
                    "median": float(self.median[i]), "iqr": float(self.iqr[i])}
                for i, f in enumerate(self.feature_order)
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "PlausibilityProfile":
        order = d["feature_order"]
        feats = d["features"]
        return cls(
            dataset=d["dataset"],
            feature_order=order,
            low=np.array([feats[f]["low"] for f in order], np.float64),
            high=np.array([feats[f]["high"] for f in order], np.float64),
            median=np.array([feats[f]["median"] for f in order], np.float64),
            iqr=np.array([feats[f]["iqr"] for f in order], np.float64),
            quantiles=(d["quantiles"]["low"], d["quantiles"]["high"]),
            fit_split=d.get("fit_split", "train"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PlausibilityProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def fit(cls, X_train: np.ndarray, feature_order: list[str], dataset: str,
            low_q: float = 0.001, high_q: float = 0.999) -> "PlausibilityProfile":
        X = np.asarray(X_train, np.float64)
        lo = np.percentile(X, low_q * 100, axis=0)
        hi = np.percentile(X, high_q * 100, axis=0)
        med = np.median(X, axis=0)
        q75, q25 = np.percentile(X, 75, axis=0), np.percentile(X, 25, axis=0)
        iqr = q75 - q25
        return cls(dataset=dataset, feature_order=list(feature_order),
                   low=lo, high=hi, median=med, iqr=iqr, quantiles=(low_q, high_q))
