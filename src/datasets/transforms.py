"""Manifest-aware feature scaling, decoupled from the VAE.

The model must not own preprocessing or carry hidden scaler state. ``FeatureTransform``
is the one abstraction that maps raw <-> model space. It is fit on TRAIN data only
and refuses to transform before being fit (no silent center=0 / scale=1 identity).

For CICIoT2023 the existing pipeline fit a single sklearn ``RobustScaler`` over all
39 columns; :meth:`from_sklearn_scaler` wraps that fitted object so existing arrays
and checkpoints remain bit-compatible, while :meth:`fit` supports fresh, per-feature
train-only fitting for new datasets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from datasets.feature_manifest import FeatureManifest, ManifestError

if TYPE_CHECKING:  # pragma: no cover
    from sklearn.preprocessing import RobustScaler


class TransformNotFittedError(RuntimeError):
    """Raised when transform/inverse_transform is called before fitting."""


# scaling family -> whether it centers/scales; "none" leaves the column untouched.
_KNOWN_SCALINGS = frozenset({"robust", "standard", "none", "identity"})


class FeatureTransform:
    """Affine per-feature transform ``x_scaled = (x_raw - center) / scale``.

    ``center``/``scale`` are full-length (one entry per manifest feature); columns
    whose ``scaling`` is ``none``/``identity`` use center=0, scale=1 by design (an
    explicit, manifest-declared decision, not an accidental fallback).
    """

    def __init__(self, manifest: FeatureManifest) -> None:
        self.manifest = manifest
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._manifest_hash: str | None = None
        self._fitted_on: str | None = None  # provenance label, e.g. "train"

    # -- state ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self._center is not None and self._scale is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise TransformNotFittedError(
                "FeatureTransform used before fit(); refusing implicit identity "
                "scaling. Call fit(X_train) or from_sklearn_scaler(...)."
            )

    # -- fitting (TRAIN ONLY) ---------------------------------------------------
    def fit(self, x_train: np.ndarray, *, provenance: str = "train") -> "FeatureTransform":
        """Fit per-feature center/scale on training data only.

        ``provenance`` is recorded purely as a leakage-audit breadcrumb; passing a
        non-train label does not change behaviour but is preserved in saved state.
        """
        x = np.asarray(x_train, dtype=np.float64)
        if x.ndim != 2:
            raise ManifestError("fit expects a 2D (N, F) array")
        self.manifest.assert_matches_array(x)
        n_features = self.manifest.n_features
        center = np.zeros(n_features, dtype=np.float64)
        scale = np.ones(n_features, dtype=np.float64)
        for spec in self.manifest.specs:
            fam = spec.scaling
            if fam not in _KNOWN_SCALINGS:
                raise ManifestError(f"{spec.name}: unknown scaling {fam!r}")
            col = x[:, spec.model_index]
            if fam == "robust":
                med = float(np.median(col))
                q75, q25 = np.percentile(col, [75, 25])
                iqr = float(q75 - q25)
                center[spec.model_index] = med
                scale[spec.model_index] = iqr if iqr > 0.0 else 1.0
            elif fam == "standard":
                center[spec.model_index] = float(col.mean())
                std = float(col.std())
                scale[spec.model_index] = std if std > 0.0 else 1.0
            # "none"/"identity": leave center=0, scale=1
        self._center = center
        self._scale = scale
        self._manifest_hash = self.manifest.content_hash
        self._fitted_on = provenance
        return self

    @classmethod
    def from_sklearn_scaler(
        cls, scaler: "RobustScaler", manifest: FeatureManifest
    ) -> "FeatureTransform":
        """Wrap an already-fitted sklearn scaler (e.g. the existing CICIoT scaler)."""
        manifest.assert_matches_scaler(scaler)
        obj = cls(manifest)
        center = np.asarray(scaler.center_, dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        scale = np.where(scale == 0.0, 1.0, scale)
        obj._center = center
        obj._scale = scale
        obj._manifest_hash = manifest.content_hash
        obj._fitted_on = "external_sklearn"
        return obj

    # -- application ------------------------------------------------------------
    def transform(self, x_raw: np.ndarray) -> np.ndarray:
        self._require_fitted()
        x = np.asarray(x_raw, dtype=np.float64)
        self.manifest.assert_matches_array(x)
        return (x - self._center) / self._scale

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        self._require_fitted()
        x = np.asarray(x_scaled, dtype=np.float64)
        self.manifest.assert_matches_array(x)
        return x * self._scale + self._center

    # -- torch-friendly views (for the decoder's raw<->scaled round trip) -------
    @property
    def center(self) -> np.ndarray:
        self._require_fitted()
        return self._center  # type: ignore[return-value]

    @property
    def scale(self) -> np.ndarray:
        self._require_fitted()
        return self._scale  # type: ignore[return-value]

    # -- (de)serialization ------------------------------------------------------
    def to_state(self) -> dict:
        self._require_fitted()
        return {
            "manifest_hash": self._manifest_hash,
            "dataset_name": self.manifest.dataset_name,
            "fitted_on": self._fitted_on,
            "center": self._center.tolist(),  # type: ignore[union-attr]
            "scale": self._scale.tolist(),  # type: ignore[union-attr]
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_state(), indent=2), encoding="utf-8")

    @classmethod
    def from_state(cls, state: dict, manifest: FeatureManifest) -> "FeatureTransform":
        manifest.assert_compatible_hash(
            state["manifest_hash"], context="FeatureTransform.from_state"
        )
        obj = cls(manifest)
        obj._center = np.asarray(state["center"], dtype=np.float64)
        obj._scale = np.asarray(state["scale"], dtype=np.float64)
        obj._manifest_hash = state["manifest_hash"]
        obj._fitted_on = state.get("fitted_on")
        obj.manifest.assert_matches_array(np.zeros((1, obj._center.shape[0])))
        return obj

    @classmethod
    def load(cls, path: str | Path, manifest: FeatureManifest) -> "FeatureTransform":
        return cls.from_state(
            json.loads(Path(path).read_text(encoding="utf-8")), manifest
        )
