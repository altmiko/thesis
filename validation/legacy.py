"""Legacy validator adapter -- preserved for auditable comparison, NOT modified.

The CICIDS2017 "current" structural validator is the mined ConstraintEngine
Layer-2 check driven by ``old_constraints/cicids2017_distrinet/mined.json`` (see
docs/validator_v2_audit.md). Rather than importing the heavy attack/VAE stack
(torch + manifest), this adapter is a faithful, self-contained reimplementation
of that engine's ``validate()`` semantics for the two rule types actually present
in the artifact:

* ``MonotoneNondecreasing`` -- consecutive diffs >= -tol
* ``ProductEquality``       -- ``|target - prod(factors)| / (|target| + 1) < rtol``

matching ``src/constraints/layer1.py``. It exists ONLY so validator_v2 can be
compared against the legacy behaviour on identical inputs. The original
implementation under ``src/`` is left untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINED = REPO_ROOT / "old_constraints" / "cicids2017_distrinet" / "mined.json"


class LegacyValidator:
    def __init__(self, feature_order: list[str], mined_path: Path = DEFAULT_MINED):
        self.feature_order = list(feature_order)
        self.idx = {f: i for i, f in enumerate(self.feature_order)}
        self.constraints = json.loads(Path(mined_path).read_text())["constraints"]

    def _col(self, X, f):
        return X[:, self.idx[f]].astype(np.float64)

    def per_rule(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """rule name -> per-sample VIOLATION mask (True == violates)."""
        X = np.asarray(X, np.float64)
        if X.ndim == 1:
            X = X[None, :]
        out: dict[str, np.ndarray] = {}
        for c in self.constraints:
            t, name, p = c["type"], c["name"], c["params"]
            if t == "MonotoneNondecreasing":
                feats = p["features"]; tol = p.get("tol", 1e-6)
                viol = np.zeros(X.shape[0], bool)
                for a, b in zip(feats[:-1], feats[1:]):
                    viol |= self._col(X, a) > self._col(X, b) + tol
                out[name] = viol
            elif t == "ProductEquality":
                tgt = p["target"]; factors = p["factors"]; rtol = p.get("rtol", 0.05)
                prod = np.ones(X.shape[0], np.float64)
                for f in factors:
                    prod = prod * self._col(X, f)
                tv = self._col(X, tgt)
                rel = np.abs(tv - prod) / (np.abs(tv) + 1.0)
                out[name] = rel >= rtol
        return out

    def validate_batch(self, X: np.ndarray) -> np.ndarray:
        """Per-sample VALID mask (True == passes all legacy mined constraints)."""
        pr = self.per_rule(X)
        n = (np.asarray(X).shape[0] if np.asarray(X).ndim > 1 else 1)
        if not pr:
            return np.ones(n, bool)
        return ~np.stack(list(pr.values()), axis=1).any(axis=1)
