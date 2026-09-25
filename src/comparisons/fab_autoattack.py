"""FAB (Fast Adaptive Boundary, Croce & Hein 2020) from the frozen AutoAttack clone.

The attack implementation is imported unmodified from ``external/auto-attack``
(``autoattack.fab_pt.FABAttack_PT``).  FAB is an *unconstrained*, minimum-norm
input-space attack: it moves every feature and knows nothing about flow semantics,
so it belongs with the PGD/C&W baselines.  validator_v2 remains the independent
validity authority.

FAB hard-codes the input domain ``[0, 1]^d`` (``clamp(0, 1)`` after every step and
box-aware hyperplane projections).  The attack space here is therefore the
TRAIN-fitted per-feature min-max box of the pristine features (the same box as the
CAPGD comparison): ``u = (raw - train_min) / span``.  Train-constant features
(``span == 0``) are pinned: the wrapper ignores ``u`` for them, so FAB cannot move
them.  Clean rows outside the train box are rejected instead of silently clamped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

AUTOATTACK_COMMIT = "a39220048b3c9f2cca9a4d3a54604793c68eca7e"


def load_fab_class(repo_root: str | Path) -> Any:
    """Return ``FABAttack_PT`` imported from the frozen ``external/auto-attack`` clone."""

    package_root = Path(repo_root).resolve() / "external" / "auto-attack"
    if not (package_root / "autoattack" / "fab_pt.py").exists():
        raise FileNotFoundError(f"missing frozen AutoAttack clone: {package_root}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    import autoattack.fab_pt as fab_pt

    origin = Path(fab_pt.__file__).resolve()
    if package_root not in origin.parents:
        raise ImportError(f"autoattack resolved to {origin}, not the frozen clone {package_root}")
    return fab_pt.FABAttack_PT


@dataclass(frozen=True)
class FABBox:
    """Train-fitted min-max attack box in pristine feature units."""

    train_min: torch.Tensor
    span: torch.Tensor       # train_max - train_min; 0 for train-constant features
    span_safe: torch.Tensor  # span with 0 -> 1 (normalization only)

    @classmethod
    def from_minmax(cls, train_min: np.ndarray, train_max: np.ndarray, device) -> "FABBox":
        lo = torch.as_tensor(train_min, dtype=torch.float32, device=device)
        span = torch.as_tensor(train_max, dtype=torch.float32, device=device) - lo
        if bool((span < 0).any()):
            raise ValueError("train_max < train_min")
        return cls(lo, span, torch.where(span > 0, span, torch.ones_like(span)))

    def to_unit(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.where(self.span > 0, (raw - self.train_min) / self.span_safe,
                           torch.zeros_like(raw))


class UnitBoxVictim(nn.Module):
    """Expose a RobustScaler-space victim on FAB's ``[0, 1]^d`` attack space."""

    def __init__(self, victim: nn.Module, box: FABBox, center: torch.Tensor,
                 scale: torch.Tensor) -> None:
        super().__init__()
        self.victim = victim
        self.box = box
        self.center = center
        self.scale = scale

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        raw = self.box.train_min + u * self.box.span
        return self.victim((raw - self.center) / self.scale)


def run_fab(
    fab_cls: Any,
    victim: nn.Module,
    raw: torch.Tensor,
    labels: torch.Tensor,
    *,
    box: FABBox,
    center: torch.Tensor,
    scale: torch.Tensor,
    norm: str,
    eps: float,
    n_iter: int,
    n_restarts: int,
    seed: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Untargeted FAB on ``raw`` (pristine units); returns adversarial rows in pristine units.

    Rows FAB fails on (or that need a perturbation larger than ``eps`` in the unit
    box) are returned bit-identical to the clean input.
    """

    tol = 1e-6 * box.span_safe
    outside = torch.where(
        box.span > 0,
        (raw < box.train_min - tol) | (raw > box.train_min + box.span + tol),
        raw != box.train_min,  # train-constant features are pinned to their train value
    )
    if bool(outside.any()):
        raise ValueError(f"{int(outside.any(1).sum())} clean rows lie outside the train "
                         "min-max box; FAB would silently clamp them")

    model = UnitBoxVictim(victim, box, center, scale).eval()
    attack = fab_cls(model, norm=norm, n_restarts=n_restarts, n_iter=n_iter, eps=eps,
                     seed=seed, targeted=False, device=device)
    chunks = []
    for start in range(0, raw.shape[0], batch_size):
        r = raw[start:start + batch_size]
        u0 = box.to_unit(r)
        u_adv = attack.perturb(u0, labels[start:start + batch_size])
        # Additive delta keeps untouched features (and failed rows) exactly equal to raw.
        chunks.append((r + (u_adv - u0) * box.span).detach())
    return torch.cat(chunks, dim=0)
