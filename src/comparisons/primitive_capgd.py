"""CAPGD adapted to PrimAttack's exact three normalized primitive controls.

This is an optimizer-control experiment, not native tabular CAPGD. CAPGD searches
``q_padding,q_delay,q_shape in [0,1]`` and maps them into PrimAttack's exact
per-flow control box. The canonical transform remains responsible for generation,
projection, quantization, and final evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from comparisons.capgd_cicids2017 import load_tabularbench_api, numpy_predict_proba


@dataclass(frozen=True)
class PrimitiveCAPGDResult:
    normalized_controls: torch.Tensor
    requested: dict[str, torch.Tensor]
    projected: dict[str, torch.Tensor]
    adversarial_raw: torch.Tensor


class PrimitiveControlVictim(nn.Module):
    """Map normalized CAPGD controls through PrimAttack's differentiable transform."""

    def __init__(
        self,
        primitive_model: Any,
        victim: nn.Module,
        raw: torch.Tensor,
        bounds: dict[str, torch.Tensor],
        capabilities: Any,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.primitive_model = primitive_model
        self.victim = victim
        self.register_buffer("raw", raw)
        self.capabilities = capabilities
        self.register_buffer("p_hi", bounds["p"])
        self.register_buffer("delay_hi", bounds["delay"])
        self.register_buffer("shape_hi", bounds["shape"])
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    def controls(self, normalized: torch.Tensor) -> dict[str, torch.Tensor]:
        if normalized.ndim != 2 or normalized.shape != (self.raw.shape[0], 3):
            raise ValueError(
                f"normalized controls must have shape {(self.raw.shape[0], 3)}, "
                f"got {tuple(normalized.shape)}"
            )
        q = normalized.clamp(0.0, 1.0)
        return {
            "p": self.p_hi * q[:, 0],
            "delay": self.delay_hi * q[:, 1],
            "shape": self.shape_hi * q[:, 2],
        }

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        controls = self.controls(normalized)
        generated = self.primitive_model.generate(
            self.raw, controls, quantize=False, capabilities=self.capabilities
        )
        return self.victim((generated - self.center) / self.scale)


def _unit_scaler(api: Any, device: torch.device) -> Any:
    class DeviceAwareUnitScaler(api.TabScaler):
        def transform(self, x_in, cat_encode_method: str = "elu"):
            out = super().transform(x_in, cat_encode_method)
            if isinstance(out, torch.Tensor):
                out = out.to(device)
            return out

    scaler = DeviceAwareUnitScaler(num_scaler="min_max", one_hot_encode=False)
    scaler.fit_scaler_data(
        api.ScalerData(
            x_min=torch.zeros(3),
            x_max=torch.ones(3),
            categories=[],
            cat_idx=[],
            num_idx=[0, 1, 2],
        )
    )
    return scaler


def run_primitive_capgd(
    *,
    repo_root: str | Path,
    primitive_model: Any,
    victim: nn.Module,
    raw: torch.Tensor,
    bounds: dict[str, torch.Tensor],
    capabilities: Any,
    center: torch.Tensor,
    scale: torch.Tensor,
    true_labels: torch.Tensor,
    seed: int,
    steps: int,
) -> PrimitiveCAPGDResult:
    """Run upstream CAPGD in the full normalized PrimAttack p75 box.

    ``Linf eps=1`` over a clean control vector ``(0,0,0)`` exposes the complete
    per-flow hard box. ``eps_margin=0`` avoids shrinking PrimAttack's bounds.
    """

    if raw.shape[0] != true_labels.shape[0]:
        raise ValueError("raw/label batch mismatch")
    device = raw.device
    api = load_tabularbench_api(repo_root)
    scaler = _unit_scaler(api, device)
    constraints = api.Constraints(
        feature_types=np.asarray(["real", "real", "real"]),
        mutable_features=np.asarray([True, True, True]),
        lower_bounds=np.asarray([0.0, 0.0, 0.0]),
        upper_bounds=np.asarray([1.0, 1.0, 1.0]),
        relation_constraints=None,
        feature_names=np.asarray(["q_padding", "q_delay", "q_shape"]),
    )
    wrapped = PrimitiveControlVictim(
        primitive_model, victim, raw, bounds, capabilities, center, scale
    ).to(device).eval()
    objective = numpy_predict_proba(wrapped, device=device)
    attack = api.CAPGD(
        constraints=constraints,
        scaler=scaler,
        model=wrapped,
        model_objective=objective,
        norm="Linf",
        eps=1.0,
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
        eps_margin=0.0,
        verbose=False,
    )
    attack.set_device(str(device))

    # Drive CAPGD's genuine adaptive core (attack_single_run) directly. Upstream
    # ``forward``/``perturb`` route restart selection through an ObjectiveCalculator
    # that reshapes and CHUNKS rows without indices, which breaks the fixed per-row
    # primitive mapping; the ``best_loss`` path is separately bugged upstream
    # (missing ``n_restart``). Selecting per-sample by best loss across restarts uses
    # the exact same optimizer, momentum, adaptive step schedule, and initialization.
    identity = torch.zeros((raw.shape[0], 3), dtype=raw.dtype, device=device)
    x_scaled = scaler.transform(identity)
    best_loss = torch.full((raw.shape[0],), -float("inf"), device=device)
    best_scaled = x_scaled.clone()
    for restart in range(2):
        _, _, loss_restart, adv_restart = attack.attack_single_run(
            x_scaled, true_labels, restart
        )
        improved = loss_restart > best_loss
        best_loss = torch.where(improved, loss_restart, best_loss)
        best_scaled = torch.where(improved.unsqueeze(1), adv_restart, best_scaled)
    normalized = scaler.inverse_transform(best_scaled).detach().clamp(0.0, 1.0)
    requested = wrapped.controls(normalized)
    projected = primitive_model.project_controls(
        raw, requested, bounds, capabilities=capabilities
    )
    adversarial_raw = primitive_model.generate(
        raw, projected, quantize=True, capabilities=capabilities
    ).detach()
    return PrimitiveCAPGDResult(
        normalized_controls=normalized,
        requested={key: value.detach() for key, value in requested.items()},
        projected={key: value.detach() for key, value in projected.items()},
        adversarial_raw=adversarial_raw,
    )
