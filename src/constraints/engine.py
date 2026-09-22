"""ConstraintEngine — composes Layer 0 / 1 / 2 with per-run on/off toggles.

Which layers are *active* is a runtime knob (``active_layers``), so one built engine
can be reconfigured per experiment: ``{0}``, ``{0,1}``, ``{0,1,2}``, ``{0,2}``, etc.
The active set is honored consistently in:

* ``project``  — Layer-0 hard projection is applied only if 0 is active;
* ``penalty``  — C1 counts only if 1 active, C2 only if 2 active;
* ``validate`` — an inactive layer contributes an all-pass mask, so the hierarchical
  rates (rate_l0 / rate_l0_l1 / rate_l0_l1_l2) reflect exactly the active layers.

Generation constraints and evaluation validators remain independent (validate does
its own checks; it does not trust the projector).
"""
from __future__ import annotations

from typing import Iterable

import torch

from constraints.base import Constraint
from constraints.layer0 import Layer0Projector
from datasets.feature_manifest import FeatureManifest

_ALL_LAYERS = (0, 1, 2)


def parse_layers(spec: "str | Iterable[int] | None", default: Iterable[int]) -> set[int]:
    """Parse a layer selector: None -> default; "012"/"0,1,2"/[0,1,2] -> {0,1,2}."""
    if spec is None:
        return set(default)
    if isinstance(spec, str):
        digits = spec.replace(",", " ").split() if ("," in spec or " " in spec) else list(spec)
        vals = {int(d) for d in digits if d.strip() != ""}
    else:
        vals = {int(v) for v in spec}
    bad = vals - set(_ALL_LAYERS)
    if bad:
        raise ValueError(f"unknown constraint layers {sorted(bad)}; allowed {list(_ALL_LAYERS)}")
    return vals


class ConstraintEngine:
    def __init__(
        self,
        manifest: FeatureManifest,
        *,
        layer0: Layer0Projector | None = None,
        layer1: list[Constraint] | None = None,
        layer2: list[Constraint] | None = None,
        active_layers: "str | Iterable[int] | None" = None,
    ) -> None:
        self.manifest = manifest
        self.layer0 = layer0 if layer0 is not None else Layer0Projector(manifest)
        self.layer1 = list(layer1 or [])
        self.layer2 = list(layer2 or [])
        # layers that actually carry something (0 is always available as a projector)
        self._present = {0} | ({1} if self.layer1 else set()) | ({2} if self.layer2 else set())
        self.set_active_layers(active_layers)

    # -- toggles ----------------------------------------------------------------
    def set_active_layers(self, spec: "str | Iterable[int] | None") -> "ConstraintEngine":
        """Select which layers are active this run. None -> all present layers."""
        self._active = parse_layers(spec, self._present)
        return self

    @property
    def active_layers(self) -> set[int]:
        return set(self._active)

    def layer_active(self, layer: int) -> bool:
        return layer in self._active

    # -- generation -------------------------------------------------------------
    def project(self, x_raw, x_source=None, mutable_mask=None) -> torch.Tensor:
        if 0 not in self._active:
            return x_raw
        return self.layer0.project(x_raw, x_source=x_source, mutable_mask=mutable_mask)

    def penalty(
        self,
        x_raw: torch.Tensor,
        ctx: dict | None = None,
        *,
        w1: float = 1.0,
        w2: float = 1.0,
    ) -> dict:
        zero = x_raw.new_tensor(0.0)
        use1 = 1 in self._active
        use2 = 2 in self._active
        c1 = sum((c.penalty(x_raw, ctx) for c in self.layer1), zero) if use1 else zero
        c2 = sum((c.penalty(x_raw, ctx) for c in self.layer2), zero) if use2 else zero
        per = {}
        if use1:
            per.update({c.name: c.penalty(x_raw, ctx) for c in self.layer1})
        if use2:
            per.update({c.name: c.penalty(x_raw, ctx) for c in self.layer2})
        return {"c1": c1, "c2": c2, "total": w1 * c1 + w2 * c2, "per_constraint": per}

    # -- evaluation (independent validators) ------------------------------------
    def validate(self, x_raw: torch.Tensor, ctx: dict | None = None) -> dict:
        n = x_raw.shape[0]
        device = x_raw.device
        all_true = torch.ones(n, dtype=torch.bool, device=device)

        l0 = self.layer0.validate(x_raw) if 0 in self._active else all_true

        def _combine(constraints, active):
            mask = torch.ones(n, dtype=torch.bool, device=device)
            per = {}
            if active:
                for c in constraints:
                    m = c.validate(x_raw, ctx)
                    per[c.name] = m
                    mask = mask & m
            return mask, per

        l1_mask, l1_per = _combine(self.layer1, 1 in self._active)
        l2_mask, l2_per = _combine(self.layer2, 2 in self._active)

        pass_l0 = l0
        pass_l0_l1 = l0 & l1_mask
        pass_l0_l1_l2 = pass_l0_l1 & l2_mask

        def rate(m):
            return float(m.float().mean().item()) if m.numel() else 1.0

        return {
            "active_layers": sorted(self._active),
            "layer0": l0,
            "layer1": l1_mask,
            "layer2": l2_mask,
            "pass_l0": pass_l0,
            "pass_l0_l1": pass_l0_l1,
            "pass_l0_l1_l2": pass_l0_l1_l2,
            "rate_l0": rate(pass_l0),
            "rate_l0_l1": rate(pass_l0_l1),
            "rate_l0_l1_l2": rate(pass_l0_l1_l2),
            "per_constraint": {**l1_per, **l2_per},
        }


def build_engine(
    manifest: FeatureManifest,
    layers: "str | Iterable[int] | None" = "012",
    *,
    layer1_fit_x_raw=None,
    layer1_features: list[str] | None = None,
    layer1_tau: float = 6.0,
    layer2_source=None,
) -> ConstraintEngine:
    """Build an engine with ONLY the requested layers, and mark them active.

    ``layers`` accepts "0", "01", "012", "0,2", or an iterable like {0, 1}.
    Layer 1 requires ``layer1_fit_x_raw`` (TRAIN raw features); Layer 2 requires
    ``layer2_source`` (path / dict / list of rule configs).
    """
    from constraints.layer1 import RobustTailBound
    from constraints.layer2 import load_layer2

    want = parse_layers(layers, _ALL_LAYERS)
    l1: list[Constraint] = []
    l2: list[Constraint] = []
    if 1 in want:
        if layer1_fit_x_raw is None:
            raise ValueError("layer 1 requested but layer1_fit_x_raw (TRAIN) not provided")
        l1 = [RobustTailBound.fit(manifest, layer1_fit_x_raw, feature_names=layer1_features, tau=layer1_tau)]
    if 2 in want:
        if layer2_source is None:
            raise ValueError("layer 2 requested but layer2_source not provided")
        l2 = load_layer2(layer2_source, manifest)
    return ConstraintEngine(
        manifest, layer0=Layer0Projector(manifest), layer1=l1, layer2=l2, active_layers=want
    )
