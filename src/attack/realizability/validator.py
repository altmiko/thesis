"""Dataset-agnostic categorized realizability validator.

Separates the *internal* realizability categories so a failing sample reports WHY:

* ``algebraic_dependency`` -- every exact identity the model declares (Level B). Recomputed
  from the *adversarial* vector, so a generator that forgets to update a dependent feature
  is caught here rather than silently passing.
* ``packet_summary``       -- packet-length ordering/consistency (min<=mean<=max, max<=total,
  std>=0, non-negativity).
* ``timing``               -- IAT ordering (min<=mean<=max<=total), directional total<=
  duration, flow-IAT<=duration, duration>0.
* ``negative_rate``        -- every rate >= 0.
* ``discreteness``         -- integer-valued features are integral (after projection).
* ``frozen``               -- features the model marks frozen exactly match the pristine row.

These are deliberately independent of the external PAVE (Level-A feature-domain) and mined
(dataset-density) validators, so 100% realizability does not imply the external validators
were embedded into the generator (thesis separation requirement).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from attack.realizability.base import DatasetPrimitiveModel, FeatureRole


@dataclass(frozen=True)
class RealizabilityReport:
    """Per-sample FAIL masks (True == violation) by category, plus the union."""

    categories: dict[str, torch.Tensor]
    frozen_atol: float

    @property
    def any_fail(self) -> torch.Tensor:
        masks = list(self.categories.values())
        out = torch.zeros_like(masks[0])
        for m in masks:
            out = out | m
        return out

    @property
    def valid(self) -> torch.Tensor:
        return ~self.any_fail

    def counts(self, row_mask: torch.Tensor | None = None) -> dict[str, int]:
        sel = (lambda m: m if row_mask is None else (m & row_mask))
        out = {k: int(sel(v).sum().item()) for k, v in self.categories.items()}
        out["any"] = int(sel(self.any_fail).sum().item())
        return out


class RealizabilityValidator:
    """Checks the internal realizability categories for a :class:`DatasetPrimitiveModel`."""

    def __init__(self, model: DatasetPrimitiveModel, *, atol: float = 1e-3, rtol: float = 1e-4,
                 frozen_atol: float = 1e-6, int_atol: float = 1e-3) -> None:
        self.model = model
        self.atol = atol
        self.rtol = rtol
        self.frozen_atol = frozen_atol
        self.int_atol = int_atol
        self.i = {n: j for j, n in enumerate(model.feature_names)}
        roles = model.roles()
        # Preserved-exactly set = every feature the transform does not write:
        # genuinely frozen (F), proven-invariant (I), and Level-C held-constant (Fᶜ).
        self.frozen_names = tuple(
            n for n, (r, _) in roles.items()
            if r in (FeatureRole.FROZEN, FeatureRole.INVARIANT, FeatureRole.LEVEL_C)
        )

    def validate(self, adv: torch.Tensor, raw: torch.Tensor) -> RealizabilityReport:
        i = self.i
        c = lambda n: adv[:, i[n]]
        n_rows = adv.shape[0]
        dev = adv.device
        atol, rtol = self.atol, self.rtol

        def le(a, b):
            return a <= b + (atol + rtol * b.abs())

        ok = lambda: torch.ones(n_rows, dtype=torch.bool, device=dev)

        # --- algebraic dependency (Level B): recompute identities from adv ---
        dep = ok()
        for chk in self.model.algebraic_identities():
            expected = chk.fn(c)
            dep &= (c(chk.target) - expected).abs() <= (chk.atol + chk.rtol * expected.abs())

        # --- packet-summary realizability ---
        pk = ok()
        pk &= c("Fwd Packet Length Min") >= -atol
        pk &= le(c("Fwd Packet Length Min"), c("Fwd Packet Length Mean"))
        pk &= le(c("Fwd Packet Length Mean"), c("Fwd Packet Length Max"))
        pk &= c("Fwd Packet Length Std") >= -atol
        pk &= le(c("Fwd Packet Length Max"), c("Total Length of Fwd Packet"))
        pk &= c("Packet Length Min") >= -atol
        pk &= le(c("Packet Length Min"), c("Packet Length Mean"))
        pk &= le(c("Packet Length Mean"), c("Packet Length Max"))
        pk &= c("Packet Length Std") >= -atol
        pk &= c("Packet Length Variance") >= -atol

        # --- timing realizability ---
        tm = ok()
        tm &= c("Fwd IAT Min") >= -atol
        tm &= c("Fwd IAT Std") >= -atol
        tm &= le(c("Fwd IAT Min"), c("Fwd IAT Mean"))
        tm &= le(c("Fwd IAT Mean"), c("Fwd IAT Max"))
        tm &= le(c("Fwd IAT Max"), c("Fwd IAT Total"))
        tm &= le(c("Flow IAT Min"), c("Flow IAT Mean"))
        tm &= le(c("Flow IAT Mean"), c("Flow IAT Max"))
        tm &= c("Flow Duration") > 0
        tm &= le(c("Fwd IAT Total"), c("Flow Duration"))
        tm &= le(c("Bwd IAT Total"), c("Flow Duration"))
        tm &= le(c("Flow IAT Max"), c("Flow Duration"))

        # --- negative rate ---
        rt = ok()
        for r in ("Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"):
            rt &= c(r) >= -atol

        # --- discreteness (integer-valued features integral) ---
        disc = ok()
        for name in self.model.integer_features():
            v = c(name)
            disc &= (v - torch.round(v)).abs() <= self.int_atol

        # --- frozen preservation (exact vs pristine raw) ---
        fz = ok()
        if self.frozen_names:
            fidx = torch.tensor([i[n] for n in self.frozen_names], dtype=torch.long, device=dev)
            fz = (adv[:, fidx] - raw[:, fidx]).abs().le(
                self.frozen_atol + rtol * raw[:, fidx].abs()
            ).all(dim=1)

        return RealizabilityReport(
            categories={
                "algebraic_dependency_fail": ~dep,
                "packet_summary_fail": ~pk,
                "timing_fail": ~tm,
                "negative_rate_fail": ~rt,
                "discreteness_fail": ~disc,
                "frozen_fail": ~fz,
            },
            frozen_atol=self.frozen_atol,
        )
