"""Dataset-agnostic contracts for realizability-aware primitive attacks.

A concrete dataset (e.g. CICIDS2017-DistriNet) implements :class:`DatasetPrimitiveModel`.
The generic optimizer, projector, validator, and metrics never import a specific dataset;
they consume this interface. Adding another flow dataset (CICIoT2023, which is windowed and
has different feature semantics) means writing one more model -- not editing the attack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

import torch


class FeatureRole(str, Enum):
    """How a feature relates to the attacker-controlled primitives (p, alpha).

    NOTE: the 79 CICFlowMeter features are NEVER optimized directly; only the primitives
    ``p`` (forward packet-length augmentation) and ``alpha`` (forward timing dilation) are.
    These roles therefore describe how each feature is *derived from* / *reacts to* the
    primitives, not that any feature is itself an attack variable. ``tag`` is the short
    label used in the example tables.
    """

    DERIVED_P = "derived_p"          # Dp: exactly derived from the packet-length primitive p
    DERIVED_T = "derived_t"          # Dt: exactly derived from the timing primitive alpha
    DERIVED = "derived"              # D : other exact algebraic derived feature
    CONDITIONAL = "conditional"      # C : conditionally / conservatively reconstructed feature
    RATE = "rate"                    # R : rate-derived feature (count/byte over projected duration)
    INVARIANT = "invariant"          # I : PROVEN invariant under the relevant primitive
    FROZEN = "frozen"                # F : genuinely unaffected by any primitive
    LEVEL_C = "level_c"              # Fᶜ: UNRESOLVED and held constant -- would change under real
    #                                     packet-level reconstruction but is not reconstructable
    #                                     from the aggregate flow (NOT claimed physically invariant)

    @property
    def tag(self) -> str:
        return {
            "derived_p": "Dp", "derived_t": "Dt", "derived": "D", "conditional": "C",
            "rate": "R", "invariant": "I", "frozen": "F", "level_c": "Fᶜ",
        }[self.value]


@dataclass(frozen=True)
class PrimitiveSpec:
    """One attacker-controlled primitive.

    ``identity`` is the value that means "no change" (0 for additive padding, 1 for a
    multiplicative dilation). ``lower``/``upper`` are absolute hard limits; per-flow feasible
    bounds are produced by :meth:`DatasetPrimitiveModel.per_flow_bounds`. ``integer`` marks a
    primitive whose realizable value is discrete (rounded during projection).
    """

    name: str
    identity: float
    lower: float
    upper: float | None
    integer: bool
    description: str
    units: str


@dataclass(frozen=True)
class IdentityCheck:
    """An exact algebraic identity ``target ≈ fn(adv)`` used by the Level-B validator.

    ``fn`` receives a column accessor ``c(name) -> tensor`` and returns the expected value
    of ``target``. It is the single source of truth for both recomputation intent and
    independent validation, but the validator recomputes from the *adversarial* vector so a
    generator bug cannot hide (an identity is only satisfied if the generator actually wrote
    the consistent value).
    """

    target: str
    parents: tuple[str, ...]
    fn: Callable[[Callable[[str], torch.Tensor]], torch.Tensor]
    atol: float = 1e-3
    rtol: float = 1e-4


@runtime_checkable
class DatasetPrimitiveModel(Protocol):
    """Contract implemented once per dataset.

    Controls are passed as a name->tensor mapping so a dataset may expose any number of
    primitives (CICIDS2017 uses two: ``p``, ``alpha``).
    """

    dataset: str

    def primitives(self) -> tuple[PrimitiveSpec, ...]:
        ...

    @property
    def feature_names(self) -> tuple[str, ...]:
        ...

    def roles(self) -> Mapping[str, tuple[FeatureRole, str]]:
        """feature name -> (role, human-readable reason)."""
        ...

    def integer_features(self) -> frozenset[str]:
        """Names whose realizable values are integral (checked by discreteness validity)."""
        ...

    def algebraic_identities(self) -> tuple[IdentityCheck, ...]:
        """Exact identities the adversarial vector must satisfy (Level-B)."""
        ...

    def active_mask(self, raw: torch.Tensor, primitive: str) -> torch.Tensor:
        """Per-flow bool mask: where a primitive is meaningful (else clamped to identity).

        E.g. forward timing dilation is undefined for single-packet flows (``Total Fwd
        Packet < 2``): there is no forward inter-arrival sequence to dilate.
        """
        ...

    def per_flow_bounds(
        self, raw: torch.Tensor, config: "Mapping[str, float]"
    ) -> dict[str, torch.Tensor]:
        """primitive name -> per-flow upper bound (lower is the identity)."""
        ...

    def infer_primitives_from_decoded(
        self,
        raw0: torch.Tensor,
        decoded_adv_raw: torch.Tensor,
        decoded_base_raw: torch.Tensor,
        bounds: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Differentiably collapse decoder movement into feasible primitive controls.

        Dataset-specific feature semantics stay behind this interface. The generic latent
        attack never names CICFlowMeter fields.
        """
        ...

    def generate(
        self,
        raw: torch.Tensor,
        controls: Mapping[str, torch.Tensor],
        *,
        quantize: bool = False,
    ) -> torch.Tensor:
        """Map raw rows + per-flow controls to adversarial raw rows.

        Differentiable in every control when ``quantize=False``. When ``quantize=True`` the
        primitives are first projected to their realizable (discrete) values and every
        dependent feature is recomputed from the projected primitives (no independent
        per-feature clipping).
        """
        ...

    def project_controls(
        self, raw: torch.Tensor, controls: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Project continuous controls to their realizable values (used by ``generate``)."""
        ...


# --------------------------------------------------------------------------------------
# Packet-level (Level-C) verification interface -- section 16.
# The feature-space attack produces primitive controls; a future backend can replay them
# against the original packet trace, re-run CICFlowMeter, and return the re-extracted
# vector. The attack is structured so this plugs in without a rewrite; none of the current
# feature-space experiments assert Level-C realizability.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PacketEditPlan:
    """A dataset-agnostic description of the packet edits implied by primitive controls."""

    flow_id: str
    controls: dict[str, float]
    notes: str = ""


@runtime_checkable
class PacketVerificationBackend(Protocol):
    """Replays primitive controls at the packet level and re-extracts features.

    Concrete implementation (out of scope for the feature-space thesis experiments) would:
    original PCAP + PacketEditPlan -> packet editor -> modified PCAP ->
    DistriNet CICFlowMeter -> re-extracted 79-vector -> classifier.
    """

    def available(self) -> bool:
        ...

    def plan_edits(
        self, model: DatasetPrimitiveModel, controls: Mapping[str, torch.Tensor]
    ) -> list[PacketEditPlan]:
        ...

    def reextract(self, plans: Sequence[PacketEditPlan]) -> torch.Tensor:
        ...


class NullPacketBackend:
    """Default backend: declares Level-C unavailable (feature-space only)."""

    reason: str = (
        "No packet trace / CICFlowMeter re-extraction backend is configured. "
        "Results are feature-space (Level A+B) realizability-aware, not packet-verified."
    )

    def available(self) -> bool:
        return False

    def plan_edits(self, model, controls):  # type: ignore[no-untyped-def]
        return [
            PacketEditPlan(flow_id=str(i), controls={k: float(v[i]) for k, v in controls.items()})
            for i in range(next(iter(controls.values())).shape[0])
        ]

    def reextract(self, plans):  # type: ignore[no-untyped-def]
        raise NotImplementedError(self.reason)
