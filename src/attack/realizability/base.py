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
    """How a feature relates to the attacker-controlled primitives (p, delay, shape).

    NOTE: the 79 CICFlowMeter features are NEVER optimized directly; only the primitives
    ``p`` (forward packet-length augmentation) and ``delay``/``shape`` (forward timing
    delay allocation) are.
    These roles therefore describe how each feature is *derived from* / *reacts to* the
    primitives, not that any feature is itself an attack variable. ``tag`` is the short
    label used in the example tables.
    """

    DERIVED_P = "derived_p"          # Dp: exactly derived from the packet-length primitive p
    DERIVED_T = "derived_t"          # Dt: exactly derived from the timing primitive (delay, shape)
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
    """Frozen contract for one attacker-controlled flow primitive.

    ``identity`` is the no-op value. Absolute bounds encode the threat-model direction;
    train-calibrated and per-flow bounds may only narrow them. ``dependencies`` lists every
    CICFlowMeter feature the canonical transform is allowed to write for this primitive.
    ``projection_function`` names the final discrete/continuous projection implemented by the
    dataset model. It is provenance, not an executable callback.
    """

    name: str
    units: str
    dtype: str
    direction: str
    identity: float
    absolute_lower_bound: float
    absolute_upper_bound: float | None
    dependencies: tuple[str, ...]
    projection_function: str
    semantic_risk: str
    description: str

    def __post_init__(self) -> None:
        if self.dtype not in {"continuous", "discrete_integer"}:
            raise ValueError(f"unsupported primitive dtype {self.dtype!r}")
        if self.direction not in {"increase_only", "decrease_only", "bidirectional"}:
            raise ValueError(f"unsupported primitive direction {self.direction!r}")
        if self.identity < self.absolute_lower_bound:
            raise ValueError("primitive identity is below its absolute lower bound")
        if (
            self.absolute_upper_bound is not None
            and self.identity > self.absolute_upper_bound
        ):
            raise ValueError("primitive identity is above its absolute upper bound")
        if not self.dependencies:
            raise ValueError("primitive dependency set must not be empty")


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


# --------------------------------------------------------------------------------------
# Semantic primitive capabilities -- section: "infer semantic capabilities before bounds".
# A primitive can be *numerically* feasible (inside the train envelope) yet *semantically*
# unsupported by the source flow: e.g. forward-length augmentation on a flow whose forward
# direction carries no payload (Total Length of Fwd Packet == 0), or on a flow that contains a
# zero-length forward packet (Fwd Packet Length Min == 0; padding adds p bytes to EVERY forward
# packet, so it would put bytes into an empty packet). Capability inference is a conservative,
# per-flow gate applied to the bounds BEFORE the optimizer ever sees them, so admissibility
# means "numerically feasible AND the operation is semantically supported by the source flow",
# not merely "inside the envelope".
# --------------------------------------------------------------------------------------
# Padding (p) capability reason codes.
PAD_ALLOWED = "PAD_ALLOWED"
NO_FORWARD_PAYLOAD = "NO_FORWARD_PAYLOAD"          # no forward bytes/mean to augment
INSUFFICIENT_FWD_PACKETS = "INSUFFICIENT_FWD_PACKETS"  # too few forward packets for evidence
EMPTY_FWD_PACKET = "EMPTY_FWD_PACKET"              # >= 1 zero-length fwd packet (fwd min == 0)
# Timing (delay, shape) capability reason codes.
TIMING_ALLOWED = "TIMING_ALLOWED"
SINGLE_FWD_PACKET = "SINGLE_FWD_PACKET"            # < 2 forward packets: no fwd IAT sequence
ZERO_TIMING_HEADROOM = "ZERO_TIMING_HEADROOM"      # forward IAT total is 0: nothing to delay


@dataclass(frozen=True)
class PrimitiveCapabilities:
    """Per-flow semantic admissibility of each primitive for a batch of source flows.

    ``pad_allowed`` / ``timing_allowed`` are boolean tensors (shape ``[n]``); a ``False`` entry
    means the source flow does not provide evidence that the primitive is realizable, so its
    per-flow cap is forced to the identity (``p_hi = 0`` / ``delay_hi = 0``). ``pad_reason`` /
    ``timing_reason`` carry one machine-readable reason code per flow (see the ``*_ALLOWED`` /
    disable constants above) for auditable artifacts.
    """

    pad_allowed: torch.Tensor
    timing_allowed: torch.Tensor
    pad_reason: list[str]
    timing_reason: list[str]


@runtime_checkable
class DatasetPrimitiveModel(Protocol):
    """Contract implemented once per dataset.

    Controls are passed as a name->tensor mapping so a dataset may expose any number of
    primitives (CICIDS2017 uses padding plus total-delay and delay-shape controls).
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

    def infer_capabilities(self, raw: torch.Tensor) -> PrimitiveCapabilities:
        """Per-flow semantic admissibility of each primitive for the source flows.

        Conservative gate applied to the bounds BEFORE optimization: a primitive is admissible
        only if the source flow provides evidence the operation is realizable (e.g. forward
        payload present and no zero-length forward packet for padding, a forward IAT sequence
        for timing dilation).
        """
        ...

    def active_mask(
        self, raw: torch.Tensor, primitive: str,
        capabilities: "PrimitiveCapabilities | None" = None,
    ) -> torch.Tensor:
        """Per-flow bool mask: where a primitive is semantically admissible (else identity).

        E.g. forward timing dilation is undefined for single-packet flows (``Total Fwd
        Packet < 2``); forward-length augmentation is unsupported when the source carries no
        forward payload or contains a zero-length forward packet. ``capabilities`` (from
        :meth:`infer_capabilities`) may be passed to
        avoid recomputation; when omitted it is inferred from ``raw``.
        """
        ...

    def per_flow_bounds(
        self, raw: torch.Tensor, config: "Mapping[str, float]",
        capabilities: "PrimitiveCapabilities | None" = None,
    ) -> dict[str, torch.Tensor]:
        """Control name -> per-flow upper bound (lower is the identity).

        Semantic capabilities gate the numeric train-envelope caps: an inadmissible
        primitive gets its cap forced to the identity. Pre-gate numeric caps may be
        returned separately for provenance.
        """
        ...

    def infer_primitives_from_decoded(
        self,
        raw0: torch.Tensor,
        decoded_adv_raw: torch.Tensor,
        decoded_base_raw: torch.Tensor,
        bounds: Mapping[str, torch.Tensor],
        capabilities: "PrimitiveCapabilities | None" = None,
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
        capabilities: "PrimitiveCapabilities | None" = None,
    ) -> torch.Tensor:
        """Map raw rows + per-flow controls to adversarial raw rows.

        Differentiable in every control when ``quantize=False``. When ``quantize=True`` the
        dependent integer-valued features are quantized after the controls have been projected.
        ``capabilities`` avoids recomputation and device synchronization in attack loops.
        """
        ...

    def project_controls(
        self,
        raw: torch.Tensor,
        controls: Mapping[str, torch.Tensor],
        bounds: Mapping[str, torch.Tensor],
        *,
        capabilities: "PrimitiveCapabilities | None" = None,
    ) -> dict[str, torch.Tensor]:
        """Project controls to the declared per-flow budget and discrete feasible set."""
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
