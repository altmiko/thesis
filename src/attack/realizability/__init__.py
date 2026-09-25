"""Realizability-aware primitive-control attack framework.

Dataset-agnostic core:

* :mod:`attack.realizability.base` -- feature roles, primitive specs, the
  :class:`DatasetPrimitiveModel` protocol every dataset adapter implements, and the
  :class:`PacketVerificationBackend` interface a future PCAP re-extraction backend plugs
  into (Level-C verification).
* :mod:`attack.realizability.validator` -- dataset-agnostic categorized validator that
  separates Level-A (feature domain), Level-B (algebraic/dependency), and discreteness
  from the external mined/PAVE validators.
* :mod:`attack.realizability.cicids2017` -- the concrete CICIDS2017-DistriNet realization
  of the primitive model (forward packet-length augmentation ``p`` + forward timing
  delay allocation ``(delay, shape)``), with the empirically-mined dependency graph.

Only true attacker-controlled primitives are optimized; every aggregate/derived feature
that a primitive affects is deterministically recomputed. See ``docs`` and the module
docstrings for the full dependency map and the Level-A/B/C distinction.
"""
from __future__ import annotations

from attack.realizability.base import (
    DatasetPrimitiveModel,
    FeatureRole,
    IdentityCheck,
    NullPacketBackend,
    PacketVerificationBackend,
    PrimitiveSpec,
)
from attack.realizability.validator import (
    RealizabilityReport,
    RealizabilityValidator,
)

__all__ = [
    "DatasetPrimitiveModel",
    "FeatureRole",
    "IdentityCheck",
    "NullPacketBackend",
    "PacketVerificationBackend",
    "PrimitiveSpec",
    "RealizabilityReport",
    "RealizabilityValidator",
]
