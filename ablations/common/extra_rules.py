"""Candidate validator rules that are not (yet) part of validator_v2.

validator_v2 is the locked FINAL-suite validator and is not modified. Rules here are evaluated
next to it: an ablation ``Condition`` toggles a rule on by listing it in ``extra_rules``, which
ANDs it into the search success predicate and into the condition's validity. The runner stores
every registered rule's per-row verdict for every condition, so ``extended_valid_success``
(validator_v2 AND all rules below) is available for rule-on and rule-off arms alike.

``single_fwd_packet_no_fwd_iat`` (definitional, CICFlowMeter):
a flow with at most one forward packet has no forward inter-arrival gap, so
``Total Fwd Packet <= 1  =>  Fwd IAT Total = Fwd IAT Mean = Fwd IAT Std = Fwd IAT Max =
Fwd IAT Min = 0``. It closes the gap found by ablation D6 (forward delay added to single
forward-packet flows passes all four validator_v2 layers). The packet counts are immutable
under PrimAttack, so the rule depends on the adversarial flow alone.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

FWD_IAT_COLUMNS = ("Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min")


def single_fwd_packet_no_fwd_iat(x: np.ndarray, index: Mapping[str, int]) -> np.ndarray:
    """True where the rule holds (row valid)."""
    nf = x[:, index["Total Fwd Packet"]]
    iat = x[:, [index[c] for c in FWD_IAT_COLUMNS]]
    return (nf > 1) | np.all(iat == 0, axis=1)


EXTRA_RULES: dict[str, Callable[[np.ndarray, Mapping[str, int]], np.ndarray]] = {
    "single_fwd_packet_no_fwd_iat": single_fwd_packet_no_fwd_iat,
}


def extra_rule_masks(x: np.ndarray, index: Mapping[str, int]) -> dict[str, np.ndarray]:
    return {name: np.asarray(fn(x, index), dtype=bool) for name, fn in EXTRA_RULES.items()}
