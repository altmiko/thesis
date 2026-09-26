"""Non-destructive attack-evaluation interface (§20, §21).

A thin, optional adapter so an existing attack path can be evaluated with
validator_v2 WITHOUT modifying any attack code. Give it the raw (pristine-space)
adversarial matrix and the per-sample evasion mask over the eligible
clean-correct malicious set; it returns the explainable validity flags and the
full targeted-ASR suite with one shared denominator.

Example (in an analysis script, not in the attack itself)::

    from validation.attack_interface import evaluate_attack
    out = evaluate_attack(X_adv_raw, evasion, source_raw=X_clean_raw)
    print(out["asr"])            # raw / hard_valid / hybrid_valid / valid_in_distribution
    print(out["rates"])          # per-layer validity rates over the adversarial set

Adversarial flows are always validated together with their unperturbed source flows
(``source_raw``, same row order): transition rules such as "an empty forward packet stays
empty" compare the two.
"""
from __future__ import annotations

import numpy as np

from validation import load_validator
from validation.metrics import targeted_asr_suite


def evaluate_attack(X_adv_raw: np.ndarray, evasion: np.ndarray, *, source_raw: np.ndarray,
                    dataset: str = "cicids2017_distrinet", eligible: int | None = None) -> dict:
    """Gate an attack's raw adversarial vectors through validator_v2.

    ``X_adv_raw``: (N, F) raw/pristine-space adversarial feature matrix (same
    feature order as the schema profile); ``source_raw``: the (N, F) source flows they were
    derived from. ``evasion``: (N,) bool, target->Benign
    success per eligible sample. Returns validity flags, per-layer rates, and the
    targeted-ASR suite (raw / hard-valid / hybrid-valid / valid+in-distribution).
    """
    v = load_validator(dataset)
    b = v.validate_batch(X_adv_raw, source_raw)
    asr = targeted_asr_suite(
        evasion=np.asarray(evasion, bool),
        hard_valid=b.hard_structural_valid,
        hybrid_valid=b.hybrid_valid,
        in_distribution=b.in_distribution,
        eligible=eligible,
    )
    return {
        "n": int(b.n),
        "rates": b.rates(),
        "flags": {
            "hard_structural_valid": b.hard_structural_valid,
            "hybrid_valid": b.hybrid_valid,
            "in_distribution": b.in_distribution,
        },
        "asr": asr.to_dict(),
        "rules_rejecting": b.rules_rejecting_any(),
    }


_VALIDATOR_CACHE: dict = {}


def get_validator(dataset: str = "cicids2017_distrinet"):
    """Process-cached validator_v2 instance (rules/profiles loaded once)."""
    if dataset not in _VALIDATOR_CACHE:
        _VALIDATOR_CACHE[dataset] = load_validator(dataset)
    return _VALIDATOR_CACHE[dataset]


def structural_masks(X_adv_raw: np.ndarray, dataset: str = "cicids2017_distrinet", *,
                     source_raw: np.ndarray) -> dict:
    """Per-sample validator_v2 masks for an attack path (numpy bool arrays).

    Drop-in replacement for the legacy ConstraintEngine validity leg
    (`engine.validate(x)["pass_l0_l1_l2"]`): use ``hybrid_valid`` as the structural
    validity mask. ``source_raw`` is the unperturbed source flow of each row (pass the flows
    themselves to validate unperturbed flows). Returns every layer plus plausibility so
    runners can record them. This reads ONLY the ``validation/`` profiles -- no dependency on
    the old root ``constraints/`` artifact.
    """
    v = get_validator(dataset)
    b = v.validate_batch(np.asarray(X_adv_raw), np.asarray(source_raw))
    return {
        "schema_valid": b.schema_valid,
        "extractor_valid": b.extractor_valid,
        "protocol_valid": b.protocol_valid,
        "mined_valid": b.mined_valid,
        "hard_structural_valid": b.hard_structural_valid,
        "hybrid_valid": b.hybrid_valid,
        "in_distribution": b.in_distribution,
    }
