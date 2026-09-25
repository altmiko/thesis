"""CSE-CIC-IDS-2018 (DistriNet) dataset adapter.

Processed artifacts live under ``data/processed/CSECICIDS_2018_Distrinet/`` and follow the
CICIDS2017 layout exactly (``X_{split}.npy`` RobustScaler-scaled, ``X_{split}_pristine.npy``,
``y_{split}_{cat,bin}.npy``, ``label_encoders.json``, ``scaler.pkl``,
``preprocessing_manifest.json``), written by
``src/preprocessing/preprocess_cicids2018_distrinet.py``. The 79 modelling features and their
order are the CICIDS2017 ones, so this adapter reuses the CICIDS2017 extractor typing with two
dataset-specific differences:

* ``Fwd/Bwd Header Length`` are kept as the extractor emitted them: a signed 16-bit field
  that wraps on long flows (observed range [-32768, 32767]). Their semantic bounds are the
  int16 range, not ``[0, inf)``; a ``[0, inf)`` bound would make Layer 0 clamp genuine
  (wrapped) training values.
* Only CICFlowMeter identities with zero violations on the CICIDS2018 TRAIN split are
  declared as derived (see ``_DERIVATIONS`` below).
"""
from __future__ import annotations

from datasets.cicids2017 import _DERIVATIONS as _CICIDS2017_DERIVATIONS
from datasets.cicids2017 import _TYPING as _CICIDS2017_TYPING
from datasets.cicids2017 import CICIDS2017Adapter

_INT16_MIN = -32768.0
_INT16_MAX = 32767.0

_TYPING = {
    **_CICIDS2017_TYPING,
    "Fwd Header Length": ("integer_count", _INT16_MIN, _INT16_MAX, "header_int16_wrapped"),
    "Bwd Header Length": ("integer_count", _INT16_MIN, _INT16_MAX, "header_int16_wrapped"),
}

# The six CICIDS2017 identities, re-verified on CICIDS2018: zero violations
# (|target - f(parents)| > 1e-3 + 1e-4 * |target|) on all 583,487 TRAIN rows and all
# 125,033 validation rows of the Benign=250k/DoS=200k/DDoS=200k run.
_DERIVATIONS = dict(_CICIDS2017_DERIVATIONS)


class CICIDS2018Adapter(CICIDS2017Adapter):
    name = "cicids2018_distrinet"
    processed_dirname = "CSECICIDS_2018_Distrinet"
    typing = _TYPING
    derivations = _DERIVATIONS
