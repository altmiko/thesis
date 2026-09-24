"""Public-method adversarial comparisons for the CICIDS2017-DistriNet study."""

from comparisons.capgd_cicids2017 import (
    CAPGD_METHOD_ID,
    RawCICIDSVictim,
    build_capgd_resources,
    load_tabularbench_api,
)

__all__ = [
    "CAPGD_METHOD_ID",
    "RawCICIDSVictim",
    "build_capgd_resources",
    "load_tabularbench_api",
]
