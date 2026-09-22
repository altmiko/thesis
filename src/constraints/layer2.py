"""Layer 2 — optional dataset-specific / mined constraints.

Layer 2 rules are data/extractor-specific and are supplied as a serializable rule
set (JSON/dict), not baked into the VAE. Each entry is ``{"type", "name", "params"}``
and is rebuilt via the registry. An automatic constraint-mining script can later emit
this exact format (e.g. ``constraints/ciciot2023/mined.json``) with no code changes.
"""
from __future__ import annotations

import json
from pathlib import Path

from constraints.base import Constraint
from constraints.registry import build_constraint
from datasets.feature_manifest import FeatureManifest


def load_layer2(source: str | Path | dict | list, manifest: FeatureManifest) -> list[Constraint]:
    """Build Layer-2 constraints from a path, a dict, or a list of rule configs."""
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = source
    if isinstance(payload, dict):
        # allow {"schema_version":..., "dataset":..., "constraints":[...]}
        rules = payload.get("constraints", [])
        declared = payload.get("dataset")
        if declared is not None and declared != manifest.dataset_name:
            raise ValueError(
                f"Layer-2 rule set declares dataset {declared!r} but manifest is "
                f"{manifest.dataset_name!r}"
            )
    else:
        rules = payload
    constraints: list[Constraint] = []
    for cfg in rules:
        c = build_constraint(manifest, cfg)
        c.layer = 2  # tag as dataset-specific regardless of underlying class
        constraints.append(c)
    return constraints


def dump_layer2(constraints: list[Constraint], dataset_name: str) -> dict:
    return {
        "schema_version": "1.0",
        "dataset": dataset_name,
        "constraints": [c.to_config() for c in constraints],
    }
