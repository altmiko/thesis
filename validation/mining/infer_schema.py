"""Automatic SCHEMA inference (the unary/type stage).

For every feature, classify its representation type on the TRAIN split
(finite / integer / binary / constant / categorical / numeric), confirm the
fact holds on VAL, and merge the inferred facts with the authored feature
registry (category / description / units). The result is the schema profile
serialized to ``validation/schema/<dataset>.yaml`` and consumed by the engine.

Observed train min/max are recorded as *evidence only* -- never converted into
hard validity bounds (methodology §9/§18).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from validation.mining.feature_registry import registry_for
from validation.mining.mine_unary import classify_feature


def infer_schema(X_train: np.ndarray, X_val: np.ndarray, feature_order: list[str],
                 dataset: str) -> dict:
    Xtr = np.asarray(X_train, np.float64)
    Xva = np.asarray(X_val, np.float64)
    features: dict[str, dict] = {}
    for i, name in enumerate(feature_order):
        facts = classify_feature(Xtr[:, i])
        reg = registry_for(name)
        # confirm the inferred type fact holds on val
        val_facts = classify_feature(Xva[:, i])
        # Guard the observed-value-set trap (methodology §5/§18): a small observed
        # set of integers is only treated as a hard categorical DOMAIN for genuine
        # code/enumeration features (flow identity, e.g. Protocol). For count-like
        # features that merely happen to show few values in train (e.g. CWR Flag
        # Count in {0,1,2}), we keep them as `integer` so a rarer count on
        # held-out data is not falsely rejected.
        if facts["value_type"] == "categorical" and reg.get("category") != "flow_identity":
            facts = dict(facts); facts["value_type"] = "integer"; facts["domain"] = None
        confirmed = val_facts["value_type"] == facts["value_type"]
        entry = {
            "index": i,
            "category": reg.get("category"),
            "units": reg.get("units"),
            "derived": reg.get("derived"),
            "uncertain": bool(reg.get("uncertain", False)),
            "description": reg.get("description", ""),
            "value_type": facts["value_type"],
            "finite": facts["finite"],
            "constant_value": facts["constant_value"],
            "domain": facts["domain"],
            "nonnegative": facts["nonnegative"],
            "inferred_on": "X_train_pristine",
            "evidence": {
                "train": facts["evidence"],
                "val_value_type": val_facts["value_type"],
                "val_type_confirmed": confirmed,
            },
        }
        features[name] = entry
    return {"dataset": dataset, "n_features": len(feature_order),
            "feature_order": list(feature_order), "features": features}


def write_schema_yaml(profile: dict, path: str | Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(profile, sort_keys=False, default_flow_style=False))
