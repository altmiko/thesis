"""The dataset-agnostic validation engine.

``Validator`` loads a set of :class:`Rule` objects from per-dataset profiles
(SCHEMA from the schema YAML, MINED from ``mined_rules.json``, EXTRACTOR and
PROTOCOL from YAML) plus an optional :class:`PlausibilityProfile`, and validates
raw feature vectors. The engine itself contains NO dataset knowledge -- swapping
datasets means swapping profiles, not code (methodology §22).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from validation.validator.plausibility import PlausibilityProfile
from validation.validator.result import BatchResult
from validation.validator.rule import Rule
from validation.validator.tolerance import Tolerance

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PACKAGE_ROOT / "schema"
RULES_DIR = PACKAGE_ROOT / "rules"


def _load_yaml(path: Path) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML profiles")
    return yaml.safe_load(path.read_text())


# ---- SCHEMA rule synthesis from the schema profile ------------------------
def schema_rules_from_profile(profile: dict) -> list[Rule]:
    """Turn inferred per-feature type facts (schema YAML) into SCHEMA rules."""
    rules: list[Rule] = []
    n = 0
    for name in profile["feature_order"]:
        spec = profile["features"][name]
        vt = spec.get("value_type", "numeric")
        base_prov = {"origin": "inferred from train split", "automatically_mined": True,
                     "inferred_on": spec.get("inferred_on", "X_train_pristine")}
        # finite always
        n += 1
        rules.append(Rule(id=f"SCH_{n:04d}", name=f"finite::{name}", source_type="SCHEMA",
                          rule_type="finite", params={"feature": name}, features=[name],
                          hardness="HARD", description=f"{name} must be a finite real number.",
                          provenance=dict(base_prov), evidence=spec.get("evidence", {})))
        if vt == "constant":
            n += 1
            rules.append(Rule(id=f"SCH_{n:04d}", name=f"constant::{name}", source_type="SCHEMA",
                              rule_type="constant", params={"feature": name, "value": spec["constant_value"]},
                              features=[name], hardness="HARD",
                              tolerance=Tolerance.from_dict(spec.get("tolerance")),
                              description=f"{name} is constant at {spec['constant_value']} across the whole train split.",
                              provenance=dict(base_prov), evidence=spec.get("evidence", {})))
        elif vt == "binary":
            n += 1
            rules.append(Rule(id=f"SCH_{n:04d}", name=f"binary::{name}", source_type="SCHEMA",
                              rule_type="binary", params={"feature": name}, features=[name],
                              hardness="HARD", description=f"{name} takes only the values 0 or 1.",
                              provenance=dict(base_prov), evidence=spec.get("evidence", {})))
        elif vt == "categorical":
            n += 1
            rules.append(Rule(id=f"SCH_{n:04d}", name=f"categorical::{name}", source_type="SCHEMA",
                              rule_type="categorical", params={"feature": name, "domain": spec["domain"]},
                              features=[name], hardness="HARD",
                              description=f"{name} takes one of the observed categorical codes {sorted(spec['domain'])}.",
                              provenance=dict(base_prov), evidence=spec.get("evidence", {})))
        elif vt == "integer":
            n += 1
            rules.append(Rule(id=f"SCH_{n:04d}", name=f"integer::{name}", source_type="SCHEMA",
                              rule_type="integer", params={"feature": name}, features=[name],
                              hardness="HARD", description=f"{name} is integer-valued.",
                              provenance=dict(base_prov), evidence=spec.get("evidence", {})))
    return rules


def _rules_from_yaml_list(path: Path, source_type: str) -> list[Rule]:
    if not path.exists():
        return []
    doc = _load_yaml(path)
    out = []
    for d in doc.get("rules", []):
        d = dict(d)
        d["source_type"] = source_type
        out.append(Rule.from_dict(d))
    return out


def _rules_from_json_list(path: Path, source_type: str) -> list[Rule]:
    if not path.exists():
        return []
    doc = json.loads(path.read_text())
    out = []
    for d in doc.get("rules", []):
        d = dict(d)
        d["source_type"] = source_type
        out.append(Rule.from_dict(d))
    return out


class Validator:
    """Composed rule set + optional plausibility profile."""

    def __init__(self, dataset: str, feature_order: list[str], rules: list[Rule],
                 plausibility: PlausibilityProfile | None = None,
                 include_mined_in_structural: bool = True):
        self.dataset = dataset
        self.feature_order = list(feature_order)
        self.idx = {f: i for i, f in enumerate(self.feature_order)}
        self.rules = rules
        self.plausibility = plausibility
        self.include_mined_in_structural = include_mined_in_structural

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_profiles(cls, dataset: str, schema_dir: Path = SCHEMA_DIR,
                      rules_dir: Path = RULES_DIR,
                      include_mined_in_structural: bool = True) -> "Validator":
        prof = _load_yaml(schema_dir / f"{dataset}.yaml")
        feature_order = prof["feature_order"]
        rd = rules_dir / dataset
        rules: list[Rule] = []
        rules += schema_rules_from_profile(prof)
        rules += _rules_from_yaml_list(rd / "extractor_rules.yaml", "EXTRACTOR")
        rules += _rules_from_yaml_list(rd / "protocol_rules.yaml", "PROTOCOL")
        rules += _rules_from_json_list(rd / "mined_rules.json", "MINED")
        plaus = None
        pp = rd / "plausibility_profile.json"
        if pp.exists():
            plaus = PlausibilityProfile.load(pp)
        return cls(dataset, feature_order, rules, plaus, include_mined_in_structural)

    # ---- validation -------------------------------------------------------
    def _as_matrix(self, X) -> np.ndarray:
        X = np.asarray(X, np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != len(self.feature_order):
            raise ValueError(f"expected {len(self.feature_order)} features, got {X.shape[1]}")
        return X

    def validate_batch(self, X) -> BatchResult:
        X = self._as_matrix(X)
        satisfied, eligible = {}, {}
        for r in self.rules:
            s, e = r.evaluate(X, self.idx)
            satisfied[r.id] = s
            eligible[r.id] = e
        plaus = self.plausibility.evaluate(X) if self.plausibility else None
        return BatchResult(rules=self.rules, satisfied=satisfied, eligible=eligible,
                           n=X.shape[0], include_mined_in_structural=self.include_mined_in_structural,
                           plausibility=plaus)

    def validate(self, x):
        b = self.validate_batch(x)
        return b.result(0, self._as_matrix(x), self.idx)

    # ---- introspection ----------------------------------------------------
    def rules_by_source(self) -> dict[str, list[Rule]]:
        out: dict[str, list[Rule]] = {"SCHEMA": [], "MINED": [], "EXTRACTOR": [], "PROTOCOL": []}
        for r in self.rules:
            out[r.source_type].append(r)
        return out

    def counts_by_source(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.rules_by_source().items()}


def load_validator(dataset: str = "cicids2017_distrinet", **kw) -> Validator:
    return Validator.from_profiles(dataset, **kw)
