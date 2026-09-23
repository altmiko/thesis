"""Generate docs/cicids2017_feature_reference.md (§17, §26).

Explains all 79 CICIDS2017 model features in plain language, grouped by family,
with the inferred type, units, direct/derived flag, and -- crucially -- which
validator_v2 rule(s) reference the feature and where each came from
(SCHEMA/MINED/EXTRACTOR/PROTOCOL). Uncertain CICFlowMeter semantics are flagged.

Run:  python -m validation.evaluation.feature_reference
"""
from __future__ import annotations

from pathlib import Path

import yaml

from validation import load_validator
from validation.mining.feature_registry import registry_for

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = Path(__file__).resolve().parents[1] / "schema" / "cicids2017_distrinet.yaml"

CATEGORY_ORDER = [
    ("flow_identity", "Flow identity / context"),
    ("timing", "Timing"),
    ("packet_count", "Packet counts"),
    ("byte_count", "Byte counts"),
    ("packet_size", "Packet lengths / sizes"),
    ("rate", "Rates"),
    ("iat", "Inter-arrival times"),
    ("tcp_flags", "TCP flags"),
    ("header_stats", "Header statistics"),
    ("ratio", "Ratios"),
    ("window_stats", "TCP window / data"),
    ("bulk_stats", "Bulk statistics"),
    ("subflow_stats", "Subflow statistics"),
    ("active_idle", "Active / idle statistics"),
]


def run() -> str:
    prof = yaml.safe_load(SCHEMA.read_text())
    v = load_validator("cicids2017_distrinet")
    # feature -> list of (source_type, expression)
    ref: dict[str, list] = {f: [] for f in prof["feature_order"]}
    for r in v.rules:
        for f in r.features:
            if f in ref:
                ref[f].append((r.source_type, r.id, r.expression()))

    L = ["# CICIDS2017 feature reference (validator_v2)", "",
         "All 79 DistriNet-corrected CICFlowMeter model features, in schema order, with "
         "the automatically inferred type, plain-English meaning, and the validator_v2 "
         "rules that reference each. Type facts are inferred on the train split; where "
         "exact CICFlowMeter semantics are uncertain in this release the entry says so.", ""]

    feats = prof["features"]
    for cat, title in CATEGORY_ORDER:
        members = [f for f in prof["feature_order"] if feats[f]["category"] == cat]
        if not members:
            continue
        L += [f"## {title}", ""]
        for f in members:
            s = feats[f]
            L.append(f"### {f}  (index {s['index']})")
            L.append("")
            L.append(f"- {s['description']}")
            units = s.get("units")
            L.append(f"- Units: {units if units else 'n/a'}; "
                     f"{'derived' if s.get('derived') else 'directly observed'}; "
                     f"inferred type: **{s['value_type']}**"
                     + (f" domain={sorted(s['domain'])}" if s.get("domain") else "")
                     + (f" constant={s['constant_value']}" if s['value_type'] == 'constant' else ""))
            if s.get("uncertain"):
                L.append("- **Uncertain:** exact CICFlowMeter semantics/relationships are not "
                         "reliably established for this release; no hard relational rule is asserted.")
            rules = ref[f]
            if rules:
                by_src = {}
                for src, rid, expr in rules:
                    by_src.setdefault(src, []).append(f"{rid} (`{expr}`)")
                parts = [f"{src}: " + "; ".join(v[:4]) + (" ..." if len(v) > 4 else "")
                         for src, v in by_src.items()]
                L.append("- Validator rules: " + " | ".join(parts))
            else:
                L.append("- Validator rules: none reference this feature directly.")
            L.append("")
    out = REPO_ROOT / "docs" / "cicids2017_feature_reference.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    print(run())
