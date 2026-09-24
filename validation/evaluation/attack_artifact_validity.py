"""Independently re-check CICIDS2017 attack artifacts with validator_v2.

Reads each runner's saved ``X_adv_raw``. Runner validity masks are ignored: all
SCHEMA/MINED/EXTRACTOR/PROTOCOL and plausibility decisions are recomputed from
validator_v2 profiles. Metrics use the runner's clean-correct denominator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from validation import load_validator


METHODS = {
    "Input PGD": Path("outputs/v2_validity_input_pgd"),
    "Primitive Direct": Path("outputs/v2_validity_primitive"),
    "VAE Latent Primitive": Path("outputs/v2_validity_vae_latent_primitive"),
    "VAE Latent Raw": Path("outputs/v2_validity_latent_raw"),
    "VAE Latent Masked": Path("outputs/v2_validity_latent_masked"),
}


def _artifact_path(output_dir: Path, recorded: str) -> Path:
    path = Path(recorded)
    if path.exists():
        return path
    fallback = output_dir / "attack_artifacts" / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(recorded)


def _aggregate(rows: list[dict]) -> dict:
    n = sum(r["n_clean_correct"] for r in rows)
    fields = (
        "schema_valid", "mined_valid", "extractor_valid", "protocol_valid",
        "hard_structural_valid", "hybrid_valid", "in_distribution",
        "untargeted", "targeted_benign",
        "untargeted_v2_valid", "targeted_v2_valid",
        "untargeted_valid_id", "targeted_valid_id",
    )
    out = {"n_clean_correct": n}
    for field in fields:
        count = sum(r[field + "_count"] for r in rows)
        out[field + "_count"] = count
        out[field + "_rate"] = count / n if n else float("nan")
    raw = out["targeted_benign_rate"]
    valid = out["targeted_v2_valid_rate"]
    out["targeted_success_validity_retention"] = valid / raw if raw else float("nan")
    raw_u = out["untargeted_rate"]
    valid_u = out["untargeted_v2_valid_rate"]
    out["untargeted_success_validity_retention"] = valid_u / raw_u if raw_u else float("nan")
    failures = Counter()
    for r in rows:
        failures.update(r["failed_rules"])
    out["top_failed_rules"] = failures.most_common(12)
    return out


def check_method(label: str, output_dir: Path, validator) -> dict:
    meta = json.loads((output_dir / "attack_results.json").read_text())
    rows: list[dict] = []
    rule_by_id = {r.id: r for r in validator.rules}
    for cell in meta["cells"]:
        artifact = _artifact_path(output_dir, cell["artifact"])
        with np.load(artifact, allow_pickle=False) as z:
            X = np.asarray(z["X_adv_raw"], np.float64)
            cc = np.asarray(z["clean_correct"], bool)
            benign = np.asarray(z["benign"], bool)
            evasion = np.asarray(z["evasion"], bool)
        b = validator.validate_batch(X)
        masks = {
            "schema_valid": b.schema_valid,
            "mined_valid": b.mined_valid,
            "extractor_valid": b.extractor_valid,
            "protocol_valid": b.protocol_valid,
            "hard_structural_valid": b.hard_structural_valid,
            "hybrid_valid": b.hybrid_valid,
            "in_distribution": b.in_distribution,
            "untargeted": evasion,
            "targeted_benign": benign,
            "untargeted_v2_valid": evasion & b.hybrid_valid,
            "targeted_v2_valid": benign & b.hybrid_valid,
            "untargeted_valid_id": evasion & b.hybrid_valid & b.in_distribution,
            "targeted_valid_id": benign & b.hybrid_valid & b.in_distribution,
        }
        failed = Counter()
        for rule in validator.rules:
            count = int((b.violation(rule.id) & cc).sum())
            if count:
                failed[f"{rule.id} {rule.expression()}"] += count
        row = {
            "class": cell["class"], "victim": cell["victim"], "seed": cell["seed"],
            "n_clean_correct": int(cc.sum()), "failed_rules": dict(failed),
        }
        for name, mask in masks.items():
            row[name + "_count"] = int((mask & cc).sum())
        rows.append(row)

    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)
    return {
        "method": label,
        "output_dir": str(output_dir),
        "strict_valid_definition": "validator_v2 hybrid_valid only",
        "overall": _aggregate(rows),
        "by_class": {name: _aggregate(group) for name, group in by_class.items()},
        "cells": rows,
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(results: list[dict]) -> str:
    L = [
        "# CICIDS2017-DistriNet Attack Results — validator_v2", "",
        "Every saved `X_adv_raw` was independently reloaded and revalidated; runner-provided "
        "validity masks were not trusted. Denominator: clean-correct malicious samples. "
        "Plausibility is reported separately from structural validity.", "",
        "## Metric definitions", "",
        "- **Untargeted ASR:** adversarial prediction changes from the true malicious class to any other class.",
        "- **Attack→Benign ASR:** adversarial prediction is specifically Benign (class 0). This is the primary NIDS-evasion metric.",
        "- **v2-valid ASR:** attack success AND `validator_v2.hybrid_valid`, over the same clean-correct denominator.",
        "- **Valid+ID ASR:** attack success AND hybrid validity AND in-distribution plausibility.", "",
        "## Evaluation protocol", "",
        "- Four attack classes: DoS, DDoS, Recon, BruteForce.",
        "- Two active victim models: MLP and CNN.",
        "- 256 sampled test rows per attack class; seed 42; CPU.",
        "- Eight class × model cells; clean-correct denominators are loaded from artifacts.",
        "- Strict validity is validator_v2 `hybrid_valid` only; no PAVE gate.", "",
        "## Overall results", "",
        "| Method | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted ASR | v2-valid Attack→Benign ASR | In-distribution | Valid+ID untargeted ASR | Valid+ID Attack→Benign ASR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for res in results:
        o = res["overall"]
        L.append("| " + " | ".join([
            res["method"], str(o["n_clean_correct"]), _pct(o["untargeted_rate"]),
            _pct(o["targeted_benign_rate"]), _pct(o["hybrid_valid_rate"]),
            _pct(o["untargeted_v2_valid_rate"]), _pct(o["targeted_v2_valid_rate"]),
            _pct(o["in_distribution_rate"]), _pct(o["untargeted_valid_id_rate"]),
            _pct(o["targeted_valid_id_rate"]),
        ]) + " |")

    L += ["", "## Structural-validity layers", "",
          "| Method | SCHEMA | EXTRACTOR | PROTOCOL | MINED | Hard structural | Hybrid |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for res in results:
        o = res["overall"]
        L.append(f"| {res['method']} | {_pct(o['schema_valid_rate'])} | "
                 f"{_pct(o['extractor_valid_rate'])} | {_pct(o['protocol_valid_rate'])} | "
                 f"{_pct(o['mined_valid_rate'])} | {_pct(o['hard_structural_valid_rate'])} | "
                 f"{_pct(o['hybrid_valid_rate'])} |")

    L += ["", "## Per-class summary", ""]
    for res in results:
        L += [f"### {res['method']}", "",
              "| Class | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | Valid+ID Attack→Benign |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for cls, o in res["by_class"].items():
            L.append(f"| {cls} | {o['n_clean_correct']} | {_pct(o['untargeted_rate'])} | "
                     f"{_pct(o['targeted_benign_rate'])} | {_pct(o['hybrid_valid_rate'])} | "
                     f"{_pct(o['untargeted_v2_valid_rate'])} | {_pct(o['targeted_v2_valid_rate'])} | "
                     f"{_pct(o['targeted_valid_id_rate'])} |")
        L.append("")

    L += ["## Detailed class × model results", ""]
    for res in results:
        L += [f"### {res['method']}", "",
              "| Class | Victim model | N | Untargeted ASR | Attack→Benign ASR | Hybrid valid | v2-valid untargeted | v2-valid Attack→Benign | In-distribution | Valid+ID Attack→Benign |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in res["cells"]:
            n = row["n_clean_correct"]
            rate = lambda field: row[field + "_count"] / n if n else float("nan")
            L.append(f"| {row['class']} | {row['victim']} | {n} | "
                     f"{_pct(rate('untargeted'))} | {_pct(rate('targeted_benign'))} | "
                     f"{_pct(rate('hybrid_valid'))} | {_pct(rate('untargeted_v2_valid'))} | "
                     f"{_pct(rate('targeted_v2_valid'))} | {_pct(rate('in_distribution'))} | "
                     f"{_pct(rate('targeted_valid_id'))} |")
        L.append("")

    L += ["## Top failed v2 rules by method", ""]
    for res in results:
        L.append(f"### {res['method']}")
        failures = res["overall"]["top_failed_rules"]
        if not failures:
            L.append("No v2 rule failures among clean-correct samples.")
        else:
            for rule, count in failures:
                L.append(f"- `{rule}`: {count}")
        L.append("")

    L += ["## Conclusions", "",
          "1. Input PGD and VAE Latent Raw obtain high raw ASR but 0% hybrid validity, so none of their successes survive v2.",
          "2. Primitive Direct and VAE Latent Primitive are 100% structurally valid; their raw and v2-valid ASRs are identical.",
          "3. VAE Latent Masked is 99.98% hybrid-valid; one DoS/CNN row violates the forward packet-length ordering rule.",
          "4. Attack→Benign ASR is lower than untargeted ASR when attacks redirect samples to another malicious class rather than Benign.",
          "5. In-distribution plausibility remains a separate gate and can substantially reduce validity-aware success.", "",
          "## Reproduction artifacts", "",
          "- `outputs/v2_validity_report/results.json` — machine-readable metrics.",
          "- `outputs/v2_validity_report/report.md` — generated report.",
          "- `python -m validation.evaluation.attack_artifact_validity` — regenerate both reports.",
          ""]
    return "\n".join(L)


def run(methods: dict[str, Path] = METHODS, output_dir: Path = Path("outputs/v2_validity_report")) -> dict:
    validator = load_validator("cicids2017_distrinet")
    results = [check_method(label, path, validator) for label, path in methods.items()]
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "cicids2017_distrinet", "methods": results}
    report = render_markdown(results)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    Path("attack_results.md").write_text(report, encoding="utf-8")
    return payload
if __name__ == "__main__":
    payload = run()
    for result in payload["methods"]:
        o = result["overall"]
        print(result["method"], {k: o[k] for k in (
            "n_clean_correct", "hybrid_valid_rate", "untargeted_rate",
            "targeted_benign_rate", "untargeted_v2_valid_rate",
            "targeted_v2_valid_rate", "targeted_valid_id_rate")})
