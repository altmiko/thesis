"""Analyze the full paired primitive-CAPGD experiment against PrimAttack.

Protocol mirrors ``scripts/analyze_capgd_primattack_comparison.py``: mean+/-sample-SD
over seeds, and within-victim paired McNemar tests (reference seed, four classes
concatenated) with Holm correction over the full comparison family.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _path in (str(REPO_ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluation.paired_validity_gap import (  # noqa: E402
    holm_adjust,
    mcnemar_test,
    newcombe_paired_ci,
)

DEFAULT_VICTIMS = ("mlp", "cnn", "ft_transformer")
DEFAULT_CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")
PRIM_REFERENCE = "prim_opt_joint_p75"


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _rate(mask: np.ndarray) -> float:
    return float(np.asarray(mask, bool).mean())


def _mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, np.float64)
    return float(arr.mean()), (float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)


def _pm(pair: tuple[float, float]) -> str:
    return f"{100 * pair[0]:.2f}\u00b1{100 * pair[1]:.2f}%"


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError(f"paired shape mismatch: {a.shape} vs {b.shape}")
    n11 = int((a & b).sum()); n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum()); n00 = int((~a & ~b).sum())
    lo, hi = newcombe_paired_ci(n11, n10, n01, n00)
    return {
        "n": len(a), "n11": n11, "n10_capgd_only": n10, "n01_primattack_only": n01,
        "n00": n00, "capgd_rate": _rate(a), "primattack_rate": _rate(b),
        "difference": _rate(a) - _rate(b), "ci95": [lo, hi], **mcnemar_test(n10, n01),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads((args.experiment_dir / "config.json").read_text(encoding="utf-8"))
    victims = list(config["victims"])
    classes = list(config["classes"])
    seeds = [int(value) for value in config["seeds"]]
    methods = [f"primitive_capgd_{steps}step" for steps in config["steps"]]
    ref_seed = args.reference_seed

    cap_metrics = (
        "raw_untargeted_asr", "valid_untargeted_asr", "feasible_untargeted_asr",
        "semantic_untargeted_asr", "targeted_benign_asr", "valid_targeted_benign_asr",
        "domain_validity", "primitive_feasibility",
    )

    # PrimAttack reference outcomes, on the same prefix rows.
    n_used = {v: {c: len(sel) for c, sel in {}.items()} for v in victims}
    prim_rows: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    prim_summary: dict[str, dict[str, tuple[float, float]]] = {}
    selection = json.loads((args.experiment_dir / "selection.json").read_text(encoding="utf-8"))
    for victim in victims:
        prim_rows[victim] = {}
        per_seed_metrics = {metric: [] for metric in cap_metrics}
        for seed in seeds:
            e_parts, t_parts, d_parts, f_parts, s_parts = [], [], [], [], []
            for class_name in classes:
                path = args.primattack_dir / "artifacts" / f"{victim}__{class_name}__{PRIM_REFERENCE}__seed{seed}.npz"
                data = _load(path)
                n = len(selection[victim][class_name]["sample_ids"])
                expected = np.asarray(selection[victim][class_name]["sample_ids"], dtype="U128")
                if not np.array_equal(data["sample_id"][:n].astype("U128"), expected):
                    raise ValueError(f"PrimAttack prefix pairing mismatch: {victim}/{class_name}/seed{seed}")
                e = np.asarray(data["evasion"][:n], bool)
                t = np.asarray(data["targeted_success"][:n], bool)
                d = np.asarray(data["domain_valid"][:n], bool)
                f = np.asarray(data["primitive_feasible"][:n], bool)
                s = np.asarray(data["semantic_pass"][:n], bool)
                e_parts.append(e); t_parts.append(t); d_parts.append(d)
                f_parts.append(f); s_parts.append(s)
            e = np.concatenate(e_parts); t = np.concatenate(t_parts)
            d = np.concatenate(d_parts); f = np.concatenate(f_parts); s = np.concatenate(s_parts)
            per_seed_metrics["raw_untargeted_asr"].append(_rate(e))
            per_seed_metrics["valid_untargeted_asr"].append(_rate(e & d))
            per_seed_metrics["feasible_untargeted_asr"].append(_rate(e & d & f))
            per_seed_metrics["semantic_untargeted_asr"].append(_rate(e & d & f & s))
            per_seed_metrics["targeted_benign_asr"].append(_rate(t))
            per_seed_metrics["valid_targeted_benign_asr"].append(_rate(t & d))
            per_seed_metrics["domain_validity"].append(_rate(d))
            per_seed_metrics["primitive_feasibility"].append(_rate(f))
            if seed == ref_seed:
                prim_rows[victim] = {"evasion": e, "valid_untargeted": e & d}
        prim_summary[victim] = {m: _mean_sd(v) for m, v in per_seed_metrics.items()}

    # Primitive-CAPGD outcomes.
    cap_summary: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    cap_ref: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for victim in victims:
        cap_summary[victim] = {}
        cap_ref[victim] = {}
        for method in methods:
            per_seed_metrics = {metric: [] for metric in cap_metrics}
            for seed in seeds:
                e_parts, t_parts, d_parts, f_parts, s_parts = [], [], [], [], []
                for class_name in classes:
                    path = args.experiment_dir / "artifacts" / f"{victim}__{class_name}__{method}__seed{seed}.npz"
                    data = _load(path)
                    expected = np.asarray(selection[victim][class_name]["sample_ids"], dtype="U128")
                    if not np.array_equal(data["sample_id"].astype("U128"), expected):
                        raise ValueError(f"CAPGD pairing mismatch: {victim}/{class_name}/{method}/seed{seed}")
                    if not bool(np.asarray(data["clean_correct"], bool).all()):
                        raise ValueError(f"non-clean-correct denominator: {victim}/{class_name}")
                    e_parts.append(np.asarray(data["evasion"], bool))
                    t_parts.append(np.asarray(data["targeted_success"], bool))
                    d_parts.append(np.asarray(data["domain_valid"], bool))
                    f_parts.append(np.asarray(data["primitive_feasible"], bool))
                    s_parts.append(np.asarray(data["semantic_pass"], bool))
                e = np.concatenate(e_parts); t = np.concatenate(t_parts)
                d = np.concatenate(d_parts); f = np.concatenate(f_parts); s = np.concatenate(s_parts)
                per_seed_metrics["raw_untargeted_asr"].append(_rate(e))
                per_seed_metrics["valid_untargeted_asr"].append(_rate(e & d))
                per_seed_metrics["feasible_untargeted_asr"].append(_rate(e & d & f))
                per_seed_metrics["semantic_untargeted_asr"].append(_rate(e & d & f & s))
                per_seed_metrics["targeted_benign_asr"].append(_rate(t))
                per_seed_metrics["valid_targeted_benign_asr"].append(_rate(t & d))
                per_seed_metrics["domain_validity"].append(_rate(d))
                per_seed_metrics["primitive_feasibility"].append(_rate(f))
                if seed == ref_seed:
                    cap_ref[victim][method] = {"evasion": e, "valid_untargeted": e & d}
            cap_summary[victim][method] = {m: _mean_sd(v) for m, v in per_seed_metrics.items()}

    tests: list[dict[str, Any]] = []
    for victim in victims:
        for method in methods:
            for outcome in ("evasion", "valid_untargeted"):
                result = _paired(cap_ref[victim][method][outcome], prim_rows[victim][outcome])
                label = "raw_untargeted" if outcome == "evasion" else "valid_untargeted"
                result.update({"victim": victim, "method": method, "outcome": label})
                tests.append(result)
    adjusted = holm_adjust([float(row["p_value"]) for row in tests])
    for row, value in zip(tests, adjusted):
        row["holm_p_value"] = value

    result = {
        "config": config,
        "reference_seed": ref_seed,
        "primattack_reference": PRIM_REFERENCE,
        "primattack_summary": {v: {m: list(x) for m, x in s.items()} for v, s in prim_summary.items()},
        "capgd_summary": {
            v: {method: {m: list(x) for m, x in metrics.items()} for method, metrics in methods_by.items()}
            for v, methods_by in cap_summary.items()
        },
        "paired_tests": tests,
    }
    (args.experiment_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_report(args, result, victims, methods)
    return result


def _write_report(args, result, victims, methods) -> None:
    config = result["config"]
    prim = result["primattack_summary"]
    cap = result["capgd_summary"]
    tests = result["paired_tests"]
    L: list[str] = []
    L.append("# CAPGD over PrimAttack's two primitive controls (full paired test)\n")
    L.append("## Question\n")
    L.append(
        "The earlier CAPGD experiment gave CAPGD a nine-feature feature-space threat model, "
        "which is broader than PrimAttack's two controls. This experiment removes that capability "
        "gap: CAPGD optimizes only PrimAttack's normalized padding and timing controls, inside the "
        "exact same per-flow p75 box, and every generated flow passes through PrimAttack's canonical "
        "capability gates, differentiable transform, projection, quantization, validator-v2, primitive "
        "feasibility, and semantic proxy. Only the optimizer and its objective differ."
    )
    L.append("")
    L.append("## Design\n")
    L.append(f"- Rows: the full frozen clean-correct selection ({config.get('n_per_class') or 800} per victim/class).")
    L.append(f"- Victims: {', '.join(victims)}. Classes: {', '.join(config['classes'])}. Seeds: {config['seeds']}.")
    L.append("- CAPGD controls: `q_padding, q_timing in [0,1]`, mapped per row to `p = p_hi*q_padding`, `alpha = 1 + (alpha_hi-1)*q_timing`.")
    L.append("- CAPGD config: untargeted CE, two starts, adaptive momentum/step schedule, normalized Linf radius 1, no epsilon margin.")
    L.append(f"- CAPGD step budgets: {config['steps']} (10 = CAA default; 40 matches PrimAttack's optimizer budget).")
    L.append(f"- Reference: existing `{result['primattack_reference']}` (targeted-Benign Adam, 40 steps), evaluated on the identical rows.")
    L.append("- Final controls use PrimAttack integer padding projection and the quantized canonical transform; validity is independently evaluated.")
    L.append("")
    L.append("## Results\n")
    L.append("Rates pool the four malicious classes within each victim, reported as mean+/-sample-SD over seeds. `Valid` = validator-v2 hybrid validity; `Feasible` also requires primitive consistency/budget; `Semantic` also requires the flow-proxy PASS.")
    L.append("")
    L.append("| Victim | Method | Raw untargeted | Valid untargeted | Feasible untargeted | Semantic untargeted | Targeted Benign | Valid targeted | Domain-validity | Primitive-feasibility |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = [(PRIM_REFERENCE, prim)] + [(method, cap) for method in methods]
    for victim in victims:
        for name, table in order:
            metrics = table[victim][name] if name in methods else prim[victim]
            L.append(
                f"| {victim} | {name} | {_pm(tuple(metrics['raw_untargeted_asr']))} | "
                f"{_pm(tuple(metrics['valid_untargeted_asr']))} | {_pm(tuple(metrics['feasible_untargeted_asr']))} | "
                f"{_pm(tuple(metrics['semantic_untargeted_asr']))} | {_pm(tuple(metrics['targeted_benign_asr']))} | "
                f"{_pm(tuple(metrics['valid_targeted_benign_asr']))} | {_pm(tuple(metrics['domain_validity']))} | "
                f"{_pm(tuple(metrics['primitive_feasibility']))} |"
            )
    L.append("")
    L.append("## Paired tests\n")
    L.append(
        f"Reference seed `{result['reference_seed']}`; four classes concatenated within each victim; one row per independent flow. "
        "McNemar tests are corrected together with Holm across the whole family."
    )
    L.append("")
    L.append("| Victim | CAPGD variant | Outcome | N | CAPGD | PrimAttack | Difference [95% CI] | n10 | n01 | McNemar p | Holm p |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in tests:
        lo, hi = row["ci95"]
        L.append(
            f"| {row['victim']} | {row['method']} | {row['outcome']} | {row['n']} | {_pct(row['capgd_rate'])} | "
            f"{_pct(row['primattack_rate'])} | {_pct(row['difference'])} [{_pct(lo)}, {_pct(hi)}] | "
            f"{row['n10_capgd_only']} | {row['n01_primattack_only']} | {row['p_value']:.3g} | {row['holm_p_value']:.3g} |"
        )
    L.append("")
    L.append("## Interpretation\n")
    L.append("With the feasible set equalized to PrimAttack's two controls and p75 box, any remaining difference reflects the optimizer and objective, not the feature-space capability. Public CAPGD stays untargeted CE with adaptive step sizing; PrimAttack uses a targeted-Benign Adam objective with a primitive-cost penalty. Compare each CAPGD variant against `prim_opt_joint_p75` on the valid-untargeted rows for the fair optimizer contrast.")
    L.append("")
    L.append("## Limitations\n")
    L.append("1. This is a custom CAPGD-over-primitives adaptation, not native TabularBench CAPGD or full CAA.")
    L.append("2. CAPGD (untargeted CE) and PrimAttack (targeted-Benign + cost) optimize different objectives; the comparison isolates capability, not objective.")
    L.append("3. Normalized Linf radius 1 exposes the complete p75 box; it is a box parameterization, not a claim that padding and timing share physical units.")
    L.append("4. Padding is integer-projected and timing is quantized exactly as in PrimAttack, so both methods realize the same discrete primitive space.")
    L.append("5. Packet-level realization, CICFlowMeter re-extraction, replay, and malicious-function preservation remain unavailable; the semantic proxy is flow-level and NOT_FULLY_TESTABLE for Recon/BruteForce.")
    L.append("6. Validity means schema/extractor/protocol/mined consistency, not distributional or causal realism.")
    L.append("7. CUDA backward for CNN/FT-Transformer may be nondeterministic under the installed stack; seed variability is reported rather than assumed absent.")
    L.append("8. Statistical tests remain within victim; victim replicas are not treated as independent flows.")
    L.append("")
    L.append("## Artifacts\n")
    L.append(f"- Config: `{args.experiment_dir.as_posix()}/config.json`")
    L.append(f"- Analysis: `{args.experiment_dir.as_posix()}/analysis.json`")
    L.append(f"- Per-sample artifacts: `{args.experiment_dir.as_posix()}/artifacts/`")
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=REPO_ROOT / "outputs/comparisons/primitive_capgd_vs_primattack",
    )
    parser.add_argument(
        "--primattack-dir", type=Path, default=REPO_ROOT / "outputs/full_adv_eval",
    )
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument(
        "--report", type=Path,
        default=REPO_ROOT / "outputs/comparisons/PRIMITIVE_CAPGD_COMPARISON.md",
    )
    args = parser.parse_args()
    analyze(args)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
