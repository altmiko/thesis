"""Analyze paired CAPGD and PrimAttack artifacts and generate the thesis report."""
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


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _capgd_path(root: Path, victim: str, class_name: str, seed: int) -> Path:
    return root / "artifacts" / f"{victim}__{class_name}__capgd__seed{seed}.npz"


def _prim_path(root: Path, victim: str, class_name: str, seed: int, attack: str) -> Path:
    return root / "artifacts" / f"{victim}__{class_name}__{attack}__seed{seed}.npz"


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _pm(pair: tuple[float, float]) -> str:
    return f"{100.0 * pair[0]:.2f}±{100.0 * pair[1]:.2f}%"


def _paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"paired outcome shape mismatch: {a.shape} vs {b.shape}")
    n11 = int((a & b).sum())
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n00 = int((~a & ~b).sum())
    test = mcnemar_test(n10, n01)
    low, high = newcombe_paired_ci(n11, n10, n01, n00)
    return {
        "n": len(a),
        "n11": n11,
        "n10_capgd_only": n10,
        "n01_primattack_only": n01,
        "n00": n00,
        "capgd_rate": float(a.mean()),
        "primattack_rate": float(b.mean()),
        "difference_capgd_minus_primattack": float(a.mean() - b.mean()),
        "ci95": [low, high],
        **test,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads((args.capgd_dir / "config.json").read_text(encoding="utf-8"))
    victims = _parse_csv(args.victims)
    classes = _parse_csv(args.classes)
    seeds = [int(value) for value in config["seeds"]]
    if set(victims) - set(config["victims"]) or set(classes) - set(config["classes"]):
        raise ValueError("requested analysis cells are absent from CAPGD config")

    per_seed: dict[str, dict[int, dict[str, float]]] = {victim: {} for victim in victims}
    per_class: list[dict[str, Any]] = []
    paired_inputs: dict[str, dict[str, list[np.ndarray]]] = {
        victim: {"cap_raw": [], "cap_valid": [], "prim_raw": [], "prim_valid": []}
        for victim in victims
    }

    for victim in victims:
        for seed in seeds:
            accumulated = {
                "cap_raw": [], "cap_valid": [], "cap_constrained": [],
                "cap_domain": [], "cap_internal": [], "cap_distance": [],
                "prim_raw": [], "prim_valid": [], "prim_feasible": [],
            }
            for class_name in classes:
                cap = _load(_capgd_path(args.capgd_dir, victim, class_name, seed))
                prim = _load(_prim_path(args.primattack_dir, victim, class_name, seed, args.primattack))
                cap_ids = cap["sample_id"].astype("U128")
                prim_ids = prim["sample_id"].astype("U128")
                if not np.array_equal(cap_ids, prim_ids):
                    raise ValueError(f"sample pairing mismatch: {victim}/{class_name}/seed{seed}")
                if not bool(np.asarray(cap["clean_correct"], bool).all()):
                    raise ValueError(f"CAPGD denominator is not wholly clean-correct: {victim}/{class_name}")
                if not bool(np.asarray(prim["clean_correct"], bool).all()):
                    raise ValueError(f"PrimAttack denominator is not wholly clean-correct: {victim}/{class_name}")

                cap_raw = np.asarray(cap["evasion"], bool)
                cap_domain = np.asarray(cap["domain_valid"], bool)
                cap_valid = cap_raw & cap_domain
                cap_constrained = np.asarray(cap["constrained_success"], bool)
                prim_raw = np.asarray(prim["evasion"], bool)
                prim_domain = np.asarray(prim["domain_valid"], bool)
                prim_valid = prim_raw & prim_domain
                prim_feasible = prim_valid & np.asarray(prim["primitive_feasible"], bool)

                for key, values in (
                    ("cap_raw", cap_raw), ("cap_valid", cap_valid),
                    ("cap_constrained", cap_constrained), ("cap_domain", cap_domain),
                    ("cap_internal", cap["internal_constraint_valid"]),
                    ("cap_distance", cap["distance_ok"]), ("prim_raw", prim_raw),
                    ("prim_valid", prim_valid), ("prim_feasible", prim_feasible),
                ):
                    accumulated[key].append(np.asarray(values, bool))

                per_class.append({
                    "victim": victim,
                    "class": class_name,
                    "seed": seed,
                    "n": len(cap_raw),
                    "capgd_raw_asr": float(cap_raw.mean()),
                    "capgd_validator_valid_asr": float(cap_valid.mean()),
                    "capgd_constrained_asr": float(cap_constrained.mean()),
                    "capgd_domain_validity": float(cap_domain.mean()),
                    "capgd_internal_validity": float(np.asarray(cap["internal_constraint_valid"], bool).mean()),
                    "capgd_distance_validity": float(np.asarray(cap["distance_ok"], bool).mean()),
                    "primattack_raw_asr": float(prim_raw.mean()),
                    "primattack_validator_valid_asr": float(prim_valid.mean()),
                    "primattack_primitive_feasible_asr": float(prim_feasible.mean()),
                })

                if seed == args.reference_seed:
                    paired_inputs[victim]["cap_raw"].append(cap_raw)
                    paired_inputs[victim]["cap_valid"].append(cap_valid)
                    paired_inputs[victim]["prim_raw"].append(prim_raw)
                    paired_inputs[victim]["prim_valid"].append(prim_valid)

            pooled = {key: np.concatenate(parts) for key, parts in accumulated.items()}
            per_seed[victim][seed] = {key: float(value.mean()) for key, value in pooled.items()}

    summaries: dict[str, dict[str, tuple[float, float]]] = {}
    for victim in victims:
        summaries[victim] = {}
        keys = next(iter(per_seed[victim].values())).keys()
        for key in keys:
            summaries[victim][key] = _mean_sd([per_seed[victim][seed][key] for seed in seeds])

    class_metrics = (
        "capgd_raw_asr",
        "capgd_validator_valid_asr",
        "capgd_constrained_asr",
        "capgd_domain_validity",
        "capgd_internal_validity",
        "capgd_distance_validity",
        "primattack_raw_asr",
        "primattack_validator_valid_asr",
        "primattack_primitive_feasible_asr",
    )
    per_class_summaries: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for victim in victims:
        per_class_summaries[victim] = {}
        for class_name in classes:
            rows = [
                row for row in per_class
                if row["victim"] == victim and row["class"] == class_name
            ]
            per_class_summaries[victim][class_name] = {
                metric: _mean_sd([float(row[metric]) for row in rows])
                for metric in class_metrics
            }

    tests: list[dict[str, Any]] = []
    for victim in victims:
        inputs = paired_inputs[victim]
        for outcome, cap_key, prim_key in (
            ("raw_untargeted", "cap_raw", "prim_raw"),
            ("validator_valid_untargeted", "cap_valid", "prim_valid"),
        ):
            result = _paired(np.concatenate(inputs[cap_key]), np.concatenate(inputs[prim_key]))
            result.update({"victim": victim, "outcome": outcome})
            tests.append(result)
    adjusted = holm_adjust([float(row["p_value"]) for row in tests])
    for row, value in zip(tests, adjusted):
        row["holm_p_value"] = value

    result = {
        "capgd_config": config,
        "primattack": args.primattack,
        "reference_seed": args.reference_seed,
        "summaries": {
            victim: {key: list(value) for key, value in metrics.items()}
            for victim, metrics in summaries.items()
        },
        "per_class_summaries": {
            victim: {
                class_name: {
                    metric: list(value) for metric, value in metrics.items()
                }
                for class_name, metrics in classes_by_name.items()
            }
            for victim, classes_by_name in per_class_summaries.items()
        },
        "per_seed": per_seed,
        "per_class": per_class,
        "paired_tests": tests,
    }
    (args.capgd_dir / "comparison_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_report(args, result)
    return result


def _write_report(args: argparse.Namespace, result: dict[str, Any]) -> None:
    cfg = result["capgd_config"]
    summaries = result["summaries"]
    per_class_summaries = result["per_class_summaries"]
    tests = result["paired_tests"]
    lines: list[str] = []
    lines.append("# CAPGD versus PrimAttack on CICIDS2017-DistriNet\n")
    lines.append("## Scope\n")
    lines.append(
        "This report documents and compares the upstream TabularBench CAPGD component with "
        "PrimAttack on the exact frozen clean-correct rows used by `outputs/full_adv_eval`. "
        "CAPGD is imported from the frozen `external/tabularbench` checkout; the thesis code "
        "adds only the CICIDS attack-space, constraints, victim wrapper, artifact runner, and "
        "independent validator-v2 evaluation."
    )
    lines.append("")
    lines.append("## Source identity\n")
    lines.append(f"- TabularBench commit: `{cfg['tabularbench']['commit']}` (MIT).")
    lines.append("- Upstream implementation: `external/tabularbench/tabularbench/attacks/capgd/capgd.py`.")
    lines.append("- Local adapter: `src/comparisons/capgd_cicids2017.py`.")
    lines.append("- Runner: `scripts/run_capgd_primattack_comparison.py`.")
    lines.append("- Analysis: `scripts/analyze_capgd_primattack_comparison.py`.")
    lines.append("## Upstream reuse and compatibility adaptations\n")
    lines.append(
        "The optimizer, restart logic, adaptive step schedule, momentum update, "
        "constraint objective, type/immutable repair, candidate selection, and distance "
        "objective execute from the frozen TabularBench source. No external file was edited. "
        "The local adapter applies four runtime compatibility corrections required by the "
        "current NumPy/CUDA/Windows environment: restore NumPy's removed `np.float_` alias; "
        "use integral constants in relation arithmetic to prevent TabularBench's float32 "
        "`SafeDivision` from receiving float64 quotients; move scaler tensor outputs to the "
        "CAPGD CUDA device during repaired-candidate reinsertion; and convert CUDA index "
        "tensors to CPU arrays for upstream `np.setdiff1d`. Joblib's constraint checks use "
        "the threading backend so workers share these compatibility definitions."
    )
    lines.append("")
    lines.append("")
    lines.append("## Exact CAPGD configuration\n")
    lines.append(f"- Goal: untargeted evasion.")
    lines.append(f"- Norm: `{cfg['norm']}` in train-fit min-max attack space.")
    lines.append(f"- Nominal epsilon: `{cfg['epsilon']}`; gradient radius uses the upstream 1% epsilon margin.")
    lines.append(f"- Steps: `{cfg['steps']}`; restarts/starts: `{cfg['n_restarts']}` (clean then random).")
    lines.append("- Loss: cross-entropy; momentum coefficient 0.75; adaptive oscillation/no-improvement step halving with rho=0.75.")
    lines.append("- Equality repair runs each iteration and at the end; types and immutables are repaired at the end.")
    lines.append(f"- Seeds: `{cfg['seeds']}`.")
    lines.append("")
    lines.append("## CICIDS constraint adaptation\n")
    lines.append(
        "The direct attack coordinates are the repository-native nine-feature config mask: "
        "Flow Duration, Total Length of Fwd Packet, Fwd Packet Length Max/Min/Std, and "
        "Fwd IAT Total/Std/Max/Min. Seven dependent features are repairable outputs: "
        "Fwd Packet Length Mean, Fwd Segment Size Avg, Fwd IAT Mean, Flow Bytes/s, "
        "Flow Packets/s, Fwd Packets/s, and Bwd Packets/s. All remaining features are copied "
        "from the clean row. Min/max attack scaling and finite search bounds are fitted on "
        "`X_train_pristine.npy` only. Final validity is not trusted to CAPGD: validator v2 "
        "independently evaluates SCHEMA, EXTRACTOR, PROTOCOL, and MINED rules."
    )
    lines.append("")
    lines.append("## Denominator and pairing\n")
    lines.append(
        "For each victim and malicious class, the denominator is the same 800 clean-correct "
        "test flows selected in `outputs/full_adv_eval/selection.json`. CAPGD and PrimAttack "
        "artifacts are joined by victim, class, seed, and ordered `sample_id`; any mismatch aborts analysis."
    )
    lines.append("")
    lines.append("## Metrics\n")
    lines.append("- **Raw ASR:** adversarial prediction differs from the true malicious class.")
    lines.append("- **Validator-valid ASR:** raw evasion AND validator-v2 `hybrid_valid`.")
    lines.append("- **CAPGD constrained ASR:** raw evasion AND `hybrid_valid` AND TabularBench constraints AND nominal distance.")
    lines.append("- **PrimAttack primitive-feasible ASR:** raw evasion AND `hybrid_valid` AND primitive feasibility.")
    lines.append("All rates use the unchanged clean-correct denominator; invalid candidates count as failures, not removed rows.")
    lines.append("")
    lines.append("## Results\n")
    lines.append("Rates are pooled over the four malicious classes within each victim, then reported as mean±sample-SD over seeds.")
    lines.append("")
    lines.append("| Victim | CAPGD raw ASR | CAPGD validator-valid ASR | CAPGD constrained ASR | CAPGD domain-validity | PrimAttack raw ASR | PrimAttack validator-valid ASR | PrimAttack primitive-feasible ASR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for victim, metrics in summaries.items():
        lines.append(
            f"| {victim} | {_pm(tuple(metrics['cap_raw']))} | {_pm(tuple(metrics['cap_valid']))} | "
            f"{_pm(tuple(metrics['cap_constrained']))} | {_pm(tuple(metrics['cap_domain']))} | "
            f"{_pm(tuple(metrics['prim_raw']))} | {_pm(tuple(metrics['prim_valid']))} | "
            f"{_pm(tuple(metrics['prim_feasible']))} |"
        )
    lines.append("")
    lines.append("## Per-class results\n")
    lines.append("Each cell is mean±sample-SD over the three seeds; every class cell has 800 rows per seed.")
    lines.append("")
    lines.append("| Victim | Class | CAPGD raw | CAPGD validator-valid | CAPGD constrained | CAPGD domain-validity | CAPGD internal-validity | CAPGD distance-validity | PrimAttack raw/valid/feasible |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for victim, classes_by_name in per_class_summaries.items():
        for class_name, metrics in classes_by_name.items():
            prim = _pm(tuple(metrics["primattack_raw_asr"]))
            prim_valid = _pm(tuple(metrics["primattack_validator_valid_asr"]))
            prim_feasible = _pm(tuple(metrics["primattack_primitive_feasible_asr"]))
            lines.append(
                f"| {victim} | {class_name} | {_pm(tuple(metrics['capgd_raw_asr']))} | "
                f"{_pm(tuple(metrics['capgd_validator_valid_asr']))} | "
                f"{_pm(tuple(metrics['capgd_constrained_asr']))} | "
                f"{_pm(tuple(metrics['capgd_domain_validity']))} | "
                f"{_pm(tuple(metrics['capgd_internal_validity']))} | "
                f"{_pm(tuple(metrics['capgd_distance_validity']))} | "
                f"{prim} / {prim_valid} / {prim_feasible} |"
            )
    lines.append("")
    lines.append("## Paired tests\n")
    lines.append(
        f"Reference seed `{result['reference_seed']}`; four classes concatenated within each victim. "
        "A success vector remains one row per independent flow. McNemar tests are corrected together with Holm."
    )
    lines.append("")
    lines.append("| Victim | Outcome | N | CAPGD | PrimAttack | Difference [95% CI] | n10 | n01 | McNemar p | Holm p |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in tests:
        lo, hi = row["ci95"]
        lines.append(
            f"| {row['victim']} | {row['outcome']} | {row['n']} | {_pct(row['capgd_rate'])} | "
            f"{_pct(row['primattack_rate'])} | {_pct(row['difference_capgd_minus_primattack'])} "
            f"[{_pct(lo)}, {_pct(hi)}] | {row['n10_capgd_only']} | {row['n01_primattack_only']} | "
            f"{row['p_value']:.3g} | {row['holm_p_value']:.3g} |"
        )
    lines.append("")
    lines.append("## Limitations\n")
    lines.append("1. **Different feasible sets.** CAPGD directly optimizes nine aggregate features plus repaired dependencies. PrimAttack optimizes only padding and timing primitives. CAPGD's feasible set is a feature-space relaxation, so ASR differences are not pure optimizer effects.")
    lines.append("2. **Different native goals.** CAPGD is natively untargeted; PrimAttack is optimized toward Benign. Untargeted predictions are used for the primary comparison, but PrimAttack was not optimized for that exact objective.")
    lines.append("3. **Empirical bounds.** CAPGD's finite min-max bounds are fitted from training observations because universal physical maxima are unavailable. They are not proofs of packet-level feasibility.")
    lines.append("4. **Incomplete attack-time relations.** CAPGD repairs the seven config-mask dependencies and penalizes the two directly affected monotone chains. Validator v2 checks the larger independent rule set only after generation; therefore internal CAPGD validity and `hybrid_valid` can differ.")
    lines.append("5. **No packet realization.** Neither method is validated by constructing packets, rerunning CICFlowMeter, replaying traffic, or proving malicious functionality. PrimAttack's semantic proxy remains flow-level only; it is not applicable to CAPGD outputs.")
    lines.append("6. **Validity is not realism.** `hybrid_valid` establishes schema/extractor/protocol/mined consistency, not distributional or causal realism. The train-fit plausibility result is stored separately.")
    lines.append("7. **Published CAA is broader.** This experiment runs CAPGD only. It does not run CAA's MOEVA fallback and must not be reported as CAA.")
    lines.append("8. **Software compatibility adaptations.** Frozen TabularBench assumes NumPy <2 and contains CPU/GPU crossings in its repaired-candidate path. The runtime corrections are enumerated above and do not alter CAPGD's mathematical update or acceptance criteria.")
    lines.append("9. **GPU reproducibility is best-effort.** Seeds and deterministic settings are applied, but PyTorch warns that CNN adaptive-max-pool backward and FT-Transformer memory-efficient attention backward lack deterministic CUDA implementations under the installed stack. Seed variability is therefore reported rather than assumed absent.")
    lines.append("10. **Single dataset and checkpoint set.** Conclusions apply to this CICIDS2017-DistriNet preprocessing, these three victims, and this frozen sample roster.")
    lines.append("11. **No cross-victim pooling for inference.** Victim eligibility sets differ. Statistical tests remain within victim; the table does not treat victim replicas as independent flows.")
    lines.append("")
    lines.append("## Reproduction\n")
    lines.append("```powershell")
    lines.append('$python = "C:/Users/user6/.local/share/mamba/envs/thesis/python.exe"')
    lines.append('& $python scripts/run_capgd_primattack_comparison.py --device cuda')
    lines.append('& $python scripts/analyze_capgd_primattack_comparison.py')
    lines.append("```")
    lines.append("")
    lines.append("Machine-readable results: `outputs/comparisons/capgd_vs_primattack/comparison_analysis.json`.")

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capgd-dir", type=Path,
        default=REPO_ROOT / "outputs/comparisons/capgd_vs_primattack",
    )
    parser.add_argument(
        "--primattack-dir", type=Path,
        default=REPO_ROOT / "outputs/full_adv_eval_primattack_v2",
    )
    parser.add_argument("--primattack", default="prim_search_joint_p75")
    parser.add_argument("--victims", default=",".join(DEFAULT_VICTIMS))
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument(
        "--report", type=Path,
        default=REPO_ROOT / "outputs/comparisons/CAPGD_PRIMATTACK_COMPARISON.md",
    )
    args = parser.parse_args()
    analyze(args)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
