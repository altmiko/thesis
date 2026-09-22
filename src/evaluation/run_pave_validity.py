"""Fit and run the independent PAVE-style validity baseline.

Examples:
    PYTHONPATH=src python -m evaluation.run_pave_validity --dataset ciciot2023
    PYTHONPATH=src python -m evaluation.run_pave_validity --dataset ciciot2023 \
        --attack-file results/attacks/attack_mlp_pgd_0.3.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from constraints import ConstraintEngine, load_layer2
from datasets import get_adapter
from evaluation.pave_style_validator import PAVEStyleValidator

_REPO = Path(__file__).resolve().parents[2]


def _raw_split(adapter: Any, split_name: str, limit: int | None = None) -> np.ndarray:
    """Load pristine raw data when present, otherwise use the saved train-fit transform."""

    pristine = getattr(adapter, "_processed", Path()) / f"X_{split_name}_pristine.npy"
    if pristine.exists():
        raw = np.load(pristine, mmap_mode="r")
        return raw if limit is None else np.asarray(raw[:limit])
    split = adapter.load_split(split_name)
    scaled = split.x if limit is None else split.x[:limit]
    return adapter.feature_transform().inverse_transform(np.asarray(scaled))


def _build_mined_checker(adapter: Any, source: Path | None) -> ConstraintEngine | None:
    manifest = adapter.feature_manifest()
    rule_path = source or (_REPO / "constraints" / manifest.dataset_name / "mined.json")
    if not rule_path.exists():
        return None
    return ConstraintEngine(
        manifest,
        layer1=[],
        layer2=load_layer2(rule_path, manifest),
        active_layers={2},
    )


def _compact_validation(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_samples",
        "valid_count",
        "validity_rate",
        "range_validity_rate",
        "type_validity_rate",
        "combined_validity_rate",
        "mined_valid_count",
        "mined_constraint_validity_rate",
        "strict_valid_count",
        "strict_validity_rate",
        "violation_counts_by_feature",
        "violation_counts_by_reason",
        "mined_violation_counts_by_constraint",
    )
    return {key: result[key] for key in keys if key in result}


def evaluate_attack_arrays(
    *,
    attack: str,
    validation: dict[str, Any],
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_adv: np.ndarray,
) -> dict[str, Any]:
    """Combine independent validity with predictions using clean-correct ASR denominator."""

    y = np.asarray(y_true).reshape(-1)
    clean = np.asarray(y_pred_clean).reshape(-1)
    adv = np.asarray(y_pred_adv).reshape(-1)
    if not (y.shape == clean.shape == adv.shape):
        raise ValueError("y_true, y_pred_clean, and y_pred_adv must have the same shape")
    pave_valid = np.asarray(validation["valid_mask"], dtype=np.bool_)
    if pave_valid.shape != y.shape:
        raise ValueError("validity mask does not match prediction arrays")
    mined_valid = validation.get("mined_valid_mask")
    strict_valid = np.asarray(validation.get("strict_valid_mask", pave_valid), dtype=np.bool_)

    clean_correct = clean == y
    successful = clean_correct & (adv != y)
    denominator = int(clean_correct.sum())

    def over_clean_correct(mask: np.ndarray) -> float:
        return float((mask & clean_correct).sum() / denominator) if denominator else 0.0

    result = {
        "attack": attack,
        "n_samples": int(len(y)),
        "originally_correct_samples": denominator,
        "raw_asr": over_clean_correct(successful),
        "pave_validity_rate": float(pave_valid.mean()),
        "range_validity_rate": float(validation["range_validity_rate"]),
        "type_validity_rate": float(validation["type_validity_rate"]),
        "combined_validity_rate": float(validation["combined_validity_rate"]),
        "pave_valid_successful_attacks": int((successful & pave_valid).sum()),
        "pave_valid_asr": over_clean_correct(successful & pave_valid),
        "valid_successful_attacks": int((successful & strict_valid).sum()),
        "valid_asr": over_clean_correct(successful & strict_valid),
    }
    if mined_valid is not None:
        result.update(
            {
                "mined_validity_rate": float(np.asarray(mined_valid, dtype=np.bool_).mean()),
                "strict_validity_rate": float(strict_valid.mean()),
            }
        )
    else:
        result.update({"mined_validity_rate": None, "strict_validity_rate": None})
    return result


def _npz_value(archive: Any, candidates: tuple[str, ...]) -> np.ndarray:
    for key in candidates:
        if key in archive:
            return np.asarray(archive[key])
    raise KeyError(f"attack artifact lacks all supported fields: {', '.join(candidates)}")


def evaluate_attack_file(
    path: Path,
    *,
    validator: PAVEStyleValidator,
    transform: Any,
    mined_checker: Any,
    attack_space: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an NPZ attack artifact without changing its stored adversarial vectors."""

    with np.load(path, allow_pickle=False) as archive:
        x_adv = _npz_value(archive, ("X_adv", "x_adv"))
        y_true = _npz_value(archive, ("y_true",))
        y_pred_clean = _npz_value(archive, ("y_pred_clean", "y_pred_before"))
        y_pred_adv = _npz_value(archive, ("y_pred_adv", "y_pred_after"))
        if "attack_name" in archive:
            attack_value = np.asarray(archive["attack_name"]).reshape(-1)[0]
            attack = str(attack_value)
        else:
            attack = path.stem

    if attack_space == "scaled":
        validation = validator.validate_scaled_batch(x_adv, transform, mined_checker=mined_checker)
    else:
        validation = validator.validate_batch(x_adv, mined_checker=mined_checker)
    metrics = evaluate_attack_arrays(
        attack=attack,
        validation=validation,
        y_true=y_true,
        y_pred_clean=y_pred_clean,
        y_pred_adv=y_pred_adv,
    )
    return metrics, _compact_validation(validation)


def run(args: argparse.Namespace) -> dict[str, Any]:
    adapter = get_adapter(args.dataset)
    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()

    if args.load_validator is not None:
        validator = PAVEStyleValidator.load(args.load_validator)
        if validator.feature_names != manifest.names:
            raise ValueError("loaded validator feature order does not match dataset manifest")
    else:
        train_raw = _raw_split(adapter, "train")
        validator = PAVEStyleValidator(
            integer_tolerance=args.integer_tolerance,
            range_tolerance=args.range_tolerance,
        ).fit(train_raw, manifest.names, schema=manifest)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    validator_path = output_dir / f"{manifest.dataset_name}_pave_validator.json"
    validator.save(validator_path)
    audit_path = output_dir / f"{manifest.dataset_name}_feature_audit.txt"
    audit_path.write_text(validator.format_audit() + "\n", encoding="utf-8")

    mined_checker = None if args.no_mined else _build_mined_checker(adapter, args.mined_constraints)
    genuine_raw = _raw_split(adapter, args.heldout_split, args.heldout_limit)
    genuine = validator.validate_batch(genuine_raw, mined_checker=mined_checker)

    attack_paths = list(args.attack_file)
    for attack_dir in args.attack_dir:
        attack_paths.extend(sorted(attack_dir.glob("*.npz")))
    attack_paths = list(dict.fromkeys(attack_paths))
    attack_reports: list[dict[str, Any]] = []
    for attack_path in attack_paths:
        metrics, validation = evaluate_attack_file(
            attack_path,
            validator=validator,
            transform=transform,
            mined_checker=mined_checker,
            attack_space=args.attack_space,
        )
        attack_reports.append({"metrics": metrics, "validation": validation})

    report = {
        "dataset": manifest.dataset_name,
        "validator_fit_split": "loaded" if args.load_validator is not None else "train",
        "heldout_split": args.heldout_split,
        "heldout_rows": int(len(genuine_raw)),
        "genuine": _compact_validation(genuine),
        "attacks": attack_reports,
        "validator_path": str(validator_path),
        "feature_audit_path": str(audit_path),
        "mined_constraints_enabled": mined_checker is not None,
    }
    report_path = output_dir / f"{manifest.dataset_name}_pave_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("\nMost common genuine-data violations:")
    for feature, count in list(genuine["violation_counts_by_feature"].items())[:10]:
        print(f"  {feature}: {count}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent PAVE-style raw-space validity audit")
    parser.add_argument("--dataset", default="ciciot2023", choices=("ciciot2023", "cicids2017"))
    parser.add_argument("--attack-file", type=Path, action="append", default=[])
    parser.add_argument(
        "--attack-dir",
        type=Path,
        action="append",
        default=[],
        help="Evaluate every .npz attack artifact in this directory.",
    )
    parser.add_argument("--attack-space", choices=("scaled", "raw"), default="scaled")
    parser.add_argument("--heldout-split", choices=("val", "test"), default="test")
    parser.add_argument("--heldout-limit", type=int, default=10000)
    parser.add_argument("--integer-tolerance", type=float, default=1e-6)
    parser.add_argument("--range-tolerance", type=float, default=1e-6)
    parser.add_argument("--mined-constraints", type=Path)
    parser.add_argument("--no-mined", action="store_true")
    parser.add_argument("--load-validator", type=Path)
    parser.add_argument("--output-dir", type=Path, default=_REPO / "outputs" / "pave_validity")
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
