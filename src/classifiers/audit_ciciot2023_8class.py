"""Audit the CICIoT2023 eight-category classifier evaluation path.

The audit checks label alignment and artifact hashes, quantifies sampling and
class/subtype shift, measures feature drift on deterministic category-stratified
samples, reevaluates every checkpoint on full validation, and analyzes the
existing full-test confusion matrices. It does not train or modify models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from src.classifiers.prior_corrected_evaluation import (
    MODEL_DISPLAY_NAMES,
    MODEL_NAMES,
    confusion_metrics,
    load_model,
)
from src.preprocessing.schema import FEATURE_NAMES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_confusion(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    names = rows[0][1:]
    matrix = np.asarray([[int(value) for value in row[1:]] for row in rows[1:]], dtype=np.int64)
    return names, matrix


def exact_label_audit(processed_dir: Path, chunk_size: int) -> dict[str, Any]:
    fine_encoder = load_pickle(processed_dir / "label_encoder.pkl")
    category_encoder = load_pickle(processed_dir / "category_encoder.pkl")
    class_to_category = json.loads((processed_dir / "class_to_category.json").read_text())
    category_names = [str(value) for value in category_encoder.classes_]
    category_to_id = {name: index for index, name in enumerate(category_names)}
    fine_to_category = np.asarray(
        [category_to_id[class_to_category[str(name)]] for name in fine_encoder.classes_],
        dtype=np.int64,
    )
    benign_id = category_to_id["Benign"]
    split_rows: dict[str, int] = {}
    for split in ("train", "val", "test"):
        x = np.load(processed_dir / f"X_{split}.npy", mmap_mode="r")
        fine = np.load(processed_dir / f"y_{split}.npy", mmap_mode="r")
        category = np.load(processed_dir / f"y_{split}_cat.npy", mmap_mode="r")
        binary = np.load(processed_dir / f"y_{split}_bin.npy", mmap_mode="r")
        n = len(fine)
        if x.shape != (n, len(FEATURE_NAMES)) or category.shape != (n,) or binary.shape != (n,):
            raise AssertionError(f"Shape mismatch in split={split}")
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            expected_category = fine_to_category[np.asarray(fine[start:end], dtype=np.int64)]
            actual_category = np.asarray(category[start:end], dtype=np.int64)
            if not np.array_equal(expected_category, actual_category):
                raise AssertionError(f"Fine/category label misalignment in {split} at {start}:{end}")
            expected_binary = (expected_category != benign_id).astype(np.int32)
            if not np.array_equal(expected_binary, np.asarray(binary[start:end], dtype=np.int32)):
                raise AssertionError(f"Category/binary label misalignment in {split} at {start}:{end}")
        split_rows[split] = n
    return {
        "split_rows": split_rows,
        "fine_classes": len(fine_encoder.classes_),
        "categories": category_names,
        "fine_to_category": fine_to_category.tolist(),
        "exact_alignment": True,
    }


def artifact_hash_audit(processed_dir: Path) -> dict[str, Any]:
    manifest = json.loads((processed_dir / "run_manifest.json").read_text())
    expected = manifest["artifact_hashes"]
    actual = {
        f"X_{split}": sha256_file(processed_dir / f"X_{split}.npy")
        for split in ("train", "val", "test")
    }
    scaler = load_pickle(processed_dir / "scaler.pkl")
    actual["scaler"] = hashlib.sha256(
        scaler.center_.tobytes() + scaler.scale_.tobytes()
    ).hexdigest()
    if actual != expected:
        raise AssertionError(f"Artifact hash mismatch: expected={expected}, actual={actual}")
    return {"expected": expected, "actual": actual, "all_match": True}


def counts(values: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(values, dtype=np.int64), minlength=n_classes).astype(np.int64)


def distribution_shift_audit(processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    fine_encoder = load_pickle(processed_dir / "label_encoder.pkl")
    category_encoder = load_pickle(processed_dir / "category_encoder.pkl")
    fine_names = [str(value) for value in fine_encoder.classes_]
    category_names = [str(value) for value in category_encoder.classes_]
    run_manifest = json.loads((processed_dir / "run_manifest.json").read_text())
    class_to_category = json.loads((processed_dir / "class_to_category.json").read_text())
    category_to_id = {name: index for index, name in enumerate(category_names)}
    fine_to_category = np.asarray(
        [category_to_id[class_to_category[name]] for name in fine_names], dtype=np.int64
    )

    y_fine = {
        split: np.load(processed_dir / f"y_{split}.npy", mmap_mode="r")
        for split in ("train", "val", "test")
    }
    y_category = {
        split: np.load(processed_dir / f"y_{split}_cat.npy", mmap_mode="r")
        for split in ("train", "val", "test")
    }
    fine_counts = {split: counts(values, len(fine_names)) for split, values in y_fine.items()}
    category_counts = {
        split: counts(values, len(category_names)) for split, values in y_category.items()
    }
    natural_train_fine = np.asarray(
        [int(run_manifest["per_class_split"][name]["n_train"]) for name in fine_names],
        dtype=np.int64,
    )
    natural_train_category = np.bincount(
        fine_to_category,
        weights=natural_train_fine,
        minlength=len(category_names),
    ).astype(np.int64)

    category_rows: list[dict[str, Any]] = []
    for category_id, name in enumerate(category_names):
        sampled = int(category_counts["train"][category_id])
        natural = int(natural_train_category[category_id])
        category_rows.append(
            {
                "category_id": category_id,
                "category": name,
                "natural_train": natural,
                "sampled_train": sampled,
                "retention_rate": sampled / natural,
                "sampled_train_share": sampled / len(y_category["train"]),
                "validation": int(category_counts["val"][category_id]),
                "validation_share": int(category_counts["val"][category_id]) / len(y_category["val"]),
                "test": int(category_counts["test"][category_id]),
                "test_share": int(category_counts["test"][category_id]) / len(y_category["test"]),
            }
        )
    write_csv(output_dir / "category_distribution_shift.csv", category_rows)

    subtype_rows: list[dict[str, Any]] = []
    subtype_summary: dict[str, Any] = {}
    for category_id, category_name in enumerate(category_names):
        fine_ids = np.flatnonzero(fine_to_category == category_id)
        split_distributions: dict[str, np.ndarray] = {}
        for split in ("train", "val", "test"):
            selected = fine_counts[split][fine_ids].astype(np.float64)
            split_distributions[split] = selected / selected.sum()
        natural_selected = natural_train_fine[fine_ids].astype(np.float64)
        natural_distribution = natural_selected / natural_selected.sum()
        train_test_tv = float(0.5 * np.abs(split_distributions["train"] - split_distributions["test"]).sum())
        val_test_tv = float(0.5 * np.abs(split_distributions["val"] - split_distributions["test"]).sum())
        natural_train_test_tv = float(0.5 * np.abs(natural_distribution - split_distributions["test"]).sum())
        train_test_js = float(jensenshannon(split_distributions["train"], split_distributions["test"], base=2.0) ** 2)
        val_test_js = float(jensenshannon(split_distributions["val"], split_distributions["test"], base=2.0) ** 2)
        subtype_summary[category_name] = {
            "fine_labels": [fine_names[index] for index in fine_ids],
            "sampled_train_vs_test_tv": train_test_tv,
            "natural_train_vs_test_tv": natural_train_test_tv,
            "validation_vs_test_tv": val_test_tv,
            "sampled_train_vs_test_js": train_test_js,
            "validation_vs_test_js": val_test_js,
        }
        for local_index, fine_id in enumerate(fine_ids):
            subtype_rows.append(
                {
                    "category": category_name,
                    "fine_label": fine_names[fine_id],
                    "natural_train_count": int(natural_train_fine[fine_id]),
                    "sampled_train_count": int(fine_counts["train"][fine_id]),
                    "sampled_train_share_within_category": float(split_distributions["train"][local_index]),
                    "validation_count": int(fine_counts["val"][fine_id]),
                    "validation_share_within_category": float(split_distributions["val"][local_index]),
                    "test_count": int(fine_counts["test"][fine_id]),
                    "test_share_within_category": float(split_distributions["test"][local_index]),
                }
            )
    write_csv(output_dir / "fine_subtype_distribution_shift.csv", subtype_rows)
    return {
        "category_rows": category_rows,
        "subtype_summary": subtype_summary,
        "natural_train_fine_counts": natural_train_fine.tolist(),
        "sampled_train_fine_counts": fine_counts["train"].tolist(),
    }


def stratified_indices(labels: np.ndarray, per_category: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(labels)
    selected: list[np.ndarray] = []
    for category_id in np.unique(values):
        candidates = np.flatnonzero(values == category_id)
        take = min(per_category, len(candidates))
        selected.append(rng.choice(candidates, size=take, replace=False))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def feature_shift_audit(
    processed_dir: Path,
    output_dir: Path,
    per_category: int,
    seed: int,
) -> dict[str, Any]:
    category_names = [
        str(value) for value in json.loads((processed_dir / "category_names.json").read_text())
    ]
    x = {
        split: np.load(processed_dir / f"X_{split}.npy", mmap_mode="r")
        for split in ("train", "val", "test")
    }
    y = {
        split: np.load(processed_dir / f"y_{split}_cat.npy", mmap_mode="r")
        for split in ("train", "val", "test")
    }
    sample_indices = {
        split: stratified_indices(y[split], per_category, seed + offset)
        for offset, split in enumerate(("train", "val", "test"))
    }
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for category_id, category_name in enumerate(category_names):
        category_samples: dict[str, np.ndarray] = {}
        for split in ("train", "val", "test"):
            indices = sample_indices[split]
            category_indices = indices[np.asarray(y[split][indices]) == category_id]
            category_samples[split] = np.asarray(x[split][category_indices], dtype=np.float32)
        category_rows: list[dict[str, Any]] = []
        for feature_id, feature in enumerate(FEATURE_NAMES):
            train_values = category_samples["train"][:, feature_id]
            val_values = category_samples["val"][:, feature_id]
            test_values = category_samples["test"][:, feature_id]
            train_test_ks = float(ks_2samp(train_values, test_values, method="asymp").statistic)
            val_test_ks = float(ks_2samp(val_values, test_values, method="asymp").statistic)
            row = {
                "category": category_name,
                "feature": feature,
                "train_rows": int(len(train_values)),
                "validation_rows": int(len(val_values)),
                "test_rows": int(len(test_values)),
                "train_median_scaled": float(np.median(train_values)),
                "validation_median_scaled": float(np.median(val_values)),
                "test_median_scaled": float(np.median(test_values)),
                "train_test_median_shift": float(abs(np.median(train_values) - np.median(test_values))),
                "validation_test_median_shift": float(abs(np.median(val_values) - np.median(test_values))),
                "train_test_ks": train_test_ks,
                "validation_test_ks": val_test_ks,
            }
            rows.append(row)
            category_rows.append(row)
        summaries[category_name] = {
            "mean_train_test_ks": float(np.mean([row["train_test_ks"] for row in category_rows])),
            "max_train_test_ks": float(np.max([row["train_test_ks"] for row in category_rows])),
            "mean_validation_test_ks": float(np.mean([row["validation_test_ks"] for row in category_rows])),
            "max_validation_test_ks": float(np.max([row["validation_test_ks"] for row in category_rows])),
            "top_train_test_features": [
                {"feature": row["feature"], "ks": row["train_test_ks"]}
                for row in sorted(category_rows, key=lambda item: item["train_test_ks"], reverse=True)[:5]
            ],
            "top_validation_test_features": [
                {"feature": row["feature"], "ks": row["validation_test_ks"]}
                for row in sorted(category_rows, key=lambda item: item["validation_test_ks"], reverse=True)[:5]
            ],
        }
    write_csv(output_dir / "feature_distribution_shift.csv", rows)
    return {
        "sample_per_category": per_category,
        "sample_rows": {split: int(len(indices)) for split, indices in sample_indices.items()},
        "category_summary": summaries,
    }


def evaluate_validation(
    processed_dir: Path,
    classifier_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    x_val = np.load(processed_dir / "X_val.npy", mmap_mode="r")
    y_val = np.load(processed_dir / "y_val_cat.npy", mmap_mode="r")
    class_names = [
        str(value) for value in json.loads((processed_dir / "category_names.json").read_text())
    ]
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        model = load_model(
            classifier_dir / "models" / f"{model_name}_8class.pt",
            model_name,
            len(class_names),
            device,
        )
        matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(y_val), batch_size):
                end = min(start + batch_size, len(y_val))
                xb = torch.from_numpy(
                    np.array(x_val[start:end], dtype=np.float32, copy=True)
                ).to(device, non_blocking=True)
                output = model(xb)
                logits = output[0] if isinstance(output, tuple) else output
                prediction = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
                truth = np.asarray(y_val[start:end], dtype=np.int64)
                matrix += np.bincount(
                    truth * len(class_names) + prediction,
                    minlength=len(class_names) ** 2,
                ).reshape(len(class_names), len(class_names))
        metrics = confusion_metrics(matrix, class_names)
        metrics["runtime_seconds"] = float(time.perf_counter() - started)
        metrics["confusion_matrix"] = matrix.tolist()
        results[model_name] = metrics
        rows.append(
            {
                "model": model_name,
                "model_display": MODEL_DISPLAY_NAMES[model_name],
                "split": "validation",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "rows": int(matrix.sum()),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(output_dir / "validation_metrics.csv", rows)
    return results


def test_confusion_audit(
    processed_dir: Path,
    classifier_dir: Path,
    output_dir: Path,
    validation_results: dict[str, Any],
) -> dict[str, Any]:
    class_names = [
        str(value) for value in json.loads((processed_dir / "category_names.json").read_text())
    ]
    confusion_dir = classifier_dir / "prior_correction" / "confusion_matrices"
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        names, matrix = read_confusion(confusion_dir / f"8class_{model_name}_raw.csv")
        if names != class_names:
            raise AssertionError(f"Class order mismatch for {model_name}")
        metrics = confusion_metrics(matrix, class_names)
        majority_baseline = float(matrix.sum(axis=1).max() / matrix.sum())
        total_errors = int(matrix.sum() - np.trace(matrix))
        off_diagonal = matrix.copy()
        np.fill_diagonal(off_diagonal, 0)
        pairs = np.dstack(np.unravel_index(np.argsort(off_diagonal.ravel())[::-1], off_diagonal.shape))[0]
        top_pairs: list[dict[str, Any]] = []
        for true_id, predicted_id in pairs:
            count = int(off_diagonal[true_id, predicted_id])
            if count == 0 or len(top_pairs) >= 10:
                break
            item = {
                "model": model_name,
                "true_class": class_names[int(true_id)],
                "predicted_class": class_names[int(predicted_id)],
                "count": count,
                "share_of_all_errors": count / total_errors,
            }
            top_pairs.append(item)
            error_rows.append(item)
        prediction_counts = matrix.sum(axis=0)
        for item in metrics["per_class"]:
            category_id = int(item["class_index"])
            per_class_rows.append(
                {
                    "model": model_name,
                    "category": item["class_name"],
                    "support": item["support"],
                    "true_share": item["support"] / matrix.sum(),
                    "predicted": int(prediction_counts[category_id]),
                    "predicted_share": int(prediction_counts[category_id]) / matrix.sum(),
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1": item["f1"],
                }
            )
        validation = validation_results[model_name]
        row = {
            "model": model_name,
            "model_display": MODEL_DISPLAY_NAMES[model_name],
            "validation_accuracy": validation["accuracy"],
            "test_accuracy": metrics["accuracy"],
            "accuracy_change": metrics["accuracy"] - validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "test_macro_f1": metrics["macro_f1"],
            "macro_f1_change": metrics["macro_f1"] - validation["macro_f1"],
            "test_weighted_f1": metrics["weighted_f1"],
            "test_majority_baseline": majority_baseline,
            "accuracy_minus_majority_baseline": metrics["accuracy"] - majority_baseline,
        }
        rows.append(row)
        results[model_name] = {
            "validation": validation,
            "test": {**metrics, "confusion_matrix": matrix.tolist()},
            "test_majority_baseline": majority_baseline,
            "total_errors": total_errors,
            "top_error_pairs": top_pairs,
        }
    write_csv(output_dir / "validation_test_comparison.csv", rows)
    write_csv(output_dir / "test_per_class_prediction_audit.csv", per_class_rows)
    write_csv(output_dir / "largest_test_error_pairs.csv", error_rows)
    return {"comparison_rows": rows, "models": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("outputs") / "ciciot2023_fixed")
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=Path("outputs") / "ciciot2023_fixed" / "classifier_results",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--feature-sample-per-category", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.feature_sample_per_category < 1:
        parser.error("batch-size and feature sample must be positive")

    processed_dir = args.processed_dir
    classifier_dir = args.classifier_dir
    output_dir = args.output_dir or classifier_dir / "audit_8class"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    started = time.perf_counter()
    label_audit = exact_label_audit(processed_dir, chunk_size=1_000_000)
    hash_audit = artifact_hash_audit(processed_dir)
    distribution_audit = distribution_shift_audit(processed_dir, output_dir)
    feature_audit = feature_shift_audit(
        processed_dir,
        output_dir,
        per_category=args.feature_sample_per_category,
        seed=args.seed,
    )
    validation_results = evaluate_validation(
        processed_dir,
        classifier_dir,
        output_dir,
        args.batch_size,
        device,
    )
    confusion_audit = test_confusion_audit(
        processed_dir,
        classifier_dir,
        output_dir,
        validation_results,
    )
    eda_report = json.loads((processed_dir / "eda/eda_report.json").read_text())
    classifier_manifest = json.loads((classifier_dir / "neural_run_manifest.json").read_text())
    training_epochs = classifier_manifest.get("completed_epochs", {}).get("8class")
    if training_epochs is None:
        training_epochs = {
            model_name: int(metrics["completed_epochs"])
            for model_name, metrics in classifier_manifest["tasks"]["8class"].items()
        }
    payload = {
        "processed_dir": str(processed_dir.resolve()),
        "classifier_dir": str(classifier_dir.resolve()),
        "device": str(device),
        "seed": args.seed,
        "runtime_seconds": float(time.perf_counter() - started),
        "label_alignment": label_audit,
        "artifact_hashes": hash_audit,
        "distribution_shift": distribution_audit,
        "feature_shift": feature_audit,
        "validation_test": confusion_audit,
        "constant_training_features": eda_report["constant_train_features"],
        "loss_weighting": classifier_manifest["loss_weighting"],
        "loss_weighting_decision": classifier_manifest["loss_weighting_decision"],
        "training_epochs": training_epochs,
    }
    (output_dir / "audit_results.json").write_text(json.dumps(payload, indent=2))
    print(f"Eight-class audit complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
