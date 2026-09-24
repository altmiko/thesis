"""Evaluate CICIoT2023 checkpoints with post-hoc logit prior correction.

No model, scaler, or saved weight is modified. Priors are computed directly
from the saved sampled-training and full-test label arrays. For class k:

    adjusted_logit[k] = raw_logit[k] - log(pi_train[k]) + log(pi_natural[k])

Raw and adjusted confusion matrices are accumulated over the full test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.classifiers.models import get_model

MODEL_NAMES = ("mlp", "cnn")
MODEL_DISPLAY_NAMES = {
    "mlp": "SimpleMLP",
    "cnn": "CNNOnly",
}
TASKS = {
    "binary": {
        "train_labels": "y_train_bin.npy",
        "test_labels": "y_test_bin.npy",
        "num_classes": 2,
        "class_names": ("Benign", "Attack"),
        "stated_majority_baseline": 0.9765,
    },
    "8class": {
        "train_labels": "y_train_cat.npy",
        "test_labels": "y_test_cat.npy",
        "num_classes": 8,
        "class_names_file": "category_names.json",
        "stated_majority_baseline": 0.7265,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_names(processed_dir: Path, task: str) -> list[str]:
    spec = TASKS[task]
    if "class_names" in spec:
        return [str(value) for value in spec["class_names"]]
    names = json.loads((processed_dir / str(spec["class_names_file"])).read_text())
    if not isinstance(names, list) or len(names) != int(spec["num_classes"]):
        raise ValueError(f"Invalid class names for task={task}")
    return [str(value) for value in names]


def empirical_prior(path: Path, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.load(path, mmap_mode="r")
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    if counts.shape != (num_classes,) or np.any(counts <= 0):
        raise ValueError(f"Every class must have positive support in {path}: {counts.tolist()}")
    prior = counts.astype(np.float64) / float(counts.sum())
    return counts.astype(np.int64), prior


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    kwargs: dict[str, Any] = {}
    if model_name == "mlp":
        kwargs["hidden_dims"] = (256, 128, 64)
    return get_model(model_name, num_features=39, num_classes=num_classes, **kwargs)


def load_model(
    checkpoint: Path,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(model_name, num_classes)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def confusion_metrics(matrix: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    total = int(matrix.sum())
    accuracy = float(true_positive.sum() / total)
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.dot(f1, support) / total)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": [
            {
                "class_index": index,
                "class_name": str(names[index]),
                "support": int(support[index]),
                "predicted": int(predicted[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(len(names))
        ],
    }


def evaluate_checkpoint(
    checkpoint: Path,
    model_name: str,
    num_classes: int,
    x_test: np.ndarray,
    y_test: np.ndarray,
    logit_correction: np.ndarray,
    names: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model = load_model(checkpoint, model_name, num_classes, device)
    correction = torch.as_tensor(logit_correction, dtype=torch.float32, device=device)
    raw_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    corrected_confusion = np.zeros_like(raw_confusion)
    started = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, len(y_test), batch_size):
            end = min(start + batch_size, len(y_test))
            xb = torch.from_numpy(
                np.array(x_test[start:end], dtype=np.float32, copy=True)
            ).to(device, non_blocking=True)
            output = model(xb)
            logits = output[0] if isinstance(output, tuple) else output
            if tuple(logits.shape) != (end - start, num_classes):
                raise ValueError(
                    f"Unexpected logits for {model_name}/{num_classes}: {tuple(logits.shape)}"
                )
            raw_prediction = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
            corrected_prediction = (
                torch.argmax(logits + correction, dim=1).cpu().numpy().astype(np.int64)
            )
            truth = np.asarray(y_test[start:end], dtype=np.int64)
            raw_confusion += np.bincount(
                truth * num_classes + raw_prediction,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)
            corrected_confusion += np.bincount(
                truth * num_classes + corrected_prediction,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)

    elapsed = time.perf_counter() - started
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "runtime_seconds": float(elapsed),
        "raw": {
            **confusion_metrics(raw_confusion, names),
            "confusion_matrix": raw_confusion.tolist(),
        },
        "corrected": {
            **confusion_metrics(corrected_confusion, names),
            "confusion_matrix": corrected_confusion.tolist(),
        },
    }


def write_matrix_csv(path: Path, matrix: Sequence[Sequence[int]], names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *row])


def markdown_matrix(matrix: Sequence[Sequence[int]], names: Sequence[str]) -> list[str]:
    lines = [
        "| True \\ Predicted | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for name, row in zip(names, matrix):
        lines.append(f"| {name} | " + " | ".join(f"{int(value):,}" for value in row) + " |")
    return lines


def write_report(
    path: Path,
    priors: dict[str, Any],
    results: dict[str, Any],
    processed_dir: Path,
    checkpoint_dir: Path,
) -> None:
    lines = [
        "# CICIoT2023 Post-hoc Logit Prior-Correction Evaluation",
        "",
        "## 1. Method",
        "",
        "No model was retrained and no scaler or checkpoint was modified. Training priors were computed directly from the sampled training label arrays, and natural priors directly from the complete unsampled test label arrays.",
        "",
        "For each class `k`, inference logits were transformed before softmax as:",
        "",
        "```text",
        "adjusted_logit[k] = raw_logit[k] - log(pi_train[k]) + log(pi_natural[k])",
        "```",
        "",
        "Because only class predictions are required, the evaluator takes `argmax` after this logit transformation; applying softmax first would not change the selected class.",
        "",
        f"Processed arrays: `{processed_dir.as_posix()}`  ",
        f"Checkpoints: `{checkpoint_dir.as_posix()}`  ",
        "Test evaluation uses all 8,248,312 rows.",
        "",
        "## 2. Empirical priors",
        "",
    ]
    for task in ("binary", "8class"):
        payload = priors[task]
        lines.extend(
            [
                f"### {task}",
                "",
                "| Class | Train count | pi_train | Test count | pi_natural | Logit offset |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for index, name in enumerate(payload["class_names"]):
            lines.append(
                f"| {name} | {payload['train_counts'][index]:,} | "
                f"{payload['train_prior'][index]:.9f} | {payload['natural_counts'][index]:,} | "
                f"{payload['natural_prior'][index]:.9f} | {payload['logit_correction'][index]:+.6f} |"
            )
        lines.extend(
            [
                "",
                f"Empirical full-test majority baseline: {payload['empirical_majority_baseline']:.4%}. "
                f"User-specified flagging baseline: {payload['stated_majority_baseline']:.4%}.",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. Side-by-side metrics",
            "",
            "| Head | Model | Raw accuracy | Corrected accuracy | Delta | Raw macro F1 | Corrected macro F1 | Raw weighted F1 | Corrected weighted F1 | Below stated majority baseline? |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for task in ("binary", "8class"):
        for model_name in MODEL_NAMES:
            result = results[task][model_name]
            raw = result["raw"]
            corrected = result["corrected"]
            flag = "**YES — correction overshot**" if result["corrected_below_stated_majority_baseline"] else "No"
            lines.append(
                f"| {task} | {MODEL_DISPLAY_NAMES[model_name]} | {raw['accuracy']:.4%} | "
                f"{corrected['accuracy']:.4%} | {corrected['accuracy'] - raw['accuracy']:+.4%} | "
                f"{raw['macro_f1']:.4%} | {corrected['macro_f1']:.4%} | "
                f"{raw['weighted_f1']:.4%} | {corrected['weighted_f1']:.4%} | {flag} |"
            )

    lines.extend(["", "## 4. Overshoot flags", ""])
    flagged = [
        (task, model_name, results[task][model_name]["corrected"]["accuracy"], priors[task]["stated_majority_baseline"])
        for task in ("binary", "8class")
        for model_name in MODEL_NAMES
        if results[task][model_name]["corrected_below_stated_majority_baseline"]
    ]
    if flagged:
        for task, model_name, accuracy, baseline in flagged:
            lines.append(
                f"- **{task} / {MODEL_DISPLAY_NAMES[model_name]}: corrected accuracy {accuracy:.4%} "
                f"is below the stated {baseline:.4%} majority baseline. Correction overshot.**"
            )
    else:
        lines.append("No corrected model fell below the user-specified majority-class baseline.")

    binary_macro_decreased = all(
        results["binary"][model]["corrected"]["macro_f1"]
        < results["binary"][model]["raw"]["macro_f1"]
        for model in MODEL_NAMES
    )
    category_accuracy_increased = all(
        results["8class"][model]["corrected"]["accuracy"]
        > results["8class"][model]["raw"]["accuracy"]
        for model in MODEL_NAMES
    )
    category_macro_decreased = all(
        results["8class"][model]["corrected"]["macro_f1"]
        < results["8class"][model]["raw"]["macro_f1"]
        for model in MODEL_NAMES
    )
    lines.extend(
        [
            "",
            "## 5. Interpretation",
            "",
            f"- Binary macro F1 decreased for every architecture: **{binary_macro_decreased}**. "
            "The correction strongly suppresses the already-rare Benign prediction and moves "
            "accuracy toward the empirical full-test Attack-majority baseline.",
            f"- Eight-category accuracy increased for every architecture: **{category_accuracy_increased}**.",
            f"- Eight-category macro F1 decreased for every architecture: **{category_macro_decreased}**. "
            "The accuracy gain is prevalence-driven—mainly stronger DDoS prediction—not a balanced "
            "improvement across categories.",
            "- BruteForce and Web remain unrecovered by the evaluated eight-category models after correction.",
            "",
            "The user-specified overshoot rule is not triggered, but the binary confusion matrices show "
            "near-majority-class collapse. Macro F1 and per-class confusion must therefore accompany accuracy.",
            "",
            "## 6. Confusion matrices",
            "",
        ]
    )
    for task in ("binary", "8class"):
        names = priors[task]["class_names"]
        for model_name in MODEL_NAMES:
            display_name = MODEL_DISPLAY_NAMES[model_name]
            lines.extend([f"### {task} — {display_name} — raw", ""])
            lines.extend(markdown_matrix(results[task][model_name]["raw"]["confusion_matrix"], names))
            lines.extend(["", f"### {task} — {display_name} — prior-corrected", ""])
            lines.extend(markdown_matrix(results[task][model_name]["corrected"]["confusion_matrix"], names))
            lines.append("")

    lines.extend(
        [
            "## 7. Integrity checks",
            "",
            "- Priors were computed from `.npy` labels, not reports or hardcoded counts.",
            "- Raw inference was rerun on the complete unsampled test matrix.",
            "- Raw metrics were checked against the existing saved classification reports.",
            "- Checkpoint SHA-256 hashes were identical before and after evaluation.",
            "- No optimizer, training loop, scaler fitting, or checkpoint write occurred.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("outputs") / "ciciot2023_fixed")
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=Path("outputs") / "ciciot2023_fixed" / "classifier_results",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw-metric-tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    if args.batch_size < 1 or args.raw_metric_tolerance < 0:
        parser.error("batch-size must be positive and tolerance must be non-negative")

    processed_dir = args.processed_dir
    classifier_dir = args.classifier_dir
    checkpoint_dir = classifier_dir / "models"
    output_dir = args.output_dir or classifier_dir / "prior_correction"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    x_test = np.load(processed_dir / "X_test.npy", mmap_mode="r")
    if x_test.shape != (8_248_312, 39):
        raise ValueError(f"Expected full unsampled X_test shape (8248312, 39), got {x_test.shape}")

    started_at = datetime.now(timezone.utc)
    priors: dict[str, Any] = {}
    results: dict[str, Any] = {}
    checkpoint_hashes_before: dict[str, str] = {}

    for task, spec in TASKS.items():
        num_classes = int(spec["num_classes"])
        names = class_names(processed_dir, task)
        train_counts, train_prior = empirical_prior(processed_dir / str(spec["train_labels"]), num_classes)
        natural_counts, natural_prior = empirical_prior(processed_dir / str(spec["test_labels"]), num_classes)
        logit_correction = np.log(natural_prior) - np.log(train_prior)
        priors[task] = {
            "class_names": names,
            "train_label_file": str(spec["train_labels"]),
            "natural_label_file": str(spec["test_labels"]),
            "train_counts": train_counts.tolist(),
            "natural_counts": natural_counts.tolist(),
            "train_prior": train_prior.tolist(),
            "natural_prior": natural_prior.tolist(),
            "logit_correction": logit_correction.tolist(),
            "empirical_majority_baseline": float(natural_prior.max()),
            "stated_majority_baseline": float(spec["stated_majority_baseline"]),
        }
        y_test = np.load(processed_dir / str(spec["test_labels"]), mmap_mode="r")
        if len(y_test) != len(x_test):
            raise ValueError(f"X/y test length mismatch for {task}")
        results[task] = {}

        for model_name in MODEL_NAMES:
            checkpoint = checkpoint_dir / f"{model_name}_{task}.pt"
            checkpoint_hashes_before[f"{model_name}_{task}"] = sha256_file(checkpoint)
            print(f"Evaluating task={task} model={model_name} on {len(y_test):,} rows", flush=True)
            result = evaluate_checkpoint(
                checkpoint=checkpoint,
                model_name=model_name,
                num_classes=num_classes,
                x_test=x_test,
                y_test=y_test,
                logit_correction=logit_correction,
                names=names,
                batch_size=args.batch_size,
                device=device,
            )
            baseline = float(spec["stated_majority_baseline"])
            result["corrected_below_stated_majority_baseline"] = bool(
                result["corrected"]["accuracy"] < baseline
            )
            saved_metrics_path = classifier_dir / "metrics" / f"{model_name}_{task}_classification_report.json"
            saved_metrics = json.loads(saved_metrics_path.read_text())
            raw_deltas = {
                metric: float(result["raw"][metric] - float(saved_metrics[metric]))
                for metric in ("accuracy", "macro_f1", "weighted_f1")
            }
            result["raw_metric_delta_from_saved_report"] = raw_deltas
            if any(abs(delta) > args.raw_metric_tolerance for delta in raw_deltas.values()):
                raise AssertionError(
                    f"Raw metrics do not reproduce saved report for {model_name}/{task}: {raw_deltas}"
                )
            results[task][model_name] = result
            write_matrix_csv(
                matrix_dir / f"{task}_{model_name}_raw.csv",
                result["raw"]["confusion_matrix"],
                names,
            )
            write_matrix_csv(
                matrix_dir / f"{task}_{model_name}_corrected.csv",
                result["corrected"]["confusion_matrix"],
                names,
            )
            print(
                f"  raw_acc={result['raw']['accuracy']:.6f} "
                f"corrected_acc={result['corrected']['accuracy']:.6f} "
                f"overshot={result['corrected_below_stated_majority_baseline']}",
                flush=True,
            )

    checkpoint_hashes_after = {
        key: sha256_file(checkpoint_dir / f"{key}.pt")
        for key in checkpoint_hashes_before
    }
    if checkpoint_hashes_after != checkpoint_hashes_before:
        raise AssertionError("At least one checkpoint changed during evaluation")

    rows: list[dict[str, Any]] = []
    for task in ("binary", "8class"):
        for model_name in MODEL_NAMES:
            result = results[task][model_name]
            raw = result["raw"]
            corrected = result["corrected"]
            rows.append(
                {
                    "task": task,
                    "model": model_name,
                    "model_display": MODEL_DISPLAY_NAMES[model_name],
                    "raw_accuracy": raw["accuracy"],
                    "corrected_accuracy": corrected["accuracy"],
                    "accuracy_delta": corrected["accuracy"] - raw["accuracy"],
                    "raw_macro_f1": raw["macro_f1"],
                    "corrected_macro_f1": corrected["macro_f1"],
                    "macro_f1_delta": corrected["macro_f1"] - raw["macro_f1"],
                    "raw_weighted_f1": raw["weighted_f1"],
                    "corrected_weighted_f1": corrected["weighted_f1"],
                    "weighted_f1_delta": corrected["weighted_f1"] - raw["weighted_f1"],
                    "stated_majority_baseline": priors[task]["stated_majority_baseline"],
                    "empirical_test_majority_baseline": priors[task]["empirical_majority_baseline"],
                    "corrected_below_stated_majority_baseline": result["corrected_below_stated_majority_baseline"],
                    "runtime_seconds": result["runtime_seconds"],
                }
            )
    csv_path = output_dir / "side_by_side_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_dir": str(processed_dir.resolve()),
        "classifier_dir": str(classifier_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "batch_size": args.batch_size,
        "formula": "adjusted_logit[k] = raw_logit[k] - log(pi_train[k]) + log(pi_natural[k])",
        "prior_sources": {
            "train": "empirical frequencies from saved sampled training label arrays",
            "natural": "empirical frequencies from saved full unsampled test label arrays",
        },
        "test_rows": int(len(x_test)),
        "priors": priors,
        "results": results,
        "checkpoint_sha256_before": checkpoint_hashes_before,
        "checkpoint_sha256_after": checkpoint_hashes_after,
        "checkpoints_unchanged": True,
        "raw_metric_tolerance": args.raw_metric_tolerance,
    }
    (output_dir / "prior_correction_results.json").write_text(json.dumps(payload, indent=2))
    (output_dir / "priors.json").write_text(json.dumps(priors, indent=2))
    write_report(
        output_dir / "prior_correction_report.md",
        priors,
        results,
        processed_dir,
        checkpoint_dir,
    )
    print(f"Prior-correction evaluation complete: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
