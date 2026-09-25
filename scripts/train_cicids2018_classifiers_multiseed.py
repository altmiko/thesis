#!/usr/bin/env python3
"""Train and aggregate CICIDS2018-DistriNet neural classifiers across independent seeds.

Each seed gets an isolated output directory containing checkpoints, predictions, metrics,
confusion matrices and a run manifest produced by ``cicids2017d_experiments.py``. This driver
then writes per-run and mean/sample-standard-deviation summaries across training seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed" / "CSECICIDS_2018_Distrinet"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "cicids2018distrinet" / "classifiers_multiseed"
DEFAULT_SEEDS = (42, 123, 2024)
MODELS = ("mlp", "cnn", "ft_transformer")
MODEL_NAMES = {"mlp": "SimpleMLP", "cnn": "CNNOnly", "ft_transformer": "FTTransformer"}
TASKS = ("binary", "category")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
)


def parse_csv(value: str, allowed: Iterable[str], label: str) -> list[str]:
    allowed_set = set(allowed)
    if value.lower() == "all":
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - allowed_set)
    if unknown or not selected:
        raise ValueError(f"invalid {label}: selected={selected}, unknown={unknown}, allowed={sorted(allowed_set)}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--models", default="all")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ft-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ft-weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Retrain complete seed directories.")
    args = parser.parse_args()
    args.seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.models = parse_csv(args.models, MODELS, "models")
    args.tasks = parse_csv(args.tasks, TASKS, "tasks")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("--seeds must contain distinct integers")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("--epochs, --patience and --batch-size must be >= 1")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_stems(models: list[str], tasks: list[str]) -> list[str]:
    return [f"{model}_{task}" for task in tasks for model in models]


def complete_run(path: Path, seed: int, models: list[str], tasks: list[str]) -> bool:
    manifest_path = path / "classifier_run_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("dataset_id") != "cicids2018_distrinet":
        return False
    if manifest.get("run_config", {}).get("seed") != seed:
        return False
    if sorted(manifest.get("models", [])) != sorted(models):
        return False
    for stem in expected_stems(models, tasks):
        if not (path / "models" / f"{stem}.pt").exists() or not (path / "metrics" / f"{stem}_metrics.json").exists():
            return False
    return bool(manifest.get("checkpoint_reload_verified"))


def seed_command(args: argparse.Namespace, seed: int, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "src" / "classifiers" / "cicids2017d_experiments.py"),
        "--processed-dir",
        str(args.processed_dir.resolve()),
        "--output-dir",
        str(run_dir.resolve()),
        "--dataset-id",
        "cicids2018_distrinet",
        "--models",
        ",".join(args.models),
        "--tasks",
        ",".join(args.tasks),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--ft-learning-rate",
        str(args.ft_learning_rate),
        "--ft-weight-decay",
        str(args.ft_weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(seed),
    ]
    for name in ("train", "val", "test"):
        limit = getattr(args, f"limit_{name}")
        if limit is not None:
            command.extend((f"--limit-{name}", str(limit)))
    return command


def train_seed(args: argparse.Namespace, seed: int, run_dir: Path) -> list[str]:
    command = seed_command(args, seed, run_dir)
    env = os.environ.copy()
    path_entries = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if env.get("PYTHONPATH"):
        path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    print(f"Training seed {seed}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return command


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def collect(args: argparse.Namespace, runs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    matrices: dict[tuple[str, str, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for seed in args.seeds:
        run_dir = runs_dir / f"seed_{seed}"
        for stem in expected_stems(args.models, args.tasks):
            payload = read_json(run_dir / "metrics" / f"{stem}_metrics.json")
            model, task = payload["model"], payload["task"]
            for split in ("validation", "test"):
                metrics = payload[split]
                row: dict[str, Any] = {
                    "seed": seed,
                    "model": model,
                    "display_name": payload["display_name"],
                    "task": task,
                    "split": split,
                    "n": metrics["n"],
                    "loss": metrics["loss"],
                    **{metric: metrics[metric] for metric in METRICS},
                    "best_epoch": payload["configuration"]["best_epoch"],
                    "epochs_completed": payload["configuration"]["epochs_completed"],
                    "training_seconds": payload["timing"]["training_seconds"],
                }
                run_rows.append(row)
                for class_name, values in metrics["per_class"].items():
                    class_rows.append(
                        {
                            "seed": seed,
                            "model": model,
                            "display_name": payload["display_name"],
                            "task": task,
                            "split": split,
                            "class": class_name,
                            "support": values["support"],
                            "precision": values["precision"],
                            "recall": values["recall"],
                            "f1": values["f1"],
                        }
                    )
                matrices[(model, task, split)].append(
                    (seed, np.asarray(metrics["confusion_matrix"], dtype=np.float64))
                )
    matrix_payload: dict[str, Any] = {}
    for (model, task, split), records in matrices.items():
        stack = np.stack([matrix for _, matrix in records])
        matrix_payload[f"{model}_{task}_{split}"] = {
            "seeds": [seed for seed, _ in records],
            "per_seed": {str(seed): matrix.astype(int).tolist() for seed, matrix in records},
            "mean": stack.mean(axis=0).tolist(),
            "sample_std": stack.std(axis=0, ddof=1).tolist() if len(stack) > 1 else np.zeros_like(stack[0]).tolist(),
        }
    return run_rows, class_rows, matrix_payload


def aggregate_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["model"], row["task"], row["split"])].append(row)
    output: list[dict[str, Any]] = []
    for (model, task, split), rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "model": model,
            "display_name": rows[0]["display_name"],
            "task": task,
            "split": split,
            "n_seeds": len(rows),
            "seeds": ";".join(str(row["seed"]) for row in rows),
            "n_rows_per_seed": rows[0]["n"],
        }
        for metric in METRICS:
            aggregate[f"{metric}_mean"], aggregate[f"{metric}_sample_std"] = mean_std(row[metric] for row in rows)
        aggregate["training_seconds_mean"], aggregate["training_seconds_sample_std"] = mean_std(
            row["training_seconds"] for row in rows
        )
        output.append(aggregate)
    return output


def aggregate_class_rows(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in class_rows:
        grouped[(row["model"], row["task"], row["split"], row["class"])].append(row)
    output: list[dict[str, Any]] = []
    for (model, task, split, class_name), rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "model": model,
            "display_name": rows[0]["display_name"],
            "task": task,
            "split": split,
            "class": class_name,
            "support_per_seed": rows[0]["support"],
            "n_seeds": len(rows),
        }
        for metric in ("precision", "recall", "f1"):
            aggregate[f"{metric}_mean"], aggregate[f"{metric}_sample_std"] = mean_std(
                row[metric] for row in rows
            )
        output.append(aggregate)
    return output


def source_label_recall(args: argparse.Namespace, runs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Category-head test recall per original source label (subtype), per run and aggregated."""
    if "category" not in args.tasks:
        return [], []
    import pandas as pd

    meta = pd.read_parquet(args.processed_dir / "test.parquet", columns=["source_label"])
    labels = meta["source_label"].to_numpy(dtype=str)
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        for model in args.models:
            path = runs_dir / f"seed_{seed}" / "predictions" / f"{model}_category_test_predictions.npz"
            payload = np.load(path)
            y_true, y_pred = payload["y_true"], payload["y_pred"]
            class_names = [str(name) for name in payload["class_names"]]
            if len(y_true) != len(labels):
                raise AssertionError(f"{path}: predictions not aligned with test.parquet")
            for source in sorted(set(labels)):
                mask = labels == source
                wrong = y_pred[mask] != y_true[mask]
                errors = np.bincount(y_pred[mask][wrong], minlength=len(class_names))
                runs.append(
                    {
                        "seed": seed,
                        "model": model,
                        "display_name": MODEL_NAMES[model],
                        "source_label": source,
                        "support": int(mask.sum()),
                        "recall": float((~wrong).mean()),
                        "errors": int(wrong.sum()),
                        "errors_by_predicted_class": ";".join(
                            f"{class_names[i]}={int(n)}" for i, n in enumerate(errors) if n
                        ),
                    }
                )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row["source_label"], row["model"])].append(row)
    aggregate = []
    for (source, model), rows in sorted(grouped.items()):
        mean, std = mean_std(row["recall"] for row in rows)
        aggregate.append(
            {
                "source_label": source,
                "model": model,
                "display_name": MODEL_NAMES[model],
                "support_per_seed": rows[0]["support"],
                "recall_mean": mean,
                "recall_sample_std": std,
                "errors_per_seed": ";".join(str(row["errors"]) for row in rows),
            }
        )
    return runs, aggregate


def plot_mean_confusions(matrix_payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, payload in matrix_payload.items():
        if not key.endswith("_test"):
            continue
        matrix = np.asarray(payload["mean"])
        task = "binary" if matrix.shape[0] == 2 else "category"
        names = ("Benign", "Attack") if task == "binary" else ("Benign", "DoS", "DDoS", "Recon", "BruteForce")
        fig, ax = plt.subplots(figsize=(6 if len(names) == 2 else 8, 6 if len(names) == 2 else 8))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{key.removesuffix('_test')} mean test confusion matrix ({len(payload['seeds'])} seeds)")
        threshold = matrix.max() / 2
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{matrix[i, j]:,.1f}", ha="center", va="center", color="white" if matrix[i, j] > threshold else "black", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"{key}_mean_confusion.png", dpi=180)
        plt.close(fig)


def pct(mean: float, std: float) -> str:
    return f"{100 * mean:.3f}% ± {100 * std:.3f}%"


def write_report(
    path: Path,
    args: argparse.Namespace,
    aggregate: list[dict[str, Any]],
    per_class: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    manifests: dict[int, dict[str, Any]],
    source_recall: list[dict[str, Any]],
) -> None:
    test_rows = [row for row in aggregate if row["split"] == "test"]
    class_counts = next(iter(manifests.values()))["split_class_counts"]["category"]
    classes = ("Benign", "DoS", "DDoS", "Recon", "BruteForce")
    lines = [
        "# CICIDS2018-DistriNet multi-seed classifier results",
        "",
        f"Independent classifier-training seeds: **{args.seeds}**. Values are mean ± sample standard deviation across seeds.",
        "",
        "## Experimental contract",
        "",
        f"- Models: {', '.join(MODEL_NAMES[model] for model in args.models)}.",
        f"- Heads: {', '.join(args.tasks)}.",
        f"- Requested epochs: {args.epochs}; early-stopping patience: {args.patience}; batch size: {args.batch_size}.",
        "- Loss: train-only balanced inverse-frequency CrossEntropyLoss.",
        "- Model selection: validation macro-F1, validation loss tie-break. Test is evaluated only after checkpoint selection.",
        "- Inputs: 79 RobustScaler-space features with model-side `asinh` transform.",
        "- Class composition is a controlled design (see preprocessing manifest); these metrics do not estimate natural CICIDS2018 prevalence.",
        "- Softmax outputs are not interpreted as naturally calibrated real-world probabilities.",
        "",
        "## Data used",
        "",
        "| Split | " + " | ".join(classes) + " | Total |",
        "|---|" + "---:|" * (len(classes) + 1),
    ]
    for split in ("train", "val", "test"):
        counts = [int(class_counts[split].get(name, 0)) for name in classes]
        lines.append(f"| {split} | " + " | ".join(f"{v:,}" for v in counts) + f" | {sum(counts):,} |")
    lines.extend(
        [
            "",
            "## Test metrics",
            "",
            "| Head | Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(test_rows, key=lambda item: (item["task"], -item["macro_f1_mean"])):
        lines.append(
            f"| {row['task']} | {row['display_name']} | {pct(row['accuracy_mean'], row['accuracy_sample_std'])} | "
            f"{pct(row['balanced_accuracy_mean'], row['balanced_accuracy_sample_std'])} | "
            f"{pct(row['macro_precision_mean'], row['macro_precision_sample_std'])} | "
            f"{pct(row['macro_recall_mean'], row['macro_recall_sample_std'])} | "
            f"{pct(row['macro_f1_mean'], row['macro_f1_sample_std'])} | "
            f"{pct(row['weighted_f1_mean'], row['weighted_f1_sample_std'])} |"
        )
    lines.extend(["", "## Test per-class metrics", ""])
    for task in args.tasks:
        lines.extend(
            [
                f"### {task.title()}",
                "",
                "| Model | Class | Support/seed | Precision | Recall | F1 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        rows = [row for row in per_class if row["split"] == "test" and row["task"] == task]
        for row in sorted(rows, key=lambda item: (item["model"], item["class"])):
            lines.append(
                f"| {row['display_name']} | {row['class']} | {row['support_per_seed']:,} | "
                f"{pct(row['precision_mean'], row['precision_sample_std'])} | "
                f"{pct(row['recall_mean'], row['recall_sample_std'])} | "
                f"{pct(row['f1_mean'], row['f1_sample_std'])} |"
            )
        lines.append("")
    if source_recall:
        models = [model for model in args.models]
        lines.extend(
            [
                "## Test recall per source label (category head)",
                "",
                "Class-level recall can hide subtype failures; errors per seed are listed in seed order.",
                "",
                "| Source label | Support/seed | " + " | ".join(MODEL_NAMES[m] for m in models) + " |",
                "|---|---:|" + "---:|" * len(models),
            ]
        )
        by_key = {(row["source_label"], row["model"]): row for row in source_recall}
        for source in sorted({row["source_label"] for row in source_recall}):
            cells = []
            for model in models:
                row = by_key[(source, model)]
                cells.append(f"{pct(row['recall_mean'], row['recall_sample_std'])} (err {row['errors_per_seed']})")
            lines.append(f"| {source} | {by_key[(source, models[0])]['support_per_seed']:,} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "## Per-run test metrics",
            "",
            "| Seed | Head | Model | Accuracy | Macro precision | Macro recall | Macro F1 | Best epoch | Epochs completed |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted((row for row in run_rows if row["split"] == "test"), key=lambda item: (item["seed"], item["task"], item["model"])):
        lines.append(
            f"| {row['seed']} | {row['task']} | {row['display_name']} | {row['accuracy']:.6f} | "
            f"{row['macro_precision']:.6f} | {row['macro_recall']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['best_epoch']} | {row['epochs_completed']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Macro metrics are primary because they weight every class equally.",
            "- Per-class recall exposes failures hidden by aggregate accuracy.",
            "- Ordinary accuracy is supplementary: the class composition is a controlled experimental design.",
            "- Across-seed standard deviations quantify model-training variability for these three initialisations only; they do not establish broader dataset or campaign robustness.",
            "",
            "## Verification and artifacts",
            "",
            "Every checkpoint was reloaded by the training program and exercised on held-out feature rows; output widths and finite logits were asserted.",
            "",
            "- `runs/seed_<seed>/models/`: seed-specific checkpoints for every model/head.",
            "- `runs/seed_<seed>/metrics/`, `predictions/`, `histories/`, `plots/`: complete per-seed evidence.",
            "- `per_run_metrics.csv`: metrics for every seed/model/head/split.",
            "- `multiseed_summary.csv`: mean and sample standard deviation by model/head/split.",
            "- `multiseed_per_class.csv`: per-class precision, recall and F1 mean/sample standard deviation.",
            "- `aggregate_confusion_matrices.json`: exact per-seed and elementwise mean/std matrices.",
            "- `plots/`: mean test confusion matrices.",
            "- `multiseed_manifest.json`: source hashes, commands, environment and checkpoint verification.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.processed_dir = args.processed_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.processed_dir.is_dir():
        raise FileNotFoundError(args.processed_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    commands: dict[str, list[str]] = {}
    for seed in args.seeds:
        run_dir = runs_dir / f"seed_{seed}"
        if not args.force and complete_run(run_dir, seed, args.models, args.tasks):
            print(f"Seed {seed}: complete; reusing {run_dir}", flush=True)
            commands[str(seed)] = seed_command(args, seed, run_dir)
            continue
        commands[str(seed)] = train_seed(args, seed, run_dir)
        if not complete_run(run_dir, seed, args.models, args.tasks):
            raise AssertionError(f"seed {seed}: training returned without complete verified artifacts")

    run_rows, class_rows, matrices = collect(args, runs_dir)
    aggregate = aggregate_rows(run_rows)
    aggregate_class = aggregate_class_rows(class_rows)
    write_csv(args.output_dir / "per_run_metrics.csv", run_rows)
    write_csv(args.output_dir / "per_run_per_class.csv", class_rows)
    write_csv(args.output_dir / "multiseed_summary.csv", aggregate)
    write_csv(args.output_dir / "multiseed_per_class.csv", aggregate_class)
    (args.output_dir / "aggregate_confusion_matrices.json").write_text(
        json.dumps(matrices, indent=2) + "\n", encoding="utf-8"
    )
    plot_mean_confusions(matrices, args.output_dir / "plots")
    source_runs, source_recall = source_label_recall(args, runs_dir)
    write_csv(args.output_dir / "per_run_source_label_recall.csv", source_runs)
    write_csv(args.output_dir / "multiseed_source_label_recall.csv", source_recall)
    manifests = {
        seed: read_json(runs_dir / f"seed_{seed}" / "classifier_run_manifest.json") for seed in args.seeds
    }
    report = args.output_dir / "cicids2018_multiseed_classifier_results.md"
    write_report(report, args, aggregate, aggregate_class, run_rows, manifests, source_recall)
    manifest = {
        "dataset_id": "cicids2018_distrinet",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "models": args.models,
        "tasks": args.tasks,
        "training_protocol": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "ft_learning_rate": args.ft_learning_rate,
            "ft_weight_decay": args.ft_weight_decay,
            "class_weighting": "balanced, train only",
            "checkpoint_selection": "validation macro-F1; validation loss tie-break",
        },
        "limits": {name: getattr(args, f"limit_{name}") for name in ("train", "val", "test")},
        "processed_dir": str(args.processed_dir),
        "preprocessing_manifest_sha256": sha256(args.processed_dir / "preprocessing_manifest.json"),
        "label_encoders_sha256": sha256(args.processed_dir / "label_encoders.json"),
        "commands": commands,
        "seed_run_manifests": {
            str(seed): {
                "path": str(runs_dir / f"seed_{seed}" / "classifier_run_manifest.json"),
                "sha256": sha256(runs_dir / f"seed_{seed}" / "classifier_run_manifest.json"),
                "checkpoint_reload_verified": manifests[seed]["checkpoint_reload_verified"],
                "device": manifests[seed]["device"],
                "gpu": manifests[seed]["gpu"],
            }
            for seed in args.seeds
        },
        "aggregation": "mean and sample standard deviation (ddof=1) across independent classifier-training seeds",
        "report": str(report),
        "all_checkpoints_reload_verified": all(manifests[seed]["checkpoint_reload_verified"] for seed in args.seeds),
    }
    (args.output_dir / "multiseed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Multi-seed report: {report}", flush=True)


if __name__ == "__main__":
    main()
