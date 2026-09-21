"""Train binary and five-category CICIDS2017-DistriNet classifiers.

Each supported architecture is trained once per head:
- binary: Benign, Attack
- category: Benign, DoS, DDoS, Recon, BruteForce

The fine-grained target is intentionally absent. Model selection uses train/validation
only; test data is evaluated after checkpoint selection. Outputs default to
``outputs/cicids2017distrinet``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

from config.paths import SEED
from src.classifiers.models import get_model

NN_MODELS = ("mlp", "cnn", "lstm", "serial")
EFFECTIVE_NUMBER_BETA = 0.999
DISPLAY_NAMES = {
    "mlp": "SimpleMLP",
    "cnn": "CNNOnly",
    "lstm": "LSTMOnly",
    "serial": "SerialCNNLSTM",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target_suffix: str
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


TASKS = (
    TaskSpec("binary", "bin", ("Benign", "Attack")),
    TaskSpec("category", "cat", ("Benign", "DoS", "DDoS", "Recon", "BruteForce")),
)
TASK_BY_NAME = {task.name: task for task in TASKS}


@dataclass(frozen=True)
class RunConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    early_stopping_patience: int
    class_weighting: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/CICIDS_2017_Distrinet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cicids2017distrinet"),
    )
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "effective", "none"),
        default="balanced",
        help="Train-only loss weighting. Balanced is inverse-frequency weighting.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None, help="Category-stratified smoke cap.")
    parser.add_argument("--limit-val", type=int, default=None, help="Category-stratified smoke cap.")
    parser.add_argument("--limit-test", type=int, default=None, help="Category-stratified smoke cap.")
    return parser.parse_args()


def parse_models(value: str) -> list[str]:
    if value.lower() == "all":
        return list(NN_MODELS)
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - set(NN_MODELS))
    if unknown:
        raise ValueError(f"unknown models {unknown}; valid={list(NN_MODELS)}")
    if not models:
        raise ValueError("at least one model is required")
    return models


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("models", "metrics", "predictions", "plots", "histories", "logs"):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    for name in (
        "cicids2017_classifier_results.md",
        "classifier_run_manifest.json",
        "classifier_metrics_summary.csv",
        "per_class_metrics.csv",
        "confusion_matrices.json",
        "dos_ddos_error_audit.csv",
        "fine_label_detection_rates.csv",
        "class_weights.json",
        "effective_number_weights.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("cicids2017d_classifiers")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "logs" / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stratified_limit_indices(y: np.ndarray, limit: int | None, num_classes: int) -> np.ndarray | None:
    if limit is None or limit >= len(y):
        return None
    if limit < num_classes:
        raise ValueError(f"sample limit must be >= {num_classes} to retain every category")
    labels = np.asarray(y, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"cannot stratify absent classes: {counts.tolist()}")
    desired = limit * counts / counts.sum()
    allocation = np.maximum(1, np.floor(desired).astype(np.int64))
    while allocation.sum() > limit:
        candidates = np.flatnonzero(allocation > 1)
        index = candidates[np.argmax(allocation[candidates] - desired[candidates])]
        allocation[index] -= 1
    while allocation.sum() < limit:
        room = counts - allocation
        candidates = np.flatnonzero(room > 0)
        index = candidates[np.argmax(desired[candidates] - allocation[candidates])]
        allocation[index] += 1
    picked: list[np.ndarray] = []
    for label in range(num_classes):
        candidates = np.flatnonzero(labels == label)
        positions = np.linspace(0, len(candidates) - 1, int(allocation[label]), dtype=int)
        picked.append(candidates[positions])
    return np.sort(np.concatenate(picked)).astype(np.int64)


def load_data(processed_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    required = ["preprocessing_manifest.json", "label_encoders.json"]
    for split in ("train", "val", "test"):
        required.extend(
            (f"X_{split}.npy", f"y_{split}_bin.npy", f"y_{split}_cat.npy", f"{split}.parquet")
        )
    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing processed artifacts: {missing}")

    with (processed_dir / "label_encoders.json").open("r", encoding="utf-8") as handle:
        encoders = json.load(handle)
    expected_encoders = {
        "binary": {name: index for index, name in enumerate(TASK_BY_NAME["binary"].class_names)},
        "category": {name: index for index, name in enumerate(TASK_BY_NAME["category"].class_names)},
    }
    if encoders != expected_encoders:
        raise ValueError(f"label encoder mismatch: expected {expected_encoders}, found {encoders}")

    data: dict[str, Any] = {"label_encoders": encoders}
    limits = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}
    for split in ("train", "val", "test"):
        x = np.load(processed_dir / f"X_{split}.npy", mmap_mode="r")
        y_category = np.load(processed_dir / f"y_{split}_cat.npy", mmap_mode="r")
        y_binary = np.load(processed_dir / f"y_{split}_bin.npy", mmap_mode="r")
        if x.ndim != 2 or len({len(x), len(y_category), len(y_binary)}) != 1:
            raise ValueError(
                f"misaligned {split} arrays: X={x.shape}, category={y_category.shape}, binary={y_binary.shape}"
            )
        metadata = pd.read_parquet(processed_dir / f"{split}.parquet", columns=["source_label"])
        source_labels = metadata["source_label"].to_numpy(dtype=str)
        if len(source_labels) != len(x):
            raise ValueError(
                f"misaligned {split} source labels: metadata={len(source_labels)}, X={len(x)}"
            )
        indices = stratified_limit_indices(
            y_category, limits[split], TASK_BY_NAME["category"].num_classes
        )
        if indices is not None:
            x = np.ascontiguousarray(x[indices])
            y_category = np.ascontiguousarray(y_category[indices])
            y_binary = np.ascontiguousarray(y_binary[indices])
            source_labels = source_labels[indices]
        data[f"x_{split}"] = x
        data[f"y_category_{split}"] = np.asarray(y_category, dtype=np.int64)
        data[f"y_binary_{split}"] = np.asarray(y_binary, dtype=np.int64)
        data[f"source_label_{split}"] = source_labels

    if data["x_train"].shape[1] != 79:
        raise ValueError(f"expected 79 DistriNet features, got {data['x_train'].shape[1]}")
    for split in ("train", "val", "test"):
        if not np.isfinite(np.asarray(data[f"x_{split}"])).all():
            raise ValueError(f"{split} features contain NaN/Inf")
        for task in TASKS:
            labels = data[f"y_{task.name}_{split}"]
            expected = set(range(task.num_classes))
            if set(np.unique(labels).tolist()) != expected:
                raise ValueError(f"{split}/{task.name} labels are not {sorted(expected)}")
        derived_binary = (data[f"y_category_{split}"] != 0).astype(np.int64)
        if not np.array_equal(derived_binary, data[f"y_binary_{split}"]):
            raise ValueError(f"{split}: binary labels are inconsistent with category labels")

    with (processed_dir / "preprocessing_manifest.json").open("r", encoding="utf-8") as handle:
        data["preprocessing_manifest"] = json.load(handle)
    return data


def effective_number_class_weights(
    labels: np.ndarray,
    num_classes: int,
    beta: float = EFFECTIVE_NUMBER_BETA,
) -> tuple[np.ndarray, dict[str, float]]:
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"cannot weight empty classes: {counts.tolist()}")
    denominator = -np.expm1(counts * np.log(beta))
    weights = ((1.0 - beta) / denominator).astype(np.float32)
    return weights, {str(index): float(value) for index, value in enumerate(weights)}

def balanced_class_weights(
    labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, dict[str, float]]:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"cannot weight empty classes: {counts.tolist()}")
    weights = (len(labels) / (num_classes * counts)).astype(np.float32)
    return weights, {str(index): float(value) for index, value in enumerate(weights)}


def class_weights_for_labels(
    labels: np.ndarray,
    num_classes: int,
    method: str,
) -> tuple[np.ndarray, dict[str, float]]:
    if method == "balanced":
        return balanced_class_weights(labels, num_classes)
    if method == "effective":
        return effective_number_class_weights(labels, num_classes)
    if method == "none":
        weights = np.ones(num_classes, dtype=np.float32)
        return weights, {str(index): 1.0 for index in range(num_classes)}
    raise ValueError(f"unknown class-weighting method: {method}")


def make_tensor_data(data: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for split in ("train", "val", "test"):
        tensors[f"x_{split}"] = torch.from_numpy(
            np.array(data[f"x_{split}"], dtype=np.float32, copy=True)
        )
        for task in TASKS:
            tensors[f"y_{task.name}_{split}"] = torch.from_numpy(
                np.array(data[f"y_{task.name}_{split}"], dtype=np.int64, copy=True)
            )
    return tensors


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def model_kwargs(model_type: str) -> dict[str, Any]:
    common: dict[str, Any] = {"input_transform": "asinh"}
    if model_type == "mlp":
        return {**common, "hidden_dims": (256, 128, 64)}
    if model_type == "cnn":
        return {**common, "pool_size": 8}
    if model_type == "lstm":
        return {
            **common,
            "feature_sequence": True,
            "feature_embedding_dim": 16,
        }
    return common


def build_nn(model_type: str, num_features: int, num_classes: int) -> nn.Module:
    return get_model(
        model_type,
        num_features=num_features,
        num_classes=num_classes,
        **model_kwargs(model_type),
    )


def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total_loss = 0.0
    total = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            total_loss += float(criterion(logits, yb).item()) * len(yb)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(yb.cpu().numpy())
            total += len(yb)
    return (
        np.concatenate(labels).astype(np.int64),
        np.concatenate(probabilities).astype(np.float32),
        total_loss / max(total, 1),
        time.perf_counter() - started,
    )


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    task: TaskSpec,
    loss: float,
) -> dict[str, Any]:
    y_pred = probabilities.argmax(axis=1).astype(np.int64)
    labels = np.arange(task.num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": int(len(y_true)),
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(task.class_names)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(task.class_names),
            output_dict=True,
            zero_division=0,
        ),
    }


def train_nn(
    model_type: str,
    task: TaskSpec,
    tensors: dict[str, torch.Tensor],
    class_weights: np.ndarray,
    config: RunConfig,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    set_seed(config.seed)
    num_features = int(tensors["x_train"].shape[1])
    model = build_nn(model_type, num_features, task.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=1, factor=0.5
    )
    pin_memory = device.type == "cuda"
    loaders = {
        split: make_loader(
            tensors[f"x_{split}"],
            tensors[f"y_{task.name}_{split}"],
            config.batch_size,
            split == "train",
            config.seed,
            args.num_workers,
            pin_memory,
        )
        for split in ("train", "val", "test")
    }

    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    logger.info("Starting %s/%s on %s", DISPLAY_NAMES[model_type], task.name, device)

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loaders["train"]:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(yb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += len(yb)

        val_true, val_prob, val_loss, val_seconds = predict_probabilities(
            model, loaders["val"], device, criterion
        )
        val_metrics = compute_metrics(val_true, val_prob, task, val_loss)
        scheduler.step(val_metrics["macro_f1"])
        record = {
            "epoch": epoch,
            "train_loss": running_loss / total,
            "train_accuracy": correct / total,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "val_inference_seconds": val_seconds,
        }
        history.append(record)
        logger.info(
            "%s/%s epoch %d/%d train_loss=%.6f train_acc=%.4f val_loss=%.6f val_acc=%.4f val_macro_f1=%.4f",
            DISPLAY_NAMES[model_type], task.name, epoch, config.epochs,
            record["train_loss"], record["train_accuracy"], val_loss,
            val_metrics["accuracy"], val_metrics["macro_f1"],
        )

        improved = (
            val_metrics["macro_f1"] > best_val_macro_f1 + 1e-12
            or (
                abs(val_metrics["macro_f1"] - best_val_macro_f1) <= 1e-12
                and val_loss < best_val_loss
            )
        )
        if improved:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= config.early_stopping_patience:
                logger.info("%s/%s early-stopped after epoch %d", DISPLAY_NAMES[model_type], task.name, epoch)
                break

    if best_state is None:
        raise RuntimeError(f"{model_type}/{task.name}: no checkpoint selected")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started
    val_true, val_prob, val_loss, val_seconds = predict_probabilities(
        model, loaders["val"], device, criterion
    )
    test_true, test_prob, test_loss, test_seconds = predict_probabilities(
        model, loaders["test"], device, criterion
    )
    validation = compute_metrics(val_true, val_prob, task, val_loss)
    test = compute_metrics(test_true, test_prob, task, test_loss)

    stem = f"{model_type}_{task.name}"
    model_path = output_dir / "models" / f"{stem}.pt"
    checkpoint = {
        "state_dict": best_state,
        "model_type": model_type,
        "model_kwargs": model_kwargs(model_type),
        "num_features": num_features,
        "num_classes": task.num_classes,
    }
    torch.save(checkpoint, model_path)
    history_path = output_dir / "histories" / f"{stem}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    reloaded = build_nn(model_type, num_features, task.num_classes).to(device)
    saved = torch.load(model_path, map_location=device, weights_only=True)
    reloaded.load_state_dict(saved["state_dict"])
    reloaded.eval()
    with torch.inference_mode():
        probe = tensors["x_test"][: min(32, len(tensors["x_test"]))].to(device)
        probe_logits = reloaded(probe)
        if isinstance(probe_logits, tuple):
            probe_logits = probe_logits[0]
    if probe_logits.shape != (len(probe), task.num_classes) or not torch.isfinite(probe_logits).all():
        raise AssertionError(f"{stem}: reloaded checkpoint failed output contract")

    return {
        "model": model_type,
        "display_name": DISPLAY_NAMES[model_type],
        "task": task.name,
        "class_names": list(task.class_names),
        "configuration": {
            "num_features": num_features,
            "num_classes": task.num_classes,
            "epochs_requested": config.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "early_stopping_patience": config.early_stopping_patience,
            "optimizer": "Adam",
            "loss": f"{config.class_weighting} weighted CrossEntropyLoss",
            "checkpoint_selection": "validation macro-F1; validation loss tie-break",
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "model_kwargs": model_kwargs(model_type),
            "gradient_clip_norm": 5.0,
        },
        "validation": validation,
        "test": test,
        "timing": {
            "training_seconds": training_seconds,
            "validation_inference_seconds": val_seconds,
            "test_inference_seconds": test_seconds,
        },
        "model_path": str(model_path),
        "history_path": str(history_path),
        "val_probability": val_prob,
        "test_probability": test_prob,
        "checkpoint_reload_verified": True,
    }


def save_model_outputs(result: dict[str, Any], data: dict[str, Any], output_dir: Path) -> None:
    stem = f"{result['model']}_{result['task']}"
    task = TASK_BY_NAME[str(result["task"])]
    for split in ("val", "test"):
        y_true = np.asarray(data[f"y_{task.name}_{split}"], dtype=np.int64)
        probability = np.asarray(result[f"{split}_probability"], dtype=np.float32)
        y_pred = probability.argmax(axis=1).astype(np.int8)
        np.savez_compressed(
            output_dir / "predictions" / f"{stem}_{split}_predictions.npz",
            y_true=y_true.astype(np.int8),
            y_pred=y_pred,
            probabilities=probability,
            class_names=np.asarray(task.class_names),
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=np.arange(task.num_classes),
            target_names=list(task.class_names),
            zero_division=0,
        )
        (output_dir / "metrics" / f"{stem}_{split}_classification_report.txt").write_text(
            report, encoding="utf-8"
        )
    serializable = {
        key: value for key, value in result.items() if not key.endswith("_probability")
    }
    (output_dir / "metrics" / f"{stem}_metrics.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )


def plot_confusion(result: dict[str, Any], output_dir: Path) -> None:
    matrix = np.asarray(result["test"]["confusion_matrix"], dtype=np.int64)
    names = list(result["class_names"])
    size = 5.5 if len(names) == 2 else 7.5
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{result['display_name']} {result['task']} test confusion matrix")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(names)):
        for column in range(len(names)):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    stem = f"{result['model']}_{result['task']}"
    fig.savefig(output_dir / "plots" / f"{stem}_test_confusion.png", dpi=220)
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "task": result["task"],
            "model": result["model"],
            "display_name": result["display_name"],
            "num_classes": len(result["class_names"]),
            "training_seconds": result["timing"]["training_seconds"],
            "validation_inference_seconds": result["timing"]["validation_inference_seconds"],
            "test_inference_seconds": result["timing"]["test_inference_seconds"],
        }
        for split in ("validation", "test"):
            for metric in ("loss", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
                row[f"{split}_{metric}"] = result[split][metric]
        rows.append(row)
    return rows


def per_class_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for split in ("validation", "test"):
            for class_name, metrics in result[split]["per_class"].items():
                rows.append(
                    {
                        "task": result["task"],
                        "model": result["model"],
                        "display_name": result["display_name"],
                        "split": split,
                        "class": class_name,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "support": metrics["support"],
                    }
                )
    return rows

def dos_ddos_audit_rows(
    results: Sequence[dict[str, Any]],
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    category_results = [result for result in results if result["task"] == "category"]
    source_labels = np.asarray(data["source_label_test"], dtype=str)
    y_true = np.asarray(data["y_category_test"], dtype=np.int64)
    for result in category_results:
        y_pred = np.asarray(result["test_probability"]).argmax(axis=1)
        for true_id, true_name, other_id in ((1, "DoS", 2), (2, "DDoS", 1)):
            mask = y_true == true_id
            rows.append(
                {
                    "model": result["model"],
                    "display_name": result["display_name"],
                    "scope": true_name,
                    "support": int(mask.sum()),
                    "recall": float((y_pred[mask] == true_id).mean()),
                    "predicted_as_other_rate": float((y_pred[mask] == other_id).mean()),
                    "predicted_as_benign_rate": float((y_pred[mask] == 0).mean()),
                }
            )
        for source_label in sorted(np.unique(source_labels[y_true == 1])):
            mask = (y_true == 1) & (source_labels == source_label)
            rows.append(
                {
                    "model": result["model"],
                    "display_name": result["display_name"],
                    "scope": source_label,
                    "support": int(mask.sum()),
                    "recall": float((y_pred[mask] == 1).mean()),
                    "predicted_as_other_rate": float((y_pred[mask] == 2).mean()),
                    "predicted_as_benign_rate": float((y_pred[mask] == 0).mean()),
                }
            )
    return rows


def format_percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def write_markdown_report(
    output_dir: Path,
    results: Sequence[dict[str, Any]],
    data: dict[str, Any],
    weights: dict[str, dict[str, float]],
    run_config: RunConfig,
    device: torch.device,
    started_at: str,
    finished_at: str,
    dos_ddos_rows: Sequence[dict[str, Any]],
) -> Path:
    lines = [
        "# CIC-IDS-2017 DistriNet classifier results",
        "",
        "## Experiment",
        "",
        "All four supported neural architectures were trained independently for exactly two heads: binary and five-category. No fine-grained target, output layer, checkpoint, or report is produced.",
        "",
        f"- Started: `{started_at}`",
        f"- Finished: `{finished_at}`",
        f"- Device: `{device}`",
        f"- Features: **{data['x_train'].shape[1]}**",
        f"- Rows: train **{len(data['x_train']):,}**, validation **{len(data['x_val']):,}**, test **{len(data['x_test']):,}**",
        "- Category mapping/encoding: `Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4`.",
        "- Binary encoding: `Benign=0, Attack=1`.",
        f"- Loss class weighting: `{run_config.class_weighting}` using training labels only.",
        "- Checkpoint selection uses validation macro-F1 with validation loss as tie-break; test is held out until selection.",
        "",
    ]

    for task in TASKS:
        task_results = [result for result in results if result["task"] == task.name]
        ordered = sorted(task_results, key=lambda item: float(item["test"]["macro_f1"]), reverse=True)
        counts = {
            split: {
                task.class_names[int(key)]: int(value)
                for key, value in Counter(data[f"y_{task.name}_{split}"]).items()
            }
            for split in ("train", "val", "test")
        }
        named_weights = {
            task.class_names[int(key)]: value for key, value in weights[task.name].items()
        }
        lines.extend(
            [
                f"## {task.name.title()} head",
                "",
                f"- Train counts: `{counts['train']}`",
                f"- Validation counts: `{counts['val']}`",
                f"- Test counts: `{counts['test']}`",
                f"- Training-only `{run_config.class_weighting}` weights: `{named_weights}`",
                "",
                "### Test summary",
                "",
                "| Rank | Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Train time | Test inference |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, result in enumerate(ordered, start=1):
            metrics = result["test"]
            lines.append(
                f"| {rank} | {result['display_name']} | {format_percent(metrics['accuracy'])} | "
                f"{format_percent(metrics['balanced_accuracy'])} | {format_percent(metrics['macro_f1'])} | "
                f"{format_percent(metrics['weighted_f1'])} | {result['timing']['training_seconds']:.1f}s | "
                f"{result['timing']['test_inference_seconds']:.2f}s |"
            )

        lines.extend(
            [
                "",
                "### Test per-class precision, recall, and F1",
                "",
                "| Model | Class | Support | Precision | Recall | F1 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for result in ordered:
            for class_name in task.class_names:
                metrics = result["test"]["per_class"][class_name]
                lines.append(
                    f"| {result['display_name']} | {class_name} | {metrics['support']:,} | "
                    f"{metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} |"
                )

        lines.extend(["", "### Test confusion matrices", ""])
        for result in ordered:
            matrix = np.asarray(result["test"]["confusion_matrix"], dtype=np.int64)
            lines.extend(
                [
                    f"#### {result['display_name']}",
                    "",
                    "Rows are true classes; columns are predicted classes.",
                    "",
                    "| True \\ Predicted | " + " | ".join(task.class_names) + " |",
                    "|---|" + "---:|" * task.num_classes,
                ]
            )
            for index, class_name in enumerate(task.class_names):
                lines.append(
                    f"| {class_name} | " + " | ".join(f"{value:,}" for value in matrix[index]) + " |"
                )
            lines.append("")

    lines.extend(["## DoS/DDoS error audit", ""])
    lines.extend(
        [
            "Rates are row-normalized within each true class/source label.",
            "",
            "| Model | Scope | Support | Recall | Predicted as paired class | Predicted as Benign |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in dos_ddos_rows:
        lines.append(
            f"| {row['display_name']} | {row['scope']} | {row['support']:,} | "
            f"{row['recall']:.6f} | {row['predicted_as_other_rate']:.6f} | "
            f"{row['predicted_as_benign_rate']:.6f} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Output artifacts",
            "",
            "- `models/`: packaged state, architecture kwargs, and dimensions per architecture/head.",
            "- `metrics/`: JSON metrics and validation/test text classification reports.",
            "- `predictions/`: aligned labels, hard predictions, full probability matrices, and class names.",
            "- `plots/`: test confusion-matrix images for both heads.",
            "- `histories/`: per-epoch training histories.",
            "- `classifier_metrics_summary.csv`: aggregate validation/test metrics.",
            "- `per_class_metrics.csv`: precision, recall, F1, and support for both heads.",
            "- `confusion_matrices.json`: numeric validation/test matrices for both heads.",
            "- `dos_ddos_error_audit.csv`: normalized DoS/DDoS and DoS-subtype error rates.",
            "- `label_encoders.json`: exact label-to-ID mappings used by training.",
            "- `class_weights.json`: training counts, method, formula, and exact weights used.",
            "- `classifier_run_manifest.json`: configuration and provenance.",
            "- `logs/training.log`: timestamped training log.",
            "",
            "## Verification",
            "",
            "Every saved checkpoint was reloaded into its source architecture and exercised on held-out feature rows. Output widths were verified as 2 for binary and 5 for category; all probe logits were finite.",
            "",
            "## Run configuration",
            "",
            "```json",
            json.dumps(asdict(run_config), indent=2),
            "```",
            "",
        ]
    )
    path = output_dir / "cicids2017_classifier_results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    device = resolve_device(args.device)
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    prepare_output_dir(output_dir)
    logger = setup_logging(output_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    config = RunConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.patience,
        class_weighting=args.class_weighting,
    )

    data = load_data(processed_dir, args)
    (output_dir / "label_encoders.json").write_text(
        json.dumps(data["label_encoders"], indent=2) + "\n", encoding="utf-8"
    )
    weights: dict[str, np.ndarray] = {}
    weight_maps: dict[str, dict[str, float]] = {}
    weight_counts: dict[str, dict[str, int]] = {}
    for task in TASKS:
        labels = data[f"y_{task.name}_train"]
        weights[task.name], weight_maps[task.name] = class_weights_for_labels(
            labels, task.num_classes, args.class_weighting
        )
        counts = np.bincount(labels, minlength=task.num_classes)
        weight_counts[task.name] = {
            task.class_names[index]: int(count)
            for index, count in enumerate(counts)
        }

    class_weight_payload = {
        "method": args.class_weighting,
        "beta": EFFECTIVE_NUMBER_BETA if args.class_weighting == "effective" else None,
        "formula": {
            "balanced": "n_samples / (n_classes * class_count)",
            "effective": "(1 - beta) / (1 - beta ** class_count)",
            "none": "1",
        }[args.class_weighting],
        "normalization": None,
        "heads": {
            task.name: {
                "class_counts": weight_counts[task.name],
                "weights": {
                    task.class_names[int(index)]: value
                    for index, value in weight_maps[task.name].items()
                },
            }
            for task in TASKS
        },
    }
    (output_dir / "class_weights.json").write_text(
        json.dumps(class_weight_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Models: %s", models)
    logger.info("Heads: %s", [task.name for task in TASKS])
    logger.info("Device: %s", device)
    logger.info(
        "Shapes train=%s val=%s test=%s",
        data["x_train"].shape,
        data["x_val"].shape,
        data["x_test"].shape,
    )
    logger.info(
        "Training-only %s class weights: %s",
        args.class_weighting,
        weight_maps,
    )

    tensors = make_tensor_data(data)
    results: list[dict[str, Any]] = []
    for task in TASKS:
        for model_type in models:
            result = train_nn(
                model_type,
                task,
                tensors,
                weights[task.name],
                config,
                args,
                device,
                output_dir,
                logger,
            )
            save_model_outputs(result, data, output_dir)
            plot_confusion(result, output_dir)
            logger.info(
                "Completed %s/%s: test_acc=%.6f macro_f1=%.6f weighted_f1=%.6f",
                DISPLAY_NAMES[model_type], task.name, result["test"]["accuracy"],
                result["test"]["macro_f1"], result["test"]["weighted_f1"],
            )
            results.append(result)

    write_csv(output_dir / "classifier_metrics_summary.csv", summary_rows(results))
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows(results))
    dos_ddos_rows = dos_ddos_audit_rows(results, data)
    write_csv(output_dir / "dos_ddos_error_audit.csv", dos_ddos_rows)
    confusion_payload = {
        f"{result['model']}_{result['task']}": {
            "class_names": result["class_names"],
            "validation": result["validation"]["confusion_matrix"],
            "test": result["test"]["confusion_matrix"],
        }
        for result in results
    }
    (output_dir / "confusion_matrices.json").write_text(
        json.dumps(confusion_payload, indent=2), encoding="utf-8"
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    report_path = write_markdown_report(
        output_dir,
        results,
        data,
        weight_maps,
        config,
        device,
        started_at,
        finished_at,
        dos_ddos_rows,
    )
    manifest = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "models": models,
        "model_display_names": {model: DISPLAY_NAMES[model] for model in models},
        "heads": {
            task.name: {"num_classes": task.num_classes, "classes": list(task.class_names)}
            for task in TASKS
        },
        "features": int(data["x_train"].shape[1]),
        "split_rows": {
            split: int(len(data[f"x_{split}"])) for split in ("train", "val", "test")
        },
        "split_class_counts": {
            task.name: {
                split: {
                    task.class_names[int(key)]: int(value)
                    for key, value in Counter(data[f"y_{task.name}_{split}"]).items()
                }
                for split in ("train", "val", "test")
            }
            for task in TASKS
        },
        "limits": {
            "train": args.limit_train,
            "val": args.limit_val,
            "test": args.limit_test,
        },
        "training_only_class_weights": weight_maps,
        "class_weight_method": args.class_weighting,
        "class_weight_formula": class_weight_payload["formula"],
        "class_weight_beta": class_weight_payload["beta"],
        "class_weight_normalization": None,
        "model_kwargs": {model: model_kwargs(model) for model in models},
        "run_config": asdict(config),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "preprocessing_manifest": str(processed_dir / "preprocessing_manifest.json"),
        "preprocessing_manifest_sha256": sha256_file(processed_dir / "preprocessing_manifest.json"),
        "label_encoders_sha256": sha256_file(processed_dir / "label_encoders.json"),
        "result_report": str(report_path),
        "checkpoint_reload_verified": all(result["checkpoint_reload_verified"] for result in results),
    }
    (output_dir / "classifier_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info("All classifier runs complete")
    logger.info("Results report: %s", report_path)


if __name__ == "__main__":
    main()
