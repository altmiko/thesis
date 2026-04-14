"""
Baseline adversarial sample generation for CICIoT2023 using torchattacks.

This script creates untargeted PGD and C&W adversarial examples for the
existing CNN-LSTM NIDS model and writes a preview of generated samples to:
  - terminal output
  - generated_baseline_samples.txt

Outputs are saved under results/baseline by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import torchattacks
except ImportError as exc:
    raise SystemExit(
        "torchattacks is required. Install with: pip install torchattacks"
    ) from exc


LOGGER = logging.getLogger("baseline_attacks")


class SimpleCNNLSTM(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


class UnitIntervalToStandardizedModel(nn.Module):
    """Wraps the classifier so attacks operate in [0,1] feature space."""

    def __init__(
        self,
        base_model: nn.Module,
        feature_min: torch.Tensor,
        feature_range: torch.Tensor,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("feature_min", feature_min)
        self.register_buffer("feature_range", feature_range)

    def forward(self, x_unit: torch.Tensor) -> torch.Tensor:
        x_std = x_unit * self.feature_range + self.feature_min
        return self.base_model(x_std)


@dataclass(frozen=True)
class AttackConfig:
    pgd_eps: float
    pgd_alpha: float
    pgd_steps: int
    cw_c: float
    cw_kappa: float
    cw_steps: int
    cw_lr: float


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_label_encoder(root: Path):
    import pickle

    candidate_paths = [
        root / "models" / "nids" / "label_encoder.pkl",
        root / "data" / "processed" / "label_encoder.pkl",
    ]
    for path in candidate_paths:
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(
        "label_encoder.pkl not found under models/nids or data/processed"
    )


def load_feature_names(root: Path, num_features: int) -> list[str]:
    import pickle

    feature_path = root / "data" / "processed" / "feature_names.pkl"
    if feature_path.exists():
        with feature_path.open("rb") as f:
            payload = pickle.load(f)
        names = payload.get("all_features") if isinstance(payload, dict) else None
        if isinstance(names, list) and len(names) == num_features:
            return [str(name) for name in names]

    return [f"feature_{i}" for i in range(num_features)]


def load_passthrough_indices(root: Path, num_features: int) -> np.ndarray:
    import pickle

    feature_path = root / "data" / "processed" / "feature_names.pkl"
    if feature_path.exists():
        with feature_path.open("rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict):
            binary_idx = np.asarray(payload.get("binary_idx", []), dtype=np.int64)
            passthrough_idx = np.asarray(payload.get("passthrough_idx", []), dtype=np.int64)
            idx = np.union1d(binary_idx, passthrough_idx)
            idx = idx[(idx >= 0) & (idx < num_features)]
            if idx.size > 0:
                return idx

    # Backward-compatible fallback for older metadata.
    return np.arange(23, 36, dtype=np.int64)


def load_data(
    root: Path,
    max_samples: int,
    shared_indices_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = root / "data" / "processed"
    x_train_path = data_dir / "X_train.npy"
    x_test_path = data_dir / "X_test.npy"
    y_test_path = data_dir / "y_test.npy"

    for required in (x_train_path, x_test_path, y_test_path):
        if not required.exists():
            raise FileNotFoundError(f"Required file missing: {required}")

    x_train = np.load(x_train_path).astype(np.float32)
    x_test = np.load(x_test_path).astype(np.float32)
    y_test_raw = np.load(y_test_path, allow_pickle=True)

    n_total = x_test.shape[0]
    target_n = n_total if max_samples <= 0 else min(max_samples, n_total)

    selected_idx = None
    if shared_indices_path.exists():
        loaded_idx = np.load(shared_indices_path).astype(np.int64)
        valid = (loaded_idx >= 0) & (loaded_idx < n_total)
        loaded_idx = loaded_idx[valid]
        loaded_idx = np.unique(loaded_idx)
        if loaded_idx.shape[0] == target_n:
            selected_idx = loaded_idx
            LOGGER.info("Using shared evaluation indices from %s", shared_indices_path)
        else:
            LOGGER.warning(
                "Shared index count (%d) does not match requested sample size (%d); regenerating.",
                loaded_idx.shape[0],
                target_n,
            )

    if selected_idx is None:
        if target_n >= n_total:
            selected_idx = np.random.permutation(n_total)
        else:
            # Seeded stratified subsampling for representative class coverage.
            labels, y_indices = np.unique(y_test_raw, return_inverse=True)
            n_classes = labels.shape[0]
            base_per_class = target_n // n_classes
            remainder = target_n % n_classes

            per_class_idx = []
            leftovers = []

            for class_id in range(n_classes):
                idx = np.where(y_indices == class_id)[0]
                idx = idx[np.random.permutation(idx.shape[0])]
                take = min(base_per_class, idx.shape[0])
                per_class_idx.append(idx[:take])
                if take < idx.shape[0]:
                    leftovers.append(idx[take:])

            selected_idx = np.concatenate(per_class_idx) if per_class_idx else np.empty((0,), dtype=np.int64)

            if remainder > 0 and leftovers:
                pool = np.concatenate(leftovers)
                pool = pool[np.random.permutation(pool.shape[0])]
                selected_idx = np.concatenate([selected_idx, pool[:remainder]])

            if selected_idx.shape[0] < target_n:
                all_idx = np.arange(n_total)
                chosen_mask = np.zeros(n_total, dtype=bool)
                chosen_mask[selected_idx] = True
                pool = all_idx[~chosen_mask]
                pool = pool[np.random.permutation(pool.shape[0])]
                need = target_n - selected_idx.shape[0]
                selected_idx = np.concatenate([selected_idx, pool[:need]])

            selected_idx = selected_idx[np.random.permutation(selected_idx.shape[0])]

        np.save(shared_indices_path, selected_idx.astype(np.int64))
        LOGGER.info("Saved shared evaluation indices to %s", shared_indices_path)

    x_test = x_test[selected_idx]
    y_test_raw = y_test_raw[selected_idx]

    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Input data has NaN/Inf values; clean data before attack generation")

    return x_train, x_test, y_test_raw, selected_idx


def load_model(root: Path, num_features: int, num_classes: int, device: torch.device) -> nn.Module:
    model_path = root / "models" / "nids" / "nids_cnnlstm.pt"
    if not model_path.exists():
        fallback = root / "models" / "nids" / "nids_cnnlstm_best.pt"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError("Could not find nids_cnnlstm.pt or nids_cnnlstm_best.pt")

    model = SimpleCNNLSTM(num_features=num_features, num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def to_unit_interval(
    x_train: np.ndarray,
    x: np.ndarray,
    passthrough_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_min = x_train.min(axis=0)
    feature_max = x_train.max(axis=0)
    feature_range = feature_max - feature_min
    feature_range = np.where(feature_range < 1e-8, 1.0, feature_range)

    x_unit = np.clip((x - feature_min) / feature_range, 0.0, 1.0)
    # Keep metadata-defined passthrough features unscaled only if they naturally live in [0,1].
    # This preserves binary-like features while avoiding out-of-range nominal codes in unit-space.
    safe_idx = passthrough_idx[
        (feature_min[passthrough_idx] >= 0.0) & (feature_max[passthrough_idx] <= 1.0)
    ]
    if safe_idx.size > 0:
        x_unit[:, safe_idx] = x[:, safe_idx]
    return x_unit.astype(np.float32), feature_min.astype(np.float32), feature_range.astype(np.float32)


def compute_attack_metrics(
    clean_unit: np.ndarray,
    adv_unit: np.ndarray,
    y_true: np.ndarray,
    model_wrapper: nn.Module,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    def batched_predict(x_np: np.ndarray) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for start in range(0, x_np.shape[0], batch_size):
                end = min(start + batch_size, x_np.shape[0])
                x_batch = torch.from_numpy(x_np[start:end]).to(device)
                pred = model_wrapper(x_batch).argmax(dim=1).cpu().numpy()
                preds.append(pred)
        return np.concatenate(preds, axis=0)

    adv_pred = batched_predict(adv_unit)
    clean_pred = batched_predict(clean_unit)

    success_mask = adv_pred != y_true
    clean_correct_mask = clean_pred == y_true

    delta = adv_unit - clean_unit
    l2 = np.linalg.norm(delta, axis=1)
    linf = np.max(np.abs(delta), axis=1)

    asr_on_all = float(success_mask.mean())
    if clean_correct_mask.any():
        asr_on_clean_correct = float((success_mask & clean_correct_mask).sum() / clean_correct_mask.sum())
    else:
        asr_on_clean_correct = 0.0

    return {
        "clean_acc": float(clean_correct_mask.mean()),
        "asr_all": asr_on_all,
        "asr_clean_correct": asr_on_clean_correct,
        "l2_mean": float(l2.mean()),
        "linf_mean": float(linf.mean()),
    }


def run_attack(
    attack_name: str,
    attack,
    x_unit: np.ndarray,
    y_encoded: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    adv_batches = []
    n = x_unit.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = torch.from_numpy(x_unit[start:end]).to(device)
        y_batch = torch.from_numpy(y_encoded[start:end]).long().to(device)

        # cuDNN RNN backward can fail in eval mode; disable cuDNN only for attack step.
        with torch.backends.cudnn.flags(enabled=False):
            adv_batch = attack(x_batch, y_batch).detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(adv_batch).all():
            raise ValueError(f"{attack_name} produced NaN/Inf values at batch {start}:{end}")
        adv_batches.append(adv_batch)

    return np.concatenate(adv_batches, axis=0)


def format_preview(
    x_clean_unit: np.ndarray,
    x_adv_pgd: np.ndarray,
    x_adv_cw: np.ndarray,
    feature_names: list[str],
    preview_count: int,
) -> str:
    def format_feature_changes(clean_row: np.ndarray, adv_row: np.ndarray, label: str) -> list[str]:
        delta = adv_row - clean_row
        changed_idx = np.where(np.abs(delta) > 1e-9)[0]
        lines = [f"  {label} changed_features={changed_idx.shape[0]}"]
        if changed_idx.shape[0] == 0:
            lines.append("    (no changes)")
            return lines

        for idx in changed_idx.tolist():
            lines.append(
                "    "
                f"{feature_names[idx]}: "
                f"{clean_row[idx]:.6f} -> {adv_row[idx]:.6f} "
                f"(delta={delta[idx]:+.6f})"
            )
        return lines

    lines = []
    count = min(preview_count, x_clean_unit.shape[0])
    for i in range(count):
        lines.append(f"Sample #{i}")
        lines.append(
            f"  Original(all) = "
            f"{np.array2string(x_clean_unit[i], precision=6, separator=', ')}"
        )
        lines.append(f"  Clean[:8] = {np.array2string(x_clean_unit[i, :8], precision=4, separator=', ')}")
        lines.append(f"  PGD[:8]   = {np.array2string(x_adv_pgd[i, :8], precision=4, separator=', ')}")
        lines.append(f"  CW[:8]    = {np.array2string(x_adv_cw[i, :8], precision=4, separator=', ')}")
        lines.extend(format_feature_changes(x_clean_unit[i], x_adv_pgd[i], "PGD"))
        lines.extend(format_feature_changes(x_clean_unit[i], x_adv_cw[i], "CW"))
        lines.append("")
    return "\n".join(lines).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline PGD/C&W attacks for CICIoT2023")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pgd-eps", type=float, default=0.08)
    parser.add_argument("--pgd-alpha", type=float, default=0.01)
    parser.add_argument("--pgd-steps", type=int, default=40)

    parser.add_argument("--cw-c", type=float, default=1.0)
    parser.add_argument("--cw-kappa", type=float, default=0.0)
    parser.add_argument("--cw-steps", type=int, default=100)
    parser.add_argument("--cw-lr", type=float, default=0.01)

    parser.add_argument(
        "--output-samples-file",
        type=Path,
        default=Path("generated_baseline_samples.txt"),
        help="Text file with sample previews and metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "baseline",
        help="Directory for generated adversarial arrays and metadata",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Loading CICIoT2023 processed arrays from %s", root / "data" / "processed")

    shared_indices_path = output_dir / "shared_eval_indices.npy"
    x_train, x_test, y_test_raw, selected_idx = load_data(
        root=root,
        max_samples=args.n_samples,
        shared_indices_path=shared_indices_path,
    )
    label_encoder = load_label_encoder(root)
    y_test = label_encoder.transform(y_test_raw).astype(np.int64)

    num_features = x_test.shape[1]
    num_classes = len(label_encoder.classes_)
    feature_names = load_feature_names(root, num_features)
    passthrough_idx = load_passthrough_indices(root, num_features)
    LOGGER.info("Loaded %d evaluation samples | features=%d | classes=%d", x_test.shape[0], num_features, num_classes)

    base_model = load_model(root=root, num_features=num_features, num_classes=num_classes, device=device)

    x_test_unit, feat_min, feat_range = to_unit_interval(
        x_train=x_train,
        x=x_test,
        passthrough_idx=passthrough_idx,
    )
    model_wrapper = UnitIntervalToStandardizedModel(
        base_model=base_model,
        feature_min=torch.from_numpy(feat_min).to(device),
        feature_range=torch.from_numpy(feat_range).to(device),
    ).to(device)
    model_wrapper.eval()

    config = AttackConfig(
        pgd_eps=args.pgd_eps,
        pgd_alpha=args.pgd_alpha,
        pgd_steps=args.pgd_steps,
        cw_c=args.cw_c,
        cw_kappa=args.cw_kappa,
        cw_steps=args.cw_steps,
        cw_lr=args.cw_lr,
    )

    pgd = torchattacks.PGD(
        model_wrapper,
        eps=config.pgd_eps,
        alpha=config.pgd_alpha,
        steps=config.pgd_steps,
        random_start=True,
    )
    cw = torchattacks.CW(
        model_wrapper,
        c=config.cw_c,
        kappa=config.cw_kappa,
        steps=config.cw_steps,
        lr=config.cw_lr,
    )

    LOGGER.info("Running PGD attack...")
    x_adv_pgd = run_attack("PGD", pgd, x_test_unit, y_test, args.batch_size, device)

    LOGGER.info("Running C&W attack...")
    x_adv_cw = run_attack("CW", cw, x_test_unit, y_test, args.batch_size, device)

    pgd_metrics = compute_attack_metrics(
        x_test_unit,
        x_adv_pgd,
        y_test,
        model_wrapper,
        device,
        args.batch_size,
    )
    cw_metrics = compute_attack_metrics(
        x_test_unit,
        x_adv_cw,
        y_test,
        model_wrapper,
        device,
        args.batch_size,
    )

    np.save(output_dir / "x_clean_unit.npy", x_test_unit)
    np.save(output_dir / "y_test_encoded.npy", y_test)
    np.save(output_dir / "adv_pgd_unit.npy", x_adv_pgd)
    np.save(output_dir / "adv_cw_unit.npy", x_adv_cw)
    np.save(output_dir / "feat_min.npy", feat_min)
    np.save(output_dir / "feat_range.npy", feat_range)
    np.save(output_dir / "shared_eval_indices.npy", selected_idx.astype(np.int64))

    report = {
        "dataset": "CICIoT2023 (processed split from data/processed)",
        "n_samples": int(x_test_unit.shape[0]),
        "device": str(device),
        "classes": [str(c) for c in label_encoder.classes_],
        "config": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "pgd": {
                "eps": config.pgd_eps,
                "alpha": config.pgd_alpha,
                "steps": config.pgd_steps,
            },
            "cw": {
                "c": config.cw_c,
                "kappa": config.cw_kappa,
                "steps": config.cw_steps,
                "lr": config.cw_lr,
            },
        },
        "metrics": {
            "pgd": pgd_metrics,
            "cw": cw_metrics,
        },
    }

    with (output_dir / "baseline_attack_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    preview_text = format_preview(
        x_clean_unit=x_test_unit,
        x_adv_pgd=x_adv_pgd,
        x_adv_cw=x_adv_cw,
        feature_names=feature_names,
        preview_count=args.preview_count,
    )

    summary_lines = [
        "Baseline adversarial generation using torchattacks",
        f"Root: {root}",
        f"Samples: {x_test_unit.shape[0]}",
        f"Device: {device}",
        "",
        "Metrics",
        f"  PGD -> clean_acc={pgd_metrics['clean_acc']:.4f}, asr_all={pgd_metrics['asr_all']:.4f}, "
        f"asr_clean_correct={pgd_metrics['asr_clean_correct']:.4f}, l2_mean={pgd_metrics['l2_mean']:.5f}, "
        f"linf_mean={pgd_metrics['linf_mean']:.5f}",
        f"  CW  -> clean_acc={cw_metrics['clean_acc']:.4f}, asr_all={cw_metrics['asr_all']:.4f}, "
        f"asr_clean_correct={cw_metrics['asr_clean_correct']:.4f}, l2_mean={cw_metrics['l2_mean']:.5f}, "
        f"linf_mean={cw_metrics['linf_mean']:.5f}",
        "",
        "Preview samples (unit-space features)",
        preview_text,
        "",
        f"Array outputs saved to: {output_dir}",
        f"JSON report saved to: {output_dir / 'baseline_attack_report.json'}",
    ]

    final_text = "\n".join(summary_lines)
    print(final_text)

    output_samples_path = (root / args.output_samples_file).resolve()
    output_samples_path.write_text(final_text + "\n", encoding="utf-8")

    LOGGER.info("Saved sample preview text to %s", output_samples_path)
    LOGGER.info("Saved numpy outputs and report under %s", output_dir)


if __name__ == "__main__":
    main()
