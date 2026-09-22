"""Train per-attack-class typed beta-VAEs for CICIDS2017-DistriNet.

Stage A is victim-independent. VAE weights are fit on TRAIN rows of one attack class;
validation rows select the checkpoint and calibrate the Mahalanobis realism gate.
Test data is never loaded here.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from constraints.engine import ConstraintEngine
from constraints.layer0 import Layer0Projector
from datasets.cicids2017 import CICIDS2017Adapter
from vae.losses import BetaScheduler, compute_manifold_elbo
from vae.model import MixedInputBetaVAE

ATTACK_CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")


@dataclass(frozen=True)
class StageAConfig:
    latent_dim: int = 16
    encoder_input_transform: str = "asinh"
    epochs: int = 30
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    beta_target: float = 0.5
    beta_warmup_epochs: int = 10
    free_bits_lambda: float = 0.1
    continuous_likelihood: str = "laplace"
    grad_clip: float = 5.0
    seed: int = 42


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _class_array(split, class_id: int) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(split.y) == class_id)
    return np.ascontiguousarray(np.asarray(split.x[idx]), dtype=np.float32)


def _loader(x: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _fit_loss_feature_weights(x_train: np.ndarray) -> np.ndarray:
    """TRAIN-fit RMS weights prevent zero-IQR rare columns dominating the ELBO."""
    rms = np.sqrt(np.mean(np.square(x_train, dtype=np.float64), axis=0))
    weights = 1.0 / np.maximum(rms, 1.0)
    return (weights / weights.mean()).astype(np.float32)


def _mean_loss(
    model,
    loader,
    engine,
    config: StageAConfig,
    feature_weights: torch.Tensor,
    device: str,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            terms = compute_manifold_elbo(
                xb,
                out,
                engine,
                beta=config.beta_target,
                continuous_likelihood=config.continuous_likelihood,
                free_bits_lambda=config.free_bits_lambda,
                constraint_l1_weight=0.0,
                constraint_l2_weight=0.0,
                continuous_feature_weights=feature_weights,
            )
            size = len(xb)
            totals["loss"] += float(terms["loss"]) * size
            totals["recon"] += float(terms["recon_continuous"]) * size
            totals["kl"] += float(terms["kl"]) * size
            n += size
    return {key: value / n for key, value in totals.items()}


def _encode(model, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    loader = _loader(x, batch_size, False, 42)
    encoded: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            mu, _ = model.encode(xb.to(device, non_blocking=True))
            encoded.append(mu.cpu().numpy())
    return np.concatenate(encoded, axis=0)


def fit_idr(model, x_val: np.ndarray, batch_size: int, device: str) -> dict:
    """Fit the val-anchored 95% Mahalanobis gate used by IDR/True-IDSR."""
    latent = _encode(model, x_val, batch_size, device).astype(np.float64)
    mean = latent.mean(axis=0)
    covariance = np.cov(latent - mean, rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance_reg = covariance + 1e-4 * np.eye(covariance.shape[0])
    precision = np.linalg.inv(covariance_reg)
    delta = latent - mean
    distance_sq = np.einsum("ni,ij,nj->n", delta, precision, delta)
    threshold = float(np.quantile(distance_sq, 0.95, method="higher"))
    return {
        "mean": mean,
        "covariance": covariance,
        "precision": precision,
        "threshold_sq": threshold,
        "threshold_calibration": "val_empirical_p95",
        "val_in_distribution_rate": float((distance_sq <= threshold).mean()),
        "fit_rows": int(len(latent)),
    }


def train_class(
    adapter: CICIDS2017Adapter,
    class_name: str,
    output_dir: Path,
    config: StageAConfig,
    device: str,
) -> dict:
    mapping = adapter.class_mapping()
    if class_name not in ATTACK_CLASSES:
        raise ValueError(f"Stage A class must be one of {ATTACK_CLASSES}, got {class_name!r}")
    class_id = mapping.name_to_id[class_name]
    train_split = adapter.load_split("train")
    val_split = adapter.load_split("val")
    x_train = _class_array(train_split, class_id)
    x_val = _class_array(val_split, class_id)

    manifest = adapter.feature_manifest()
    transform = adapter.feature_transform()
    model = MixedInputBetaVAE(
        manifest=manifest,
        latent_dim=config.latent_dim,
        encoder_input_transform=config.encoder_input_transform,
    ).to(device)
    model.register_feature_transform(transform)
    engine = ConstraintEngine(
        manifest,
        layer0=Layer0Projector(manifest),
        active_layers=set(),
    )
    train_loader = _loader(x_train, config.batch_size, True, config.seed)
    val_loader = _loader(x_val, config.batch_size, False, config.seed)
    loss_feature_weights = _fit_loss_feature_weights(x_train)
    loss_feature_weights_tensor = torch.tensor(
        loss_feature_weights, dtype=torch.float32, device=device
    )
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = BetaScheduler(
        config.beta_target,
        total_steps=config.epochs * steps_per_epoch,
        warmup_steps=config.beta_warmup_epochs * steps_per_epoch,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            beta = scheduler.step()
            terms = compute_manifold_elbo(
                xb,
                out,
                engine,
                beta=beta,
                continuous_likelihood=config.continuous_likelihood,
                free_bits_lambda=config.free_bits_lambda,
                constraint_l1_weight=0.0,
                constraint_l2_weight=0.0,
                continuous_feature_weights=loss_feature_weights_tensor,
            )
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_loss += float(terms["loss"]) * len(xb)
            seen += len(xb)

        val = _mean_loss(
            model,
            val_loader,
            engine,
            config,
            loss_feature_weights_tensor,
            device,
        )
        row = {
            "epoch": epoch,
            "beta": scheduler.current_beta,
            "train_loss": train_loss / seen,
            "val_loss": val["loss"],
            "val_recon": val["recon"],
            "val_kl": val["kl"],
        }
        history.append(row)
        print(json.dumps({"class": class_name, **row}), flush=True)
        if val["loss"] < best_val_loss - 1e-6:
            best_val_loss = val["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    if best_state is None:
        raise RuntimeError(f"{class_name}: no Stage-A checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"vae_{class_name}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "class_name": class_name,
            "class_id": class_id,
            "manifest_hash": manifest.content_hash,
            "model_config": {
                "latent_dim": config.latent_dim,
                "encoder_hidden": (128, 64),
                "decoder_hidden": (64, 128),
                "decoder_kind": "typed",
                "encoder_input_transform": config.encoder_input_transform,
            },
            "training_config": asdict(config),
            "loss_feature_weights": loss_feature_weights.tolist(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_rows": len(x_train),
            "val_rows": len(x_val),
        },
        checkpoint_path,
    )

    idr = fit_idr(model, x_val, config.batch_size, device)
    idr_path = output_dir / f"idr_{class_name}.npz"
    np.savez(
        idr_path,
        mean=idr["mean"],
        covariance=idr["covariance"],
        precision=idr["precision"],
        threshold_calibration=np.asarray(idr["threshold_calibration"]),
        threshold_sq=np.asarray(idr["threshold_sq"]),
        val_in_distribution_rate=np.asarray(idr["val_in_distribution_rate"]),
        fit_rows=np.asarray(idr["fit_rows"]),
    )
    return {
        "class_name": class_name,
        "class_id": class_id,
        "checkpoint": str(checkpoint_path),
        "idr": str(idr_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_rows": len(x_train),
        "val_rows": len(x_val),
        "val_in_distribution_rate": idr["val_in_distribution_rate"],
        "history": history,
    }


def load_stage_a(
    adapter: CICIDS2017Adapter,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MixedInputBetaVAE, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    manifest = adapter.feature_manifest()
    manifest.assert_compatible_hash(checkpoint["manifest_hash"], context=str(checkpoint_path))
    cfg = checkpoint["model_config"]
    model = MixedInputBetaVAE(
        manifest=manifest,
        latent_dim=int(cfg["latent_dim"]),
        encoder_hidden=tuple(cfg["encoder_hidden"]),
        decoder_hidden=tuple(cfg["decoder_hidden"]),
        decoder_kind=str(cfg["decoder_kind"]),
        encoder_input_transform=str(cfg.get("encoder_input_transform", "none")),
    ).to(device)
    model.register_feature_transform(adapter.feature_transform())
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", default=",".join(ATTACK_CLASSES))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cicids2017_vae_attacks/stage_a"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    config = StageAConfig(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
    _seed(config.seed)
    adapter = CICIDS2017Adapter()
    results = [train_class(adapter, name, args.output_dir, config, args.device) for name in classes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stage_a_summary.json").write_text(
        json.dumps({"config": asdict(config), "classes": results}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
