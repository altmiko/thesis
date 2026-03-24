"""
train_vae.py
Mixed-input VAE for CICIoT2023 adversarial attack pipeline.

Trains one VAE per attack class (DDoS, DoS, Mirai, Recon).
Each VAE learns the manifold of its attack class for use in
latent-space adversarial example generation.

Architecture:
    Input (37 features) split into:
        - Continuous (24 features) → MSE reconstruction loss
        - Binary (13 features)     → BCE reconstruction loss
    Encoder: MLP → (mu, log_var) in latent space (dim=16)
    Decoder: MLP → continuous head (MSE) + binary head (sigmoid)

Usage:
    python train_vae.py
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────
DATA_DIR = r"D:\thesis\data\processed"
MODEL_DIR = r"D:\thesis\models\vae"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Feature split — indices into the 37-feature vector (Telnet & IGMP dropped)
# Continuous: 0-22 + 36 (Protocol Type) = 24 features
# Binary: 23-35 = 13 features
CONTINUOUS_IDX = list(range(0, 23)) + [36]  # 24 continuous
BINARY_IDX = list(range(23, 36))  # 13 binary
NUM_CONTINUOUS = len(CONTINUOUS_IDX)  # 24
NUM_BINARY = len(BINARY_IDX)  # 13
INPUT_DIM = NUM_CONTINUOUS + NUM_BINARY  # 37

# Global VAE hyperparameters
LATENT_DIM = 16
DROPOUT = 0.2
BATCH_SIZE = 512
LR = 1e-3
LAMBDA_REG = 1e-6
MAX_SAMPLES = 100000

# ── Per-class config ──────────────────────────────────────────────
# Recon gets a bigger network and more epochs due to higher intra-class variance
PER_CLASS_CONFIG = {
    "DDoS": {"hidden_dims": [128, 64], "epochs": 50, "kl_weight": 1.5},
    "DoS": {"hidden_dims": [128, 64], "epochs": 50, "kl_weight": 1.5},
    "Mirai": {"hidden_dims": [128, 64], "epochs": 50, "kl_weight": 1.5},
    "Recon": {"hidden_dims": [256, 128], "epochs": 100, "kl_weight": 1.5},
}

ATTACK_CLASSES = ["DDoS", "DoS", "Mirai", "Recon"]

print(f"Device: {DEVICE}")
print(f"Continuous features: {NUM_CONTINUOUS}, Binary features: {NUM_BINARY}")
print(f"Latent dim: {LATENT_DIM}")


# ── VAE ARCHITECTURE ─────────────────────────────────────────────
class MixedInputVAE(nn.Module):
    """
    Mixed-input VAE for IoT network traffic tabular data.
    - Continuous features: MSE reconstruction loss
    - Binary features: BCE reconstruction loss (sigmoid output)
    One instance trained per attack class.
    """

    def __init__(
        self,
        num_continuous: int,
        num_binary: int,
        hidden_dims: list,
        latent_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_continuous = num_continuous
        self.num_binary = num_binary
        self.input_dim = num_continuous + num_binary
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # ── Encoder ───────────────────────────────────────────────
        enc_layers = []
        in_dim = self.input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)

        # ── Decoder ───────────────────────────────────────────────
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        self.decoder = nn.Sequential(*dec_layers)

        # Separate output heads
        self.cont_head = nn.Linear(in_dim, num_continuous)
        self.binary_head = nn.Sequential(nn.Linear(in_dim, num_binary), nn.Sigmoid())

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        return self.cont_head(h), self.binary_head(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        cont_recon, binary_recon = self.decode(z)
        return cont_recon, binary_recon, mu, log_var

    def get_latent(self, x):
        """Get mu (deterministic) for attack pipeline."""
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


# ── LOSS FUNCTION ─────────────────────────────────────────────────
def vae_loss(x, cont_recon, binary_recon, mu, log_var, kl_weight=1.0):
    """
    Combined VAE loss:
        - MSE for continuous features
        - BCE for binary features
        - KL divergence (beta-annealed)
    """
    x_cont = x[:, :NUM_CONTINUOUS]
    x_binary = x[:, NUM_CONTINUOUS:].clamp(0.0, 1.0)

    cont_loss = F.mse_loss(cont_recon, x_cont, reduction="sum")
    binary_loss = F.binary_cross_entropy(binary_recon, x_binary, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total = cont_loss + binary_loss + kl_weight * kl_loss
    return total, cont_loss, binary_loss, kl_loss


# ── TRAINING FUNCTION ─────────────────────────────────────────────
def train_vae(
    attack_class: str,
    X_class: np.ndarray,
    hidden_dims: list,
    epochs: int,
    kl_weight: float,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> MixedInputVAE:

    print(f"\n{'=' * 60}")
    print(f"Training VAE for class: {attack_class}")
    print(
        f"  Samples: {X_class.shape[0]} | Hidden: {hidden_dims} | "
        f"Epochs: {epochs} | KL weight: {kl_weight}"
    )
    print(f"{'=' * 60}")

    # Reorder features: continuous first, binary last
    X_reordered = np.concatenate(
        [X_class[:, CONTINUOUS_IDX], X_class[:, BINARY_IDX]], axis=1
    ).astype(np.float32)

    # Verify binary features are clean
    assert X_reordered[:, NUM_CONTINUOUS:].min() >= 0.0
    assert X_reordered[:, NUM_CONTINUOUS:].max() <= 1.0

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_reordered)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    model = MixedInputVAE(
        num_continuous=NUM_CONTINUOUS,
        num_binary=NUM_BINARY,
        hidden_dims=hidden_dims,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=LAMBDA_REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_loss = float("inf")
    best_state = None
    history = {"total": [], "cont": [], "binary": [], "kl": []}

    pbar = tqdm(range(1, epochs + 1), desc=f"VAE [{attack_class}]")
    for epoch in pbar:
        model.train()
        epoch_total = epoch_cont = epoch_binary = epoch_kl = 0.0
        n_samples = 0

        # KL annealing: ramp from 0 → kl_weight over first 50% of epochs
        beta = min(kl_weight, kl_weight * (epoch / (epochs * 0.5)))

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            cont_recon, binary_recon, mu, log_var = model(batch)
            loss, cont_l, binary_l, kl_l = vae_loss(
                batch, cont_recon, binary_recon, mu, log_var, kl_weight=beta
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = batch.size(0)
            epoch_total += loss.item()
            epoch_cont += cont_l.item()
            epoch_binary += binary_l.item()
            epoch_kl += kl_l.item()
            n_samples += bs

        scheduler.step()

        avg_total = epoch_total / n_samples
        avg_cont = epoch_cont / n_samples
        avg_binary = epoch_binary / n_samples
        avg_kl = epoch_kl / n_samples

        history["total"].append(avg_total)
        history["cont"].append(avg_cont)
        history["binary"].append(avg_binary)
        history["kl"].append(avg_kl)

        pbar.set_postfix(
            {
                "loss": f"{avg_total:.4f}",
                "cont": f"{avg_cont:.4f}",
                "bin": f"{avg_binary:.4f}",
                "kl": f"{avg_kl:.4f}",
                "beta": f"{beta:.3f}",
            }
        )

        if avg_total < best_loss:
            best_loss = avg_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    save_path = os.path.join(MODEL_DIR, f"vae_{attack_class.lower()}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "attack_class": attack_class,
            "num_continuous": NUM_CONTINUOUS,
            "num_binary": NUM_BINARY,
            "latent_dim": LATENT_DIM,
            "hidden_dims": hidden_dims,
            "continuous_idx": CONTINUOUS_IDX,
            "binary_idx": BINARY_IDX,
            "best_loss": best_loss,
            "history": history,
        },
        save_path,
    )

    print(f"\n  Best loss: {best_loss:.4f} | Saved to: {save_path}")
    return model


# ── RECONSTRUCTION EVALUATION ─────────────────────────────────────
@torch.no_grad()
def evaluate_reconstruction(
    model: MixedInputVAE, X_class: np.ndarray, attack_class: str
):
    model.eval()
    X_reordered = np.concatenate(
        [X_class[:, CONTINUOUS_IDX], X_class[:, BINARY_IDX]], axis=1
    ).astype(np.float32)

    # Evaluate in batches to avoid OOM on large classes
    batch_size = 4096
    all_cont_recon, all_binary_recon, all_mu, all_log_var = [], [], [], []
    x_full = torch.from_numpy(X_reordered)

    for i in range(0, len(x_full), batch_size):
        batch = x_full[i : i + batch_size].to(DEVICE)
        cr, br, mu, lv = model(batch)
        all_cont_recon.append(cr.cpu())
        all_binary_recon.append(br.cpu())
        all_mu.append(mu.cpu())
        all_log_var.append(lv.cpu())

    cont_recon = torch.cat(all_cont_recon)
    binary_recon = torch.cat(all_binary_recon)
    mu = torch.cat(all_mu)
    log_var = torch.cat(all_log_var)
    x_tensor = x_full

    # Continuous MAE
    x_cont = x_tensor[:, :NUM_CONTINUOUS]
    cont_mae = F.l1_loss(cont_recon, x_cont).item()

    # Binary accuracy (threshold at 0.5)
    x_binary = x_tensor[:, NUM_CONTINUOUS:].clamp(0, 1)
    bin_pred = (binary_recon > 0.5).float()
    bin_acc = (bin_pred == x_binary.round()).float().mean().item()

    # KL
    kl = (
        -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x_tensor.size(0)
    ).item()

    print(f"\n  [{attack_class}] Reconstruction quality:")
    print(f"    Continuous MAE:  {cont_mae:.6f}")
    print(f"    Binary accuracy: {bin_acc:.4f} ({bin_acc * 100:.1f}%)")
    print(f"    KL divergence:   {kl:.4f}")

    return {"cont_mae": cont_mae, "bin_acc": bin_acc, "kl": kl}


# ── MAIN ──────────────────────────────────────────────────────────
def main():
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train_raw = np.load(os.path.join(DATA_DIR, "y_train.npy"), allow_pickle=True)

    with open(os.path.join(r"D:\thesis\models\nids", "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    y_train = le.transform(y_train_raw)
    print(f"  Total training samples: {X_train.shape[0]}")
    print(f"  Classes: {le.classes_}")

    class_to_idx = {c: i for i, c in enumerate(le.classes_)}
    results = {}

    for attack_class in ATTACK_CLASSES:
        if attack_class not in class_to_idx:
            print(f"  WARNING: {attack_class} not found in label encoder, skipping.")
            continue

        cfg = PER_CLASS_CONFIG[attack_class]
        class_idx = class_to_idx[attack_class]
        mask = y_train == class_idx
        X_class = X_train[mask]

        # Cap samples
        if len(X_class) > MAX_SAMPLES:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(X_class), MAX_SAMPLES, replace=False)
            X_class = X_class[idx]

        print(
            f"\n{attack_class}: {X_class.shape[0]} samples | "
            f"hidden={cfg['hidden_dims']} epochs={cfg['epochs']} "
            f"kl={cfg['kl_weight']}"
        )

        model = train_vae(
            attack_class=attack_class,
            X_class=X_class,
            hidden_dims=cfg["hidden_dims"],
            epochs=cfg["epochs"],
            kl_weight=cfg["kl_weight"],
        )

        metrics = evaluate_reconstruction(model, X_class, attack_class)
        results[attack_class] = metrics

    # Summary
    print(f"\n{'=' * 60}")
    print("VAE Training Summary")
    print(f"{'=' * 60}")
    print(f"{'Class':<10} {'Cont MAE':>10} {'Bin Acc':>10} {'KL':>10}")
    print("-" * 45)
    for cls, m in results.items():
        print(
            f"{cls:<10} {m['cont_mae']:>10.6f} {m['bin_acc']:>10.4f} {m['kl']:>10.4f}"
        )

    print(f"\nAll VAE models saved to {MODEL_DIR}")
    print("VAE training complete. Ready for attack pipeline.")


if __name__ == "__main__":
    main()
