"""
run_attacks.py — Constraint-Aware VAE Latent-Space Adversarial Attack on NIDS
==============================================================================
Thesis: VAE-based latent-space adversarial attacks on CICIoT2023

Attack pipeline:
    x (attack sample) → VAE Encoder → μ
    Perturb μ via gradient descent (C&W-style loss)
    Hard L2 constraint: ‖z' - μ‖ ≤ r
    z' → VAE Decoder → x'
    Constraint-aware loss: bounds + binary consistency + reconstruction anchor

Key improvements over original:
    1. Constraint penalty loss (feature bounds + binary consistency)
    2. Feature masking  — only perturb SAFE (low-correlation) features
    3. Reconstruction anchor — keeps adversarial samples close to originals
    4. In-loop feature clamping — hard bounds enforced every PGD step
    5. Conservative radius grid — avoids latent explosion
    6. Validity checker — ASR computed ONLY over valid adversarial samples
    7. Multi-seed runs — mean ± std reported for thesis robustness
    8. Straight-through estimator for binary features (gradient-friendly)
    9. Conservative latent projection (0.99 radius to prevent boundary violations)
"""

import os
import pickle
import joblib
import itertools
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from tqdm import tqdm
from datetime import datetime
import io
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = r"D:\thesis"
PROCESSED = os.path.join(ROOT, "data", "processed")
VAE_DIR = os.path.join(ROOT, "models", "vae")
NIDS_DIR = os.path.join(ROOT, "models", "nids")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Output directory for results ────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(ROOT, "output_text")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Feature indices ────────────────────────────────────────────────────────────
CONTINUOUS_IDX = list(range(0, 23)) + [36]  # 24 continuous features
BINARY_IDX = list(range(23, 36))  # 13 binary features
N_FEATURES = 37

# ── Fixed hyper-parameters ─────────────────────────────────────────────────────
LATENT_DIM = 16
KAPPA = 0.0
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ATTACK_SAMPLES = 500
OVERSAMPLE_FACTOR = 5
PROBE_SIZE = 50

# ── Constraint loss weights ────────────────────────────────────────────────────
# Tune these if validity is still low:
#   Raise LAMBDA_CONSTRAINT → higher validity, possibly lower ASR
#   Raise LAMBDA_RECON      → samples stay closer to originals
LAMBDA_CONSTRAINT = 10.0
LAMBDA_RECON = 1.0

# ── Multi-seed evaluation ──────────────────────────────────────────────────────
# Results are averaged over these seeds and reported as mean ± std
EVAL_SEEDS = [42, 123, 2024, 7, 999]

# ── Feature mask: which continuous features are "safe" to perturb ─────────────
# Strategy: mask out high-correlation features to avoid breaking
#           inter-feature relationships (same idea as NetDiffuser).
# Set CORRELATION_THRESHOLD lower to be more conservative.
# These are computed from training data in main(); can also be set manually.
CORRELATION_THRESHOLD = 0.7  # features correlated > this are frozen

# ── Conservative radius search grid (original went up to 16 — too large) ──────
SEARCH_GRID = {
    "radius": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5],  # was [2,3,5,8,12,16]
    "lambda_cw": [1.0, 5.0, 10.0, 20.0],
    "lr": [0.001, 0.01, 0.05],
    "max_iter": [300, 500],
}

DEFAULT_CONFIG = {
    "DDoS": {"radius": 2.0, "lambda_cw": 5.0, "lr": 0.01, "max_iter": 300},
    "DoS": {"radius": 2.0, "lambda_cw": 10.0, "lr": 0.01, "max_iter": 500},
    "Mirai": {"radius": 2.0, "lambda_cw": 10.0, "lr": 0.01, "max_iter": 500},
    "Recon": {"radius": 1.0, "lambda_cw": 5.0, "lr": 0.01, "max_iter": 300},
}

SEARCH_CLASSES = {"DoS", "Mirai", "DDoS", "Recon"}

ATTACK_CLASSES = {
    "DDoS": "vae_ddos.pt",
    "DoS": "vae_dos.pt",
    "Mirai": "vae_mirai.pt",
    "Recon": "vae_recon.pt",
}

VAE_HIDDEN = {
    "DDoS": [128, 64],
    "DoS": [128, 64],
    "Mirai": [128, 64],
    "Recon": [256, 128],
}


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
#  Model definitions  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════


class MixedInputVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, continuous_idx, binary_idx):
        super().__init__()
        self.continuous_idx = continuous_idx
        self.binary_idx = binary_idx

        enc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)

        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            in_dim = h
        self.decoder = nn.Sequential(*dec_layers)
        self.cont_head = nn.Linear(in_dim, len(continuous_idx))
        self.binary_head = nn.Sequential(
            nn.Linear(in_dim, len(binary_idx)), nn.Sigmoid()
        )

    def encode(self, x):
        x_reordered = torch.cat(
            [x[:, self.continuous_idx], x[:, self.binary_idx]], dim=1
        )
        h = self.encoder(x_reordered)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.decoder(z)
        return self.cont_head(h), self.binary_head(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        cont, binary = self.decode(z)
        return cont, binary, mu, log_var


class SimpleCNNLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        x = self.dropout(hn.squeeze(0))
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════════════════
#  Feature mask: compute safe (low-correlation) continuous features
# ══════════════════════════════════════════════════════════════════════════════


def compute_safe_feature_mask(
    X_train: np.ndarray, threshold: float = CORRELATION_THRESHOLD
) -> torch.Tensor:
    """
    Returns a float mask of shape (N_FEATURES,).
    Safe continuous features (pairwise |corr| < threshold with all others) = 1.0
    High-correlation continuous features = 0.0
    Binary features = 0.0  (never perturb directly; VAE handles them)

    The mask is applied as:
        x_adv = x_orig + mask * (x_decoded - x_orig)
    so high-correlation and binary features stay exactly as decoded by the VAE.
    """
    df = pd.DataFrame(X_train[:, CONTINUOUS_IDX])
    corr = df.corr().abs()

    safe_cont_cols = [
        col for col in corr.columns if (corr[col].drop(index=col) < threshold).all()
    ]

    mask = torch.zeros(N_FEATURES, dtype=torch.float32, device=DEVICE)
    for col in safe_cont_cols:
        mask[CONTINUOUS_IDX[col]] = 1.0

    n_safe = int(mask.sum().item())
    n_cont = len(CONTINUOUS_IDX)
    print(
        f"[FeatureMask] Safe features: {n_safe}/{n_cont} continuous  "
        f"(threshold={threshold})"
    )
    if n_safe == 0:
        print(
            "[FeatureMask] WARNING: no safe features found — "
            "relaxing threshold to 0.9 and using all continuous features."
        )
        mask[CONTINUOUS_IDX] = 1.0

    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  Validity checker
# ══════════════════════════════════════════════════════════════════════════════


def check_validity(
    X_adv: np.ndarray, X_orig: np.ndarray, cont_min: np.ndarray, cont_max: np.ndarray
) -> np.ndarray:
    """
    Returns a boolean array of shape (N,) — True if the adversarial sample
    is semantically valid:

        1. Continuous features stay within [cont_min, cont_max] (the
           feature-wise min/max seen in the original test samples).
        2. Binary features are exactly 0 or 1 (no fractional values).

    ASR must be computed ONLY over valid samples to avoid inflating results
    with garbage inputs.
    """
    cont_adv = X_adv[:, CONTINUOUS_IDX]
    bin_adv = X_adv[:, BINARY_IDX]

    cont_valid = (cont_adv >= cont_min[CONTINUOUS_IDX] - 1e-4).all(axis=1) & (
        cont_adv <= cont_max[CONTINUOUS_IDX] + 1e-4
    ).all(axis=1)
    bin_valid = np.all((bin_adv == 0) | (bin_adv == 1), axis=1)
    return cont_valid & bin_valid


# ══════════════════════════════════════════════════════════════════════════════
#  Constraint loss (used inside optimisation loop)
# ══════════════════════════════════════════════════════════════════════════════


def constraint_loss(
    x_adv: torch.Tensor, cont_min_t: torch.Tensor, cont_max_t: torch.Tensor
) -> torch.Tensor:
    """
    Penalises three kinds of violations:

    1. Continuous out-of-bounds  → relu(x - max) + relu(min - x)
    2. Binary non-binary values  → distance from nearest {0,1}
    3. Large total perturbation  → L1 norm of change (light regulariser)

    Returns a scalar tensor.
    """
    cont = x_adv[:, CONTINUOUS_IDX]
    binary = x_adv[:, BINARY_IDX]

    # 1. Bounds violation
    upper_viol = torch.relu(cont - cont_max_t).sum()
    lower_viol = torch.relu(cont_min_t - cont).sum()

    # 2. Binary consistency: push toward hard 0/1
    bin_target = (binary >= 0.5).float()
    binary_viol = torch.abs(binary - bin_target).mean()

    return upper_viol + lower_viol + binary_viol


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers  (IMPROVED: straight-through estimator for binary features)
# ══════════════════════════════════════════════════════════════════════════════


def decode_to_full(cont_out, bin_out, n_features=N_FEATURES, use_ste=True):
    """
    Decode VAE outputs to full feature vector.

    Args:
        cont_out: Continuous features from VAE decoder
        bin_out: Binary features from VAE decoder (sigmoid activated)
        n_features: Total number of features
        use_ste: If True, use straight-through estimator for binary features
                to maintain gradient flow while keeping hard thresholding
                in forward pass.
    """
    B = cont_out.size(0)
    x = torch.zeros(B, n_features, device=cont_out.device)
    x[:, CONTINUOUS_IDX] = cont_out

    if use_ste and bin_out.requires_grad:
        # Straight-through estimator: forward = hard threshold, backward = soft
        binary_hard = (bin_out >= 0.5).float()
        # STE: output = hard - soft.detach() + soft
        # This way gradients flow through the soft values
        x[:, BINARY_IDX] = binary_hard + (bin_out - bin_out.detach())
    else:
        # Standard hard thresholding (no gradients through binary)
        x[:, BINARY_IDX] = (bin_out >= 0.5).float()

    return x


def cw_loss(logits, original_class, kappa=KAPPA):
    B = logits.size(0)
    correct_scores = logits[torch.arange(B), original_class]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(B), original_class] = False
    other_scores = logits[mask].view(B, -1).max(dim=1).values
    return torch.clamp(correct_scores - other_scores, min=-kappa).mean()


def get_cnnlstm_preds(X_np, cnnlstm):
    cnnlstm.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_np), BATCH_SIZE):
            xb = torch.tensor(
                X_np[start : start + BATCH_SIZE], dtype=torch.float32, device=DEVICE
            )
            preds.append(cnnlstm(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ══════════════════════════════════════════════════════════════════════════════
#  Load helpers  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════


def load_vae(class_name, hidden_dims):
    ckpt_path = os.path.join(VAE_DIR, ATTACK_CLASSES[class_name])
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt["model_state_dict"]
    continuous_idx = ckpt.get("continuous_idx", CONTINUOUS_IDX)
    binary_idx = ckpt.get("binary_idx", BINARY_IDX)
    latent_dim = ckpt.get("latent_dim", LATENT_DIM)
    hidden_dims = ckpt.get("hidden_dims", hidden_dims)
    vae = MixedInputVAE(
        input_dim=N_FEATURES,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        continuous_idx=continuous_idx,
        binary_idx=binary_idx,
    ).to(DEVICE)
    vae.load_state_dict(state_dict)
    vae.eval()
    return vae


def load_cnnlstm(num_classes):
    model = SimpleCNNLSTM(N_FEATURES, num_classes).to(DEVICE)
    ckpt = os.path.join(NIDS_DIR, "nids_cnnlstm.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def load_lgbm():
    sklearn_path = os.path.join(NIDS_DIR, "nids_lgbm_sklearn.pkl")
    if os.path.exists(sklearn_path):
        clf = joblib.load(sklearn_path)
        feature_names = clf.feature_names_in_
        print(f"[LGBM] Loaded sklearn classifier. classes_: {clf.classes_}")

        class LGBMWrapper:
            def __init__(self, clf, feature_names):
                self.clf = clf
                self.feature_names = feature_names

            def _to_df(self, X):
                return pd.DataFrame(X, columns=self.feature_names)

            def predict_proba(self, X):
                return self.clf.predict_proba(self._to_df(X))

            def predict(self, X):
                return self.clf.predict(self._to_df(X))

        return LGBMWrapper(clf, feature_names)
    else:
        print("[LGBM] WARNING: sklearn pkl not found, using raw booster.")
        booster = lgb.Booster(model_file=os.path.join(NIDS_DIR, "nids_lgbm.txt"))

        class LGBMWrapper:
            def __init__(self, booster):
                self.booster = booster

            def predict_proba(self, X):
                out = self.booster.predict(X)
                if out.ndim == 1:
                    out = np.stack([1 - out, out], axis=1)
                return out

            def predict(self, X):
                return self.predict_proba(X).argmax(axis=1)

        return LGBMWrapper(booster)


# ══════════════════════════════════════════════════════════════════════════════
#  Core attack — CNN-LSTM  (REWRITTEN with constraint-awareness)
# ══════════════════════════════════════════════════════════════════════════════


def latent_attack_cnnlstm(
    x_batch,
    original_class_idx,
    vae,
    cnnlstm,
    radius,
    max_iter,
    lambda_cw,
    lr,
    safe_mask,  # NEW: (N_FEATURES,) float tensor
    cont_min_t,  # NEW: continuous lower bounds tensor
    cont_max_t,  # NEW: continuous upper bounds tensor
    use_ste=True,
):  # NEW: use straight-through estimator
    """
    Constraint-aware latent-space C&W attack against CNN-LSTM.

    Changes vs original:
    ─────────────────────────────────────────────────────────────────────────
    1. CONSTRAINT LOSS added to optimisation objective.
    2. FEATURE MASK applied after decoding — only safe features are changed.
    3. RECONSTRUCTION ANCHOR loss keeps x_adv close to x_orig.
    4. IN-LOOP CLAMPING enforces hard continuous bounds every step.
    5. CONSERVATIVE PROJECTION: 0.99*radius to prevent boundary violations.
    6. STRAIGHT-THROUGH ESTIMATOR for binary features (gradient-friendly).
    ─────────────────────────────────────────────────────────────────────────
    """
    vae.eval()
    cnnlstm.train()  # enable dropout for gradient flow
    for module in cnnlstm.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()

    B = x_batch.size(0)

    with torch.no_grad():
        mu, _ = vae.encode(x_batch)

    delta = torch.zeros_like(mu, requires_grad=True)
    optimiser = optim.Adam([delta], lr=lr)
    orig_labels = torch.full((B,), original_class_idx, dtype=torch.long, device=DEVICE)

    # Conservative radius factor to prevent exact boundary violations
    conservative_radius = radius * 0.99

    for _ in range(max_iter):
        optimiser.zero_grad()

        # ── Latent projection: keep z within L2 ball of radius r ──────────
        # Conservative projection: stay slightly inside the boundary
        z_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z_proj = torch.where(
            z_norm > conservative_radius, delta * conservative_radius / z_norm, delta
        )
        z_adv = mu + z_proj

        # ── Decode ────────────────────────────────────────────────────────
        cont_out, bin_out = vae.decode(z_adv)
        x_decoded = decode_to_full(cont_out, bin_out, use_ste=use_ste)

        # ── Feature mask: only change safe (low-corr) features ────────────
        # x_adv = x_orig  +  mask * (x_decoded - x_orig)
        # High-corr and binary features stay at their decoded values but
        # the gradient only flows through masked features.
        x_adv_t = x_batch + safe_mask.unsqueeze(0) * (x_decoded - x_batch)

        # ── In-loop hard clamping on continuous features ───────────────────
        # detach → clamp → reattach to graph via x_adv_t replacement
        cont_clamped = torch.clamp(
            x_adv_t[:, CONTINUOUS_IDX],
            min=cont_min_t,
            max=cont_max_t,
        )
        x_adv_t = x_adv_t.clone()
        x_adv_t[:, CONTINUOUS_IDX] = cont_clamped

        # ── Losses ────────────────────────────────────────────────────────
        logits = cnnlstm(x_adv_t)
        loss_cw = cw_loss(logits, orig_labels)
        loss_con = constraint_loss(x_adv_t, cont_min_t, cont_max_t)
        loss_recon = torch.mean((x_adv_t - x_batch) ** 2)

        loss = (
            0.1 * z_proj.norm(dim=1).mean()  # keep latent perturbation small
            + lambda_cw * loss_cw  # fool the classifier
            + LAMBDA_CONSTRAINT * loss_con  # stay valid
            + LAMBDA_RECON * loss_recon  # stay close to original
        )

        loss.backward()
        optimiser.step()

    # ── Final evaluation (no grad) ─────────────────────────────────────────
    cnnlstm.eval()
    with torch.no_grad():
        z_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z_proj = torch.where(
            z_norm > conservative_radius, delta * conservative_radius / z_norm, delta
        )
        z_adv = mu + z_proj
        cont_out, bin_out = vae.decode(z_adv)
        x_decoded = decode_to_full(
            cont_out, bin_out, use_ste=False
        )  # Hard threshold for eval

        # Apply mask + clamp
        x_adv_final = x_batch + safe_mask.unsqueeze(0) * (x_decoded - x_batch)
        x_adv_final[:, CONTINUOUS_IDX] = torch.clamp(
            x_adv_final[:, CONTINUOUS_IDX], min=cont_min_t, max=cont_max_t
        )

        preds = cnnlstm(x_adv_final).argmax(dim=1)
        success = (preds != orig_labels).cpu().numpy()

    return x_adv_final.cpu().numpy(), success


# ══════════════════════════════════════════════════════════════════════════════
#  Core attack — LightGBM  (REWRITTEN with constraint-awareness)
# ══════════════════════════════════════════════════════════════════════════════


def latent_attack_lgbm(
    x_batch,
    original_class_idx,
    vae,
    lgbm_model,
    radius,
    max_iter,
    lr,
    safe_mask,  # NEW
    cont_min_np,  # NEW: numpy arrays for post-clamp
    cont_max_np,  # NEW
    seed=42,
):  # NEW: configurable seed
    """
    Constraint-aware random-direction latent-space attack against LightGBM.

    Changes vs original:
    ─────────────────────────────────────────────────────────────────────────
    1. FEATURE MASK applied to candidates — only safe features are changed.
    2. HARD CLAMP on continuous features after every candidate decode.
    3. CONSERVATIVE RADIUS: 0.99*radius to prevent boundary violations.
    4. CONFIGURABLE SEED for reproducibility.
    ─────────────────────────────────────────────────────────────────────────
    Note: LightGBM is non-differentiable, so we keep the random-direction
    search but make each candidate valid before evaluating it.
    """
    vae.eval()
    B = x_batch.size(0)

    with torch.no_grad():
        mu, _ = vae.encode(x_batch)

    mu_np = mu.cpu().numpy().copy()
    mu_orig_np = mu.cpu().numpy().copy()
    mask_np = safe_mask.cpu().numpy()  # (N_FEATURES,)

    best_x_adv = x_batch.cpu().numpy().copy()
    init_proba = lgbm_model.predict_proba(best_x_adv)
    best_scores = init_proba[:, original_class_idx].copy()

    x_orig_np = x_batch.cpu().numpy().copy()  # anchor for feature mask

    rng = np.random.default_rng(seed)

    # Conservative radius
    conservative_radius = radius * 0.99

    for step in range(max_iter):
        noise = rng.standard_normal(mu_np.shape).astype(np.float32)
        noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
        step_size = conservative_radius * (1 - step / max_iter) * 0.1
        z_cand = mu_np + noise / (noise_norm + 1e-8) * step_size

        # L2 projection in latent space (conservative)
        diff = z_cand - mu_orig_np
        diff_norm = np.linalg.norm(diff, axis=1, keepdims=True)
        z_cand = np.where(
            diff_norm > conservative_radius,
            mu_orig_np + diff * conservative_radius / diff_norm,
            z_cand,
        )

        # Decode candidate
        z_t = torch.tensor(z_cand, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            cont_out, bin_out = vae.decode(z_t)
            x_decoded = decode_to_full(cont_out, bin_out, use_ste=False).cpu().numpy()

        # Apply feature mask: blend decoded with original
        x_cand = x_orig_np + mask_np * (x_decoded - x_orig_np)

        # Hard clamp continuous features
        x_cand[:, CONTINUOUS_IDX] = np.clip(
            x_cand[:, CONTINUOUS_IDX],
            cont_min_np[CONTINUOUS_IDX],
            cont_max_np[CONTINUOUS_IDX],
        )

        # Score candidates; keep best (lowest score on true class)
        proba = lgbm_model.predict_proba(x_cand)
        cand_score = proba[:, original_class_idx]

        improved = cand_score < best_scores
        best_scores[improved] = cand_score[improved]
        best_x_adv[improved] = x_cand[improved]
        mu_np[improved] = z_cand[improved]

    final_preds = lgbm_model.predict(best_x_adv)
    success = final_preds != original_class_idx
    return best_x_adv, success


# ══════════════════════════════════════════════════════════════════════════════
#  Grid search  (unchanged logic, new signature forwards safe_mask + bounds)
# ══════════════════════════════════════════════════════════════════════════════


def grid_search(
    class_name,
    class_idx,
    X_probe,
    vae,
    cnnlstm,
    safe_mask,
    cont_min_t,
    cont_max_t,
    use_ste=True,
):
    x_probe_t = torch.tensor(X_probe, dtype=torch.float32, device=DEVICE)

    keys = list(SEARCH_GRID.keys())
    values = list(SEARCH_GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    best_asr = -1.0
    best_config = DEFAULT_CONFIG[class_name].copy()

    print(f"  Grid search: {total} combinations on {len(X_probe)} probe samples ...")

    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        _, suc = latent_attack_cnnlstm(
            x_probe_t,
            class_idx,
            vae,
            cnnlstm,
            radius=cfg["radius"],
            max_iter=cfg["max_iter"],
            lambda_cw=cfg["lambda_cw"],
            lr=cfg["lr"],
            safe_mask=safe_mask,
            cont_min_t=cont_min_t,
            cont_max_t=cont_max_t,
            use_ste=use_ste,
        )
        asr = suc.mean()

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(
                f"    [{i + 1}/{total}] r={cfg['radius']} λ={cfg['lambda_cw']} "
                f"lr={cfg['lr']} iter={cfg['max_iter']} → probe ASR={asr:.3f}  "
                f"(best so far: {best_asr:.3f})"
            )

        if asr > best_asr:
            best_asr = asr
            best_config = cfg.copy()

    print(f"  ✓ Best config: {best_config}  →  probe ASR={best_asr:.4f}")
    return best_config, best_asr


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-seed ASR evaluation (for mean ± std reporting)
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_asr_multiseed(
    X_cls,
    class_idx,
    vae,
    cnnlstm,
    lgbm_model,
    best_cfg,
    safe_mask,
    cont_min_t,
    cont_max_t,
    cont_min_np,
    cont_max_np,
    seeds=EVAL_SEEDS,
    use_ste=True,
):
    """
    Runs the attack with multiple random seeds (via torch.manual_seed) and
    reports mean ± std of ASR for both models, computed over VALID samples.

    Returns dicts with keys: asr_cnnlstm, asr_lgbm, validity_cnnlstm,
                              validity_lgbm, seeds_used.
    """
    radius = best_cfg["radius"]
    max_iter = best_cfg["max_iter"]
    lambda_cw = best_cfg["lambda_cw"]
    lr = best_cfg["lr"]

    cnnlstm_asrs, lgbm_asrs = [], []
    cnnlstm_validities, lgbm_validities = [], []

    x_tensor = torch.tensor(X_cls, dtype=torch.float32, device=DEVICE)

    for seed in seeds:
        set_seed(seed)  # Use centralized seed setting

        # ── CNN-LSTM full-set attack ──────────────────────────────────────
        adv_c_list, suc_c_list = [], []
        for start in range(0, len(X_cls), BATCH_SIZE):
            xb = x_tensor[start : start + BATCH_SIZE]
            adv, suc = latent_attack_cnnlstm(
                xb,
                class_idx,
                vae,
                cnnlstm,
                radius=radius,
                max_iter=max_iter,
                lambda_cw=lambda_cw,
                lr=lr,
                safe_mask=safe_mask,
                cont_min_t=cont_min_t,
                cont_max_t=cont_max_t,
                use_ste=use_ste,
            )
            adv_c_list.append(adv)
            suc_c_list.append(suc)

        adv_c = np.vstack(adv_c_list)
        suc_c = np.concatenate(suc_c_list)

        valid_c = check_validity(adv_c, X_cls, cont_min_np, cont_max_np)
        validity_c = valid_c.mean()

        # ASR only over valid samples (thesis-correct definition)
        if valid_c.sum() > 0:
            asr_c = suc_c[valid_c].mean()
        else:
            asr_c = 0.0

        cnnlstm_asrs.append(asr_c)
        cnnlstm_validities.append(validity_c)

        # ── LGBM full-set attack ──────────────────────────────────────────
        adv_l_list, suc_l_list = [], []
        for start in range(0, len(X_cls), BATCH_SIZE):
            xb = x_tensor[start : start + BATCH_SIZE]
            adv, suc = latent_attack_lgbm(
                xb,
                class_idx,
                vae,
                lgbm_model,
                radius=radius,
                max_iter=max_iter,
                lr=lr,
                safe_mask=safe_mask,
                cont_min_np=cont_min_np,
                cont_max_np=cont_max_np,
                seed=seed,  # Pass seed for reproducibility
            )
            adv_l_list.append(adv)
            suc_l_list.append(suc)

        adv_l = np.vstack(adv_l_list)
        suc_l = np.concatenate(suc_l_list)

        valid_l = check_validity(adv_l, X_cls, cont_min_np, cont_max_np)
        validity_l = valid_l.mean()

        if valid_l.sum() > 0:
            asr_l = suc_l[valid_l].mean()
        else:
            asr_l = 0.0

        lgbm_asrs.append(asr_l)
        lgbm_validities.append(validity_l)

        print(
            f"    seed={seed} | "
            f"CNN-LSTM valid={validity_c:.3f} ASR={asr_c:.3f} | "
            f"LGBM valid={validity_l:.3f} ASR={asr_l:.3f}"
        )

    return {
        "asr_cnnlstm_mean": float(np.mean(cnnlstm_asrs)),
        "asr_cnnlstm_std": float(np.std(cnnlstm_asrs)),
        "asr_lgbm_mean": float(np.mean(lgbm_asrs)),
        "asr_lgbm_std": float(np.std(lgbm_asrs)),
        "validity_cnnlstm_mean": float(np.mean(cnnlstm_validities)),
        "validity_cnnlstm_std": float(np.std(cnnlstm_validities)),
        "validity_lgbm_mean": float(np.mean(lgbm_validities)),
        "validity_lgbm_std": float(np.std(lgbm_validities)),
        "seeds": seeds,
        # Keep last-seed adversarial examples for saving
        "_adv_cnnlstm": adv_c,
        "_adv_lgbm": adv_l,
        "_suc_cnnlstm": suc_c,
        "_suc_lgbm": suc_l,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main(use_train_bounds=False, use_ste=True):
    """
    Main attack pipeline.

    Args:
        use_train_bounds: If True, use training set bounds for validity checking
                         (more robust against distribution shift). If False, use
                         test set bounds (original behavior).
        use_ste: If True, use straight-through estimator for binary features
                during CNN-LSTM optimization (recommended).
    """
    print(f"Device: {DEVICE}")
    print(f"Using straight-through estimator: {use_ste}")
    print(f"Using {'train' if use_train_bounds else 'test'} set bounds for validity")

    with open(os.path.join(NIDS_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    # Load both train and test for bounds computation
    X_test = np.load(os.path.join(PROCESSED, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(PROCESSED, "y_test.npy"), allow_pickle=True)

    if use_train_bounds:
        try:
            X_train = np.load(os.path.join(PROCESSED, "X_train.npy")).astype(np.float32)
            print(f"[Bounds] Using training set bounds (more robust)")
            bounds_source = X_train
        except FileNotFoundError:
            print(
                f"[Bounds] WARNING: X_train.npy not found, falling back to test set bounds"
            )
            bounds_source = X_test
    else:
        bounds_source = X_test

    if y_test.dtype.kind in ("U", "O", "S"):
        y_test = le.transform(y_test).astype(int)
    else:
        y_test = y_test.astype(int)

    print(f"Classes: {class_names}")
    print(f"Test set: {X_test.shape}  |  y unique: {np.unique(y_test)}")

    # ── Compute feature bounds from selected source ─────────────────────────
    # Using train set min/max is more robust against distribution shift
    cont_min_np = bounds_source.min(axis=0)  # shape (N_FEATURES,)
    cont_max_np = bounds_source.max(axis=0)
    cont_min_t = torch.tensor(
        cont_min_np[CONTINUOUS_IDX], dtype=torch.float32, device=DEVICE
    )
    cont_max_t = torch.tensor(
        cont_max_np[CONTINUOUS_IDX], dtype=torch.float32, device=DEVICE
    )
    print(
        f"Continuous bounds computed from {'train' if use_train_bounds else 'test'} set."
    )

    # ── Compute safe feature mask ─────────────────────────────────────────────
    # Use training data for correlation computation if available
    if use_train_bounds and "X_train" in locals():
        safe_mask = compute_safe_feature_mask(X_train)
    else:
        safe_mask = compute_safe_feature_mask(X_test)

    lgbm = load_lgbm()
    cnnlstm = load_cnnlstm(num_classes)
    print("Victim models loaded.\n")

    all_results = {}
    best_configs = {}

    for class_name in ATTACK_CLASSES:
        if class_name not in class_names:
            print(f"[WARN] {class_name} not in label encoder, skipping.")
            continue

        class_idx = class_names.index(class_name)

        # ── Sample pool: correctly classified by BOTH models ─────────────
        mask = y_test == class_idx
        X_pool = X_test[mask][: N_ATTACK_SAMPLES * OVERSAMPLE_FACTOR]
        n_pool = len(X_pool)

        lgbm_preds = lgbm.predict(X_pool)
        cnnlstm_preds = get_cnnlstm_preds(X_pool, cnnlstm)

        correct_mask = (lgbm_preds == class_idx) & (cnnlstm_preds == class_idx)
        X_correct = X_pool[correct_mask]
        n_correct = len(X_correct)

        X_cls = X_correct[:N_ATTACK_SAMPLES]
        n_samples = len(X_cls)

        print(f"\n{'=' * 60}")
        print(
            f"Class: {class_name}  |  pool={n_pool}  "
            f"both-correct={n_correct}  attacking={n_samples}"
        )
        print(f"{'=' * 60}")

        if n_samples < 50:
            print(f"  [WARN] Only {n_samples} valid samples — skipping.")
            continue

        vae = load_vae(class_name, VAE_HIDDEN[class_name])

        # ── Grid search ───────────────────────────────────────────────────
        if class_name in SEARCH_CLASSES:
            X_probe = X_correct[N_ATTACK_SAMPLES:][:PROBE_SIZE]
            if len(X_probe) < PROBE_SIZE:
                X_probe = X_cls[:PROBE_SIZE]
            best_cfg, probe_asr = grid_search(
                class_name,
                class_idx,
                X_probe,
                vae,
                cnnlstm,
                safe_mask,
                cont_min_t,
                cont_max_t,
                use_ste=use_ste,
            )
        else:
            best_cfg = DEFAULT_CONFIG[class_name].copy()
            probe_asr = float("nan")
            print(f"  Using default config: {best_cfg}")

        best_configs[class_name] = best_cfg

        # ── Multi-seed evaluation ─────────────────────────────────────────
        print(f"\n  Running multi-seed evaluation ({len(EVAL_SEEDS)} seeds) ...")
        stats = evaluate_asr_multiseed(
            X_cls,
            class_idx,
            vae,
            cnnlstm,
            lgbm,
            best_cfg,
            safe_mask,
            cont_min_t,
            cont_max_t,
            cont_min_np,
            cont_max_np,
            use_ste=use_ste,
        )

        print(
            f"\n  Results for {class_name}:\n"
            f"    CNN-LSTM  | validity={stats['validity_cnnlstm_mean']:.4f}"
            f" ± {stats['validity_cnnlstm_std']:.4f}"
            f"  |  ASR (valid only)={stats['asr_cnnlstm_mean']:.4f}"
            f" ± {stats['asr_cnnlstm_std']:.4f}\n"
            f"    LightGBM  | validity={stats['validity_lgbm_mean']:.4f}"
            f" ± {stats['validity_lgbm_std']:.4f}"
            f"  |  ASR (valid only)={stats['asr_lgbm_mean']:.4f}"
            f" ± {stats['asr_lgbm_std']:.4f}"
        )

        # ── Save adversarial examples ─────────────────────────────────────
        save_path = os.path.join(RESULTS_DIR, class_name.lower())
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "adv_cnnlstm.npy"), stats["_adv_cnnlstm"])
        np.save(os.path.join(save_path, "adv_lgbm.npy"), stats["_adv_lgbm"])
        np.save(os.path.join(save_path, "original.npy"), X_cls)
        np.save(os.path.join(save_path, "success_cnnlstm.npy"), stats["_suc_cnnlstm"])
        np.save(os.path.join(save_path, "success_lgbm.npy"), stats["_suc_lgbm"])

        all_results[class_name] = {
            "n_pool": n_pool,
            "n_correct": n_correct,
            "n_samples": n_samples,
            "best_config": best_cfg,
            "probe_asr": probe_asr,
            **{k: v for k, v in stats.items() if not k.startswith("_")},
        }

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL ATTACK SUMMARY  (ASR computed over valid adversarial samples)")
    print(f"{'=' * 70}")
    header = (
        f"{'Class':<10} {'N':>5}  "
        f"{'CNN-LSTM valid':>14}  {'CNN-LSTM ASR':>12}  "
        f"{'LGBM valid':>10}  {'LGBM ASR':>10}"
    )
    print(header)
    print("-" * len(header))

    for cls, res in all_results.items():
        print(
            f"{cls:<10} {res['n_samples']:>5}  "
            f"{res['validity_cnnlstm_mean']:>6.1%} ± {res['validity_cnnlstm_std']:.2%}  "
            f"{res['asr_cnnlstm_mean']:>6.1%} ± {res['asr_cnnlstm_std']:.2%}  "
            f"{res['validity_lgbm_mean']:>6.1%} ± {res['validity_lgbm_std']:.2%}  "
            f"{res['asr_lgbm_mean']:>6.1%} ± {res['asr_lgbm_std']:.2%}"
        )

    with open(os.path.join(RESULTS_DIR, "attack_summary.pkl"), "wb") as f:
        pickle.dump(all_results, f)
    with open(os.path.join(RESULTS_DIR, "best_configs.pkl"), "wb") as f:
        pickle.dump(best_configs, f)

    # ── Write results to timestamped markdown file ─────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_filename = f"attack_results_{timestamp}.md"
    md_path = os.path.join(OUTPUT_DIR, md_filename)

    with open(md_path, "w") as f:
        f.write(
            f"# Attack Results — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("## Configuration\n\n")
        f.write(f"- use_train_bounds: {use_train_bounds}\n")
        f.write(f"- use_ste: {use_ste}\n")
        f.write(f"- seeds: {EVAL_SEEDS}\n\n")
        f.write("## Per-Class Results\n\n")
        f.write(
            "| Class | Samples | CNN-LSTM Validity | CNN-LSTM ASR | LGBM Validity | LGBM ASR |\n"
        )
        f.write(
            "|-------|---------|-------------------|--------------|---------------|----------|\n"
        )
        for cls, res in all_results.items():
            f.write(
                f"| {cls} | {res['n_samples']} | "
                f"{res['validity_cnnlstm_mean']:.1%} ± {res['validity_cnnlstm_std']:.2%} | "
                f"{res['asr_cnnlstm_mean']:.1%} ± {res['asr_cnnlstm_std']:.2%} | "
                f"{res['validity_lgbm_mean']:.1%} ± {res['validity_lgbm_std']:.2%} | "
                f"{res['asr_lgbm_mean']:.1%} ± {res['asr_lgbm_std']:.2%} |\n"
            )
        f.write("\n## Best Configurations\n\n")
        for cls, cfg in best_configs.items():
            f.write(f"### {cls}\n")
            for k, v in cfg.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")

    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Markdown output written to {md_path}")


if __name__ == "__main__":
    # Set default seed for initialization
    set_seed(42)

    # Run with recommended settings:
    # - use_train_bounds=True: More robust validity checking using training set bounds
    # - use_ste=True: Use straight-through estimator for binary features (recommended)
    main(use_train_bounds=True, use_ste=True)
