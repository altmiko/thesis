"""
attack.py — VAE Latent-Space Adversarial Attack on NIDS
========================================================
Thesis: VAE-based latent-space adversarial attacks on CICIoT2023

Attack pipeline:
    x (attack sample) → VAE Encoder → μ
    Perturb μ via gradient descent (C&W-style loss)
    Hard L2 constraint: ‖z' - μ‖ ≤ r
    z' → VAE Decoder → x'
    Post-process: threshold binary features at 0.5

Evaluated against two victim models:
    - LightGBM  (nids_lgbm_sklearn.pkl preferred, nids_lgbm.txt fallback)
    - CNN-LSTM  (nids_cnnlstm.pt)

Only samples correctly classified by BOTH models are attacked.
"""

import os
import pickle
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = r"D:\thesis"
PROCESSED = os.path.join(ROOT, "data", "processed")
VAE_DIR = os.path.join(ROOT, "models", "vae")
NIDS_DIR = os.path.join(ROOT, "models", "nids")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Feature indices (must match preprocess.py) ─────────────────────────────────
# 37 features: 23 continuous + 13 binary + Protocol Type
CONTINUOUS_IDX = list(range(0, 23)) + [36]  # 24 continuous features
BINARY_IDX = list(range(23, 36))  # 13 binary features
N_FEATURES = 37

# ── Global attack hyper-parameters ────────────────────────────────────────────
LATENT_DIM = 16
KAPPA = 0.0
LAMBDA_CW = 1.0
LR = 0.01
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Per-class attack config ────────────────────────────────────────────────────
CLASS_ATTACK_CONFIG = {
    "DDoS": {"radius": 3.0, "max_iter": 200},
    "DoS": {"radius": 8.0, "max_iter": 500},
    "Mirai": {"radius": 10.0, "max_iter": 500},
    "Recon": {"radius": 5.0, "max_iter": 300},
}

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

N_ATTACK_SAMPLES = 5000
OVERSAMPLE_FACTOR = 3  # draw N*5 candidates, keep correctly-classified ones

SCALER = None


# ══════════════════════════════════════════════════════════════════════════════
#  Model definitions  (unchanged from your working version)
# ══════════════════════════════════════════════════════════════════════════════


class MixedInputVAE(nn.Module):
    """VAE with separate continuous (MSE) and binary (BCE) output heads."""

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
        # Reorder from original feature order to [continuous | binary]
        # to match the training-time feature order used in train_vae.py
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
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════


def decode_to_full(cont_out, bin_out, n_features=N_FEATURES):
    B = cont_out.size(0)
    x = torch.zeros(B, n_features, device=cont_out.device)
    x[:, CONTINUOUS_IDX] = cont_out
    x[:, BINARY_IDX] = (bin_out >= 0.5).float()
    return x


def project_constraints(x):
    """
    Project adversarial examples back to valid protocol space.
    Only applies transforms when constraints are actually violated.
    """
    x = x.clone()
    device = x.device
    global SCALER

    if SCALER is None:
        return x

    cont_idx = list(range(0, 23)) + [36]
    x_detached = x.detach()
    x_np = x_detached.cpu().numpy().copy()

    rows_to_fix = set()

    # 1. Check Min <= AVG <= Max in raw space
    x_cont = x_np[:, cont_idx]
    x_raw = SCALER.inverse_transform(x_cont)
    for i in range(len(x_raw)):
        mn, av, mx = x_raw[i, 15], x_raw[i, 17], x_raw[i, 16]
        if not (mn <= av <= mx):
            vals = [mn, av, mx]
            vals.sort()
            x_raw[i, 15], x_raw[i, 17], x_raw[i, 16] = vals[0], vals[1], vals[2]
            rows_to_fix.add(i)

    if rows_to_fix:
        x_cont_proj = SCALER.transform(x_raw)
        x_np[:, cont_idx] = x_cont_proj

    # 2. Fix binary protocol conflicts (check ALL rows)
    binary = x_np[:, 23:36]
    tcp_idx, udp_idx, arp_idx, icmp_idx = 6, 7, 9, 10
    has_flags = (x_np[:, 3:10] > 0).any(axis=1)
    tcp_udp_both = (binary[:, tcp_idx] == 1) & (binary[:, udp_idx] == 1)
    arp_conflict = (binary[:, arp_idx] == 1) & (
        (binary[:, tcp_idx] == 1)
        | (binary[:, udp_idx] == 1)
        | (binary[:, icmp_idx] == 1)
    )
    icmp_conflict = (binary[:, icmp_idx] == 1) & (
        (binary[:, tcp_idx] == 1) | (binary[:, udp_idx] == 1)
    )

    for i in range(len(x_np)):
        changed = False
        if tcp_udp_both[i]:
            x_np[i, 30] = 0
            changed = True
        if arp_conflict[i]:
            x_np[i, 29] = 0
            x_np[i, 30] = 0
            x_np[i, 33] = 0
            changed = True
        if icmp_conflict[i]:
            x_np[i, 29] = 0
            x_np[i, 30] = 0
            changed = True
        if has_flags[i] and x_np[i, 29] != 1:
            x_np[i, 29] = 1
            changed = True
        if changed:
            rows_to_fix.add(i)

    return torch.tensor(x_np, dtype=x.dtype, device=device)


def cw_loss(logits, original_class, kappa=KAPPA):
    B = logits.size(0)
    correct_scores = logits[torch.arange(B), original_class]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(B), original_class] = False
    other_scores = logits[mask].view(B, -1).max(dim=1).values
    return torch.clamp(correct_scores - other_scores, min=-kappa).mean()


def get_cnnlstm_preds(X_np, cnnlstm):
    """Batch inference in eval mode, returns integer predictions."""
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
#  Load helpers
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
        feature_names = clf.feature_names_in_  # stored by sklearn during fit
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
        # Fallback: raw booster — label order assumed to match label_encoder.pkl
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
#  Attack functions  (unchanged from your working version)
# ══════════════════════════════════════════════════════════════════════════════


def latent_attack_cnnlstm(
    x_batch, original_class_idx, vae, cnnlstm, radius, max_iter, lam=LAMBDA_CW, lr=LR
):
    vae.eval()
    cnnlstm.train()  # cuDNN RNN backward requires training mode
    for module in cnnlstm.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()

    B = x_batch.size(0)
    with torch.no_grad():
        mu, _ = vae.encode(x_batch)

    delta = torch.zeros_like(mu, requires_grad=True)
    optimiser = optim.Adam([delta], lr=lr)
    orig_labels = torch.full((B,), original_class_idx, dtype=torch.long, device=DEVICE)

    for _ in range(max_iter):
        optimiser.zero_grad()
        z_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z_proj = torch.where(z_norm > radius, delta * radius / z_norm, delta)
        z_adv = mu + z_proj
        cont_out, bin_out = vae.decode(z_adv)
        x_adv_t = decode_to_full(cont_out, bin_out)
        x_adv_t = project_constraints(x_adv_t)
        logits = cnnlstm(x_adv_t)
        loss = z_proj.norm(dim=1).mean() + lam * cw_loss(logits, orig_labels)
        loss.backward()
        optimiser.step()

    cnnlstm.eval()
    with torch.no_grad():
        z_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z_proj = torch.where(z_norm > radius, delta * radius / z_norm, delta)
        z_adv = mu + z_proj
        cont_out, bin_out = vae.decode(z_adv)
        x_adv_final = decode_to_full(cont_out, bin_out)
        x_adv_final = project_constraints(x_adv_final)
        preds = cnnlstm(x_adv_final).argmax(dim=1)
        success = (preds != orig_labels).cpu().numpy()

    return x_adv_final.cpu().numpy(), success


def latent_attack_lgbm(
    x_batch, original_class_idx, vae, lgbm_model, radius, max_iter, lr=LR
):
    vae.eval()
    B = x_batch.size(0)

    with torch.no_grad():
        mu, _ = vae.encode(x_batch)

    mu_np = mu.cpu().numpy().copy()
    mu_orig_np = mu.cpu().numpy().copy()  # fixed anchor for L2 projection

    best_x_adv = x_batch.cpu().numpy().copy()
    init_proba = lgbm_model.predict_proba(best_x_adv)
    best_scores = init_proba[:, original_class_idx].copy()

    rng = np.random.default_rng(42)

    for step in range(max_iter):
        noise = rng.standard_normal(mu_np.shape).astype(np.float32)
        noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
        step_size = radius * (1 - step / max_iter) * 0.1
        z_cand = mu_np + noise / (noise_norm + 1e-8) * step_size

        # Always project onto ball around the ORIGINAL mu
        diff = z_cand - mu_orig_np
        diff_norm = np.linalg.norm(diff, axis=1, keepdims=True)
        z_cand = np.where(
            diff_norm > radius, mu_orig_np + diff * radius / diff_norm, z_cand
        )

        z_t = torch.tensor(z_cand, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            cont_out, bin_out = vae.decode(z_t)
            x_cand_t = decode_to_full(cont_out, bin_out)
            x_cand_t = project_constraints(x_cand_t)
            x_cand = x_cand_t.cpu().numpy()

        proba = lgbm_model.predict_proba(x_cand)
        cand_score = proba[:, original_class_idx]

        improved = cand_score < best_scores
        best_scores = np.where(improved, cand_score, best_scores)
        best_x_adv[improved] = x_cand[improved]
        mu_np[improved] = z_cand[improved]

    best_x_adv_t = project_constraints(torch.tensor(best_x_adv, device=DEVICE))
    best_x_adv = best_x_adv_t.cpu().numpy()

    final_preds = lgbm_model.predict(best_x_adv)
    success = final_preds != original_class_idx

    l2_dist = np.linalg.norm(best_x_adv - x_batch.cpu().numpy(), axis=1)
    print(
        f"    [LGBM sanity] mean L2 dist original->adv: {l2_dist.mean():.4f}  "
        f"min: {l2_dist.min():.4f}  max: {l2_dist.max():.4f}"
    )

    return best_x_adv, success


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    global SCALER
    print(f"Device: {DEVICE}")

    with open(os.path.join(PROCESSED, "scaler.pkl"), "rb") as f:
        SCALER = pickle.load(f)

    # ── Load label encoder first ──
    with open(os.path.join(NIDS_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    # ── Load test data ──
    X_test = np.load(os.path.join(PROCESSED, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(PROCESSED, "y_test.npy"), allow_pickle=True)
    if y_test.dtype.kind in ("U", "O", "S"):
        y_test = le.transform(y_test).astype(int)
    else:
        y_test = y_test.astype(int)

    print(f"Classes: {class_names}")
    print(f"Test set size: {X_test.shape}")
    print(f"y_test unique values: {np.unique(y_test)}  (expected 0–{num_classes - 1})")

    # ── Load victim models ──
    lgbm = load_lgbm()
    cnnlstm = load_cnnlstm(num_classes)
    print("Victim models loaded.")

    # ── Label alignment sanity check ──
    sample_preds = lgbm.predict(X_test[:10])
    print(f"[Sanity] LGBM predictions on first 10 samples: {sample_preds}")
    print(f"[Sanity] True labels  for first 10 samples:    {y_test[:10]}")

    all_results = {}

    for class_name in ATTACK_CLASSES:
        if class_name not in class_names:
            print(f"[WARN] {class_name} not in label encoder, skipping.")
            continue

        class_idx = class_names.index(class_name)
        cfg = CLASS_ATTACK_CONFIG[class_name]
        radius = cfg["radius"]
        max_iter = cfg["max_iter"]

        # ── Sample selection: correctly classified by BOTH models ────────────
        mask = y_test == class_idx
        X_pool = X_test[mask][: N_ATTACK_SAMPLES * OVERSAMPLE_FACTOR]
        n_pool = len(X_pool)

        lgbm_preds = lgbm.predict(X_pool)
        cnnlstm_preds = get_cnnlstm_preds(X_pool, cnnlstm)

        correct_mask = (lgbm_preds == class_idx) & (cnnlstm_preds == class_idx)
        X_cls = X_pool[correct_mask][:N_ATTACK_SAMPLES]
        n_samples = len(X_cls)

        print(f"\n{'=' * 60}")
        print(
            f"Class: {class_name}  |  pool={n_pool}  "
            f"both-correct={correct_mask.sum()}  attacking={n_samples}  "
            f"(radius={radius}, max_iter={max_iter})"
        )
        print(f"{'=' * 60}")

        if n_samples < 50:
            print(f"  [WARN] Only {n_samples} correctly-classified samples — skipping.")
            continue

        vae = load_vae(class_name, VAE_HIDDEN[class_name])
        x_tensor = torch.tensor(X_cls, device=DEVICE)

        # ── CNN-LSTM attack ──────────────────────────────────────────────────
        print("  Running latent attack vs CNN-LSTM ...")
        adv_cnnlstm_list, success_cnnlstm_list = [], []
        for start in tqdm(
            range(0, n_samples, BATCH_SIZE), desc=f"  {class_name}/CNNLSTM"
        ):
            xb = x_tensor[start : start + BATCH_SIZE]
            adv, suc = latent_attack_cnnlstm(
                xb, class_idx, vae, cnnlstm, radius=radius, max_iter=max_iter
            )
            adv_cnnlstm_list.append(adv)
            success_cnnlstm_list.append(suc)

        adv_cnnlstm = np.vstack(adv_cnnlstm_list)
        suc_cnnlstm = np.concatenate(success_cnnlstm_list)
        asr_cnnlstm = suc_cnnlstm.mean()
        print(f"  CNN-LSTM ASR: {asr_cnnlstm:.4f}  ({suc_cnnlstm.sum()}/{n_samples})")

        # ── LightGBM attack ──────────────────────────────────────────────────
        print("  Running latent attack vs LightGBM ...")
        adv_lgbm_list, success_lgbm_list = [], []
        for start in tqdm(range(0, n_samples, BATCH_SIZE), desc=f"  {class_name}/LGBM"):
            xb = x_tensor[start : start + BATCH_SIZE]
            adv, suc = latent_attack_lgbm(
                xb, class_idx, vae, lgbm, radius=radius, max_iter=max_iter
            )
            adv_lgbm_list.append(adv)
            success_lgbm_list.append(suc)

        adv_lgbm = np.vstack(adv_lgbm_list)
        suc_lgbm = np.concatenate(success_lgbm_list)
        asr_lgbm = suc_lgbm.mean()
        print(f"  LightGBM ASR: {asr_lgbm:.4f}  ({suc_lgbm.sum()}/{n_samples})")

        # ── Save ─────────────────────────────────────────────────────────────
        save_path = os.path.join(RESULTS_DIR, class_name.lower())
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "adv_cnnlstm.npy"), adv_cnnlstm)
        np.save(os.path.join(save_path, "adv_lgbm.npy"), adv_lgbm)
        np.save(os.path.join(save_path, "original.npy"), X_cls)
        np.save(os.path.join(save_path, "success_cnnlstm.npy"), suc_cnnlstm)
        np.save(os.path.join(save_path, "success_lgbm.npy"), suc_lgbm)

        all_results[class_name] = {
            "n_pool": n_pool,
            "n_correct": int(correct_mask.sum()),
            "n_samples": n_samples,
            "asr_cnnlstm": float(asr_cnnlstm),
            "asr_lgbm": float(asr_lgbm),
        }

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ATTACK SUMMARY  (correctly classified by both models)")
    print(f"{'=' * 60}")
    print(
        f"{'Class':<10} {'Pool':>6} {'BothOK':>8} {'N':>6} "
        f"{'ASR(CNN-LSTM)':>14} {'ASR(LGBM)':>10}"
    )
    print("-" * 58)
    for cls, res in all_results.items():
        print(
            f"{cls:<10} {res['n_pool']:>6} {res['n_correct']:>8} "
            f"{res['n_samples']:>6} "
            f"{res['asr_cnnlstm']:>14.4f} "
            f"{res['asr_lgbm']:>10.4f}"
        )

    with open(os.path.join(RESULTS_DIR, "attack_summary.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
