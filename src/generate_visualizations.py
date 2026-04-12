"""
generate_visualizations.py — Thesis Visualization Suite
========================================================
Generates publication-quality figures for supervisor presentation.
"""

import os
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

ROOT        = r"D:\thesis"
PROCESSED   = os.path.join(ROOT, "data", "processed")
VAE_DIR     = os.path.join(ROOT, "models", "vae")
NIDS_DIR    = os.path.join(ROOT, "models", "nids")
RESULTS_DIR = os.path.join(ROOT, "results")
VIZ_DIR     = os.path.join(ROOT, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

CONTINUOUS_IDX = list(range(0, 23)) + [36]
BINARY_IDX     = list(range(23, 36))
N_FEATURES     = 37
LATENT_DIM     = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTACK_CLASSES = ["DDoS", "DoS", "Mirai", "Recon"]
CLASS_COLORS   = {"DDoS":"#FF6B6B", "DoS":"#4ECDC4", "Mirai":"#45B7D1", "Recon":"#FFA07A"}

SAMPLE_STYLES = {
    "Original":      {"color": "#4CAF50", "marker": "o", "alpha": 0.5, "s": 15, "zorder": 2},
    "VAE Adversarial":{"color": "#2196F3", "marker": "^", "alpha": 0.6, "s": 20, "zorder": 3},
    "PGD Adversarial": {"color": "#FF5722", "marker": "x", "alpha": 0.5, "s": 18, "zorder": 4},
    "FGSM Adversarial":{"color": "#9C27B0", "marker": "s", "alpha": 0.5, "s": 15, "zorder": 4},
}

N_LATENT_SAMPLES = 2000

class MixedInputVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, continuous_idx, binary_idx):
        super().__init__()
        self.continuous_idx = continuous_idx
        self.binary_idx = binary_idx
        enc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        self.decoder = nn.Sequential(*dec_layers)
        self.cont_head = nn.Linear(in_dim, len(continuous_idx))
        self.binary_head = nn.Sequential(nn.Linear(in_dim, len(binary_idx)), nn.Sigmoid())

    def encode(self, x):
        x_reordered = torch.cat([x[:, self.continuous_idx], x[:, self.binary_idx]], dim=1)
        h = self.encoder(x_reordered)
        return self.fc_mu(h), self.fc_log_var(h)
    def forward(self, x):
        mu, log_var = self.encode(x)
        return mu, log_var

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e", "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0", "axes.grid": True, "grid.alpha": 0.15, "grid.color": "#ffffff",
        "text.color": "#e0e0e0", "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
        "legend.facecolor": "#0f3460", "legend.edgecolor": "#e0e0e0", "legend.fontsize": 9,
        "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "#1a1a2e"
    })

def load_vae(class_name):
    vae_files = {"DDoS": "vae_ddos.pt", "DoS": "vae_dos.pt", "Mirai": "vae_mirai.pt", "Recon": "vae_recon.pt"}
    hidden_dims_map = {"DDoS": [128, 64], "DoS": [128, 64], "Mirai": [128, 64], "Recon": [256, 128]}
    ckpt = torch.load(os.path.join(VAE_DIR, vae_files[class_name]), map_location=DEVICE)
    vae = MixedInputVAE(
        input_dim=N_FEATURES, hidden_dims=ckpt.get("hidden_dims", hidden_dims_map[class_name]),
        latent_dim=LATENT_DIM, continuous_idx=ckpt.get("continuous_idx", CONTINUOUS_IDX),
        binary_idx=ckpt.get("binary_idx", BINARY_IDX)
    ).to(DEVICE)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    return vae

@torch.no_grad()
def extract_latent_vectors(vae, X_np):
    vae.eval()
    mus = []
    for start in range(0, len(X_np), 512):
        xb = torch.tensor(X_np[start:start+512], dtype=torch.float32, device=DEVICE)
        mu, _ = vae.encode(xb)
        mus.append(mu.cpu().numpy())
    return np.vstack(mus)

def load_attack_results(class_name):
    cdir = os.path.join(RESULTS_DIR, class_name.lower())
    results = {}
    for k, v in [("original", "original.npy"), ("vae_adv", "adv_cnnlstm.npy"), 
                 ("pgd_adv", "adv_pgd_cnnlstm.npy"), ("fgsm_adv", "adv_fgsm_cnnlstm.npy")]:
        p = os.path.join(cdir, v)
        if os.path.exists(p):
            results[k] = np.load(p)
    return results

def plot_latent_pca(all_latents, all_labels):
    Z, labels = np.vstack(all_latents), np.concatenate(all_labels)
    pca = PCA(n_components=2, random_state=42)
    Z_2d = pca.fit_transform(Z)
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in ATTACK_CLASSES:
        mask = labels == cls
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=CLASS_COLORS[cls], label=cls, alpha=0.4, s=12, edgecolors="none")
    ax.legend()
    fig.savefig(os.path.join(VIZ_DIR, "fig1_latent_space_pca.png"))
    plt.close(fig)

def plot_latent_tsne(all_latents, all_labels):
    Z, labels = np.vstack(all_latents), np.concatenate(all_labels)
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000, init="pca")
    Z_2d = tsne.fit_transform(Z)
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in ATTACK_CLASSES:
        mask = labels == cls
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=CLASS_COLORS[cls], label=cls, alpha=0.4, s=12, edgecolors="none")
    ax.legend()
    fig.savefig(os.path.join(VIZ_DIR, "fig2_latent_space_tsne.png"))
    plt.close(fig)

def plot_onmanifold_pca(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for i, cls in enumerate(ATTACK_CLASSES):
        ax, res = axes[i], all_results[cls]
        if "original" not in res: continue
        pca = PCA(n_components=2, random_state=42)
        orig_2d = pca.fit_transform(res["original"])
        ax.scatter(orig_2d[:, 0], orig_2d[:, 1], label="Original", **SAMPLE_STYLES["Original"])
        for k, lbl in [("vae_adv", "VAE Adversarial"), ("pgd_adv", "PGD Adversarial"), ("fgsm_adv", "FGSM Adversarial")]:
            if k in res:
                p2d = pca.transform(res[k])
                ax.scatter(p2d[:, 0], p2d[:, 1], label=lbl, **SAMPLE_STYLES[lbl])
        ax.set_title(cls)
        ax.legend()
    fig.savefig(os.path.join(VIZ_DIR, "fig3_onmanifold_pca.png"))
    plt.close(fig)

def plot_onmanifold_tsne(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for i, cls in enumerate(ATTACK_CLASSES):
        ax, res = axes[i], all_results[cls]
        if "original" not in res: continue
        parts, part_labels = [res["original"]], ["Original"] * len(res["original"])
        for k, lbl in [("vae_adv", "VAE Adversarial"), ("pgd_adv", "PGD Adversarial"), ("fgsm_adv", "FGSM Adversarial")]:
            if k in res:
                parts.append(res[k]); part_labels.extend([lbl] * len(res[k]))
        X_all, part_labels = np.vstack(parts), np.array(part_labels)
        tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000, init="pca")
        X_2d = tsne.fit_transform(X_all)
        for stype, style in SAMPLE_STYLES.items():
            mask = part_labels == stype
            if mask.any(): ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=stype, **style)
        ax.set_title(cls)
        ax.legend()
    fig.savefig(os.path.join(VIZ_DIR, "fig4_onmanifold_tsne.png"))
    plt.close(fig)

def plot_feature_umap(all_results):
    if not HAS_UMAP: return
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for i, cls in enumerate(ATTACK_CLASSES):
        ax, res = axes[i], all_results[cls]
        if "original" not in res: continue
        parts, part_labels = [res["original"]], ["Original"] * len(res["original"])
        for k, lbl in [("vae_adv", "VAE Adversarial"), ("pgd_adv", "PGD Adversarial"), ("fgsm_adv", "FGSM Adversarial")]:
            if k in res:
                parts.append(res[k]); part_labels.extend([lbl] * len(res[k]))
        X_all, part_labels = np.vstack(parts), np.array(part_labels)
        reducer = UMAP(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_all)
        for stype, style in SAMPLE_STYLES.items():
            mask = part_labels == stype
            if mask.any(): ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=stype, **style)
        ax.set_title(cls)
        ax.legend()
    fig.savefig(os.path.join(VIZ_DIR, "fig5_feature_comparison_umap.png"))
    plt.close(fig)

def plot_perturbation_comparison(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, cls in enumerate(ATTACK_CLASSES):
        ax, res = axes[i], all_results[cls]
        if "original" not in res: continue
        data, labels, colors = [], [], []
        for k, lbl, col in [("vae_adv", "VAE", "#2196F3"), ("pgd_adv", "PGD", "#FF5722"), ("fgsm_adv", "FGSM", "#9C27B0")]:
            if k in res:
                data.append(np.linalg.norm(res[k] - res["original"], axis=1))
                labels.append(lbl); colors.append(col)
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, c in zip(bp["boxes"], colors): patch.set_facecolor(c)
        ax.set_title(cls)
    fig.savefig(os.path.join(VIZ_DIR, "fig6_perturbation_comparison.png"))
    plt.close(fig)

def main():
    setup_style()
    X_train = np.load(os.path.join(PROCESSED, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED, "y_train.npy"), allow_pickle=True)
    with open(os.path.join(NIDS_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    y_train_encoded = le.transform(y_train)
    class_to_idx = {c: i for i, c in enumerate(le.classes_)}

    all_latents, all_labels = [], []
    for cls in ATTACK_CLASSES:
        vae = load_vae(cls)
        mask = y_train_encoded == class_to_idx[cls]
        X_cls = X_train[mask]
        if len(X_cls) > N_LATENT_SAMPLES: X_cls = X_cls[np.random.default_rng(42).choice(len(X_cls), N_LATENT_SAMPLES, False)]
        Z_cls = extract_latent_vectors(vae, X_cls)
        all_latents.append(Z_cls); all_labels.append(np.array([cls] * len(Z_cls)))
    plot_latent_pca(all_latents, all_labels)
    plot_latent_tsne(all_latents, all_labels)

    all_results = {cls: load_attack_results(cls) for cls in ATTACK_CLASSES}
    plot_onmanifold_pca(all_results)
    plot_onmanifold_tsne(all_results)
    plot_feature_umap(all_results)
    plot_perturbation_comparison(all_results)

if __name__ == "__main__":
    main()
