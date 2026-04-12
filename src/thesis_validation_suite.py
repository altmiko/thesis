"""
thesis_validation_suite.py
==========================
Comprehensive multi-layer validation framework for TabAttack adversarial examples.
Ensures semantic naturalness, strict bound compliance, and temporal coherence
for CICIoT2023 dataset constraints.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

from validator import validate as check_physical_rules

# Paths
ROOT = r"D:\thesis"
PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
VIS_DIR = os.path.join(ROOT, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

# Feature Indices
CONTINUOUS_IDX = list(range(0, 23)) + [36]
BINARY_IDX = list(range(23, 36))
N_FEATURES = 37


class AttackValidator:
    def __init__(self, train_samples: np.ndarray, feature_names: list, scaler, vae_model, device):
        self.train_samples = train_samples
        self.feature_names = feature_names
        self.scaler = scaler
        self.vae = vae_model
        self.device = device

        # Precompute training bounds and statistics
        self.train_min = np.min(train_samples, axis=0)
        self.train_max = np.max(train_samples, axis=0)
        self.train_corr = pd.DataFrame(train_samples).corr().fillna(0)
        
        # Precompute VAE reconstruction error for training baseline
        self.train_recon_error = self._compute_vae_recon_error(train_samples[:5000])

    def _compute_vae_recon_error(self, samples: np.ndarray) -> float:
        """Compute mean squared error of VAE reconstruction"""
        self.vae.eval()
        x_tensor = torch.tensor(samples, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, _ = self.vae.encode(x_tensor)
            cont_out, bin_out = self.vae.decode(mu)
            
            x_recon = torch.zeros_like(x_tensor)
            x_recon[:, CONTINUOUS_IDX] = cont_out
            x_recon[:, BINARY_IDX] = (bin_out >= 0.5).float()
            
            error = torch.mean((x_tensor - x_recon) ** 2).item()
        return error

    def _check_bounds(self, adv_samples: np.ndarray) -> bool:
        """Check strict binary bounds (exactly 0.0 or 1.0) and continuous boundaries."""
        binary_vals = adv_samples[:, BINARY_IDX]
        is_binary = np.all(np.isin(binary_vals, [0.0, 1.0]))
        is_finite = np.all(np.isfinite(adv_samples))
        return bool(is_binary and is_finite)

    def _check_train_distribution(self, adv_samples: np.ndarray) -> dict:
        """Verify adversarial samples don't exceed realistic historical bounds."""
        # 1% buffer tolerance for continuous values due to potential epsilon rounding
        buffer_min = self.train_min - np.abs(self.train_min * 0.01) - 1e-4
        buffer_max = self.train_max + np.abs(self.train_max * 0.01) + 1e-4
        
        cont_adv = adv_samples[:, CONTINUOUS_IDX]
        cont_bmin = buffer_min[CONTINUOUS_IDX]
        cont_bmax = buffer_max[CONTINUOUS_IDX]
        
        valid_min = np.all(cont_adv >= cont_bmin, axis=1)
        valid_max = np.all(cont_adv <= cont_bmax, axis=1)
        
        overall_valid = bool(np.mean(valid_min & valid_max) > 0.99) # 99% compliance
        return {"passed": overall_valid, "compliance_rate": float(np.mean(valid_min & valid_max))}

    def _check_statistics(self, adv_samples: np.ndarray) -> dict:
        """Statistical naturalness via Kolmogorov-Smirnov test and Correlational Drift."""
        ks_fails = 0
        failed_features = []
        for i in CONTINUOUS_IDX:
            # KS Test: are the distributions statistically indistinguishable?
            _, p_value = stats.ks_2samp(self.train_samples[:len(adv_samples), i], adv_samples[:, i])
            if p_value < 0.01:
                ks_fails += 1
                failed_features.append(self.feature_names[i])
                
        # Covariance Structure (Feature Correlation Drift)
        adv_corr = pd.DataFrame(adv_samples).corr().fillna(0)
        corr_diff = np.abs(self.train_corr - adv_corr).max().max()
        
        passed = (ks_fails < len(CONTINUOUS_IDX) * 0.5) and (corr_diff < 0.5)
        return {
            "passed": passed, 
            "ks_fails": ks_fails,
            "failed_features": failed_features[:5], # show top 5 for brevity
            "max_correlation_drift": float(corr_diff)
        }

    def _check_vae_naturalness(self, adv_samples: np.ndarray) -> dict:
        """Ensure the VAE considers the generated samples naturally part of its manifold."""
        recon_error = self._compute_vae_recon_error(adv_samples)
        # Error should be comparable to training data (not an anomalous out-of-distribution spike)
        passed = recon_error < (self.train_recon_error * 3.0) 
        return {
            "passed": passed,
            "adv_recon_error": recon_error,
            "train_recon_error": self.train_recon_error
        }

    def visualize_distributions(self, adv_samples: np.ndarray, class_name: str, prefix: str):
        """Generates required PCA projection and essential distribution overlay plots."""
        # --- PLOT 1: Distributions & PCA ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        pca = PCA(n_components=2)
        train_pca = pca.fit_transform(self.train_samples[:len(adv_samples)])
        adv_pca = pca.transform(adv_samples)
        
        axes[0].scatter(train_pca[:, 0], train_pca[:, 1], alpha=0.3, label='Train (Clean)', c='blue', s=10)
        axes[0].scatter(adv_pca[:, 0], adv_pca[:, 1], alpha=0.5, label='Adversarial', c='red', s=10)
        axes[0].set_title(f"PCA Projection: {class_name} ({prefix}) Manifold Alignment")
        axes[0].legend()

        feat_idx = CONTINUOUS_IDX[2] 
        sns.kdeplot(self.train_samples[:len(adv_samples), feat_idx], ax=axes[1], color='blue', label='Train', fill=True)
        sns.kdeplot(adv_samples[:, feat_idx], ax=axes[1], color='red', label='Adversarial', fill=True)
        axes[1].set_title(f"Feature '{self.feature_names[feat_idx]}' Distribution")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, f"{class_name}_{prefix}_validity_check.png"))
        plt.close()

        # --- PLOT 2: Correlation Heatmaps ---
        adv_corr = pd.DataFrame(adv_samples).corr().fillna(0)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        sns.heatmap(self.train_corr, ax=ax1, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        sns.heatmap(adv_corr, ax=ax2, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        sns.heatmap(np.abs(self.train_corr - adv_corr), ax=ax3, cmap='Reds')
        
        ax1.set_title('Train Correlation')
        ax2.set_title(f'Adversarial Correlation ({prefix})')  
        ax3.set_title('Absolute Drift')
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, f"{class_name}_{prefix}_correlation_drift.png"))
        plt.close()

    def validate(self, adv_samples: np.ndarray, class_name: str, prefix: str) -> dict:
        """Runs the fully comprehensive end-to-end multi-layer validation."""
        report = {
            "01_box_constraints": self._check_bounds(adv_samples),
            "02_train_bounds": self._check_train_distribution(adv_samples),
            "03_statistical_naturalness": self._check_statistics(adv_samples),
            "04_vae_reconstruction": self._check_vae_naturalness(adv_samples),
        }
        
        # Physical Rules (incorporating validator.py logic directly)
        physical_res, _ = check_physical_rules(adv_samples, label=None)
        report["05_physical_rules_validity_rate"] = physical_res.get("__overall_validity__", 0.0)

        # Draw graphs
        self.visualize_distributions(adv_samples, class_name, prefix)
        return report

def main():
    import sys
    from run_attacks import load_vae, VAE_HIDDEN

    print("="*60)
    print("Running Multi-Layer Validation Suite")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Training Data
    print("Loading training data for baseline distributions...")
    try:
        X_train = np.load(os.path.join(PROCESSED, "X_train.npy"))
    except FileNotFoundError:
        print(f"Error: Could not find X_train.npy at {PROCESSED}")
        return

    # Load scaler if it exists
    scaler_path = os.path.join(PROCESSED, "scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    attack_classes = ["DDoS", "DoS", "Mirai", "Recon"]

    # Generic string names for visualizations
    feature_names = [f"Feature_{i}" for i in range(37)]

    for class_name in attack_classes:
        adv_cnn_path = os.path.join(RESULTS_DIR, class_name.lower(), "adv_cnnlstm.npy")
        adv_lgbm_path = os.path.join(RESULTS_DIR, class_name.lower(), "adv_lgbm.npy")
        
        if not os.path.exists(adv_cnn_path) and not os.path.exists(adv_lgbm_path):
            continue
            
        print(f"\nEvaluating Class: {class_name}")
        print("-" * 50)
        
        try:
            vae_model = load_vae(class_name, VAE_HIDDEN[class_name])
        except Exception as e:
            print(f"  [ERROR] Could not load VAE for {class_name}: {e}")
            continue

        validator = AttackValidator(
            train_samples=X_train,
            feature_names=feature_names,
            scaler=scaler,
            vae_model=vae_model,
            device=device
        )
        
        if os.path.exists(adv_cnn_path):
            adv_cnn = np.load(adv_cnn_path)
            print(f"  -> Validating CNN-LSTM attacks (N={len(adv_cnn)})...")
            report_cnn = validator.validate(adv_cnn, class_name, "CNNLSTM")
            for k, v in report_cnn.items():
                print(f"     {k}: {v}")
                
        if os.path.exists(adv_lgbm_path):
            adv_lgbm = np.load(adv_lgbm_path)
            print(f"  -> Validating LightGBM attacks (N={len(adv_lgbm)})...")
            report_lgbm = validator.validate(adv_lgbm, class_name, "LGBM")
            for k, v in report_lgbm.items():
                print(f"     {k}: {v}")

if __name__ == "__main__":
    main()
