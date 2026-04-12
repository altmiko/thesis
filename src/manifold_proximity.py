import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = r"D:\thesis"
RESULTS_DIR = os.path.join(ROOT, "results")
ATTACK_CLASSES = ['DDoS', 'DoS', 'Mirai', 'Recon']

def manifold_proximity(orig, adv, k=5):
    if np.isnan(adv).sum() > 0:
        return float('nan')
    nn = NearestNeighbors(n_neighbors=k).fit(orig)
    distances, _ = nn.kneighbors(adv)
    return distances.mean()

def manifold_proximity_conditioned(orig, adv, success_mask, k=5):
    """Proximity among successful attacks only."""
    adv_successful = adv[success_mask]
    if len(adv_successful) == 0:
        return float('nan')
    nn = NearestNeighbors(n_neighbors=k).fit(orig)
    distances, _ = nn.kneighbors(adv_successful)
    return distances.mean()

def main():
    print("=" * 80)
    print("MANIFOLD PROXIMITY (KNN L2 Distance to Original Data)")
    print("Lower distance means the attack is closer to the true manifold.")
    print("=" * 80)
    
    col_w = 12
    header = f"{'Class':<{col_w}}{'VAE (all)':>15}{'VAE (succ)':>15}{'PGD (all)':>15}{'PGD (succ)':>15}"
    print(header)
    print("-" * 72)
    
    for attack_class in ATTACK_CLASSES:
        cls_dir = os.path.join(RESULTS_DIR, attack_class.lower())
        
        orig_path = os.path.join(cls_dir, "original.npy")
        vae_path = os.path.join(cls_dir, "adv_cnnlstm.npy")
        vae_succ_path = os.path.join(cls_dir, "success_cnnlstm.npy")
        pgd_path = os.path.join(cls_dir, "adv_pgd_cnnlstm.npy")
        pgd_succ_path = os.path.join(cls_dir, "success_pgd_cnnlstm.npy")
        
        if not os.path.exists(orig_path):
            continue
            
        orig = np.load(orig_path)
        row = f"{attack_class:<{col_w}}"
        
        # VAE
        if os.path.exists(vae_path) and os.path.exists(vae_succ_path):
            vae_adv = np.load(vae_path)
            vae_succ = np.load(vae_succ_path)
            d_vae_all = manifold_proximity(orig, vae_adv)
            d_vae_succ = manifold_proximity_conditioned(orig, vae_adv, vae_succ)
            row += f"{d_vae_all:>15.3f}{d_vae_succ:>15.3f}"
        else:
            row += f"{'N/A':>15}{'N/A':>15}"
            
        # PGD
        if os.path.exists(pgd_path) and os.path.exists(pgd_succ_path):
            pgd_adv = np.load(pgd_path)
            pgd_succ = np.load(pgd_succ_path)
            d_pgd_all = manifold_proximity(orig, pgd_adv)
            d_pgd_succ = manifold_proximity_conditioned(orig, pgd_adv, pgd_succ)
            row += f"{d_pgd_all:>15.3f}{d_pgd_succ:>15.3f}"
        else:
            row += f"{'N/A':>15}{'N/A':>15}"
            
        print(row)
        
    print("-" * 72)

if __name__ == "__main__":
    main()
