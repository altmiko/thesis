import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch

sys.path.append(r"D:\thesis\src")
import run_attacks
from validator import validate

ROOT = r"D:\thesis"
RESULTS_DIR = os.path.join(ROOT, "results", "tuning")
VIS_DIR = os.path.join(ROOT, "visualizations")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# We will sweep across these protocol penalties
WEIGHT_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0]
TARGET_CLASS = "DDoS"

def run_tuning_sweep():
    print("=" * 80)
    print(f"TUNING SWEEP: Protocol Penalty Weight Tradeoff vs ASR on {TARGET_CLASS}")
    print("=" * 80)
    
    # ── Setup Models and Data ──────────────────────────────
    class_names = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon']
    class_idx = class_names.index(TARGET_CLASS)

    X_train = np.load(os.path.join(ROOT, "data", "processed", "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(ROOT, "data", "processed", "y_train.npy"), allow_pickle=True)
    
    with open(os.path.join(ROOT, "data", "processed", "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    if y_train.dtype.kind in ('U', 'O', 'S'):
        y_train = le.transform(y_train).astype(int)
    else:
        y_train = y_train.astype(int)

    # ── Load scaler for protocol loss ───────────────────────
    sc_path = os.path.join(ROOT, "data", "processed", "scaler.pkl")
    with open(sc_path, "rb") as f:
        scaler = pickle.load(f)
    run_attacks.SCALER_MEAN_T = torch.tensor(scaler.mean_, dtype=torch.float32, device=run_attacks.DEVICE)
    run_attacks.SCALER_SCALE_T = torch.tensor(scaler.scale_, dtype=torch.float32, device=run_attacks.DEVICE)

    # Calculate bounds & mask
    cont_min_np = X_train.min(axis=0)
    cont_max_np = X_train.max(axis=0)
    cont_min_t = torch.tensor(cont_min_np[run_attacks.CONTINUOUS_IDX], dtype=torch.float32, device=run_attacks.DEVICE)
    cont_max_t = torch.tensor(cont_max_np[run_attacks.CONTINUOUS_IDX], dtype=torch.float32, device=run_attacks.DEVICE)
    safe_mask = run_attacks.compute_safe_feature_mask(X_train)

    # Models
    vae = run_attacks.load_vae(TARGET_CLASS, run_attacks.VAE_HIDDEN[TARGET_CLASS])
    cnnlstm = run_attacks.load_cnnlstm(len(class_names))
    lgbm = run_attacks.load_lgbm()

    # Get sample pool
    mask_cls = (y_train == class_idx)
    X_cls = X_train[mask_cls]
    
    cnnlstm.eval()
    preds_c = run_attacks.get_cnnlstm_preds(X_cls, cnnlstm)
    preds_l = lgbm.predict(X_cls)
    
    correct_mask = (preds_c == class_idx) & (preds_l == class_idx)
    X_pool = X_cls[correct_mask]
    
    # Take a smaller pool for faster tuning sweep
    pool_size = min(200, len(X_pool))
    X_target = X_pool[:pool_size]
    X_target_t = torch.tensor(X_target, dtype=torch.float32, device=run_attacks.DEVICE)

    print(f"Data ready. Using N={pool_size} correctly classified {TARGET_CLASS} samples.")
    
    # Best config for DDoS from run_attacks
    cfg = {'radius': 2.0, 'lambda_cw': 5.0, 'lr': 0.01, 'max_iter': 300}
    
    results = []

    for w in WEIGHT_SWEEP:
        run_attacks.LAMBDA_PROTO = w
        print(f"\n--- Running Sweep for LAMBDA_PROTO = {w} ---")
        
        # 1. Run CNNLSTM
        adv_cnn, suc_cnn = run_attacks.latent_attack_cnnlstm(
            X_target_t, class_idx, vae, cnnlstm, 
            radius=cfg['radius'], max_iter=cfg['max_iter'], 
            lambda_cw=cfg['lambda_cw'], lr=cfg['lr'],
            safe_mask=safe_mask, cont_min_t=cont_min_t, cont_max_t=cont_max_t, use_ste=True
        )
        
        # 2. Validity of CNNLSTM
        val_res_cnn, _ = validate(adv_cnn, label=None)
        valid_cnn = val_res_cnn['__overall_validity__']
        asr_cnn = suc_cnn.mean()
        
        # 3. Run LGBM
        adv_lgbm, suc_lgbm = run_attacks.latent_attack_lgbm(
            X_target_t, class_idx, vae, lgbm,
            radius=cfg['radius'], max_iter=500, lr=cfg['lr'],
            safe_mask=safe_mask, cont_min_np=cont_min_np, cont_max_np=cont_max_np, seed=42
        )
        
        # 4. Validity of LGBM
        val_res_lgbm, _ = validate(adv_lgbm, label=None)
        valid_lgbm = val_res_lgbm['__overall_validity__']
        asr_lgbm = suc_lgbm.mean()
        
        results.append({
            'Weight': w,
            'CNN_ASR': asr_cnn,
            'CNN_Valid': valid_cnn,
            'LGBM_ASR': asr_lgbm,
            'LGBM_Valid': valid_lgbm
        })
        
        print(f"  CNN-LSTM: ASR={asr_cnn:.3%}, Valid={valid_cnn:.3%}")
        print(f"  LGBM    : ASR={asr_lgbm:.3%}, Valid={valid_lgbm:.3%}")
        
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "tuning_results.csv"), index=False)
    
    # ── Plotting IDSR Tradeoff ──────────────────────────────
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Protocol Constraint Penalty Weight (LAMBDA_PROTO)', fontsize=12)
    ax1.set_ylabel('Attack Success Rate (ASR)', color=color, fontsize=12)
    ax1.plot(df['Weight'], df['CNN_ASR'], marker='o', color=color, label='CNN-LSTM ASR', linewidth=2)
    ax1.plot(df['Weight'], df['LGBM_ASR'], marker='x', color='tab:orange', label='LGBM ASR', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:cyan'
    ax2.set_ylabel('Overall Protocol Validity Rate', color=color, fontsize=12)
    ax2.plot(df['Weight'], df['CNN_Valid'], marker='o', color=color, linestyle='--', label='CNN-LSTM Validity', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Tradeoff Curve: ASR over Protocol Validity Penalty (IDSR)', fontsize=14)
    fig.tight_layout()
    plt.grid(alpha=0.3)
    
    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    outpath = os.path.join(VIS_DIR, f"fig7_tuning_tradeoff_idsr.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nSaved tradeoff plot to {outpath}")
    print("=" * 80)

if __name__ == "__main__":
    run_tuning_sweep()
