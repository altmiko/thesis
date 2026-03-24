# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thesis project: **VAE-based latent-space adversarial attacks on NIDS (Network Intrusion Detection Systems)** using the CICIoT2023 IoT network traffic dataset. Adapts the VAE-TabAttack framework for IoT intrusion detection.

The attack pipeline: encode attack-class samples with a class-specific VAE → perturb the latent mean μ under an L2 ball constraint → decode back to feature space → evaluate whether victim classifiers are fooled (Attack Success Rate).

## Running Scripts

All scripts are standalone Python files with no package structure. Run from anywhere (paths are hardcoded):

```bash
python train_lightgbm.py     # Train LightGBM NIDS classifier (victim model)
python train_cnn-lstm.py     # Train CNN-LSTM NIDS classifier (victim model)
python train_vae.py          # Train one VAE per attack class (DDoS, DoS, Mirai, Recon)
python run_attacks.py        # Run latent-space attacks against both victim models
python debug.py              # Data sanity checks (binary/continuous feature ranges)
```

**Execution order matters**: train_lightgbm → train_cnn-lstm → train_vae → run_attacks. The LightGBM script creates `label_encoder.pkl` which all other scripts depend on.

## Architecture

### Data Layout (39 features)
- **Continuous features**: indices 0–22 and 38 (24 total)
- **Binary features**: indices 23–37 (15 total)
- This split is defined identically in `train_vae.py`, `run_attacks.py`, and `debug.py` — keep them in sync.

### Models
- **MixedInputVAE**: Encoder→(μ, log_var)→Decoder with separate output heads: continuous (MSE loss) and binary (BCE + sigmoid). One VAE trained per attack class. The VAE in `train_vae.py` takes reordered input (continuous first, binary last); the copy in `run_attacks.py` takes original 39-feature order.
- **SimpleCNNLSTM**: Conv1d→MaxPool→LSTM→FC. Victim NIDS classifier (5-class).
- **LightGBM**: Gradient-boosted tree victim classifier, loaded via `lgb.Booster` with a custom `LGBMWrapper` for predict/predict_proba interface.

### Attack Strategies
- **vs CNN-LSTM** (`latent_attack_cnnlstm`): Gradient-based C&W-style loss. Optimizes delta in latent space with Adam. Requires `cnnlstm.train()` for cuDNN RNN backward compatibility (BatchNorm layers kept in eval).
- **vs LightGBM** (`latent_attack_lgbm`): Random-search (non-differentiable model). Explores latent space with decaying step size, projects onto L2 ball around original μ.

### Per-Class Attack Config
Attack radius and max_iter are tuned per class (DDoS=3.0/200, DoS=8.0/500, Mirai=10.0/500, Recon=5.0/300) based on empirical ASR results.

## Directory Structure (D:\thesis)

```
data/processed/     X_train.npy, X_test.npy, y_train.npy, y_test.npy,
                    feature_names.pkl, scaler.pkl, label_encoder.pkl
models/nids/        nids_lgbm.txt, nids_cnnlstm.pt, label_encoder.pkl
models/vae/         vae_ddos.pt, vae_dos.pt, vae_mirai.pt, vae_recon.pt
results/            Per-class subdirs with adv_cnnlstm.npy, adv_lgbm.npy, etc.
VAE-TabAttack/      Reference implementation (the paper's original codebase)
notebooks/          Jupyter notebooks from VAE-TabAttack experiments
```

## Key Conventions

- All file paths are **hardcoded** to `D:\thesis\...` — no CLI args or config files.
- VAE checkpoints store metadata: `model_state_dict`, `hidden_dims`, `latent_dim`, `continuous_idx`, `binary_idx`, `history`.
- The VAE in `train_vae.py` reorders features to [continuous | binary] before training. The VAE copy in `run_attacks.py` accepts the original 39-dim feature vector directly. These are intentionally different interfaces.
- Device selection is automatic (CUDA if available, else CPU).
- Random seeds are set to 42 throughout.

## Dependencies

PyTorch, LightGBM (with GPU support), NumPy, scikit-learn, tqdm, pickle (stdlib).
