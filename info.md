Thesis: VAE Latent-Space Adversarial Attacks on NIDS (CICIoT2023)
Core Claim
First VAE latent-space adversarial attack on CICIoT2023. Learned manifold constraints
produce more protocol-valid adversarial IoT traffic than unconstrained PGD.
Evaluated with ASR, validity rate, and IDSR metrics.

Environment

OS: Windows 11, remote via AnyDesk
Conda env: thesis (Python 3.10)
GPU: RTX 4080 Super, CUDA 13.2, PyTorch 2.10.0+cu130
Project root: D:\thesis\


Project Structure
D:\thesis\
    data\
        ciciot2023\        ← ciciot2023_base.csv (raw)
        processed\         ← X_train.npy, X_test.npy, y_train.npy, y_test.npy,
                              scaler.pkl, feature_names.pkl
    models\
        vae\               ← vae_ddos.pt, vae_dos.pt, vae_mirai.pt, vae_recon.pt
        nids\              ← nids_lgbm.txt, nids_lgbm_sklearn.pkl,
                              nids_cnnlstm.pt, label_encoder.pkl
    results\
        ddos\              ← adv_cnnlstm.npy, adv_lgbm.npy, original.npy,
        dos\                  success_cnnlstm.npy, success_lgbm.npy
        mirai\
        recon\
        attack_summary.pkl
    src\
        preprocess.py          ✅ done
        train_lightgbm.py      ✅ done
        train_nids_cnnlstm.py  ✅ done
        train_vae.py           ✅ done
        run_attacks.py         ✅ done (VAE latent attack)
        baseline_pgd.py        ⬜ TODO
        evaluate.py            ⬜ TODO
    VAE-TabAttack\             ← reference repo (read only)

Features (39 total, order in .npy files)

Continuous (indices 0–22 + 38): Header_Length, Time_To_Live, Rate,
fin/syn/rst/psh/ack/ece/cwr flag numbers, ack/syn/fin/rst counts,
Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance, Protocol Type
Binary (indices 23–37): HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC,
TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC

pythonCONTINUOUS_IDX = list(range(0, 23)) + [38]   # 24 features
BINARY_IDX     = list(range(23, 38))          # 15 features
N_FEATURES     = 39

Preprocessing Rules (must be respected in all generated samples)

StandardScaler applied to continuous indices only
Binary features clipped to [0, 1], never scaled
Downsampling: DDoS and DoS capped at mirai_count * 1
Label encoder: models/nids/label_encoder.pkl


Model Architectures
MixedInputVAE

Mixed input: continuous MSE head + binary BCE/sigmoid head
Encoder: Linear → BN → ReLU → Dropout (×n layers) → fc_mu + fc_log_var
Decoder: reversed layers → cont_head (Linear) + binary_head (Linear → Sigmoid)
Latent dim: 16, KL annealing: beta ramps 0→kl_weight over first 50% of epochs
Checkpoints are dicts — load with:

python  ckpt = torch.load(path, map_location=device)
  state_dict     = ckpt["model_state_dict"]
  continuous_idx = ckpt.get("continuous_idx", CONTINUOUS_IDX)
  binary_idx     = ckpt.get("binary_idx",     BINARY_IDX)
  latent_dim     = ckpt.get("latent_dim",      16)
  hidden_dims    = ckpt.get("hidden_dims",     [...])
Per-class config:
Classhidden_dimsepochskl_weightDDoS[128, 64]501.5DoS[128, 64]501.5Mirai[128, 64]501.5Recon[256, 128]1001.5
SimpleCNNLSTM (victim model 2)
pythonself.cnn = nn.Sequential(Conv1d(1,32,k=3,pad=1), ReLU(), MaxPool1d(2))
self.lstm = nn.LSTM(32, 64, batch_first=True)
self.dropout = nn.Dropout(0.3)
self.fc = nn.Sequential(Dropout(0.3), Linear(64, num_classes))

def forward(self, x):
    x = x.unsqueeze(1)         # (B, 1, 39)
    x = self.cnn(x)            # (B, 32, 19)
    x = x.permute(0, 2, 1)    # (B, 19, 32)
    _, (hn, _) = self.lstm(x)
    x = self.dropout(hn.squeeze(0))
    return self.fc(x)
⚠️ cuDNN RNN backward requires model.train() during attack loop.
Freeze BN layers explicitly: set any BatchNorm submodules to .eval().
LightGBM (victim model 1)

Saved as both nids_lgbm.txt (raw booster) and nids_lgbm_sklearn.pkl (sklearn)
Always load via sklearn pkl to preserve label encoding alignment:

python  import joblib
  clf = joblib.load("models/nids/nids_lgbm_sklearn.pkl")
  # clf.predict() returns integer class indices matching label_encoder.pkl

Attack Pipeline (run_attacks.py — completed)
VAE Latent Attack
x → VAE.encode() → μ
delta = learnable zero-init tensor (shape: latent_dim)
z' = μ + project(delta, L2 ball radius r)
x' = VAE.decode(z') → threshold binary at 0.5
Loss = ‖delta‖ + λ * CW_loss(NIDS(x'), original_class)
C&W loss (untargeted):
pythonloss = clamp(logit[orig] - max_other_logit, min=-kappa)
Per-class attack config (tuned):
Classradiusmax_iterDDoS3.0200DoS8.0500Mirai10.0500Recon5.0300
Current VAE Attack Results (preliminary, N=500 per class)
ClassASR CNN-LSTMASR LGBMDDoS0.9440.986DoS0.1081.000*Mirai0.0120.798Recon0.4341.000*
*LGBM results pending label alignment fix (nids_lgbm_sklearn.pkl retraining).
DoS/Mirai CNN-LSTM ASR is low — attributed to tight latent clustering; document
as validity/ASR tradeoff finding.

TODO: baseline_pgd.py
Unconstrained PGD attack directly in feature space (no VAE).

Same C&W loss, same victim models, same N=500 per class
Continuous features: gradient step + L∞ or L2 clip
Binary features: straight-through estimator or post-hoc threshold
Save results to results/{class}/adv_pgd_cnnlstm.npy etc.
Purpose: show PGD produces higher ASR but lower protocol validity than VAE

TODO: evaluate.py
Compute all three thesis metrics on saved .npy files:

ASR — fraction of successful adversarial examples (from success masks)
Protocol Validity Rate — binary feature constraint checks:

At most one of TCP/UDP/ICMP/ARP/DHCP should be active per sample
Flag counts consistent with flag bits (e.g. fin_count > 0 ↔ fin_flag > 0)


IDSR (Intrusion Detection Survival Rate) — fraction of adversarial examples
classified as Benign by the victim model


Compare VAE attack vs PGD baseline in a single summary table


Key Known Issues / Limitations

DDoS/DoS confusion exists at dataset level — not fixable, document as limitation
Cont MAE targets (<0.03) not met; binary accuracy (>98%) is the binding constraint
Recon has highest reconstruction error (MAE 0.132, KL 10.7) — high feature variance
cuDNN requires LSTM in train() mode during backward; freeze BN manually
VAE checkpoints are saved as dicts, not raw state_dicts


Related Work (cite in this order in thesis)

VAE-TabAttack (He et al. 2025) — primary method source
Vitorino et al. 2024 (Annals of Telecom) — VAE IoT adversarial, manual constraints
SAAE (Al-Fawa'reh et al. 2025) — deterministic AE NIDS attacks, no probabilistic manifold
A2PM/TXAI-ADV (Electronics 2024) — hand-crafted constraints on CICIoT2023
NetDiffuser (Kumar et al. 2026) — concurrent diffusion NAE, different dataset
LTA (Shaar et al. 2026) — concurrent latent VAE, vision domain only
Constraint invalidity paper — up to 80.3% of NIDS adversarial examples physically invalid