"""
baseline_pgd.py — Unconstrained PGD & FGSM Baseline Attacks
=============================================================
Feature-space adversarial attacks (no VAE, no validity constraints).
Purpose: comparison baseline showing higher ASR but lower protocol validity
         than the VAE latent-space method.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = r"D:\thesis"
PROCESSED = os.path.join(ROOT, "data", "processed")
NIDS_DIR = os.path.join(ROOT, "models", "nids")
RESULTS_DIR = os.path.join(ROOT, "results")

CONTINUOUS_IDX = list(range(0, 23)) + [36]   # 24 continuous
BINARY_IDX     = list(range(23, 36))         # 13 binary
N_FEATURES     = 37

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 128
KAPPA    = 0.0

PGD_EPS      = 2.0     # L2 ball radius
PGD_ALPHA    = 0.05    # step size
PGD_ITERS    = 200     # iterations
FGSM_EPS     = 0.3     # L∞ perturbation budget

ATTACK_CLASSES = {"DDoS": 1, "DoS": 2, "Mirai": 3, "Recon": 4}

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

def cw_loss(logits, original_class, kappa=KAPPA):
    B = logits.size(0)
    correct_scores = logits[torch.arange(B, device=logits.device), original_class]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(B, device=logits.device), original_class] = False
    other_scores = logits[mask].view(B, -1).max(dim=1).values
    return torch.clamp(correct_scores - other_scores, min=-kappa).mean()

def pgd_attack(model, x_batch, class_idx, eps=PGD_EPS, alpha=PGD_ALPHA, n_iters=PGD_ITERS):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    B = x_batch.size(0)
    labels = torch.full((B,), class_idx, dtype=torch.long, device=x_batch.device)
    x_adv = x_batch.clone().detach().requires_grad_(True)
    for _ in range(n_iters):
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        logits = model(x_adv)
        loss = cw_loss(logits, labels)
        loss.backward()
        with torch.no_grad():
            grad = x_adv.grad
            grad_norm = grad.norm(dim=1, keepdim=True).clamp(min=1e-8)
            x_adv = x_adv - alpha * grad / grad_norm
            delta = x_adv - x_batch
            delta_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
            delta = torch.where(delta_norm > eps, delta * eps / delta_norm, delta)
            x_adv = (x_batch + delta).detach().requires_grad_(True)
    model.eval()
    with torch.no_grad():
        preds = model(x_adv).argmax(dim=1)
        success = (preds != labels).cpu().numpy()
    return x_adv.detach().cpu().numpy(), success

def fgsm_attack(model, x_batch, class_idx, eps=FGSM_EPS):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    B = x_batch.size(0)
    labels = torch.full((B,), class_idx, dtype=torch.long, device=x_batch.device)
    x_adv = x_batch.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = cw_loss(logits, labels)
    loss.backward()
    with torch.no_grad():
        x_adv = x_adv - eps * x_adv.grad.sign()
    model.eval()
    with torch.no_grad():
        preds = model(x_adv).argmax(dim=1)
        success = (preds != labels).cpu().numpy()
    return x_adv.detach().cpu().numpy(), success

def main():
    with open(os.path.join(PROCESSED, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)
    class_to_idx = {c: i for i, c in enumerate(le.classes_)}
    cnnlstm = SimpleCNNLSTM(N_FEATURES, num_classes).to(DEVICE)
    ckpt = os.path.join(NIDS_DIR, "nids_cnnlstm.pt")
    cnnlstm.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    cnnlstm.eval()
    for class_name, _ in ATTACK_CLASSES.items():
        class_idx = class_to_idx[class_name]
        class_dir = os.path.join(RESULTS_DIR, class_name.lower())
        orig_path = os.path.join(class_dir, "original.npy")
        if not os.path.exists(orig_path):
            continue
        X_orig = np.load(orig_path)
        N = X_orig.shape[0]
        
        adv_pgd_list, suc_pgd_list = [], []
        for start in tqdm(range(0, N, BATCH), desc=f"PGD [{class_name}]"):
            xb = torch.tensor(X_orig[start:start+BATCH], dtype=torch.float32, device=DEVICE)
            adv, suc = pgd_attack(cnnlstm, xb, class_idx)
            adv_pgd_list.append(adv)
            suc_pgd_list.append(suc)
        adv_pgd = np.vstack(adv_pgd_list)
        suc_pgd = np.concatenate(suc_pgd_list)
        np.save(os.path.join(class_dir, "adv_pgd_cnnlstm.npy"), adv_pgd)
        np.save(os.path.join(class_dir, "success_pgd_cnnlstm.npy"), suc_pgd)

        adv_fgsm_list, suc_fgsm_list = [], []
        for start in tqdm(range(0, N, BATCH), desc=f"FGSM [{class_name}]"):
            xb = torch.tensor(X_orig[start:start+BATCH], dtype=torch.float32, device=DEVICE)
            adv, suc = fgsm_attack(cnnlstm, xb, class_idx)
            adv_fgsm_list.append(adv)
            suc_fgsm_list.append(suc)
        adv_fgsm = np.vstack(adv_fgsm_list)
        suc_fgsm = np.concatenate(suc_fgsm_list)
        np.save(os.path.join(class_dir, "adv_fgsm_cnnlstm.npy"), adv_fgsm)
        np.save(os.path.join(class_dir, "success_fgsm_cnnlstm.npy"), suc_fgsm)

if __name__ == "__main__":
    main()
