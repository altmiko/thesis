import torch
import numpy as np
import sys
sys.path.append(r"d:\thesis\src")
import run_attacks

run_attacks.set_seed(42)
vae = run_attacks.load_vae("DDoS", run_attacks.VAE_HIDDEN["DDoS"])
cnnlstm = run_attacks.load_cnnlstm(5)

x_orig = np.load(f"results/ddos/original.npy")
x_batch = torch.tensor(x_orig[:50], dtype=torch.float32, device=run_attacks.DEVICE)
class_idx = 1
radius=2.0
max_iter=10
lambda_cw=5.0
lr=0.01

safe_mask = torch.ones(37, device=run_attacks.DEVICE)
cont_min_t = torch.zeros(24, device=run_attacks.DEVICE)
cont_max_t = torch.ones(24, device=run_attacks.DEVICE)
use_ste=True

vae.eval()
cnnlstm.train()
for module in cnnlstm.modules():
    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        module.eval()

B = x_batch.size(0)
with torch.no_grad():
    mu, _ = vae.encode(x_batch)

delta = torch.zeros_like(mu, requires_grad=True)
optimiser = torch.optim.Adam([delta], lr=lr)
orig_labels = torch.full((B,), class_idx, dtype=torch.long, device=run_attacks.DEVICE)
conservative_radius = radius * 0.99

for i in range(max_iter):
    optimiser.zero_grad()
    
    z_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
    z_proj = torch.where(
        z_norm > conservative_radius, delta * conservative_radius / z_norm, delta
    )
    z_adv = mu + z_proj

    cont_out, bin_out = vae.decode(z_adv)
    x_decoded = run_attacks.decode_to_full(cont_out, bin_out, use_ste=use_ste)

    x_adv_t = x_batch + safe_mask.unsqueeze(0) * (x_decoded - x_batch)

    cont_clamped = torch.clamp(
        x_adv_t[:, run_attacks.CONTINUOUS_IDX],
        min=cont_min_t,
        max=cont_max_t,
    )
    x_adv_t = x_adv_t.clone()
    x_adv_t[:, run_attacks.CONTINUOUS_IDX] = cont_clamped

    logits = cnnlstm(x_adv_t)
    loss_cw = run_attacks.cw_loss(logits, orig_labels)
    loss_con = run_attacks.constraint_loss(x_adv_t, cont_min_t, cont_max_t)
    loss_recon = torch.mean((x_adv_t - x_batch) ** 2)

    z_proj_norm_mean = z_proj.norm(dim=1).mean()
    loss = (
        0.1 * z_proj_norm_mean
        + lambda_cw * loss_cw
        + run_attacks.LAMBDA_CONSTRAINT * loss_con
        + run_attacks.LAMBDA_RECON * loss_recon
    )

    print(f"Iter {i}: loss={loss.item()} l_cw={loss_cw.item()} " 
          f"l_con={loss_con.item()} l_recon={loss_recon.item()} "
          f"l_z={z_proj_norm_mean.item()} delta_max={delta.max().item()}")

    loss.backward()
    
    print(f"  grad max={delta.grad.max().item()} isnan={torch.isnan(delta.grad).any().item()}")
    optimiser.step()
