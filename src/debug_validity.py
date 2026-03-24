# debug_validity.py
import numpy as np
import torch

# Load original sample
X_orig = np.load("results/ddos/original.npy")[0]  # First sample
print("Original sample shape:", X_orig.shape)
print("First 10 values:", X_orig[:10])

# Check feature mapping
CONTINUOUS_IDX = list(range(0, 23)) + [36]   # 24 features
BINARY_IDX = list(range(23, 36))             # 13 features

print(f"\nContinuous indices ({len(CONTINUOUS_IDX)}): {CONTINUOUS_IDX}")
print(f"Binary indices ({len(BINARY_IDX)}): {BINARY_IDX}")
print(f"Total: {len(CONTINUOUS_IDX) + len(BINARY_IDX)} (should be 37)")

# Decode a VAE output manually
vae = load_vae("DDoS", [128, 64])
x_t = torch.tensor(X_orig[:1], dtype=torch.float32, device=DEVICE)

with torch.no_grad():
    mu, _ = vae.encode(x_t)
    cont, bin_out = vae.decode(mu)
    
    print(f"\nVAE continuous output: {cont}")
    print(f"VAE binary output (sigmoid): {bin_out}")
    
    # Check if indices align
    x_recon = torch.zeros(1, 37, device=DEVICE)
    x_recon[:, CONTINUOUS_IDX] = cont
    x_recon[:, BINARY_IDX] = (bin_out >= 0.5).float()
    
    print(f"\nReconstructed sample first 10:", x_recon[0, :10])