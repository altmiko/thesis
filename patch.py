import re

with open("d:/thesis/src/run_attacks.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Import loss_protocol
if "from loss_protocol import compute_protocol_loss" not in text:
    text = text.replace("from tqdm import tqdm", "from tqdm import tqdm\nfrom loss_protocol import compute_protocol_loss")

# 2. Add globals
if "SCALER_MEAN_T = None" not in text:
    text = text.replace("OVERSAMPLE_FACTOR = 5", "OVERSAMPLE_FACTOR = 5\n\nSCALER_MEAN_T = None\nSCALER_SCALE_T = None")

# 3. Patch CNN-LSTM
if "loss_proto = compute_protocol_loss(x_adv_t" not in text:
    text = text.replace("loss_recon = torch.mean((x_adv_t - x_batch) ** 2)", 
                        "loss_recon = torch.mean((x_adv_t - x_batch) ** 2)\n        global SCALER_MEAN_T, SCALER_SCALE_T\n        loss_proto = compute_protocol_loss(x_adv_t, SCALER_MEAN_T, SCALER_SCALE_T)")
    text = text.replace("+ LAMBDA_RECON * loss_recon  # stay close to original",
                        "+ LAMBDA_RECON * loss_recon  # stay close to original\n            + 10.0 * loss_proto")

# 4. Patch LGBM
if "p_loss = compute_protocol_loss(x_cand_clamped" not in text:
    text = text.replace("c_loss = constraint_loss(x_cand_clamped, cont_min_t, cont_max_t)",
                        "c_loss = constraint_loss(x_cand_clamped, cont_min_t, cont_max_t)\n            global SCALER_MEAN_T, SCALER_SCALE_T\n            p_loss = compute_protocol_loss(x_cand_clamped, SCALER_MEAN_T, SCALER_SCALE_T)")
    
    text = text.replace("fitness = target_probs - LAMBDA_CONSTRAINT * c_loss_np",
                        "p_loss_np = p_loss.cpu().numpy()\n            fitness = target_probs - LAMBDA_CONSTRAINT * c_loss_np - 10.0 * p_loss_np")

# 5. Patch main to load scaler
if "global SCALER_MEAN_T, SCALER_SCALE_T\n    sc_path" not in text:
    patch_main = """    print(f"Using {'train' if use_train_bounds else 'test'} set bounds for validity")

    import pickle
    global SCALER_MEAN_T, SCALER_SCALE_T
    with open(os.path.join(NIDS_DIR, "..", "..", "data", "processed", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    SCALER_MEAN_T = torch.tensor(scaler.mean_[CONTINUOUS_IDX], dtype=torch.float32, device=DEVICE)
    SCALER_SCALE_T = torch.tensor(scaler.scale_[CONTINUOUS_IDX], dtype=torch.float32, device=DEVICE)"""
    
    text = text.replace("    print(f\"Using {'train' if use_train_bounds else 'test'} set bounds for validity\")", patch_main)

with open("d:/thesis/src/run_attacks.py", "w", encoding="utf-8") as f:
    f.write(text)
print("Patched successfully.")
