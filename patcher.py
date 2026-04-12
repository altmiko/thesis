import sys

filepath = "d:/thesis/src/run_attacks.py"

with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Add imports if not present
if "compute_protocol_loss" not in text[:1000]:
    text = text.replace("from tqdm import tqdm", "from tqdm import tqdm\nfrom loss_protocol import compute_protocol_loss")

# 2. Add global variables
if "SCALER_MEAN_T = None" not in text:
    text = text.replace("OVERSAMPLE_FACTOR = 5\n\n# ── Feature indices", "OVERSAMPLE_FACTOR = 5\n\nSCALER_MEAN_T = None\nSCALER_SCALE_T = None\n\n# ── Feature indices")

# 3. Add to LGBM loss (looks like CNNLSTM is already patched)
if "p_loss = compute_protocol_loss" not in text:
    target_lgbm = """            # Constraint and protocol loss to prefer valid candidates
            c_loss = constraint_loss(x_cand_clamped, cont_min_t, cont_max_t)"""
    rep_lgbm = """            # Constraint and protocol loss to prefer valid candidates
            c_loss = constraint_loss(x_cand_clamped, cont_min_t, cont_max_t)
            p_loss = compute_protocol_loss(x_cand_clamped, SCALER_MEAN_T, SCALER_SCALE_T)"""
    text = text.replace(target_lgbm, rep_lgbm, 1)

target_lgbm_fit = """            fitness = target_probs - LAMBDA_CONSTRAINT * c_loss_np"""
rep_lgbm_fit = """            p_loss_np = p_loss.cpu().numpy()
            fitness = target_probs - LAMBDA_CONSTRAINT * c_loss_np - 10.0 * p_loss_np"""
if "p_loss_np" not in text:
    text = text.replace(target_lgbm_fit, rep_lgbm_fit, 1)

# 4. Initialize SCALER_MEAN_T in main
target_main = """    with open(os.path.join(NIDS_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)"""

rep_main = """    global SCALER_MEAN_T, SCALER_SCALE_T
    sc_path = os.path.join(PROCESSED, "scaler.pkl")
    with open(sc_path, "rb") as f:
        scaler = pickle.load(f)
    SCALER_MEAN_T = torch.tensor(scaler.mean_[CONTINUOUS_IDX], dtype=torch.float32, device=DEVICE)
    SCALER_SCALE_T = torch.tensor(scaler.scale_[CONTINUOUS_IDX], dtype=torch.float32, device=DEVICE)

    with open(os.path.join(NIDS_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)"""

if "global SCALER_MEAN_T" not in text:
    text = text.replace(target_main, rep_main, 1)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(text)

print("Patched run_attacks.py successfully")
