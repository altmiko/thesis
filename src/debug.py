import numpy as np
import pickle

DATA_DIR = r"D:\thesis\data\processed"
MODEL_DIR = r"D:\thesis\models\nids"

X_train = np.load(DATA_DIR + r"\X_train.npy")
y_train_raw = np.load(DATA_DIR + r"\y_train.npy", allow_pickle=True)

with open(MODEL_DIR + r"\label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

y_train = le.transform(y_train_raw)

# Check each attack class
for cls in ["DDoS", "DoS", "Mirai", "Recon"]:
    idx = list(le.classes_).index(cls)
    X_class = X_train[y_train == idx]

    # Binary cols 23-35 (13 features, Telnet/IGMP dropped)
    binary = X_class[:, 23:36]
    cont = X_class[:, list(range(0, 23)) + [36]]

    print(f"\n{cls} ({len(X_class)} samples):")
    print(f"  Binary — min:{binary.min():.4f} max:{binary.max():.4f}")
    print(f"  Binary unique value counts (how many are exactly 0 or 1):")
    print(f"    Exactly 0: {(binary == 0).sum()}")
    print(f"    Exactly 1: {(binary == 1).sum()}")
    print(f"    Between 0-1: {((binary > 0) & (binary < 1)).sum()}")
    print(f"    Above 1: {(binary > 1).sum()}")
    print(f"  Continuous — min:{cont.min():.4f} max:{cont.max():.4f}")
