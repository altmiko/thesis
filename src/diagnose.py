import numpy as np, pickle, joblib, torch, os

PROCESSED = r"D:\thesis\data\processed"
NIDS_DIR = r"D:\thesis\models\nids"
RESULTS = r"D:\thesis\results"

with open(os.path.join(NIDS_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)
class_names = list(le.classes_)

clf = joblib.load(os.path.join(NIDS_DIR, "nids_lgbm_sklearn.pkl"))
X_test = np.load(os.path.join(PROCESSED, "X_test.npy")).astype(np.float32)
y_test = np.load(os.path.join(PROCESSED, "y_test.npy"), allow_pickle=True)
y_test = le.transform(y_test).astype(int)

for class_name in ["DDoS", "DoS", "Mirai", "Recon"]:
    idx = class_names.index(class_name)
    mask = y_test == idx
    X_cls = X_test[mask][:500]

    lgbm_preds = clf.predict(X_cls)
    lgbm_correct = (lgbm_preds == idx).mean()
    print(
        f"{class_name}: LGBM correctly classifies {lgbm_correct:.1%} of original samples before attack"
    )
