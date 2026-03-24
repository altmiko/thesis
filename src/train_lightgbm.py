import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib
import os

DATA_DIR = r"D:\thesis\data\processed"
MODEL_DIR = r"D:\thesis\models\nids"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"), allow_pickle=True)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"), allow_pickle=True)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 2. Encode labels ──────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(y_train)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)
print(f"Classes: {list(le.classes_)}")

# ── 3. Train LightGBM ─────────────────────────────────────────────────────────
print("\nTraining LightGBM NIDS classifier...")
clf = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=8,
    min_child_samples=50,
    min_data_in_bin=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

clf.fit(
    X_train,
    y_train_enc,
    eval_set=[(X_test, y_test_enc)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
print("\nEvaluating...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test_enc, y_pred)
print(f"\nTest accuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# ── 5. Save ───────────────────────────────────────────────────────────────────
clf.booster_.save_model(os.path.join(MODEL_DIR, "nids_lgbm.txt"))
joblib.dump(clf, os.path.join(MODEL_DIR, "nids_lgbm_sklearn.pkl"))
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# Sanity checks
print(f"[Sanity] LightGBM n_classes: {clf.n_classes_}")
test_proba = clf.predict_proba(X_test[:5])
print(f"[Sanity] LGBM proba shape: {test_proba.shape}")
print(f"[Sanity] LGBM proba sample:\n{test_proba}")
print(f"[Sanity] le.classes_: {le.classes_}")

print(f"\nModel saved to {MODEL_DIR}")
print("NIDS training complete.")
