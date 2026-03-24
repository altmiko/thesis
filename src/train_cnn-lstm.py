"""
train_nids_cnnlstm.py
Simple CNN-LSTM victim model for CICIoT2023 (5-class NIDS classification).
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ── CONFIG ────────────────────────────────────────────────────────
DATA_DIR    = r"D:\thesis\data\processed"
MODEL_DIR   = r"D:\thesis\models\nids"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 2048
EPOCHS      = 30
LR          = 1e-3
PATIENCE    = 5
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── LOAD DATA ─────────────────────────────────────────────────────
print("Loading data...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)

y_train_raw = np.load(os.path.join(DATA_DIR, "y_train.npy"), allow_pickle=True)
y_test_raw  = np.load(os.path.join(DATA_DIR, "y_test.npy"),  allow_pickle=True)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

y_train = le.transform(y_train_raw).astype(np.int64)
y_test  = le.transform(y_test_raw).astype(np.int64)

NUM_FEATURES = X_train.shape[1]
NUM_CLASSES  = len(le.classes_)
print(f"  Features: {NUM_FEATURES}, Classes: {NUM_CLASSES}, Device: {DEVICE}")

# Class weights
cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw_tensor = torch.tensor(cw, dtype=torch.float32).to(DEVICE)

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                          batch_size=BATCH_SIZE, shuffle=False)

# ── MODEL ─────────────────────────────────────────────────────────
class SimpleCNNLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),                          # (B, 32, F/2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64,
                            batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)                           # (B, 1, F)
        x = self.cnn(x)                              # (B, 32, F/2)
        x = x.permute(0, 2, 1)                       # (B, F/2, 32)
        _, (h, _) = self.lstm(x)                     # h: (1, B, 64)
        x = self.fc(h.squeeze(0))                    # (B, classes)
        return x

model = SimpleCNNLSTM(NUM_FEATURES, NUM_CLASSES).to(DEVICE)
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# ── TRAIN ─────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=cw_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
patience_counter = 0
best_path = os.path.join(MODEL_DIR, "nids_cnnlstm.pt")

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()

    # Validate
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            logits = model(X_b)
            val_loss += criterion(logits, y_b).item() * X_b.size(0)
            correct  += (logits.argmax(1) == y_b).sum().item()
            total    += X_b.size(0)
    val_loss /= total
    val_acc   = correct / total

    print(f"Epoch {epoch:02d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# ── EVALUATE ──────────────────────────────────────────────────────
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        preds = model(X_b.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_b.numpy())

print(f"\nTest Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels)):.4f}")
print(classification_report(all_labels, all_preds, target_names=le.classes_))
print(f"Model saved to {best_path}")