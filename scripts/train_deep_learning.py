"""
Deep Learning for PPG-based BP Estimation.

Models:
  1. 1D-CNN: 浅层卷积网络，从 1250 点原始 PPG 波形中自动学习特征
  2. ResNet1D: 深层残差网络，更强的特征表达能力

Anti-overfitting:
  - Subject-level train/val/test split
  - Early stopping on validation set
  - Dropout + BatchNorm
  - L2 regularization
  - ReduceLR on plateau
"""

import sys; sys.path.insert(0, ".")
import os, json, glob, time
import numpy as np
import h5py
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-3
PATIENCE = 30  # early stopping patience
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("DEEP LEARNING — PPG → BP End-to-End")
print(f"Device: {DEVICE}")
print("=" * 60)

# ═══════════════════════════════════════════════════
# 1. Load raw waveforms
# ═══════════════════════════════════════════════════
DATA_DIR = "data/pulsedb/Segment_Files"
WAVEFORM_CACHE = "data/processed/pulsedb_waveforms.npz"

print(f"\n[1/5] Loading raw PPG waveforms...")

if os.path.exists(WAVEFORM_CACHE):
    d = np.load(WAVEFORM_CACHE, allow_pickle=True)
    X = d["X"]
    y_sbp = d["y_sbp"]
    y_dbp = d["y_dbp"]
    subjects_arr = d["subjects"]
    sources_arr = d["sources"]
    print(f"  Loaded cached: {len(y_sbp)} segments from {len(set(subjects_arr))} subjects")
else:
    t0 = time.time()
    all_files = sorted(glob.glob(f"{DATA_DIR}/**/*.mat", recursive=True))
    print(f"  Processing {len(all_files)} .mat files...")

    X_list, y_sbp_list, y_dbp_list = [], [], []
    subjects, sources = [], []

    for fpath in all_files:
        fname = os.path.basename(fpath)
        source = "MIMIC" if "MIMIC" in fpath else "VitalDB"
        subject = fname.replace(".mat", "")
        try:
            with h5py.File(fpath, "r") as f:
                g = f["Subj_Wins"]
                ppg_refs = g["PPG_F"][0]
                sbp_refs = g["SegSBP"][0]
                dbp_refs = g["SegDBP"][0]
                for i in range(len(ppg_refs)):
                    try:
                        ppg = np.array(g[ppg_refs[i]]).flatten().astype(np.float32)
                    except:
                        continue
                    if ppg.std() < 1e-6 or len(ppg) < 100:
                        continue
                    # Ensure fixed length (1250 samples)
                    if len(ppg) != 1250:
                        continue
                    try:
                        sbp = float(np.array(g[sbp_refs[i]]).flatten()[0])
                        dbp = float(np.array(g[dbp_refs[i]]).flatten()[0])
                    except:
                        continue
                    X_list.append(ppg)
                    y_sbp_list.append(sbp)
                    y_dbp_list.append(dbp)
                    subjects.append(subject)
                    sources.append(source)
        except:
            continue

    X = np.array(X_list, dtype=np.float32)
    y_sbp = np.array(y_sbp_list, dtype=np.float32)
    y_dbp = np.array(y_dbp_list, dtype=np.float32)
    subjects_arr = np.array(subjects)
    sources_arr = np.array(sources)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s: {len(y_sbp)} segments from {len(set(subjects))} subjects")

    np.savez_compressed(WAVEFORM_CACHE, X=X, y_sbp=y_sbp, y_dbp=y_dbp,
                        subjects=subjects_arr, sources=sources_arr)

print(f"  Shape: {X.shape}, SBP range=[{y_sbp.min():.0f},{y_sbp.max():.0f}], DBP range=[{y_dbp.min():.0f},{y_dbp.max():.0f}]")

# ═══════════════════════════════════════════════════
# 2. Train / Val / Test Split (subject-level)
# ═══════════════════════════════════════════════════
print(f"\n[2/5] Subject-level split")

unique_subjects = np.unique(subjects_arr)
rng = np.random.RandomState(SEED)
rng.shuffle(unique_subjects)

n = len(unique_subjects)
n_train = int(n * 0.70)
n_val = int(n * 0.15)

train_subjs = set(unique_subjects[:n_train])
val_subjs = set(unique_subjects[n_train:n_train + n_val])
test_subjs = set(unique_subjects[n_train + n_val:])

train_idx = np.where([s in train_subjs for s in subjects_arr])[0]
val_idx = np.where([s in val_subjs for s in subjects_arr])[0]
test_idx = np.where([s in test_subjs for s in subjects_arr])[0]

print(f"  Train:  {len(train_idx):>5d} segments ({n_train} subjects)")
print(f"  Val:    {len(val_idx):>5d} segments ({n_val} subjects)")
print(f"  Test:   {len(test_idx):>5d} segments ({n - n_train - n_val} subjects)")

# ═══════════════════════════════════════════════════
# 3. Normalization (per-segment z-score)
# ═══════════════════════════════════════════════════
print(f"\n[3/5] Normalizing PPG signals (per-segment Z-score)")

# Normalize: (X - mean) / std for each segment
X_norm = np.zeros_like(X)
for i in range(len(X)):
    m = X[i].mean()
    s = X[i].std()
    if s > 1e-8:
        X_norm[i] = (X[i] - m) / s

# Add channel dimension for 1D conv: (N, 1, 1250)
X_norm = X_norm[:, np.newaxis, :]

# Target normalization
y_sbp_mean, y_sbp_std = y_sbp.mean(), y_sbp.std()
y_dbp_mean, y_dbp_std = y_dbp.mean(), y_dbp.std()

y_sbp_norm = (y_sbp - y_sbp_mean) / y_sbp_std
y_dbp_norm = (y_dbp - y_dbp_mean) / y_dbp_std

print(f"  SBP: mean={y_sbp_mean:.1f}, std={y_sbp_std:.1f}")
print(f"  DBP: mean={y_dbp_mean:.1f}, std={y_dbp_std:.1f}")

# ═══════════════════════════════════════════════════
# 4. Dataset + Models
# ═══════════════════════════════════════════════════
print(f"\n[4/5] Building models...")

class PPGDataset(Dataset):
    def __init__(self, X, y_sbp, y_dbp):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_sbp = torch.tensor(y_sbp, dtype=torch.float32)
        self.y_dbp = torch.tensor(y_dbp, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_sbp[idx], self.y_dbp[idx]


# ─── 1D-CNN ────────────────────────────────────────
class CNN1D(nn.Module):
    """Shallow 1D CNN for PPG waveform."""
    def __init__(self, input_length=1250, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)  # 1250 → 312

        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)  # 312 → 78

        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)  # 78 → 19

        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.AdaptiveAvgPool1d(1)  # → 256

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, 2)  # SBP, DBP

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ─── ResNet1D ──────────────────────────────────────
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        return F.relu(out)


class ResNet1D(nn.Module):
    """ResNet-18 style 1D architecture."""
    def __init__(self, input_length=1250, dropout=0.3):
        super().__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)  # 1250→313→157

        # Layers
        self.layer1 = self._make_layer(64, 2, stride=1)   # 157
        self.layer2 = self._make_layer(128, 2, stride=2)  # 79
        self.layer3 = self._make_layer(256, 2, stride=2)  # 40
        self.layer4 = self._make_layer(512, 2, stride=2)  # 20

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # → 512
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, 2)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [ResidualBlock1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ═══════════════════════════════════════════════════
# 5. Training
# ═══════════════════════════════════════════════════
print(f"\n[5/5] Training...")

def train_model(model, model_name, train_loader, val_loader, test_loader,
                y_sbp_test_raw, y_dbp_test_raw):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    history = []

    print(f"\n  [{model_name}] Training...")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for x, sbp, dbp in train_loader:
            x, sbp, dbp = x.to(DEVICE), sbp.to(DEVICE), dbp.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred[:, 0], sbp) + criterion(pred[:, 1], dbp)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, sbp, dbp in val_loader:
                x, sbp, dbp = x.to(DEVICE), sbp.to(DEVICE), dbp.to(DEVICE)
                pred = model(x)
                loss = criterion(pred[:, 0], sbp) + criterion(pred[:, 1], dbp)
                val_loss += loss.item() * len(x)
        val_loss /= len(val_loader.dataset)

        history.append((train_loss, val_loss))
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            # Save best model
            torch.save(model.state_dict(), f"/tmp/best_{model_name}.pt")
        else:
            no_improve += 1

        if epoch % 20 == 0 or no_improve == PATIENCE:
            # Compute denormalized MAE for readable progress
            model.eval()
            with torch.no_grad():
                all_preds = []
                for x, _, _ in val_loader:
                    x = x.to(DEVICE)
                    p = model(x).cpu().numpy()
                    all_preds.append(p)
                p = np.concatenate(all_preds)
                y_all_sbp = np.concatenate([s.numpy() for _, s, _ in val_loader])
                y_all_dbp = np.concatenate([d.numpy() for _, _, d in val_loader])
                # Denormalize
                p_sbp = p[:, 0] * y_sbp_std + y_sbp_mean
                p_dbp = p[:, 1] * y_dbp_std + y_dbp_mean
                y_sbp_d = y_all_sbp * y_sbp_std + y_sbp_mean
                y_dbp_d = y_all_dbp * y_dbp_std + y_dbp_mean
                sbp_mae = mean_absolute_error(y_sbp_d, p_sbp)
                dbp_mae = mean_absolute_error(y_dbp_d, p_dbp)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch:>3d}: Train={train_loss:.4f} Val={val_loss:.4f} "
                  f"SBP_MAE={sbp_mae:.2f} DBP_MAE={dbp_mae:.2f} "
                  f"LR={lr_now:.1e} {'★' if epoch == best_epoch else ''}")

        if no_improve >= PATIENCE:
            print(f"    Early stopping at epoch {epoch} (best: epoch {best_epoch}, val={best_val_loss:.4f})")
            break

    # ─── Load best model and evaluate on Test ───
    model.load_state_dict(torch.load(f"/tmp/best_{model_name}.pt"))

    model.eval()
    with torch.no_grad():
        # Train predictions
        train_preds = []
        for x, _, _ in train_loader:
            x = x.to(DEVICE)
            train_preds.append(model(x).cpu().numpy())
        train_preds = np.concatenate(train_preds)

        # Val predictions
        val_preds = []
        for x, _, _ in val_loader:
            x = x.to(DEVICE)
            val_preds.append(model(x).cpu().numpy())
        val_preds = np.concatenate(val_preds)

        # Test predictions
        test_preds = []
        for x, _, _ in test_loader:
            x = x.to(DEVICE)
            test_preds.append(model(x).cpu().numpy())
        test_preds = np.concatenate(test_preds)

    # Collect all true values
    def collect_y(loader):
        sbps, dbps = [], []
        for _, s, d in loader:
            sbps.append(s.numpy())
            dbps.append(d.numpy())
        return np.concatenate(sbps), np.concatenate(dbps)

    y_sbp_train, y_dbp_train = collect_y(train_loader)
    y_sbp_val, y_dbp_val = collect_y(val_loader)
    y_sbp_test, y_dbp_test = collect_y(test_loader)

    # Denormalize
    p_train_sbp = train_preds[:, 0] * y_sbp_std + y_sbp_mean
    p_train_dbp = train_preds[:, 1] * y_dbp_std + y_dbp_mean
    p_val_sbp = val_preds[:, 0] * y_sbp_std + y_sbp_mean
    p_val_dbp = val_preds[:, 1] * y_dbp_std + y_dbp_mean
    p_test_sbp = test_preds[:, 0] * y_sbp_std + y_sbp_mean
    p_test_dbp = test_preds[:, 1] * y_dbp_std + y_dbp_mean

    y_sbp_tr = y_sbp_train * y_sbp_std + y_sbp_mean
    y_dbp_tr = y_dbp_train * y_dbp_std + y_dbp_mean
    y_sbp_vl = y_sbp_val * y_sbp_std + y_sbp_mean
    y_dbp_vl = y_dbp_val * y_dbp_std + y_dbp_mean
    y_sbp_te = y_sbp_test * y_sbp_std + y_sbp_mean
    y_dbp_te = y_dbp_test * y_dbp_std + y_dbp_mean

    results = {}
    for bp, (y_tr, p_tr, y_vl, p_vl, y_te, p_te) in [
        ("SBP", (y_sbp_tr, p_train_sbp, y_sbp_vl, p_val_sbp, y_sbp_te, p_test_sbp)),
        ("DBP", (y_dbp_tr, p_train_dbp, y_dbp_vl, p_val_dbp, y_dbp_te, p_test_dbp)),
    ]:
        train_mae = mean_absolute_error(y_tr, p_tr)
        val_mae = mean_absolute_error(y_vl, p_vl)
        test_mae = mean_absolute_error(y_te, p_te)
        train_r2 = r2_score(y_tr, p_tr)
        val_r2 = r2_score(y_vl, p_vl)
        test_r2 = r2_score(y_te, p_te)
        gap = val_mae - train_mae

        print(f"\n  [{model_name}] {bp}:")
        print(f"    Train:  MAE={train_mae:.2f}, R²={train_r2:.3f}")
        print(f"    Val:    MAE={val_mae:.2f}, R²={val_r2:.3f}")
        print(f"    Test:   MAE={test_mae:.2f}, R²={test_r2:.3f}")
        print(f"    Overfit gap: {gap:+.2f} mmHg")

        results[bp] = {
            "train_mae": round(float(train_mae), 2),
            "val_mae": round(float(val_mae), 2),
            "test_mae": round(float(test_mae), 2),
            "train_r2": round(float(train_r2), 3),
            "val_r2": round(float(val_r2), 3),
            "test_r2": round(float(test_r2), 3),
            "overfit_gap": round(float(gap), 2),
            "best_epoch": int(best_epoch),
            "best_val_loss": round(float(best_val_loss), 4),
        }

    results["history"] = [(float(t), float(v)) for t, v in history]
    return results


# ─── Create data loaders ───────────────────────────
train_dataset = PPGDataset(X_norm[train_idx], y_sbp_norm[train_idx], y_dbp_norm[train_idx])
val_dataset = PPGDataset(X_norm[val_idx], y_sbp_norm[val_idx], y_dbp_norm[val_idx])
test_dataset = PPGDataset(X_norm[test_idx], y_sbp_norm[test_idx], y_dbp_norm[test_idx])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ─── Train both models ─────────────────────────────
all_results = {}

cnn = CNN1D(input_length=1250, dropout=0.3)
cnn_results = train_model(cnn, "CNN1D",
                          train_loader, val_loader, test_loader,
                          y_sbp[test_idx], y_dbp[test_idx])
all_results["CNN1D"] = cnn_results

resnet = ResNet1D(input_length=1250, dropout=0.3)
resnet_results = train_model(resnet, "ResNet1D",
                             train_loader, val_loader, test_loader,
                             y_sbp[test_idx], y_dbp[test_idx])
all_results["ResNet1D"] = resnet_results

# ─── Baseline comparison ───────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: XGBoost vs Deep Learning")
print("=" * 60)

# XGBoost result from proper split (earlier)
xgb_results = {
    "SBP": {"test_mae": 20.61, "test_r2": 0.019, "val_mae": 12.27, "val_r2": -0.003},
    "DBP": {"test_mae": 11.71, "test_r2": 0.020, "val_mae": 8.88, "val_r2": -0.161},
}

# Naive baseline
baseline_sbp = np.mean(np.abs(y_sbp[test_idx] - y_sbp_mean))
baseline_dbp = np.mean(np.abs(y_dbp[test_idx] - y_dbp_mean))

for bp in ["SBP", "DBP"]:
    bl = baseline_sbp if bp == "SBP" else baseline_dbp
    print(f"\n  {bp} Test MAE comparison:")
    print(f"    Baseline (predict mean): {bl:.2f}")
    print(f"    XGBoost (handcrafted):   {xgb_results[bp]['test_mae']:.2f}")
    for model_name in ["CNN1D", "ResNet1D"]:
        if model_name in all_results:
            r = all_results[model_name][bp]
            print(f"    {model_name}:              {r['test_mae']:.2f} (R²={r['test_r2']:.3f})")

# Save
output = {
    "timestamp": datetime.now().isoformat(),
    "device": str(DEVICE),
    "n_segments": int(len(y_sbp)),
    "n_subjects": int(len(unique_subjects)),
    "split": {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
    },
    "baseline": {
        "sbp_mae": round(float(baseline_sbp), 2),
        "dbp_mae": round(float(baseline_dbp), 2),
    },
    "xgb_baseline": xgb_results,
    **all_results,
}

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/dl_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to data/processed/dl_results.json")
