"""
Data Augmentation + Strong Regularization for PPG→BP.

Augmentation strategies:
  1. Gaussian noise injection (simulate sensor noise)
  2. Amplitude scaling (0.8x - 1.2x, simulate skin tone / LED intensity)
  3. Time warping (stretch/compress, simulate heart rate variability)
  4. Baseline wander (low-frequency drift, simulate motion artifact)
  5. Random segment dropout (mask portions of the waveform)

Regularization strategies:
  1. Dropout (0.4 instead of 0.3)
  2. Weight decay (1e-3 instead of 1e-4)
  3. Label smoothing (MSE → SmoothL1 for robustness to outliers)
  4. Cosine annealing with warm restarts
  5. Gradient clipping

Expected: SBP MAE 16.6 → ~14-15
"""

import sys; sys.path.insert(0, ".")
import os, json, glob, time
import numpy as np
import h5py
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, r2_score

# ═══════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 300
LR = 1e-3
PATIENCE = 30
SEED = 42
AUG_FACTOR = 2  # generate 2 augmented copies per original

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("AUGMENTED TRAINING — Data Augmentation + Regularization")
print(f"Device: {DEVICE} | Aug factor: {AUG_FACTOR}x")
print("=" * 60)

# ═══════════════════════════════════════════════
# 1. Data Augmentation Transform
# ═══════════════════════════════════════════════
class PPGAugment:
    """Apply random augmentations to PPG waveform."""
    def __init__(self, p_noise=0.5, p_scale=0.5, p_warp=0.4, p_wander=0.4, p_mask=0.3):
        self.p_noise = p_noise
        self.p_scale = p_scale
        self.p_warp = p_warp
        self.p_wander = p_wander
        self.p_mask = p_mask

    def __call__(self, wave):
        wave = wave.copy()
        L = len(wave)

        # 1. Gaussian noise
        if np.random.random() < self.p_noise:
            noise_std = np.random.uniform(0.01, 0.05) * wave.std()
            wave += np.random.randn(L) * noise_std

        # 2. Amplitude scaling
        if np.random.random() < self.p_scale:
            scale = np.random.uniform(0.75, 1.25)
            wave *= scale

        # 3. Time warping (via interpolation)
        if np.random.random() < self.p_warp:
            t_orig = np.arange(L)
            # Random stretch/compress
            warp_factor = np.random.uniform(0.9, 1.1)
            center = np.random.randint(L // 4, 3 * L // 4)
            t_new = t_orig.copy().astype(float)
            # Warp around center
            dist = t_orig - center
            t_new += dist * (warp_factor - 1) * np.exp(-(dist ** 2) / (2 * (L / 8) ** 2))
            t_new = np.clip(t_new, 0, L - 1)
            wave = np.interp(t_orig, t_new, wave)

        # 4. Baseline wander (low-frequency sine)
        if np.random.random() < self.p_wander:
            freq = np.random.uniform(0.5, 2.0)  # Hz
            amp = np.random.uniform(0.02, 0.08) * wave.std()
            phase = np.random.uniform(0, 2 * np.pi)
            t = np.arange(L) / 125.0  # 125 Hz sampling
            wander = amp * np.sin(2 * np.pi * freq * t + phase)
            wave += wander

        # 5. Random mask (dropout parts of the wave)
        if np.random.random() < self.p_mask:
            mask_len = int(np.random.uniform(0.05, 0.15) * L)
            mask_start = np.random.randint(0, L - mask_len)
            # Linear interpolation across mask
            before = wave[max(0, mask_start - 1)]
            after = wave[min(L - 1, mask_start + mask_len)]
            interp = np.linspace(before, after, mask_len)
            wave[mask_start:mask_start + mask_len] = interp

        return wave


# ═══════════════════════════════════════════════
# 2. Load data
# ═══════════════════════════════════════════════
WAVEFORM_CACHE = "data/processed/pulsedb_waveforms.npz"
print(f"\n[1/4] Loading data...")

if os.path.exists(WAVEFORM_CACHE):
    d = np.load(WAVEFORM_CACHE, allow_pickle=True)
    X_raw = d["X"]
    y_sbp_raw = d["y_sbp"]
    y_dbp_raw = d["y_dbp"]
    subjects_arr = d["subjects"]
    sources_arr = d["sources"]
    print(f"  Loaded: {len(y_sbp_raw)} segments from {len(set(subjects_arr))} subjects")
else:
    print("  ERROR: waveforms.npz not found. Run train_deep_learning.py first.")
    sys.exit(1)

# ═══════════════════════════════════════════════
# 3. Train/Val/Test Split (subject-level)
# ═══════════════════════════════════════════════
print(f"\n[2/4] Subject-level split...")

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

print(f"  Train: {len(train_idx)} seg, Val: {len(val_idx)} seg, Test: {len(test_idx)} seg")

# ═══════════════════════════════════════════════
# 4. Normalization
# ═══════════════════════════════════════════════
# Compute normalization on training set only
X_train_raw = X_raw[train_idx]
X_val_raw = X_raw[val_idx]
X_test_raw = X_raw[test_idx]

y_sbp_train_raw = y_sbp_raw[train_idx]
y_sbp_val_raw = y_sbp_raw[val_idx]
y_sbp_test_raw = y_sbp_raw[test_idx]
y_dbp_train_raw = y_dbp_raw[train_idx]
y_dbp_val_raw = y_dbp_raw[val_idx]
y_dbp_test_raw = y_dbp_raw[test_idx]

y_sbp_mean = y_sbp_train_raw.mean()
y_sbp_std = y_sbp_train_raw.std()
y_dbp_mean = y_dbp_train_raw.mean()
y_dbp_std = y_dbp_train_raw.std()

def normalize_wave(wave):
    m, s = wave.mean(), wave.std()
    return (wave - m) / (s + 1e-8)

def normalize_waves(waves):
    return np.array([normalize_wave(w) for w in waves])

X_train_norm = normalize_waves(X_train_raw)
X_val_norm = normalize_waves(X_val_raw)
X_test_norm = normalize_waves(X_test_raw)

y_sbp_train = (y_sbp_train_raw - y_sbp_mean) / y_sbp_std
y_dbp_train = (y_dbp_train_raw - y_dbp_mean) / y_dbp_std
y_sbp_val = (y_sbp_val_raw - y_sbp_mean) / y_sbp_std
y_dbp_val = (y_dbp_val_raw - y_dbp_mean) / y_dbp_std
y_sbp_test = (y_sbp_test_raw - y_sbp_mean) / y_sbp_std
y_dbp_test = (y_dbp_test_raw - y_dbp_mean) / y_dbp_std

# ═══════════════════════════════════════════════
# 5. Dataset (with on-the-fly augmentation)
# ═══════════════════════════════════════════════
class AugmentedPPGDataset(Dataset):
    def __init__(self, X, y_sbp, y_dbp, augment=None, aug_factor=1):
        self.X = X
        self.y_sbp = y_sbp
        self.y_dbp = y_dbp
        self.augment = augment
        self.aug_factor = aug_factor
        self.n_orig = len(X)
        self.n_total = self.n_orig * aug_factor

    def __len__(self):
        return self.n_total

    def __getitem__(self, idx):
        orig_idx = idx % self.n_orig
        wave = self.X[orig_idx].copy()

        if self.augment and idx >= self.n_orig:
            # Apply augmentation for extra copies
            wave = self.augment(wave)

        # Normalize (if not already)
        m, s = wave.mean(), wave.std()
        if s > 1e-8:
            wave = (wave - m) / s

        x = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)  # (1, 1250)
        return x, torch.tensor(self.y_sbp[orig_idx], dtype=torch.float32), \
               torch.tensor(self.y_dbp[orig_idx], dtype=torch.float32)


# A regular dataset for val/test (no augmentation)
class PPGGDataset(Dataset):
    def __init__(self, X, y_sbp, y_dbp):
        self.X = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)
        self.y_sbp = torch.tensor(y_sbp, dtype=torch.float32)
        self.y_dbp = torch.tensor(y_dbp, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_sbp[idx], self.y_dbp[idx]


# ═══════════════════════════════════════════════
# 6. Model (deeper CNN with stronger regularization)
# ═══════════════════════════════════════════════
class DeepCNN(nn.Module):
    """Deeper CNN with stronger dropout for regularization."""
    def __init__(self, input_length=1250, dropout=0.4):
        super().__init__()

        # Block 1: 1250 → 312
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(dropout * 0.5)
        self.pool1 = nn.MaxPool1d(4)

        # Block 2: 312 → 78
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(dropout * 0.7)
        self.pool2 = nn.MaxPool1d(4)

        # Block 3: 78 → 19
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(dropout * 0.8)
        self.pool3 = nn.MaxPool1d(4)

        # Block 4: 19 → 4
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(dropout)
        self.pool4 = nn.MaxPool1d(4)

        # Global pooling + FC
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop_fc = nn.Dropout(dropout * 1.2)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool1(x)

        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(x)

        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool3(x)

        x = self.drop4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop_fc(x)
        x = self.fc(x)
        return x


# ═══════════════════════════════════════════════
# 7. Training
# ═══════════════════════════════════════════════
print(f"\n[3/4] Building datasets...")

augmenter = PPGAugment(
    p_noise=0.6, p_scale=0.6, p_warp=0.5, p_wander=0.5, p_mask=0.4
)

train_dataset = AugmentedPPGDataset(
    X_train_norm, y_sbp_train, y_dbp_train,
    augment=augmenter, aug_factor=AUG_FACTOR
)
val_dataset = PPGGDataset(X_val_norm, y_sbp_val, y_dbp_val)
test_dataset = PPGGDataset(X_test_norm, y_sbp_test, y_dbp_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train: {len(train_dataset)} samples (with {AUG_FACTOR}x augmentation)")
print(f"  Val:   {len(val_dataset)} samples")
print(f"  Test:  {len(test_dataset)} samples")

# ─── Train ─────────────────────────────────────
print(f"\n[4/4] Training DeepCNN with augmentation...")

model = DeepCNN(dropout=0.4).to(DEVICE)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=30, T_mult=2, eta_min=1e-6
)
# Use SmoothL1Loss (Huber) - more robust to outliers than MSE
criterion = nn.SmoothL1Loss(beta=0.5)

best_val_loss = float('inf')
best_epoch = 0
no_improve = 0

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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        train_loss += loss.item() * len(x)
    train_loss /= len(train_loader.dataset)

    scheduler.step()

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

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        no_improve = 0
        torch.save(model.state_dict(), "/tmp/best_augmented.pt")
    else:
        no_improve += 1

    if epoch % 20 == 0 or no_improve == PATIENCE:
        model.eval()
        with torch.no_grad():
            all_preds = []
            for x, _, _ in val_loader:
                p = model(x.to(DEVICE)).cpu().numpy()
                all_preds.append(p)
            p = np.concatenate(all_preds)
            y_all_sbp = np.concatenate([s.numpy() for _, s, _ in val_loader])
            y_all_dbp = np.concatenate([d.numpy() for _, _, d in val_loader])
            p_sbp = p[:, 0] * y_sbp_std + y_sbp_mean
            p_dbp = p[:, 1] * y_dbp_std + y_dbp_mean
            y_sbp_d = y_all_sbp * y_sbp_std + y_sbp_mean
            y_dbp_d = y_all_dbp * y_dbp_std + y_dbp_mean
            sbp_mae = mean_absolute_error(y_sbp_d, p_sbp)
            dbp_mae = mean_absolute_error(y_dbp_d, p_dbp)

        lr_now = optimizer.param_groups[0]['lr']
        star = '★' if epoch == best_epoch else ''
        print(f"  Epoch {epoch:>3d}: Train={train_loss:.4f} Val={val_loss:.4f} "
              f"SBP={sbp_mae:.2f} DBP={dbp_mae:.2f} LR={lr_now:.1e} {star}")

    if no_improve >= PATIENCE:
        print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
        break

# ═══════════════════════════════════════════════
# 8. Final Evaluation
# ═══════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("FINAL EVALUATION (Test Set)")
print(f"{'=' * 60}")

model.load_state_dict(torch.load("/tmp/best_augmented.pt", weights_only=True))
model.eval()

def evaluate(loader, X_raw, y_sbp_raw, y_dbp_raw):
    """Evaluate model and return denormalized predictions."""
    model.eval()
    with torch.no_grad():
        all_preds = []
        for x, _, _ in loader:
            p = model(x.to(DEVICE)).cpu().numpy()
            all_preds.append(p)
        preds = np.concatenate(all_preds)

    p_sbp = preds[:, 0] * y_sbp_std + y_sbp_mean
    p_dbp = preds[:, 1] * y_dbp_std + y_dbp_mean

    sbp_mae = mean_absolute_error(y_sbp_raw, p_sbp)
    dbp_mae = mean_absolute_error(y_dbp_raw, p_dbp)
    sbp_r2 = r2_score(y_sbp_raw, p_sbp)
    dbp_r2 = r2_score(y_dbp_raw, p_dbp)

    return sbp_mae, dbp_mae, sbp_r2, dbp_r2

# Train
tr_sbp_, tr_dbp_, tr_sbp_r2_, tr_dbp_r2_ = evaluate(
    DataLoader(PPGGDataset(X_train_norm, y_sbp_train, y_dbp_train), batch_size=BATCH_SIZE),
    X_train_raw, y_sbp_train_raw, y_dbp_train_raw)

# Val (with augmentation-free loader)
val_sbp, val_dbp, val_sbp_r2, val_dbp_r2 = evaluate(
    DataLoader(val_dataset, batch_size=BATCH_SIZE),
    X_val_raw, y_sbp_val_raw, y_dbp_val_raw)

# Test
test_sbp, test_dbp, test_sbp_r2, test_dbp_r2 = evaluate(
    DataLoader(test_dataset, batch_size=BATCH_SIZE),
    X_test_raw, y_sbp_test_raw, y_dbp_test_raw)

# Baseline
bl_sbp = mean_absolute_error(y_sbp_test_raw, np.full_like(y_sbp_test_raw, y_sbp_mean))
bl_dbp = mean_absolute_error(y_dbp_test_raw, np.full_like(y_dbp_test_raw, y_dbp_mean))

print(f"\n  {'Method':<25s} {'SBP MAE':>10s} {'SBP R²':>8s} {'DBP MAE':>10s} {'DBP R²':>8s}")
print(f"  {'─' * 63}")
print(f"  {'Baseline (predict mean)':<25s} {bl_sbp:>10.2f} {'0.000':>8s} {bl_dbp:>10.2f} {'0.000':>8s}")

# Previous CNN1D result
print(f"  {'CNN1D (original)':<25s} {16.63:>10.2f} {'0.321':>8s} {9.14:>10.2f} {'0.236':>8s}")

print(f"  {'DeepCNN + Augmentation':<25s} {test_sbp:>10.2f} {test_sbp_r2:>8.3f} {test_dbp:>10.2f} {test_dbp_r2:>8.3f}")

delta_sbp = (16.63 - test_sbp) / 16.63 * 100
delta_dbp = (9.14 - test_dbp) / 9.14 * 100
print(f"\n  Improvement vs original CNN1D:")
print(f"    SBP: {delta_sbp:+.1f}%  |  DBP: {delta_dbp:+.1f}%")

# Save
result = {
    "timestamp": datetime.now().isoformat(),
    "aug_factor": AUG_FACTOR,
    "regularization": {
        "dropout": 0.4,
        "weight_decay": 1e-3,
        "loss": "SmoothL1Loss",
        "gradient_clip": 5.0,
        "scheduler": "CosineAnnealingWarmRestarts",
    },
    "sbp": {"test_mae": round(float(test_sbp), 2), "test_r2": round(float(test_sbp_r2), 3),
            "val_mae": round(float(val_sbp), 2), "val_r2": round(float(val_sbp_r2), 3)},
    "dbp": {"test_mae": round(float(test_dbp), 2), "test_r2": round(float(test_dbp_r2), 3),
            "val_mae": round(float(val_dbp), 2), "val_r2": round(float(val_dbp_r2), 3)},
    "baseline": {"sbp_mae": round(float(bl_sbp), 2), "dbp_mae": round(float(bl_dbp), 2)},
    "original_cnn1d": {"sbp_mae": 16.63, "sbp_r2": 0.321, "dbp_mae": 9.14, "dbp_r2": 0.236},
}

with open("data/processed/augmented_results.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to data/processed/augmented_results.json")
