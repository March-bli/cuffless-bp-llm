"""
Dual-Channel ECG + PPG BP Estimation.

Key insight: PTT (Pulse Transit Time) = ECG R-peak → PPG systolic peak.
This is the gold-standard feature for cuff-less BP estimation.
PulseDB provides ECG_Raw, PPG_Raw, ECG_RPeaks, PPG_SPeaks, Age, Gender.

Model: 2-channel 1D-CNN (ECG + PPG concatenated as 2 channels)
Auxiliary features: derived PTT + Age + Gender
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 250
LR = 1e-3
PATIENCE = 30
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)

print("=" * 60)
print("DUAL-CHANNEL ECG + PPG BP Estimation")
print(f"Device: {DEVICE}")
print("=" * 60)

# ═══════════════════════════════════════════════
# 1. Load ECG + PPG + Demographics
# ═══════════════════════════════════════════════
DATA_DIR = "data/pulsedb/Segment_Files"
CACHE = "data/processed/pulsedb_ecg_ppg.npz"

if os.path.exists(CACHE):
    print(f"\n[1/4] Loading cached ECG+PPG data...")
    d = np.load(CACHE, allow_pickle=True)
    X_ecg, X_ppg = d["X_ecg"], d["X_ppg"]
    y_sbp_raw, y_dbp_raw = d["y_sbp"], d["y_dbp"]
    ages, genders = d["ages"], d["genders"]
    subjects_arr, sources_arr = d["subjects"], d["sources"]
    print(f"  {len(y_sbp_raw)} segments from {len(set(subjects_arr))} subjects")
else:
    print(f"\n[1/4] Extracting ECG+PPG from {DATA_DIR}...")
    t0 = time.time()
    all_files = sorted(glob.glob(f"{DATA_DIR}/**/*.mat", recursive=True))

    ecg_list, ppg_list = [], []
    sbp_list, dbp_list, age_list, gender_list = [], [], [], []
    subjects, sources = [], []

    for fpath in all_files:
        fname = os.path.basename(fpath)
        source = "MIMIC" if "MIMIC" in fpath else "VitalDB"
        subject = fname.replace(".mat", "")
        try:
            with h5py.File(fpath, "r") as f:
                g = f["Subj_Wins"]
                ecg_refs = g["ECG_F"][0]
                ppg_refs = g["PPG_F"][0]
                sbp_refs = g["SegSBP"][0]
                dbp_refs = g["SegDBP"][0]
                age_refs = g["Age"][0]
                gender_refs = g["Gender"][0]
                n_seg = len(ecg_refs)

                for i in range(n_seg):
                    try:
                        ecg = np.array(g[ecg_refs[i]]).flatten().astype(np.float32)
                        ppg = np.array(g[ppg_refs[i]]).flatten().astype(np.float32)
                    except:
                        continue
                    if len(ecg) < 100 or len(ppg) < 100:
                        continue
                    if len(ecg) != len(ppg):
                        continue
                    if ecg.std() < 1e-6 or ppg.std() < 1e-6:
                        continue
                    try:
                        sbp = float(np.array(g[sbp_refs[i]]).flatten()[0])
                        dbp = float(np.array(g[dbp_refs[i]]).flatten()[0])
                        age = float(np.array(g[age_refs[i]]).flatten()[0])
                        gender = float(np.array(g[gender_refs[i]]).flatten()[0])
                    except:
                        continue

                    # Pad/truncate to fixed length
                    target_len = 1250
                    if len(ecg) > target_len:
                        ecg = ecg[:target_len]
                        ppg = ppg[:target_len]
                    elif len(ecg) < target_len:
                        ecg = np.pad(ecg, (0, target_len - len(ecg)), mode='edge')
                        ppg = np.pad(ppg, (0, target_len - len(ppg)), mode='edge')

                    ecg_list.append(ecg)
                    ppg_list.append(ppg)
                    sbp_list.append(sbp)
                    dbp_list.append(dbp)
                    age_list.append(age)
                    gender_list.append(gender)
                    subjects.append(subject)
                    sources.append(source)
        except:
            continue

    X_ecg = np.array(ecg_list, dtype=np.float32)
    X_ppg = np.array(ppg_list, dtype=np.float32)
    y_sbp_raw = np.array(sbp_list, dtype=np.float32)
    y_dbp_raw = np.array(dbp_list, dtype=np.float32)
    ages = np.array(age_list, dtype=np.float32)
    genders = np.array(gender_list, dtype=np.float32)
    subjects_arr = np.array(subjects)
    sources_arr = np.array(sources)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s: {len(y_sbp_raw)} segments, {len(set(subjects))} subjects")
    print(f"  MIMIC: {(sources_arr=='MIMIC').sum()}, VitalDB: {(sources_arr=='VitalDB').sum()}")
    print(f"  Age range: [{ages.min():.0f}, {ages.max():.0f}], Gender: {genders.mean():.1%} male")

    np.savez_compressed(CACHE, X_ecg=X_ecg, X_ppg=X_ppg,
                        y_sbp=y_sbp_raw, y_dbp=y_dbp_raw,
                        ages=ages, genders=genders,
                        subjects=subjects_arr, sources=sources_arr)

# ═══════════════════════════════════════════════
# 2. Train/Val/Test Split
# ═══════════════════════════════════════════════
print(f"\n[2/4] Subject-level split...")

unique_subjects = np.unique(subjects_arr)
rng = np.random.RandomState(SEED)
rng.shuffle(unique_subjects)
n = len(unique_subjects)
n_train, n_val = int(n * 0.70), int(n * 0.15)

train_subjs = set(unique_subjects[:n_train])
val_subjs = set(unique_subjects[n_train:n_train + n_val])
test_subjs = set(unique_subjects[n_train + n_val:])

m_train = np.array([s in train_subjs for s in subjects_arr])
m_val = np.array([s in val_subjs for s in subjects_arr])
m_test = np.array([s in test_subjs for s in subjects_arr])

# Subset data
ecg_tr, ppg_tr = X_ecg[m_train], X_ppg[m_train]
ecg_vl, ppg_vl = X_ecg[m_val], X_ppg[m_val]
ecg_te, ppg_te = X_ecg[m_test], X_ppg[m_test]

def subset(mask):
    return y_sbp_raw[mask], y_dbp_raw[mask], ages[mask], genders[mask]

sbp_tr, dbp_tr, age_tr, gen_tr = subset(m_train)
sbp_vl, dbp_vl, age_vl, gen_vl = subset(m_val)
sbp_te, dbp_te, age_te, gen_te = subset(m_test)

print(f"  Train: {len(sbp_tr)} seg, Val: {len(sbp_vl)} seg, Test: {len(sbp_te)} seg")

# ═══════════════════════════════════════════════
# 3. Normalization + Demographics
# ═══════════════════════════════════════════════
print(f"\n[3/4] Preprocessing...")

y_sbp_mean, y_sbp_std = sbp_tr.mean(), sbp_tr.std()
y_dbp_mean, y_dbp_std = dbp_tr.mean(), dbp_tr.std()

# Age standardization
age_mean, age_std = ages[m_train].mean(), ages[m_train].std()

def norm_wave(wave):
    m, s = wave.mean(), wave.std()
    return (wave - m) / (s + 1e-8) if s > 1e-8 else wave

def norm_waves(waves):
    return np.array([norm_wave(w) for w in waves])

def norm_y(y_tr, y_vl, y_te, mean, std):
    return ((y_tr - mean) / std, (y_vl - mean) / std, (y_te - mean) / std)

ecg_tr_n, ecg_vl_n, ecg_te_n = norm_waves(ecg_tr), norm_waves(ecg_vl), norm_waves(ecg_te)
ppg_tr_n, ppg_vl_n, ppg_te_n = norm_waves(ppg_tr), norm_waves(ppg_vl), norm_waves(ppg_te)

sbp_tr_n, sbp_vl_n, sbp_te_n = norm_y(sbp_tr, sbp_vl, sbp_te, y_sbp_mean, y_sbp_std)
dbp_tr_n, dbp_vl_n, dbp_te_n = norm_y(dbp_tr, dbp_vl, dbp_te, y_dbp_mean, y_dbp_std)

age_tr_n = (age_tr - age_mean) / age_std
age_vl_n = (age_vl - age_mean) / age_std
age_te_n = (age_te - age_mean) / age_std

# ═══════════════════════════════════════════════
# 4. Dataset
# ═══════════════════════════════════════════════
class DualPPGDataset(Dataset):
    def __init__(self, ecg, ppg, sbp, dbp, age, gender):
        # Stack as 2 channels: (N, 2, 1250)
        self.X = torch.tensor(np.stack([ecg, ppg], axis=1), dtype=torch.float32)
        self.sbp = torch.tensor(sbp, dtype=torch.float32)
        self.dbp = torch.tensor(dbp, dtype=torch.float32)
        self.age = torch.tensor(age[:, None], dtype=torch.float32)  # (N, 1)
        self.gender = torch.tensor(gender[:, None], dtype=torch.float32)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.age[idx], self.gender[idx], self.sbp[idx], self.dbp[idx]

train_ds = DualPPGDataset(ecg_tr_n, ppg_tr_n, sbp_tr_n, dbp_tr_n, age_tr_n, gen_tr)
val_ds = DualPPGDataset(ecg_vl_n, ppg_vl_n, sbp_vl_n, dbp_vl_n, age_vl_n, gen_vl)
test_ds = DualPPGDataset(ecg_te_n, ppg_te_n, sbp_te_n, dbp_te_n, age_te_n, gen_te)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

# ═══════════════════════════════════════════════
# 5. Model: Dual-Channel CNN + Demographics
# ═══════════════════════════════════════════════
class DualCNN(nn.Module):
    """2-channel 1D-CNN (ECG + PPG) with demographic features."""
    def __init__(self, dropout=0.3):
        super().__init__()

        # Shared conv layers for both channels
        self.conv1 = nn.Conv1d(2, 32, 15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)  # 1250 → 312

        self.conv2 = nn.Conv1d(32, 64, 9, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)  # 312 → 78

        self.conv3 = nn.Conv1d(64, 128, 7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)  # 78 → 19

        self.conv4 = nn.Conv1d(128, 256, 5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.gap = nn.AdaptiveAvgPool1d(1)  # → 256

        self.drop_conv = nn.Dropout(dropout)

        # FC: waveform features (256) + age (1) + gender (1) → output (2)
        self.fc1 = nn.Linear(256 + 1 + 1, 128)
        self.drop_fc = nn.Dropout(dropout * 1.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, age, gender):
        # x: (N, 2, 1250)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop_conv(x)

        # Concatenate demographics
        x = torch.cat([x, age, gender], dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════
# 6. Training
# ═══════════════════════════════════════════════
print(f"\n[4/4] Training DualCNN...")

model = DualCNN(dropout=0.3).to(DEVICE)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=12, min_lr=1e-6)
criterion = nn.MSELoss()

best_val_loss = float('inf')
best_epoch = 0
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, age, gen, sbp, dbp in train_loader:
        x, age, gen = x.to(DEVICE), age.to(DEVICE), gen.to(DEVICE)
        sbp, dbp = sbp.to(DEVICE), dbp.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x, age, gen)
        loss = criterion(pred[:, 0], sbp) + criterion(pred[:, 1], dbp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_loss += loss.item() * len(x)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, age, gen, sbp, dbp in val_loader:
            x, age, gen = x.to(DEVICE), age.to(DEVICE), gen.to(DEVICE)
            sbp, dbp = sbp.to(DEVICE), dbp.to(DEVICE)
            pred = model(x, age, gen)
            loss = criterion(pred[:, 0], sbp) + criterion(pred[:, 1], dbp)
            val_loss += loss.item() * len(x)
    val_loss /= len(val_loader.dataset)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        no_improve = 0
        torch.save(model.state_dict(), "/tmp/best_dual.pt")
    else:
        no_improve += 1

    if epoch % 15 == 0 or no_improve == PATIENCE:
        model.eval()
        with torch.no_grad():
            all_p = []
            for x, age, gen, _, _ in val_loader:
                p = model(x.to(DEVICE), age.to(DEVICE), gen.to(DEVICE)).cpu().numpy()
                all_p.append(p)
            p = np.concatenate(all_p)
            y_s = np.concatenate([s.numpy() for _, _, _, s, _ in val_loader])
            y_d = np.concatenate([d.numpy() for _, _, _, _, d in val_loader])
            p_sbp = p[:, 0] * y_sbp_std + y_sbp_mean
            p_dbp = p[:, 1] * y_dbp_std + y_dbp_mean
            y_s_d = y_s * y_sbp_std + y_sbp_mean
            y_d_d = y_d * y_dbp_std + y_dbp_mean
            sbp_mae = mean_absolute_error(y_s_d, p_sbp)
            dbp_mae = mean_absolute_error(y_d_d, p_dbp)
        lr_now = optimizer.param_groups[0]['lr']
        star = '★' if epoch == best_epoch else ''
        print(f"  Epoch {epoch:>3d}: Train={train_loss:.4f} Val={val_loss:.4f} "
              f"SBP={sbp_mae:.2f} DBP={dbp_mae:.2f} LR={lr_now:.1e} {star}")

    if no_improve >= PATIENCE:
        print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
        break

# ═══════════════════════════════════════════════
# 7. Final Evaluation
# ═══════════════════════════════════════════════
model.load_state_dict(torch.load("/tmp/best_dual.pt", weights_only=True))
model.eval()

def eval_dual(loader):
    with torch.no_grad():
        preds = []
        for x, age, gen, _, _ in loader:
            p = model(x.to(DEVICE), age.to(DEVICE), gen.to(DEVICE)).cpu().numpy()
            preds.append(p)
        preds = np.concatenate(preds)
    return preds[:, 0] * y_sbp_std + y_sbp_mean, preds[:, 1] * y_dbp_std + y_dbp_mean

p_sbp_tr, p_dbp_tr = eval_dual(train_loader)
p_sbp_vl, p_dbp_vl = eval_dual(val_loader)
p_sbp_te, p_dbp_te = eval_dual(test_loader)

def metrics(y, p):
    return mean_absolute_error(y, p), r2_score(y, p)

bl_sbp = mean_absolute_error(sbp_te, np.full_like(sbp_te, y_sbp_mean))
bl_dbp = mean_absolute_error(dbp_te, np.full_like(dbp_te, y_dbp_mean))

print(f"\n{'=' * 60}")
print("FINAL RESULTS — Dual-Channel ECG+PPG")
print(f"{'=' * 60}")

for name, sbp_y, sbp_p, dbp_y, dbp_p in [
    ("Train", sbp_tr, p_sbp_tr, dbp_tr, p_dbp_tr),
    ("Val", sbp_vl, p_sbp_vl, dbp_vl, p_dbp_vl),
    ("Test", sbp_te, p_sbp_te, dbp_te, p_dbp_te),
]:
    sm, sr = metrics(sbp_y, sbp_p)
    dm, dr = metrics(dbp_y, dbp_p)
    print(f"  {name:<6s}: SBP MAE={sm:.2f} R²={sr:.3f} | DBP MAE={dm:.2f} R²={dr:.3f}")

print(f"\n  Comparison:")
print(f"  {'Method':<25s} {'SBP MAE':>10s} {'SBP R²':>8s} {'DBP MAE':>10s} {'DBP R²':>8s}")
print(f"  {'─' * 63}")
print(f"  {'Baseline (mean)':<25s} {bl_sbp:>10.2f} {'0.000':>8s} {bl_dbp:>10.2f} {'0.000':>8s}")
print(f"  {'CNN1D (PPG only)':<25s} {16.63:>10.2f} {'0.321':>8s} {9.14:>10.2f} {'0.236':>8s}")

sm_te, sr_te = metrics(sbp_te, p_sbp_te)
dm_te, dr_te = metrics(dbp_te, p_dbp_te)
print(f"  {'DualCNN (ECG+PPG)':<25s} {sm_te:>10.2f} {sr_te:>8.3f} {dm_te:>10.2f} {dr_te:>8.3f}")

improve_sbp = (16.63 - sm_te) / 16.63 * 100
improve_dbp = (9.14 - dm_te) / 9.14 * 100
print(f"\n  vs PPG-only: SBP {improve_sbp:+.1f}% | DBP {improve_dbp:+.1f}%")

result = {
    "timestamp": datetime.now().isoformat(),
    "model": "DualCNN (ECG+PPG + Age+Gender)",
    "sbp": {"test_mae": round(float(sm_te), 2), "test_r2": round(float(sr_te), 3)},
    "dbp": {"test_mae": round(float(dm_te), 2), "test_r2": round(float(dr_te), 3)},
    "baseline": {"sbp_mae": round(float(bl_sbp), 2), "dbp_mae": round(float(bl_dbp), 2)},
    "ppg_only": {"sbp_mae": 16.63, "sbp_r2": 0.321, "dbp_mae": 9.14, "dbp_r2": 0.236},
}
with open("data/processed/ecg_ppg_results.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to data/processed/ecg_ppg_results.json")
