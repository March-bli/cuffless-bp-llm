"""
Master's Thesis: PPG-based Cuffless BP Estimation.

Architecture: Time-Aware 1D-CNN + Demographics (Age, Gender)
- Multi-scale convolutions capture different time scales
- Attention pooling learns which time points matter most
- Demographics provide prior information (known BP correlates)
- Subject-level train/val/test split
- Comprehensive evaluation: MAE, RMSE, R², BHS grading, Bland-Altman

Previous results (for comparison):
  - XGBoost handcrafted: SBP=20.61 MAE, R²=0.019 (random guess level)
  - CNN1D PPG only:     SBP=16.63 MAE, R²=0.321
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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEVICE = torch.device("cuda")
BATCH_SIZE = 96
EPOCHS = 350
LR = 1e-3
PATIENCE = 40
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)

print("=" * 60)
print("MASTER'S THESIS — Final PPG→BP Model")
print("Device:", DEVICE)
print("=" * 60)

# ═══════════════════════════════════════════════
# 1. Load Data
# ═══════════════════════════════════════════════
CACHE = "data/processed/pulsedb_waveforms.npz"
d = np.load(CACHE, allow_pickle=True)
X_raw = d["X"]
subjects_arr = d["subjects"]

# Extract demographics (Age, Gender) from original files
print(f"\n[1/5] Loading demographics...")
DATA_DIR = "data/pulsedb/Segment_Files"
all_files = sorted(glob.glob(f"{DATA_DIR}/**/*.mat", recursive=True))

# Build subject → age, gender mapping
subj_meta = {}
for fpath in all_files:
    try:
        with h5py.File(fpath, "r") as f:
            g = f["Subj_Wins"]
            age = float(np.array(g[g["Age"][0][0]]).flatten()[0])
            gender_code = float(np.array(g[g["Gender"][0][0]]).flatten()[0])
            subject = os.path.basename(fpath).replace(".mat", "")
            # ASCII: 77='M'(Male), 70='F'(Female)
            subj_meta[subject] = {
                "age": age,
                "gender": 1.0 if abs(gender_code - 77) < 5 else 0.0
            }
    except:
        pass

# Map to each segment
ages_arr = np.array([subj_meta.get(s, {"age": 55})["age"] for s in subjects_arr])
genders_arr = np.array([subj_meta.get(s, {"gender": 1.0})["gender"] for s in subjects_arr])

# Load BP
data_full = np.load(CACHE, allow_pickle=True)
y_sbp_raw = data_full["y_sbp"]
y_dbp_raw = data_full["y_dbp"]

print(f"  {len(y_sbp_raw)} segments, {len(subj_meta)} subjects with metadata")
print(f"  Age: [{ages_arr.min():.0f},{ages_arr.max():.0f}], mean={ages_arr.mean():.1f}")
print(f"  Gender: {genders_arr.sum():.0f}M / {(1-genders_arr).sum():.0f}F")

# ═══════════════════════════════════════════════
# 2. Subject-level Split
# ═══════════════════════════════════════════════
print(f"\n[2/5] Subject-level split...")

unique_subjects = np.unique(subjects_arr)
rng = np.random.RandomState(SEED)
rng.shuffle(unique_subjects)
n = len(unique_subjects)
n_train, n_val = int(n * 0.70), int(n * 0.15)

train_subjs = set(unique_subjects[:n_train])
val_subjs = set(unique_subjects[n_train:n_train + n_val])
test_subjs = set(unique_subjects[n_train + n_val:])

def split(mask):
    return (X_raw[mask], y_sbp_raw[mask], y_dbp_raw[mask],
            ages_arr[mask], genders_arr[mask])

X_tr, sbp_tr, dbp_tr, age_tr, gen_tr = split(
    np.array([s in train_subjs for s in subjects_arr]))
X_vl, sbp_vl, dbp_vl, age_vl, gen_vl = split(
    np.array([s in val_subjs for s in subjects_arr]))
X_te, sbp_te, dbp_te, age_te, gen_te = split(
    np.array([s in test_subjs for s in subjects_arr]))

print(f"  Train: {len(sbp_tr)} seg | Val: {len(sbp_vl)} seg | Test: {len(sbp_te)} seg")

# ═══════════════════════════════════════════════
# 3. Normalization
# ═══════════════════════════════════════════════
print(f"\n[3/5] Normalizing...")

sbp_m, sbp_s = sbp_tr.mean(), sbp_tr.std()
dbp_m, dbp_s = dbp_tr.mean(), dbp_tr.std()
age_m, age_s = age_tr.mean(), age_tr.std()

def norm_waves(X):
    return np.array([(w - w.mean()) / (w.std() + 1e-8) for w in X], dtype=np.float32)

X_tr_n = norm_waves(X_tr)
X_vl_n = norm_waves(X_vl)
X_te_n = norm_waves(X_te)

sbp_tr_n = (sbp_tr - sbp_m) / sbp_s
sbp_vl_n = (sbp_vl - sbp_m) / sbp_s
sbp_te_n = (sbp_te - sbp_m) / sbp_s
dbp_tr_n = (dbp_tr - dbp_m) / dbp_s
dbp_vl_n = (dbp_vl - dbp_m) / dbp_s
dbp_te_n = (dbp_te - dbp_m) / dbp_s

age_tr_n = (age_tr - age_m) / age_s
age_vl_n = (age_vl - age_m) / age_s
age_te_n = (age_te - age_m) / age_s

# ═══════════════════════════════════════════════
# 4. Model: Multi-Scale CNN + Attention + Demographics
# ═══════════════════════════════════════════════
class MultiScaleBP(nn.Module):
    """
    Multi-scale 1D-CNN with attention pooling and demographics.
    - Three parallel convolution branches at different kernel sizes
    - Attention pooling over time (learn which beats matter)
    - Demographics fused before final prediction
    """
    def __init__(self, input_len=1250, dropout=0.35):
        super().__init__()

        # Multi-scale branches
        self.branch_small = self._make_branch(1, 32, 7)   # short window
        self.branch_med = self._make_branch(1, 32, 31)     # medium window (1/4 second)
        self.branch_large = self._make_branch(1, 32, 125)  # long window (~1 second)

        # Fuse
        self.conv_fuse = nn.Conv1d(96, 128, 5, padding=2)
        self.bn_fuse = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(4)  # 1250 → 312

        # Deeper conv blocks
        self.conv2 = nn.Conv1d(128, 256, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(4)  # 312 → 78

        self.conv3 = nn.Conv1d(256, 256, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)  # 78 → 19

        # Attention over time
        self.attn_conv = nn.Conv1d(256, 1, 1)
        self.attn_fc = nn.Linear(256, 128)

        self.drop_cnn = nn.Dropout(dropout)

        # Feature fusion: waveform features + demographics
        self.fc1 = nn.Linear(128 + 2, 64)  # 2 = age + gender
        self.bn_fc = nn.BatchNorm1d(64)
        self.drop_fc = nn.Dropout(dropout + 0.1)
        self.fc2 = nn.Linear(64, 2)

    def _make_branch(self, in_c, out_c, kernel):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, padding=kernel//2),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
        )

    def forward(self, x, age, gender):
        # x: (N, 1, 1250)
        b1 = self.branch_small(x)
        b2 = self.branch_med(x)
        b3 = self.branch_large(x)

        x = torch.cat([b1, b2, b3], dim=1)  # (N, 96, 1250)
        x = F.relu(self.bn_fuse(self.conv_fuse(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # (N, 256, 19)

        # Attention pooling over time
        attn_w = F.softmax(self.attn_conv(x), dim=2)  # (N, 1, 19)
        x_attn = (x * attn_w).sum(dim=2)  # (N, 256)
        x_attn = F.relu(self.attn_fc(x_attn))  # (N, 128)
        x_attn = self.drop_cnn(x_attn)

        # Fuse with demographics
        demo = torch.cat([age, gender], dim=1)  # (N, 2)
        x = torch.cat([x_attn, demo], dim=1)  # (N, 130)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════
# 5. Dataset & Training
# ═══════════════════════════════════════════════
class PPGDataset(Dataset):
    def __init__(self, X, sbp, dbp, age, gender):
        self.X = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)
        self.sbp = torch.tensor(sbp, dtype=torch.float32)
        self.dbp = torch.tensor(dbp, dtype=torch.float32)
        self.age = torch.tensor(age[:, None], dtype=torch.float32)
        self.gender = torch.tensor(gender[:, None], dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.age[i], self.gender[i], self.sbp[i], self.dbp[i]


train_ds = PPGDataset(X_tr_n, sbp_tr_n, dbp_tr_n, age_tr_n, gen_tr)
val_ds = PPGDataset(X_vl_n, sbp_vl_n, dbp_vl_n, age_vl_n, gen_vl)
test_ds = PPGDataset(X_te_n, sbp_te_n, dbp_te_n, age_te_n, gen_te)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

# ─── Train ───────────────────────────────────
print(f"\n[4/5] Training MultiScaleBP...")

model = MultiScaleBP(dropout=0.35).to(DEVICE)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=40, T_mult=2, eta_min=1e-6)
criterion = nn.HuberLoss(delta=1.0)

best_val = float('inf')
best_ep = 0
no_imp = 0

for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0
    for x, age, gen, sbp, dbp in train_loader:
        x, age, gen = x.to(DEVICE), age.to(DEVICE), gen.to(DEVICE)
        sbp, dbp = sbp.to(DEVICE), dbp.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x, age, gen)
        loss = criterion(pred[:, 0], sbp) + criterion(pred[:, 1], dbp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        tr_loss += loss.item() * len(x)
    tr_loss /= len(train_ds)

    scheduler.step()

    model.eval()
    vl_loss = 0
    with torch.no_grad():
        for x, age, gen, sbp, dbp in val_loader:
            x, age, gen = x.to(DEVICE), age.to(DEVICE), gen.to(DEVICE)
            sbp, dbp = sbp.to(DEVICE), dbp.to(DEVICE)
            pred = model(x, age, gen)
            loss = criterion(pred[:, 0], sbp) + criterion(pred[:, 1], dbp)
            vl_loss += loss.item() * len(x)
    vl_loss /= len(val_ds)

    if vl_loss < best_val:
        best_val = vl_loss
        best_ep = epoch
        no_imp = 0
        torch.save(model.state_dict(), "/tmp/best_ms.pt")
    else:
        no_imp += 1

    if epoch % 25 == 0 or no_imp == PATIENCE:
        model.eval()
        with torch.no_grad():
            all_p = []
            for x, age, gen, _, _ in val_loader:
                p = model(x.to(DEVICE), age.to(DEVICE), gen.to(DEVICE)).cpu().numpy()
                all_p.append(p)
            p = np.concatenate(all_p)
            y_s = np.concatenate([s.numpy() for _, _, _, s, _ in val_loader])
            y_d = np.concatenate([d.numpy() for _, _, _, _, d in val_loader])
            p_s = p[:, 0] * sbp_s + sbp_m
            p_d = p[:, 1] * dbp_s + dbp_m
            y_s_d = y_s * sbp_s + sbp_m
            y_d_d = y_d * dbp_s + dbp_m
            sm = mean_absolute_error(y_s_d, p_s)
            dm = mean_absolute_error(y_d_d, p_d)
        lr = optimizer.param_groups[0]['lr']
        star = '★' if epoch == best_ep else ''
        print(f"  Ep {epoch:>3d}: Tr={tr_loss:.4f} Vl={vl_loss:.4f} "
              f"SBP={sm:.2f} DBP={dm:.2f} lr={lr:.1e} {star}")

    if no_imp >= PATIENCE:
        print(f"  Early stop at epoch {epoch} (best: {best_ep})")
        break

# ═══════════════════════════════════════════════
# 6. Final Evaluation
# ═══════════════════════════════════════════════
model.load_state_dict(torch.load("/tmp/best_ms.pt", weights_only=True))
model.eval()

def predict(loader):
    with torch.no_grad():
        ps = []
        for x, age, gen, _, _ in loader:
            p = model(x.to(DEVICE), age.to(DEVICE), gen.to(DEVICE)).cpu().numpy()
            ps.append(p)
        ps = np.concatenate(ps)
    return ps[:, 0] * sbp_s + sbp_m, ps[:, 1] * dbp_s + dbp_m

p_sbp_tr, p_dbp_tr = predict(train_loader)
p_sbp_vl, p_dbp_vl = predict(val_loader)
p_sbp_te, p_dbp_te = predict(test_loader)

def bhs_grade(y, p):
    e = np.abs(y - p)
    p5 = np.mean(e <= 5) * 100
    p10 = np.mean(e <= 10) * 100
    p15 = np.mean(e <= 15) * 100
    if p5 >= 60 and p10 >= 85 and p15 >= 95: return "A"
    if p5 >= 50 and p10 >= 75 and p15 >= 90: return "B"
    if p5 >= 40 and p10 >= 65 and p15 >= 85: return "C"
    return "D"

def evaluate(name, y_s, p_s, y_d, p_d):
    sm = mean_absolute_error(y_s, p_s)
    sr = np.sqrt(mean_squared_error(y_s, p_s))
    sr2 = r2_score(y_s, p_s)
    dm = mean_absolute_error(y_d, p_d)
    dr = np.sqrt(mean_squared_error(y_d, p_d))
    dr2 = r2_score(y_d, p_d)
    s_bhs = bhs_grade(y_s, p_s)
    d_bhs = bhs_grade(y_d, p_d)

    s_me = np.mean(p_s - y_s)
    s_se = np.std(p_s - y_s)
    d_me = np.mean(p_d - y_d)
    d_se = np.std(p_d - y_d)
    s_aami = abs(s_me) <= 5 and s_se <= 8
    d_aami = abs(d_me) <= 5 and d_se <= 8

    print(f"\n  {name}:")
    print(f"    SBP: MAE={sm:.2f}  RMSE={sr:.2f}  R²={sr2:.3f}  BHS={s_bhs}  AAMI={'✓' if s_aami else '✗'}")
    print(f"    DBP: MAE={dm:.2f}  RMSE={dr:.2f}  R²={dr2:.3f}  BHS={d_bhs}  AAMI={'✓' if d_aami else '✗'}")
    print(f"    SBP ≤5/≤10/≤15: {np.mean(np.abs(y_s-p_s)<=5)*100:.0f}/{np.mean(np.abs(y_s-p_s)<=10)*100:.0f}/{np.mean(np.abs(y_s-p_s)<=15)*100:.0f}%")
    print(f"    DBP ≤5/≤10/≤15: {np.mean(np.abs(y_d-p_d)<=5)*100:.0f}/{np.mean(np.abs(y_d-p_d)<=10)*100:.0f}/{np.mean(np.abs(y_d-p_d)<=15)*100:.0f}%")
    return sm, sr2, dm, dr2, s_bhs, d_bhs

print(f"\n{'=' * 60}")
print("FINAL RESULTS (Test Set — held-out subjects)")
print("=" * 60)

bl_s = mean_absolute_error(sbp_te, np.full_like(sbp_te, sbp_m))
bl_d = mean_absolute_error(dbp_te, np.full_like(dbp_te, dbp_m))

print(f"\n  Baseline (predict mean):  SBP={bl_s:.2f}  DBP={bl_d:.2f}")
print(f"  XGBoost (handcrafted):    SBP=20.61  DBP=11.71  R²≈0.02")
print(f"  CNN1D (PPG only):         SBP=16.63   DBP=9.14   R²=0.32/0.24")

sms, _, dms, _, _, _ = evaluate("MultiScaleBP (PPG+Age+Gender)", sbp_te, p_sbp_te, dbp_te, p_dbp_te)

# Also show train and val
evaluate("Train", sbp_tr, p_sbp_tr, dbp_tr, p_dbp_tr)
evaluate("Val", sbp_vl, p_sbp_vl, dbp_vl, p_dbp_vl)

# Improvement
print(f"\n  {'─' * 50}")
print(f"  vs XGBoost:   SBP {sms:.1f} ({20.61-sms:+5.1f}) | DBP {dms:.1f} ({11.71-dms:+5.1f})")
prev_s = 16.63; prev_d = 9.14
print(f"  vs CNN1D:     SBP {sms:.1f} ({prev_s-sms:+5.1f}) | DBP {dms:.1f} ({prev_d-dms:+5.1f})")
print(f"  vs Baseline:  SBP {sms:.1f} ({bl_s-sms:+5.1f}) | DBP {dms:.1f} ({bl_d-dms:+5.1f})")

# Save
result = {
    "timestamp": datetime.now().isoformat(),
    "model": "MultiScaleBP (PPG + Age + Gender)",
    "sbp_test_mae": round(float(sms), 2),
    "dbp_test_mae": round(float(dms), 2),
    "baseline": {"sbp_mae": round(float(bl_s), 2), "dbp_mae": round(float(bl_d), 2)},
    "previous": {"xgb_sbp": 20.61, "xgb_dbp": 11.71, "cnn_sbp": 16.63, "cnn_dbp": 9.14},
}
with open("data/processed/ms_final_results.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to data/processed/ms_final_results.json")
print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
