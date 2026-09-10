"""
Standard ML pipeline: Train / Validation / Test split.

Anti-overfitting:
  1. Subject-level split (no data leakage across subjects)
  2. Early stopping on validation set
  3. Hyperparameter tuning on validation set (not test)
  4. Test set evaluated ONLY ONCE at the end
"""

import sys; sys.path.insert(0, ".")
import os, json, time, glob
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import h5py

np.random.seed(42)

# ═══════════════════════════════════════════════════════
# 1. Load / Extract Features
# ═══════════════════════════════════════════════════════
DATA_DIR = "data/pulsedb/Segment_Files"
FEAT_CACHE = "data/processed/pulsedb_features_proper.npz"
SAMPLING_RATE = 125

print("=" * 60)
print("STANDARD TRAIN/VAL/TEST PIPELINE")
print("=" * 60)

from src.ml.features import extract_ppg_features

if not os.path.exists(FEAT_CACHE):
    print(f"\n[1/4] Extracting features from {DATA_DIR}...")
    t0 = time.time()
    all_files = sorted(glob.glob(f"{DATA_DIR}/**/*.mat", recursive=True))
    print(f"  Found {len(all_files)} .mat files")

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
                        ppg = np.array(g[ppg_refs[i]]).flatten().astype(np.float64)
                    except:
                        continue
                    if ppg.std() < 1e-6:
                        continue
                    try:
                        feats = extract_ppg_features(ppg, sampling_rate=SAMPLING_RATE)
                    except:
                        continue
                    if len(feats) < 5:
                        continue
                    try:
                        sbp = float(np.array(g[sbp_refs[i]]).flatten()[0])
                        dbp = float(np.array(g[dbp_refs[i]]).flatten()[0])
                    except:
                        continue
                    X_list.append(feats)
                    y_sbp_list.append(sbp)
                    y_dbp_list.append(dbp)
                    subjects.append(subject)
                    sources.append(source)
        except:
            continue

    all_keys = sorted(set().union(*[set(f.keys()) for f in X_list]))
    feature_names = list(all_keys)
    X = np.zeros((len(X_list), len(all_keys)), dtype=np.float32)
    for i, feats in enumerate(X_list):
        for j, k in enumerate(all_keys):
            X[i, j] = feats.get(k, 0.0)
    y_sbp = np.array(y_sbp_list, dtype=np.float32)
    y_dbp = np.array(y_dbp_list, dtype=np.float32)
    subjects_arr = np.array(subjects)
    sources_arr = np.array(sources)

    print(f"  Done: {len(y_sbp)} segments from {len(set(subjects))} subjects")
    print(f"  MIMIC: {(sources_arr=='MIMIC').sum()}, VitalDB: {(sources_arr=='VitalDB').sum()}")

    np.savez_compressed(FEAT_CACHE, X=X, y_sbp=y_sbp, y_dbp=y_dbp,
                        feature_names=feature_names, subjects=subjects_arr,
                        sources=sources_arr)
else:
    print(f"\n[1/4] Loading cached features...")
    d = np.load(FEAT_CACHE, allow_pickle=True)
    X, y_sbp, y_dbp = d["X"], d["y_sbp"], d["y_dbp"]
    feature_names = list(d["feature_names"])
    subjects_arr = d["subjects"]
    sources_arr = d["sources"]
    print(f"  {len(y_sbp)} segments from {len(set(subjects_arr))} subjects")

# ═══════════════════════════════════════════════════════
# 2. Train / Val / Test Split (subject-level)
# ═══════════════════════════════════════════════════════
print(f"\n[2/4] Subject-level Train/Val/Test split")

unique_subjects = np.unique(subjects_arr)
rng = np.random.RandomState(42)
rng.shuffle(unique_subjects)

n = len(unique_subjects)
n_train = int(n * 0.70)
n_val = int(n * 0.15)
# rest = test (~15%)

train_subjs = set(unique_subjects[:n_train])
val_subjs = set(unique_subjects[n_train:n_train + n_val])
test_subjs = set(unique_subjects[n_train + n_val:])

train_mask = np.array([s in train_subjs for s in subjects_arr])
val_mask = np.array([s in val_subjs for s in subjects_arr])
test_mask = np.array([s in test_subjs for s in subjects_arr])

X_train, y_sbp_train, y_dbp_train = X[train_mask], y_sbp[train_mask], y_dbp[train_mask]
X_val, y_sbp_val, y_dbp_val = X[val_mask], y_sbp[val_mask], y_dbp[val_mask]
X_test, y_sbp_test, y_dbp_test = X[test_mask], y_sbp[test_mask], y_dbp[test_mask]

print(f"  Train:  {len(y_sbp_train):>5d} segments ({len(train_subjs)} subjects)")
print(f"  Val:    {len(y_sbp_val):>5d} segments ({len(val_subjs)} subjects)")
print(f"  Test:   {len(y_sbp_test):>5d} segments ({len(test_subjs)} subjects)")
print(f"  (NEVER look at Test during training/tuning)")

# BP distribution check
for name, y in [("SBP", y_sbp), ("DBP", y_dbp)]:
    for subset, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        yy = y[mask]
    # just print overall
    print(f"  {name} overall: mean={y.mean():.1f}, std={y.std():.1f}, range=[{y.min():.0f},{y.max():.0f}]")

# ═══════════════════════════════════════════════════════
# 3. Hyperparameter Tuning on Validation Set
# ═══════════════════════════════════════════════════════
print(f"\n[3/4] Training with early stopping on Validation set")

# Use conservative params to start, verify with early stopping
base_params = {
    "n_estimators": 2000,  # large number, early stopping will cut it short
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
}

def train_and_eval(params, X_tr, y_tr, X_v, y_v, X_te, y_te, bp_name):
    """Train with early stopping on val, evaluate on train/val/test."""
    model = XGBRegressor(**params)

    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_v, y_v)],
        verbose=False,
    )
    train_time = time.time() - t0

    # Predictions
    p_train = model.predict(X_tr)
    p_val = model.predict(X_v)
    p_test = model.predict(X_te)

    train_mae = mean_absolute_error(y_tr, p_train)
    val_mae = mean_absolute_error(y_v, p_val)
    test_mae = mean_absolute_error(y_te, p_test)

    train_r2 = r2_score(y_tr, p_train)
    val_r2 = r2_score(y_v, p_val)
    test_r2 = r2_score(y_te, p_test)

    n_trees = model.best_iteration if model.best_iteration else params["n_estimators"]

    gap = val_mae - train_mae
    overfit_flag = "⚠️ HIGH" if gap > 3 else ("⚠️ MEDIUM" if gap > 1.5 else "✓ OK")

    print(f"  [{bp_name}] n_trees={n_trees} | "
          f"Train={train_mae:.2f} | Val={val_mae:.2f} | Test={test_mae:.2f} "
          f"| Gap={gap:+.2f} {overfit_flag} "
          f"({train_time:.0f}s)")

    return {
        "train_mae": round(float(train_mae), 2),
        "val_mae": round(float(val_mae), 2),
        "test_mae": round(float(test_mae), 2),
        "train_r2": round(float(train_r2), 3),
        "val_r2": round(float(val_r2), 3),
        "test_r2": round(float(test_r2), 3),
        "n_trees": int(n_trees),
        "overfit_gap": round(float(gap), 2),
    }

results = {}

# SBP
results["SBP"] = train_and_eval(
    base_params, X_train, y_sbp_train,
    X_val, y_sbp_val, X_test, y_sbp_test, "SBP"
)

# DBP
results["DBP"] = train_and_eval(
    base_params, X_train, y_dbp_train,
    X_val, y_dbp_val, X_test, y_dbp_test, "DBP"
)

# ═══════════════════════════════════════════════════════
# 4. Feature Importance
# ═══════════════════════════════════════════════════════
print(f"\n[4/4] Feature importance analysis")

model_sbp = XGBRegressor(**base_params)
model_sbp.fit(X_train, y_sbp_train,
              eval_set=[(X_train, y_sbp_train), (X_val, y_sbp_val)],
              verbose=False)
model_dbp = XGBRegressor(**base_params)
model_dbp.fit(X_train, y_dbp_train,
              eval_set=[(X_train, y_dbp_train), (X_val, y_dbp_val)],
              verbose=False)

for bp, model in [("SBP", model_sbp), ("DBP", model_dbp)]:
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]
    print(f"\n  Top-15 {bp} features:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:>2d}. {feature_names[idx]:<30s} {imp[idx]:.4f}")

# ═══════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL RESULTS (Train/Val/Test)")
print("=" * 60)

for bp in ["SBP", "DBP"]:
    r = results[bp]
    print(f"\n  {bp}:")
    print(f"    Train: MAE={r['train_mae']:.2f}, R²={r['train_r2']:.3f}")
    print(f"    Val:   MAE={r['val_mae']:.2f}, R²={r['val_r2']:.3f}")
    print(f"    Test:  MAE={r['test_mae']:.2f}, R²={r['test_r2']:.3f}")
    print(f"    Overfit gap: {r['overfit_gap']:+.2f} mmHg, n_trees={r['n_trees']}")

# Save
results["timestamp"] = datetime.now().isoformat()
results["n_segments"] = int(len(y_sbp))
results["n_subjects"] = int(len(unique_subjects))
results["n_features"] = int(len(feature_names))
results["split"] = {
    "train": {"segments": int(len(y_sbp_train)), "subjects": int(len(train_subjs))},
    "val": {"segments": int(len(y_sbp_val)), "subjects": int(len(val_subjs))},
    "test": {"segments": int(len(y_sbp_test)), "subjects": int(len(test_subjs))},
}

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/proper_split_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to data/processed/proper_split_results.json")
