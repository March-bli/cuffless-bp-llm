"""
Ablation studies for thesis:
  1. Feature group ablation — which feature categories matter most?
  2. Data volume impact — how does performance scale with training size?
"""

import sys; sys.path.insert(0, ".")
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import json

# ── load data ──────────────────────────────────────────────────────────────
data = np.load("data/processed/pulsedb_features.npz", allow_pickle=True)
X = data["X"]
y_sbp = data["y_sbp"]
y_dbp = data["y_dbp"]
feature_names = list(data["feature_names"])

# ── feature groups ─────────────────────────────────────────────────────────
GROUPS = {
    "统计特征 (statistical)": [
        "ppg_mean", "ppg_std", "ppg_median", "ppg_skew", "ppg_kurtosis",
        "ppg_min", "ppg_max", "ppg_ptp", "ppg_rms",
        "ppg_p5", "ppg_p25", "ppg_p75", "ppg_p95", "ppg_entropy",
    ],
    "时域特征 (time-domain)": [
        "sys_amp_mean", "sys_amp_std", "dia_amp_mean",
        "pulse_amp_mean", "pulse_amp_std",
        "pulse_interval_mean", "pulse_interval_std",
        "upstroke_time_mean", "upstroke_time_std", "diastolic_time_mean",
        "pulse_width_25pct", "pulse_width_50pct", "pulse_width_75pct",
        "aix", "stiffness_index", "heart_rate",
    ],
    "导数特征 (VPG+APG)": [
        "vpg_max", "vpg_min", "vpg_std", "vpg_mean_abs",
        "apg_a_mean", "apg_a_std", "apg_b_mean", "apg_b_std",
        "apg_c_mean", "apg_c_std", "apg_d_mean", "apg_d_std",
        "apg_e_mean", "apg_e_std",
        "apg_b_a_ratio", "apg_c_a_ratio", "apg_d_a_ratio", "apg_e_a_ratio",
    ],
    "频域特征 (frequency)": [
        "spectral_energy", "hr_band_power", "hr_band_ratio",
        "dominant_freq", "dominant_power", "spectral_centroid", "lf_hf_ratio",
    ],
}

# Verify coverage
all_grouped = set()
for feats in GROUPS.values():
    all_grouped.update(feats)
missing = set(feature_names) - all_grouped
if missing:
    print(f"[WARN] {len(missing)} features not in any group: {sorted(missing)}")
    GROUPS["未分类"] = sorted(missing)

# ── ablation: feature groups ───────────────────────────────────────────────
print("=" * 70)
print("EXPERIMENT 1: Feature Group Ablation")
print("=" * 70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
base_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                          random_state=42, verbosity=0)

def get_indices(feat_names_subset):
    return [feature_names.index(f) for f in feat_names_subset if f in feature_names]

results_ablation = []

for group_name, feats in GROUPS.items():
    idx = get_indices(feats)
    if len(idx) == 0:
        continue
    X_sub = X[:, idx]

    pred_sbp = cross_val_predict(base_model, X_sub, y_sbp, cv=kf)
    pred_dbp = cross_val_predict(base_model, X_sub, y_dbp, cv=kf)

    sbp_mae = mean_absolute_error(y_sbp, pred_sbp)
    dbp_mae = mean_absolute_error(y_dbp, pred_dbp)
    sbp_r2 = 1 - np.sum((y_sbp - pred_sbp)**2) / np.sum((y_sbp - y_sbp.mean())**2)
    dbp_r2 = 1 - np.sum((y_dbp - pred_dbp)**2) / np.sum((y_dbp - y_dbp.mean())**2)

    results_ablation.append({
        "group": group_name,
        "n_features": len(idx),
        "sbp_mae": round(sbp_mae, 2),
        "dbp_mae": round(dbp_mae, 2),
        "sbp_r2": round(sbp_r2, 3),
        "dbp_r2": round(dbp_r2, 3),
    })

# Full features
idx_all = list(range(X.shape[1]))
pred_sbp_all = cross_val_predict(base_model, X, y_sbp, cv=kf)
pred_dbp_all = cross_val_predict(base_model, X, y_dbp, cv=kf)
results_ablation.append({
    "group": "全部特征 (all 55)",
    "n_features": X.shape[1],
    "sbp_mae": round(mean_absolute_error(y_sbp, pred_sbp_all), 2),
    "dbp_mae": round(mean_absolute_error(y_dbp, pred_dbp_all), 2),
    "sbp_r2": round(1 - np.sum((y_sbp-pred_sbp_all)**2) / np.sum((y_sbp-y_sbp.mean())**2), 3),
    "dbp_r2": round(1 - np.sum((y_dbp-pred_dbp_all)**2) / np.sum((y_dbp-y_dbp.mean())**2), 3),
})

# Print table
print(f"\n{'Feature Group':<30s} {'#F':>4s} {'SBP MAE':>8s} {'DBP MAE':>8s} {'SBP R²':>8s} {'DBP R²':>8s}")
print("-" * 72)
for r in results_ablation:
    print(f"{r['group']:<30s} {r['n_features']:>4d} {r['sbp_mae']:>8.2f} {r['dbp_mae']:>8.2f} {r['sbp_r2']:>8.3f} {r['dbp_r2']:>8.3f}")

# ── ablation: data volume ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 2: Data Volume Impact")
print("=" * 70)

ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
results_volume = []

for ratio in ratios:
    n = int(len(y_sbp) * ratio)
    if n < 100:
        continue
    # Random subset (use same seed for reproducibility)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_sbp), n, replace=False)
    X_sub, y_sbp_sub, y_dbp_sub = X[idx], y_sbp[idx], y_dbp[idx]

    pred_sbp = cross_val_predict(base_model, X_sub, y_sbp_sub, cv=min(5, n//20))
    pred_dbp = cross_val_predict(base_model, X_sub, y_dbp_sub, cv=min(5, n//20))

    results_volume.append({
        "ratio": ratio,
        "n_samples": n,
        "sbp_mae": round(mean_absolute_error(y_sbp_sub, pred_sbp), 2),
        "dbp_mae": round(mean_absolute_error(y_dbp_sub, pred_dbp), 2),
        "sbp_r2": round(1 - np.sum((y_sbp_sub-pred_sbp)**2)/np.sum((y_sbp_sub-y_sbp_sub.mean())**2), 3),
        "dbp_r2": round(1 - np.sum((y_dbp_sub-pred_dbp)**2)/np.sum((y_dbp_sub-y_dbp_sub.mean())**2), 3),
    })

print(f"\n{'Ratio':>8s} {'Samples':>8s} {'SBP MAE':>8s} {'DBP MAE':>8s} {'SBP R²':>8s} {'DBP R²':>8s}")
print("-" * 54)
for r in results_volume:
    print(f"{r['ratio']:>8.0%} {r['n_samples']:>8d} {r['sbp_mae']:>8.2f} {r['dbp_mae']:>8.2f} {r['sbp_r2']:>8.3f} {r['dbp_r2']:>8.3f}")

# ── save ───────────────────────────────────────────────────────────────────
output = {
    "ablation_features": results_ablation,
    "ablation_volume": results_volume,
}
with open("data/processed/ablation_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to data/processed/ablation_results.json")
