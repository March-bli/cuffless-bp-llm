"""
Stacking Ensemble for BP Estimation.

Algorithm optimization #2:
  1. Multiple base learners capture different data patterns
  2. Meta-learner (Ridge) combines predictions optimally
  3. 5-fold cross-validation prevents overfitting
  4. Each base model uses its own optimal hyperparameters

Expected improvement: additional 3-8% error reduction over single best model
"""

import sys; sys.path.insert(0, ".")
import numpy as np
import json
import os
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

np.random.seed(42)

# ── load data ──────────────────────────────────────────────────────────
data = np.load("data/processed/pulsedb_features.npz", allow_pickle=True)
X = data["X"]
y_sbp = data["y_sbp"]
y_dbp = data["y_dbp"]

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

# ── base learners ──────────────────────────────────────────────────────
base_learners = {
    # Random Forest — bagging, captures variance patterns
    "RF": RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    ),
    # XGBoost tuned — gradient boosting, our strongest single model
    "XGB_tuned": XGBRegressor(
        n_estimators=738, max_depth=15, learning_rate=0.01065,
        subsample=0.728, colsample_bytree=0.913, colsample_bylevel=0.676,
        min_child_weight=1.835, reg_alpha=0.808, reg_lambda=0.00056,
        gamma=0.000123, random_state=42, verbosity=0, n_jobs=-1,
    ),
    # LightGBM — different boosting strategy
    "LGBM": None,  # try import below
    # SVR — captures non-linear relationships differently
    "SVR_rbf": SVR(kernel="rbf", C=15, gamma="scale", cache_size=500),
    # KNN — local patterns, good for edge cases
    "KNN_7": KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=-1),
}

try:
    import lightgbm as lgb
    base_learners["LGBM"] = lgb.LGBMRegressor(
        n_estimators=300, max_depth=10, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1, force_col_wise=True,
    )
    print("LightGBM: available")
except ImportError:
    print("LightGBM: not installed, skipping")
    del base_learners["LGBM"]

# ── evaluation ─────────────────────────────────────────────────────────
def bhs_grade(y, p):
    e = np.abs(y - p)
    p5 = np.mean(e <= 5) * 100
    p10 = np.mean(e <= 10) * 100
    p15 = np.mean(e <= 15) * 100
    if p5 >= 60 and p10 >= 85 and p15 >= 95: return "A"
    if p5 >= 50 and p10 >= 75 and p15 >= 90: return "B"
    if p5 >= 40 and p10 >= 65 and p15 >= 85: return "C"
    return "D"


# ── stacking with 5-fold CV ────────────────────────────────────────────
def stacking_cv(X, y, base_models, meta_model, n_folds=5, seed=42):
    """
    Stacking with k-fold cross-validation.
    For each fold:
      1. Train base models on train set
      2. Predict on val set → meta features
      3. Train meta-learner on meta features
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    n_samples = len(y)
    n_models = len(base_models)

    # Store base model predictions (out-of-fold)
    meta_features = np.zeros((n_samples, n_models))
    final_preds = np.zeros(n_samples)

    # Also collect per-model predictions for analysis
    per_model_preds = {}

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        fold_meta = np.zeros((len(val_idx), n_models))

        for model_i, (name, model_cls) in enumerate(base_models.items()):
            # Scale for SVR/KNN
            if name in ("SVR_rbf", "KNN_7"):
                scaler = StandardScaler()
                Xt = scaler.fit_transform(X_train)
                Xv = scaler.transform(X_val)
            else:
                Xt, Xv = X_train, X_val

            model = model_cls.__class__(**model_cls.get_params())  # clone
            model.fit(Xt, y_train)
            preds = model.predict(Xv)
            fold_meta[:, model_i] = preds

            # Store per-model preds for analysis
            if name not in per_model_preds:
                per_model_preds[name] = np.zeros(n_samples)
            per_model_preds[name][val_idx] = preds

        # Train meta-learner on this fold
        meta_model_clone = meta_model.__class__(**meta_model.get_params())
        meta_model_clone.fit(fold_meta, y[val_idx])

        # Store meta features for final evaluation
        meta_features[val_idx] = fold_meta
        final_preds[val_idx] = meta_model_clone.predict(fold_meta)

    # Final meta-learner (fit on all data for reporting weights)
    final_meta = meta_model.__class__(**meta_model.get_params())
    final_meta.fit(meta_features, y)

    return final_preds, meta_features, per_model_preds, final_meta


# ── run stacking for both SBP and DBP ──────────────────────────────────
meta = Ridge(alpha=1.0, random_state=42)

print("\n" + "=" * 65)
print("STACKING ENSEMBLE — 5-fold Cross-Validation")
print("=" * 65)
print(f"Base learners: {list(base_learners.keys())}")
print(f"Meta-learner: Ridge(alpha=1.0)")

results = {}
for bp_type, y in [("SBP", y_sbp), ("DBP", y_dbp)]:
    print(f"\n{'─' * 40}")
    print(f"  {bp_type}")
    print(f"{'─' * 40}")

    final_preds, meta_features, per_model, meta_model = stacking_cv(
        X, y, base_learners, meta
    )

    # ── per-model evaluation ───────────────────────────────────────
    print(f"\n  Individual models:")
    print(f"  {'Model':<15s} {'MAE':>6s} {'RMSE':>6s} {'R²':>6s} {'BHS':>4s} {'w5%':>5s}")
    print(f"  {'─' * 45}")

    for name in base_learners:
        p = per_model[name]
        mae = mean_absolute_error(y, p)
        rmse = np.sqrt(np.mean((y - p) ** 2))
        r2 = 1 - np.sum((y - p)**2) / np.sum((y - y.mean())**2)
        bhs = bhs_grade(y, p)
        w5 = np.mean(np.abs(y - p) <= 5) * 100
        print(f"  {name:<15s} {mae:>6.2f} {rmse:>6.2f} {r2:>6.3f} {bhs:>4s} {w5:>4.1f}%")

    # ── stacking evaluation ────────────────────────────────────────
    mae = mean_absolute_error(y, final_preds)
    rmse = np.sqrt(np.mean((y - final_preds) ** 2))
    r2 = 1 - np.sum((y - final_preds)**2) / np.sum((y - y.mean())**2)
    bhs = bhs_grade(y, final_preds)
    w5 = np.mean(np.abs(y - final_preds) <= 5) * 100
    w10 = np.mean(np.abs(y - final_preds) <= 10) * 100
    w15 = np.mean(np.abs(y - final_preds) <= 15) * 100
    me = np.mean(final_preds - y)
    se = np.std(final_preds - y)
    aami = abs(me) <= 5 and se <= 8

    print(f"\n  ╔═══ STACKING RESULT ═══╗")
    print(f"  ║ {'MAE':>10s}: {mae:>7.2f} mmHg  ║")
    print(f"  ║ {'RMSE':>10s}: {rmse:>7.2f} mmHg  ║")
    print(f"  ║ {'R²':>10s}: {r2:>7.3f}       ║")
    print(f"  ║ {'BHS':>10s}: {bhs:>7s}       ║")
    print(f"  ║ {'w5/w10/w15':>10s}: {w5:>4.1f}/{w10:>4.1f}/{w15:>4.1f}%  ║")
    print(f"  ║ {'AAMI':>10s}: {str(aami):>7s}       ║")
    print(f"  ╚═══════════════════════╝")

    # ── meta-learner weights ────────────────────────────────────────
    print(f"\n  Meta-learner weights (Ridge coefficients):")
    for name, coef in zip(base_learners.keys(), meta_model.coef_):
        print(f"    {name:<15s}: {coef:+.4f}")

    results[bp_type] = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 3),
        "bhs_grade": bhs,
        "within_5mmHg": round(float(w5), 1),
        "within_10mmHg": round(float(w10), 1),
        "within_15mmHg": round(float(w15), 1),
        "aami_pass": bool(aami),
        "mean_error": round(float(me), 2),
        "std_error": round(float(se), 2),
        "meta_weights": {name: round(float(coef), 4) for name, coef in zip(base_learners.keys(), meta_model.coef_)},
        "per_model": {
            name: {
                "mae": round(float(mean_absolute_error(y, p)), 2),
                "r2": round(float(1 - np.sum((y-p)**2) / np.sum((y-y.mean())**2)), 3),
                "bhs": bhs_grade(y, p),
            }
            for name, p in per_model.items()
        },
    }

# ── compare with baseline ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL COMPARISON: Baseline vs Optuna vs Stacking")
print("=" * 65)

baseline = {"SBP": {"mae": 8.02, "rmse": 11.33, "r2": 0.689, "bhs_grade": "C"},
            "DBP": {"mae": 5.00, "rmse": 7.58,  "r2": 0.628, "bhs_grade": "A"}}

optuna = {"SBP": {"mae": 7.15, "rmse": 10.58, "r2": 0.729, "bhs_grade": "C"},
          "DBP": {"mae": 4.46, "rmse": 7.22,  "r2": 0.663, "bhs_grade": "A"}}

for bp in ["SBP", "DBP"]:
    print(f"\n  {bp}:")
    print(f"  {'Method':<20s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'BHS':>5s} {'ΔMAE':>8s}")
    print(f"  {'─' * 55}")
    base = baseline[bp]
    print(f"  {'Baseline XGBoost':<20s} {base['mae']:>8.2f} {base['rmse']:>8.2f} {base['r2']:>8.3f} {base['bhs_grade']:>5s} {'':>8s}")
    opt = optuna[bp]
    delta1 = opt['mae'] - base['mae']
    print(f"  {'Optuna Tuned':<20s} {opt['mae']:>8.2f} {opt['rmse']:>8.2f} {opt['r2']:>8.3f} {opt['bhs_grade']:>5s} {delta1:>+8.2f}")
    stk = results[bp]
    delta2 = stk['mae'] - opt['mae']
    arrow = "↓" if delta2 < 0 else "↑"
    print(f"  {'Stacking Ensemble':<20s} {stk['mae']:>8.2f} {stk['rmse']:>8.2f} {stk['r2']:>8.3f} {stk['bhs_grade']:>5s} {arrow}{abs(delta2):>7.2f}")

results["baseline"] = baseline
results["optuna"] = optuna
results["timestamp"] = datetime.now().isoformat()
results["n_models"] = len(base_learners)
results["model_names"] = list(base_learners.keys())

out_path = "data/processed/stacking_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
