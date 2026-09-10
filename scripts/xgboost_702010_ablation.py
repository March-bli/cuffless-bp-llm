"""XGBoost 70/20/10 K 消融：bias correction 在 K=1/3/5/10/20 的表现"""
import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
OUT = "/users/acp25bl/bishe/xgboost_702010_ablation.json"
SEED = 42
KS = [1, 3, 5, 10, 20]
N_TEST = 5

d = np.load(DATA, allow_pickle=True)
X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
subjects = d["subjects"]
print(f"数据: {len(X)} 段, {len(np.unique(subjects))} 受试者", flush=True)

rng = np.random.RandomState(SEED)
us = np.unique(subjects)
rng.shuffle(us)
n = len(us)
n_train = int(n * 0.7); n_val = int(n * 0.2)
train_s = us[:n_train]; val_s = us[n_train:n_train + n_val]; test_s = us[n_train + n_val:]
train_mask = np.isin(subjects, train_s)
print(f"70/20/10: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}", flush=True)

# 全局模型（训练一次）
g_sbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=16)
g_dbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=16)
g_sbp.fit(X[train_mask], y_sbp[train_mask])
g_dbp.fit(X[train_mask], y_dbp[train_mask])
print("全局模型训练完成", flush=True)

out = {}
for K in KS:
    pred_sbp, pred_dbp, true_sbp, true_dbp = [], [], [], []
    for subj in test_s:
        idx = np.where(subjects == subj)[0]
        if len(idx) < K + N_TEST:
            continue
        calib_idx = idx[:K]
        test_idx = idx[K:K + N_TEST]
        gp_s = g_sbp.predict(X[test_idx]); gp_d = g_dbp.predict(X[test_idx])
        cp_s = g_sbp.predict(X[calib_idx]); cp_d = g_dbp.predict(X[calib_idx])
        bias_s = gp_s - (np.mean(cp_s) - np.mean(y_sbp[calib_idx]))
        bias_d = gp_d - (np.mean(cp_d) - np.mean(y_dbp[calib_idx]))
        pred_sbp.extend(bias_s); pred_dbp.extend(bias_d)
        true_sbp.extend(y_sbp[test_idx]); true_dbp.extend(y_dbp[test_idx])
    out[str(K)] = {
        "sbp_mae": round(float(mean_absolute_error(true_sbp, pred_sbp)), 2),
        "dbp_mae": round(float(mean_absolute_error(true_dbp, pred_dbp)), 2),
        "n": len(true_sbp),
    }
    print(f"K={K}: {out[str(K)]}", flush=True)

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"已保存 {OUT}", flush=True)
