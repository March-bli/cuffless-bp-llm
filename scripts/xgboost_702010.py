"""XGBoost 70/20/10 划分：全局 + 偏差校正 + BHS 分布（完整数据）"""
import numpy as np
import xgboost as xgb
import json

DATA = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
SEED = 42
K = 20
N_TEST = 5
OUT = "/users/acp25bl/bishe/xgboost_702010.json"

d = np.load(DATA, allow_pickle=True)
X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
subjects = d["subjects"]
print(f"数据: {len(X)} 段, {len(np.unique(subjects))} 受试者", flush=True)

rng = np.random.RandomState(SEED)
unique_subjects = np.unique(subjects)
rng.shuffle(unique_subjects)
n = len(unique_subjects)
n_train = int(n * 0.7)
n_val = int(n * 0.2)
train_subjs = unique_subjects[:n_train]
val_subjs = unique_subjects[n_train:n_train + n_val]
test_subjs = unique_subjects[n_train + n_val:]
print(f"70/20/10: train={len(train_subjs)}, val={len(val_subjs)}, test={len(test_subjs)}", flush=True)

train_mask = np.isin(subjects, train_subjs)
test_mask = np.isin(subjects, test_subjs)

# 训练全局 XGBoost
g_sbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=16)
g_dbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=16)
g_sbp.fit(X[train_mask], y_sbp[train_mask])
g_dbp.fit(X[train_mask], y_dbp[train_mask])
print("全局模型训练完成", flush=True)


def bhs(errors):
    n = len(errors)
    e = np.array(errors)
    return {"n": n, "mae": round(float(np.mean(e)), 2),
            "pct_le5": round(float(np.sum(e <= 5) / n * 100), 1),
            "pct_le10": round(float(np.sum(e <= 10) / n * 100), 1),
            "pct_le15": round(float(np.sum(e <= 15) / n * 100), 1)}


# 全局预测（测试集，后 5 段，与 LLM 一致）
global_sbp_preds, global_dbp_preds = [], []
global_sbp_true, global_dbp_true = [], []
bias_sbp_preds, bias_dbp_preds = [], []

for si, subj in enumerate(test_subjs):
    idx = np.where(subjects == subj)[0]
    if len(idx) < K + N_TEST:
        continue
    calib_idx = idx[:K]
    test_idx = idx[K:K + N_TEST]

    gp_s = g_sbp.predict(X[test_idx])
    gp_d = g_dbp.predict(X[test_idx])
    global_sbp_preds.extend(gp_s)
    global_dbp_preds.extend(gp_d)
    global_sbp_true.extend(y_sbp[test_idx])
    global_dbp_true.extend(y_dbp[test_idx])

    # 偏差校正
    X_cal = X[calib_idx]
    bias_s = np.mean(y_sbp[calib_idx] - g_sbp.predict(X_cal))
    bias_d = np.mean(y_dbp[calib_idx] - g_dbp.predict(X_cal))
    bias_sbp_preds.extend(gp_s + bias_s)
    bias_dbp_preds.extend(gp_d + bias_d)

    if (si + 1) % 100 == 0:
        print(f"  {si+1}/{len(test_subjs)} 完成", flush=True)

result = {
    "split": {"train": len(train_subjs), "val": len(val_subjs), "test": len(test_subjs)},
    "global": {
        "sbp": bhs(np.abs(np.array(global_sbp_preds) - np.array(global_sbp_true))),
        "dbp": bhs(np.abs(np.array(global_dbp_preds) - np.array(global_dbp_true))),
    },
    "bias_corr_k20": {
        "sbp": bhs(np.abs(np.array(bias_sbp_preds) - np.array(global_sbp_true))),
        "dbp": bhs(np.abs(np.array(bias_dbp_preds) - np.array(global_dbp_true))),
    },
}
print("\n=== 70/20/10 XGBoost 结果 ===", flush=True)
print("global SBP:", json.dumps(result["global"]["sbp"]), flush=True)
print("global DBP:", json.dumps(result["global"]["dbp"]), flush=True)
print("bias_corr SBP:", json.dumps(result["bias_corr_k20"]["sbp"]), flush=True)
print("bias_corr DBP:", json.dumps(result["bias_corr_k20"]["dbp"]), flush=True)

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print("已保存", flush=True)
