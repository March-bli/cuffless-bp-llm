"""完整 PulseDB 上 XGBoost 的 K 校准（偏差校正 + 微调），对比 LLM few-shot"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import json

SEED = 42
DATA = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
OUT = "/users/acp25bl/bishe/ml_calibration_full.json"
MAX_K = 20
N_TEST = 5
KS = [1, 3, 5, 10, 20]

d = np.load(DATA, allow_pickle=True)
X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
subjects = d["subjects"]
print(f"数据: {len(X)} 段, {len(np.unique(subjects))} 受试者", flush=True)

rng = np.random.RandomState(SEED)
unique_subjects = np.unique(subjects)
rng.shuffle(unique_subjects)
test_subjs = unique_subjects[:20]
train_subjs = unique_subjects[20:]
train_mask = np.isin(subjects, train_subjs)
X_train, y_sbp_train = X[train_mask], y_sbp[train_mask]
y_dbp_train = y_dbp[train_mask]
print(f"训练集: {len(train_subjs)} 受试者, {X_train.shape[0]} 段", flush=True)

# 全局模型
g_sbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=8)
g_dbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=8)
g_sbp.fit(X_train, y_sbp_train)
g_dbp.fit(X_train, y_dbp_train)
print("全局模型训练完成", flush=True)

results = {k: {"bias_corr": {"sbp": [], "dbp": []},
               "finetune": {"sbp": [], "dbp": []}} for k in KS}

for si, subj in enumerate(test_subjs):
    idx = np.where(subjects == subj)[0]
    if len(idx) < MAX_K + N_TEST:
        continue
    calib_idx = idx[:MAX_K]
    test_idx = idx[MAX_K:MAX_K + N_TEST]

    for K in KS:
        cal = calib_idx[:K]
        X_cal, s_cal = X[cal], y_sbp[cal]
        d_cal = y_dbp[cal]
        X_t, s_t = X[test_idx], y_sbp[test_idx]
        d_t = y_dbp[test_idx]

        # 偏差校正
        pred_s = g_sbp.predict(X_t)
        pred_d = g_dbp.predict(X_t)
        bias_s = np.mean(s_cal - g_sbp.predict(X_cal))
        bias_d = np.mean(d_cal - g_dbp.predict(X_cal))
        results[K]["bias_corr"]["sbp"].append(mean_absolute_error(s_t, pred_s + bias_s))
        results[K]["bias_corr"]["dbp"].append(mean_absolute_error(d_t, pred_d + bias_d))

        # 微调
        ft_s = xgb.XGBRegressor(n_estimators=20, max_depth=6, learning_rate=0.05, n_jobs=8)
        ft_d = xgb.XGBRegressor(n_estimators=20, max_depth=6, learning_rate=0.05, n_jobs=8)
        ft_s.fit(X_cal, s_cal, xgb_model=g_sbp.get_booster())
        ft_d.fit(X_cal, d_cal, xgb_model=g_dbp.get_booster())
        results[K]["finetune"]["sbp"].append(mean_absolute_error(s_t, ft_s.predict(X_t)))
        results[K]["finetune"]["dbp"].append(mean_absolute_error(d_t, ft_d.predict(X_t)))

    print(f"subject {si+1}/20 ({subj}) done", flush=True)

summary = {}
for K in KS:
    summary[str(K)] = {
        "bias_corr": {"sbp_mae": round(float(np.mean(results[K]["bias_corr"]["sbp"])), 2),
                      "dbp_mae": round(float(np.mean(results[K]["bias_corr"]["dbp"])), 2)},
        "finetune": {"sbp_mae": round(float(np.mean(results[K]["finetune"]["sbp"])), 2),
                     "dbp_mae": round(float(np.mean(results[K]["finetune"]["dbp"])), 2)},
    }

print("\n=== XGBoost K 校准结果（完整数据） ===")
for K in KS:
    bc = summary[str(K)]["bias_corr"]
    ft = summary[str(K)]["finetune"]
    print(f"K={K:<2d} bias_corr: {bc['sbp_mae']}/{bc['dbp_mae']}  finetune: {ft['sbp_mae']}/{ft['dbp_mae']}", flush=True)

with open(OUT, "w") as f:
    json.dump({"summary": summary}, f, indent=2)
print("结果已保存", flush=True)
