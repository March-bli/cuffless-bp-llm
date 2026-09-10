"""补算 XGBoost 全局 + 偏差校正 + 微调的 BHS 分布（逐样本预测）
确保公平对比：所有方法的 BHS 分布都用相同协议计算。
"""
import numpy as np
import xgboost as xgb
import json

DATA = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
SEED = 42
KS = [1, 3, 5, 10, 20]
OUT = "/users/acp25bl/bishe/xgboost_bhs_full.json"

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

X_train = X[train_mask]
y_sbp_train = y_sbp[train_mask]
y_dbp_train = y_dbp[train_mask]

# 训练全局 XGBoost
g_sbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=16)
g_dbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=16)
g_sbp.fit(X_train, y_sbp_train)
g_dbp.fit(X_train, y_dbp_train)
print("全局模型训练完成", flush=True)

def bhs(errors):
    n = len(errors)
    e = np.array(errors)
    return {
        "n": n,
        "mae": round(float(np.mean(e)), 2),
        "pct_le5": round(float(np.sum(e <= 5) / n * 100), 1),
        "pct_le10": round(float(np.sum(e <= 10) / n * 100), 1),
        "pct_le15": round(float(np.sum(e <= 15) / n * 100), 1),
    }

# 收集所有测试受试者的预测
global_sbp_preds, global_dbp_preds = [], []
global_sbp_true, global_dbp_true = [], []
bias_preds = {k: {"sbp": [], "dbp": [], "true_sbp": [], "true_dbp": []} for k in KS}
ft_preds = {k: {"sbp": [], "dbp": [], "true_sbp": [], "true_dbp": []} for k in KS}

MAX_K = max(KS)
N_TEST = 5

for si, subj in enumerate(test_subjs):
    idx = np.where(subjects == subj)[0]
    if len(idx) < MAX_K + N_TEST:
        continue
    calib_idx = idx[:MAX_K]
    test_idx = idx[MAX_K:MAX_K + N_TEST]

    # 全局预测（所有测试段）
    gp_s = g_sbp.predict(X[test_idx])
    gp_d = g_dbp.predict(X[test_idx])
    global_sbp_preds.extend(gp_s)
    global_dbp_preds.extend(gp_d)
    global_sbp_true.extend(y_sbp[test_idx])
    global_dbp_true.extend(y_dbp[test_idx])

    for K in KS:
        cal = calib_idx[:K]
        X_cal = X[cal]
        s_cal = y_sbp[cal]
        d_cal = y_dbp[cal]

        # 偏差校正
        bias_s = np.mean(s_cal - g_sbp.predict(X_cal))
        bias_d = np.mean(d_cal - g_dbp.predict(X_cal))
        bc_s = gp_s + bias_s
        bc_d = gp_d + bias_d
        bias_preds[K]["sbp"].extend(bc_s)
        bias_preds[K]["dbp"].extend(bc_d)
        bias_preds[K]["true_sbp"].extend(y_sbp[test_idx])
        bias_preds[K]["true_dbp"].extend(y_dbp[test_idx])

        # 微调
        ft_s = xgb.XGBRegressor(n_estimators=20, max_depth=6, learning_rate=0.05, n_jobs=16)
        ft_d = xgb.XGBRegressor(n_estimators=20, max_depth=6, learning_rate=0.05, n_jobs=16)
        ft_s.fit(X_cal, s_cal, xgb_model=g_sbp.get_booster())
        ft_d.fit(X_cal, d_cal, xgb_model=g_dbp.get_booster())
        ft_preds[K]["sbp"].extend(ft_s.predict(X[test_idx]))
        ft_preds[K]["dbp"].extend(ft_d.predict(X[test_idx]))
        ft_preds[K]["true_sbp"].extend(y_sbp[test_idx])
        ft_preds[K]["true_dbp"].extend(y_dbp[test_idx])

    print(f"subject {si+1}/20 done", flush=True)

result = {
    "global": {
        "sbp": bhs(np.abs(np.array(global_sbp_preds) - np.array(global_sbp_true))),
        "dbp": bhs(np.abs(np.array(global_dbp_preds) - np.array(global_dbp_true))),
    },
    "bias_corr": {},
    "finetune": {},
}
for K in KS:
    result["bias_corr"][str(K)] = {
        "sbp": bhs(np.abs(np.array(bias_preds[K]["sbp"]) - np.array(bias_preds[K]["true_sbp"]))),
        "dbp": bhs(np.abs(np.array(bias_preds[K]["dbp"]) - np.array(bias_preds[K]["true_dbp"]))),
    }
    result["finetune"][str(K)] = {
        "sbp": bhs(np.abs(np.array(ft_preds[K]["sbp"]) - np.array(ft_preds[K]["true_sbp"]))),
        "dbp": bhs(np.abs(np.array(ft_preds[K]["dbp"]) - np.array(ft_preds[K]["true_dbp"]))),
    }

print("\n=== XGBoost 系列 BHS 分布 ===", flush=True)
print("全局:", json.dumps(result["global"], indent=2), flush=True)
for K in KS:
    print(f"偏差校正 K={K}: SBP {result['bias_corr'][str(K)]['sbp']}", flush=True)
    print(f"             DBP {result['bias_corr'][str(K)]['dbp']}", flush=True)
    print(f"微调     K={K}: SBP {result['finetune'][str(K)]['sbp']}", flush=True)
    print(f"             DBP {result['finetune'][str(K)]['dbp']}", flush=True)

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print("已保存", flush=True)
