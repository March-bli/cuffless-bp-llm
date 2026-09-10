"""计算 XGBoost 全局（完整数据）的 BHS 分布，与语义方法对比"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import json

DATA = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
SEED = 42
OUT = "/users/acp25bl/bishe/xgboost_bhs.json"

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
test_mask = np.isin(subjects, test_subjs)

X_train, X_test = X[train_mask], X[test_mask]
print(f"train: {X_train.shape}, test: {X_test.shape}", flush=True)

# 训练全局 XGBoost
m_sbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=16)
m_dbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=16)
m_sbp.fit(X_train, y_sbp[train_mask])
m_dbp.fit(X_train, y_dbp[train_mask])
print("训练完成", flush=True)

pred_sbp = m_sbp.predict(X_test)
pred_dbp = m_dbp.predict(X_test)
true_sbp = y_sbp[test_mask]
true_dbp = y_dbp[test_mask]

def bhs(errors):
    n = len(errors)
    return {
        "n": n,
        "mae": round(float(np.mean(errors)), 2),
        "pct_le5": round(float(np.sum(errors <= 5) / n * 100), 1),
        "pct_le10": round(float(np.sum(errors <= 10) / n * 100), 1),
        "pct_le15": round(float(np.sum(errors <= 15) / n * 100), 1),
    }

result = {
    "sbp": bhs(np.abs(pred_sbp - true_sbp)),
    "dbp": bhs(np.abs(pred_dbp - true_dbp)),
}
print("XGBoost 全局 BHS 分布:", result, flush=True)

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print("已保存", flush=True)
