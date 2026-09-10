"""完整 PulseDB 上重跑传统 ML 基线（HPC 版）"""
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import json

SEED = 42
DATA = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
OUT = "/users/acp25bl/bishe/data/processed/ml_baseline_full.json"

d = np.load(DATA, allow_pickle=True)
X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
subjects = d["subjects"]
print(f"数据: {len(X)} 段, {len(np.unique(subjects))} 受试者", flush=True)

rng = np.random.RandomState(SEED)
unique_subjects = np.unique(subjects)
rng.shuffle(unique_subjects)
test_subjs = unique_subjects[:20]
train_subjs = unique_subjects[20:]
print(f"train: {len(train_subjs)}, test: {len(test_subjs)}", flush=True)

train_mask = np.isin(subjects, train_subjs)
test_mask = np.isin(subjects, test_subjs)
X_train, X_test = X[train_mask], X[test_mask]

results = {}
for name, model in [
    ("XGBoost", xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)),
    ("RandomForest", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
]:
    for target, y in [("SBP", y_sbp), ("DBP", y_dbp)]:
        model.fit(X_train, y[train_mask])
        pred = model.predict(X_test)
        mae = mean_absolute_error(y[test_mask], pred)
        results[f"{name}_{target}"] = round(float(mae), 2)
        print(f"{name} {target}: MAE = {mae:.2f} mmHg", flush=True)

with open(OUT, "w") as f:
    json.dump({"results": results}, f, indent=2)
print("ML 基线完成", flush=True)
