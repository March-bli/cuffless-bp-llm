"""在 857 人规模上重跑传统 ML 基线（XGBoost / RandomForest）。
使用与 LLM 实验相同的 subject-level split（seed=42，20 测试受试者）。
"""
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import json

SEED = 42
DATA = "/home/lby/projects/bishe/data/processed/pulsedb_features_proper.npz"

d = np.load(DATA, allow_pickle=True)
X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
subjects = d["subjects"]
print(f"数据: {len(X)} 段, {len(np.unique(subjects))} 受试者")

# 与 LLM 实验一致的 subject-level split
rng = np.random.RandomState(SEED)
unique_subjects = np.unique(subjects)
rng.shuffle(unique_subjects)
test_subjs = unique_subjects[:20]
train_subjs = unique_subjects[20:]
print(f"train: {len(train_subjs)} 受试者, test: {len(test_subjs)} 受试者")

train_mask = np.isin(subjects, train_subjs)
test_mask = np.isin(subjects, test_subjs)
X_train, X_test = X[train_mask], X[test_mask]
print(f"train: {X_train.shape[0]} 段, test: {X_test.shape[0]} 段")

results = {}
for name, model in [
    ("XGBoost", xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)),
    ("RandomForest", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1)),
]:
    for target, y in [("SBP", y_sbp), ("DBP", y_dbp)]:
        model.fit(X_train, y[train_mask])
        pred = model.predict(X_test)
        mae = mean_absolute_error(y[test_mask], pred)
        results[f"{name}_{target}"] = round(float(mae), 2)
        print(f"{name} {target}: MAE = {mae:.2f} mmHg")

with open("/home/lby/projects/bishe/data/processed/ml_baseline_857.json", "w") as f:
    json.dump({"config": {"seed": SEED, "n_test": 20, "n_train": len(train_subjs)},
               "results": results}, f, indent=2)
print("\n结果已保存")
