"""完整数据上的段级 XGBoost（5-fold，说明"段级划分作弊"）"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import json

DATA = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
SEED = 42

d = np.load(DATA, allow_pickle=True)
X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
print(f"数据: {X.shape}", flush=True)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
sbp_maes, dbp_maes = [], []

for fold, (tr, te) in enumerate(kf.split(X)):
    m_sbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=16)
    m_dbp = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=16)
    m_sbp.fit(X[tr], y_sbp[tr])
    m_dbp.fit(X[tr], y_dbp[tr])
    sbp_maes.append(mean_absolute_error(y_sbp[te], m_sbp.predict(X[te])))
    dbp_maes.append(mean_absolute_error(y_dbp[te], m_dbp.predict(X[te])))
    print(f"fold {fold+1}: SBP {sbp_maes[-1]:.2f}, DBP {dbp_maes[-1]:.2f}", flush=True)

res = {
    "segment_level_5fold": {
        "sbp_mae": round(float(np.mean(sbp_maes)), 2),
        "dbp_mae": round(float(np.mean(dbp_maes)), 2),
    }
}
print("段级 5-fold 结果:", res, flush=True)

with open("/users/acp25bl/bishe/ml_segment_level.json", "w") as f:
    json.dump(res, f, indent=2)
print("已保存", flush=True)
