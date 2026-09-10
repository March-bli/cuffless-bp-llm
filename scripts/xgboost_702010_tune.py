"""XGBoost 70/20/10 验证集调参：用 val 选超参数，test 评估（全局 + 偏差校正）"""
import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
OUT = "/users/acp25bl/bishe/xgboost_702010_tuned.json"
SEED = 42
K = 20
N_TEST = 5

d = np.load(DATA, allow_pickle=True)
X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
subjects = d["subjects"]
print(f"数据: {len(X)} 段, {len(np.unique(subjects))} 受试者", flush=True)

# 70/20/10 划分（seed 42，与主实验一致）
rng = np.random.RandomState(SEED)
us = np.unique(subjects)
rng.shuffle(us)
n = len(us)
n_train = int(n * 0.7)
n_val = int(n * 0.2)
train_s = us[:n_train]
val_s = us[n_train:n_train + n_val]
test_s = us[n_train + n_val:]
train_mask = np.isin(subjects, train_s)
val_mask = np.isin(subjects, val_s)
print(f"70/20/10: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}", flush=True)

# 网格搜索（9 组合：3 max_depth × 3 learning_rate）
grid = []
for md in [4, 6, 8]:
    for lr in [0.03, 0.05, 0.1]:
        grid.append((md, lr))

print(f"\n=== 网格搜索 {len(grid)} 组合 ===", flush=True)
search_results = []
for md, lr in grid:
    params = dict(n_estimators=300, max_depth=md, learning_rate=lr,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=SEED, n_jobs=16, early_stopping_rounds=20)
    m_sbp = xgb.XGBRegressor(**params)
    m_dbp = xgb.XGBRegressor(**params)
    m_sbp.fit(X[train_mask], y_sbp[train_mask],
              eval_set=[(X[val_mask], y_sbp[val_mask])], verbose=False)
    m_dbp.fit(X[train_mask], y_dbp[train_mask],
              eval_set=[(X[val_mask], y_dbp[val_mask])], verbose=False)
    val_sbp = float(mean_absolute_error(y_sbp[val_mask], m_sbp.predict(X[val_mask])))
    val_dbp = float(mean_absolute_error(y_dbp[val_mask], m_dbp.predict(X[val_mask])))
    avg = (val_sbp + val_dbp) / 2
    search_results.append({"max_depth": md, "learning_rate": lr,
                           "val_sbp": round(val_sbp, 2), "val_dbp": round(val_dbp, 2),
                           "val_avg": round(avg, 2),
                           "n_estimators": int(m_sbp.best_iteration or 300)})
    print(f"  depth={md} lr={lr}: SBP={val_sbp:.2f} DBP={val_dbp:.2f} avg={avg:.2f} "
          f"(best_iter={m_sbp.best_iteration})", flush=True)

best = min(search_results, key=lambda r: r["val_avg"])
print(f"\n最优: max_depth={best['max_depth']} lr={best['learning_rate']} "
      f"(val avg={best['val_avg']})", flush=True)

# 用最优参数在 train 上训练（最终模型），test 评估
best_params = dict(n_estimators=best["n_estimators"], max_depth=best["max_depth"],
                   learning_rate=best["learning_rate"], subsample=0.8,
                   colsample_bytree=0.8, random_state=SEED, n_jobs=16)
g_sbp = xgb.XGBRegressor(**best_params)
g_dbp = xgb.XGBRegressor(**best_params)
g_sbp.fit(X[train_mask], y_sbp[train_mask])
g_dbp.fit(X[train_mask], y_dbp[train_mask])
print("最优参数全局模型训练完成", flush=True)


def bhs(errs):
    e = np.array(errs)
    return {"n": len(e), "mae": round(float(np.mean(e)), 2),
            "pct_le5": round(float(np.sum(e <= 5) / len(e) * 100), 1),
            "pct_le10": round(float(np.sum(e <= 10) / len(e) * 100), 1),
            "pct_le15": round(float(np.sum(e <= 15) / len(e) * 100), 1)}


# 全局 + 偏差校正评估（与主实验相同的 K=20 校准协议）
g_sbp_preds, g_dbp_preds, g_sbp_true, g_dbp_true = [], [], [], []
bias_sbp_preds, bias_dbp_preds = [], []

for subj in test_s:
    idx = np.where(subjects == subj)[0]
    if len(idx) < K + N_TEST:
        continue
    calib_idx = idx[:K]
    test_idx = idx[K:K + N_TEST]
    # 全局
    gp_s = g_sbp.predict(X[test_idx]); gp_d = g_dbp.predict(X[test_idx])
    g_sbp_preds.extend(gp_s); g_dbp_preds.extend(gp_d)
    g_sbp_true.extend(y_sbp[test_idx]); g_dbp_true.extend(y_dbp[test_idx])
    # 偏差校正
    cp_s = g_sbp.predict(X[calib_idx]); cp_d = g_dbp.predict(X[calib_idx])
    bias_s = gp_s - (np.mean(cp_s) - np.mean(y_sbp[calib_idx]))
    bias_d = gp_d - (np.mean(cp_d) - np.mean(y_dbp[calib_idx]))
    bias_sbp_preds.extend(bias_s); bias_dbp_preds.extend(bias_d)

g_sbp_err = np.abs(np.array(g_sbp_preds) - np.array(g_sbp_true))
g_dbp_err = np.abs(np.array(g_dbp_preds) - np.array(g_dbp_true))
bias_sbp_err = np.abs(np.array(bias_sbp_preds) - np.array(g_sbp_true))
bias_dbp_err = np.abs(np.array(bias_dbp_preds) - np.array(g_dbp_true))

out = {
    "split": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
    "grid_search": search_results,
    "best_params": best,
    "global": {"sbp": bhs(g_sbp_err), "dbp": bhs(g_dbp_err)},
    "bias_corr_k20": {"sbp": bhs(bias_sbp_err), "dbp": bhs(bias_dbp_err)},
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print("\n=== 调参后结果 ===", flush=True)
print(f"global SBP: {out['global']['sbp']}", flush=True)
print(f"global DBP: {out['global']['dbp']}", flush=True)
print(f"bias_corr SBP: {out['bias_corr_k20']['sbp']}", flush=True)
print(f"bias_corr DBP: {out['bias_corr_k20']['dbp']}", flush=True)
print(f"已保存 {OUT}", flush=True)
