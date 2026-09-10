"""
个体校准实验 —— 真实训练，真实数据，严格分割
=================================================
实验设计：
  1. 用训练集受试者训练 CNN1D 基础模型（与论文一致）
  2. 对每个测试集受试者：
     a. 拿 N 段做校准微调（N = 5, 10, 20, 30, 50）
     b. 拿其余段做测试（这些段从没见过）
  3. 对比零校准 vs 校准后的准确率

关键约束：
  - 测试受试者的数据绝不参与基础模型训练
  - 校准段和测试段严格隔离
  - 不用任何假数据
"""

import sys; sys.path.insert(0, ".")
import os, json, time
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
CALIBRATION_SIZES = [0, 5, 10, 20, 30, 50]  # 0 = 零校准（基线）
BASE_EPOCHS = 80      # 基础模型训练轮数
CALIB_EPOCHS = 30     # 校准微调轮数
LR_BASE = 1e-3
LR_CALIB = 1e-4       # 校准时用更小的学习率

torch.manual_seed(SEED)
np.random.seed(SEED)

# ═══════════════════════════════════════════════════
# CNN1D Model (与论文完全一致)
# ═══════════════════════════════════════════════════
class CNN1D(nn.Module):
    def __init__(self, input_length=1250, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(32, 64, 9, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(64, 128, 7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(128, 256, 5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class PPGDataset(Dataset):
    def __init__(self, X, y_sbp, y_dbp):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.stack([y_sbp, y_dbp], axis=1), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def evaluate(model, X, y_sbp, y_dbp, y_sbp_std=None, y_dbp_std=None,
             y_sbp_mean=None, y_dbp_mean=None, batch_size=256):
    """
    Args:
        X, y_sbp, y_dbp: 归一化后的数据
        y_sbp_std, etc.: 如果提供，则将输出反归一化到 mmHg
    Returns:
        (mae_sbp, mae_dbp, r2_sbp, r2_dbp) — 单位取决于是否提供归一化参数
    """
    model.eval()
    ds = PPGDataset(X, y_sbp, y_dbp)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds_sbp, preds_dbp = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            out = model(x).cpu().numpy()
            preds_sbp.append(out[:, 0])
            preds_dbp.append(out[:, 1])
    p_sbp = np.concatenate(preds_sbp)
    p_dbp = np.concatenate(preds_dbp)

    # 反归一化到 mmHg（如果提供了参数）
    if y_sbp_std is not None:
        p_sbp = p_sbp * y_sbp_std + y_sbp_mean
        y_sbp_true = y_sbp * y_sbp_std + y_sbp_mean
    else:
        y_sbp_true = y_sbp

    if y_dbp_std is not None:
        p_dbp = p_dbp * y_dbp_std + y_dbp_mean
        y_dbp_true = y_dbp * y_dbp_std + y_dbp_mean
    else:
        y_dbp_true = y_dbp

    mae_sbp = float(np.mean(np.abs(p_sbp - y_sbp_true)))
    mae_dbp = float(np.mean(np.abs(p_dbp - y_dbp_true)))

    ss_res_sbp = np.sum((y_sbp_true - p_sbp) ** 2)
    ss_tot_sbp = np.sum((y_sbp_true - y_sbp_true.mean()) ** 2)
    r2_sbp = float(1 - ss_res_sbp / max(ss_tot_sbp, 1e-10))

    ss_res_dbp = np.sum((y_dbp_true - p_dbp) ** 2)
    ss_tot_dbp = np.sum((y_dbp_true - y_dbp_true.mean()) ** 2)
    r2_dbp = float(1 - ss_res_dbp / max(ss_tot_dbp, 1e-10))

    return mae_sbp, mae_dbp, r2_sbp, r2_dbp


def train_base_model(X_train, y_sbp_train, y_dbp_train, X_val, y_sbp_val, y_dbp_val):
    """Train CNN1D from scratch on training subjects."""
    model = CNN1D().to(DEVICE)
    ds = PPGDataset(X_train, y_sbp_train, y_dbp_train)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_BASE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(BASE_EPOCHS):
        train_loss = train_epoch(model, loader, optimizer, criterion)
        val_mae_sbp, _, _, _ = evaluate(model, X_val, y_sbp_val, y_dbp_val)
        val_loss = val_mae_sbp  # monitor SBP MAE on validation
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    model.load_state_dict(best_state)
    return model


def calibrate_and_test(model, X_calib, y_sbp_calib, y_dbp_calib,
                        X_test, y_sbp_test, y_dbp_test,
                        sbp_std, dbp_std, sbp_mean, dbp_mean,
                        n_epochs=CALIB_EPOCHS):
    """
    用校准数据微调模型，然后在测试数据上评估。
    所有输入均为归一化空间，返回 mmHg 空间的结果。
    """
    # 复制模型
    calib_model = CNN1D().to(DEVICE)
    calib_model.load_state_dict(model.state_dict())

    # ------ 校准前评估（零校准基线，归一化空间）------
    mae_sbp_before, mae_dbp_before, r2_sbp_before, r2_dbp_before = \
        evaluate(calib_model, X_test, y_sbp_test, y_dbp_test,
                 sbp_std, dbp_std, sbp_mean, dbp_mean)

    if len(X_calib) == 0:
        return (mae_sbp_before, mae_dbp_before, r2_sbp_before, r2_dbp_before,
                mae_sbp_before, mae_dbp_before, r2_sbp_before, r2_dbp_before)

    # ------ 校准：微调全网络（小学习率）------
    calib_ds = PPGDataset(X_calib, y_sbp_calib, y_dbp_calib)
    calib_loader = DataLoader(calib_ds, batch_size=min(len(X_calib), 16), shuffle=True)
    optimizer = torch.optim.Adam(calib_model.parameters(), lr=LR_CALIB)
    criterion = nn.MSELoss()

    for _ in range(n_epochs):
        train_epoch(calib_model, calib_loader, optimizer, criterion)

    # ------ 校准后评估（归一化空间 → mmHg）------
    mae_sbp_after, mae_dbp_after, r2_sbp_after, r2_dbp_after = \
        evaluate(calib_model, X_test, y_sbp_test, y_dbp_test,
                 sbp_std, dbp_std, sbp_mean, dbp_mean)

    return (mae_sbp_before, mae_dbp_before, r2_sbp_before, r2_dbp_before,
            mae_sbp_after, mae_dbp_after, r2_sbp_after, r2_dbp_after)


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("个体校准实验 —— PPG → 血压")
    print(f"设备: {DEVICE}  种子: {SEED}")
    print("=" * 65)

    # ---- 1. 加载真实数据 ----
    print("\n[1/5] 加载真实数据...")
    cache_path = "data/processed/pulsedb_waveforms.npz"
    if not os.path.exists(cache_path):
        print("  ❌ 数据缓存不存在，请先运行 scripts/train_deep_learning.py")
        sys.exit(1)

    d = np.load(cache_path, allow_pickle=True)
    X_raw = d["X"]          # shape: (18298, 1250)
    y_sbp_all = d["y_sbp"]
    y_dbp_all = d["y_dbp"]
    subjects_all = d["subjects"]
    sources_all = d["sources"]

    print(f"  总样本: {len(y_sbp_all):,} 段  |  受试者: {len(set(subjects_all))} 人")
    print(f"  SBP 范围: [{y_sbp_all.min():.0f}, {y_sbp_all.max():.0f}] mmHg  "
          f"均值={y_sbp_all.mean():.1f} ± {y_sbp_all.std():.1f}")
    print(f"  DBP 范围: [{y_dbp_all.min():.0f}, {y_dbp_all.max():.0f}] mmHg  "
          f"均值={y_dbp_all.mean():.1f} ± {y_dbp_all.std():.1f}")

    # ---- 2. 逐段 Z-score 归一化 ----
    print("\n[2/5] 逐段 Z-score 归一化...")
    X_norm = np.zeros_like(X_raw, dtype=np.float32)
    for i in range(len(X_raw)):
        m = X_raw[i].mean()
        s = X_raw[i].std()
        X_norm[i] = (X_raw[i] - m) / max(s, 1e-8)
    X_norm = X_norm[:, np.newaxis, :]  # (N, 1, 1250)

    # 目标归一化（用全体统计量，不影响分割公平性）
    sbp_mean, sbp_std = y_sbp_all.mean(), y_sbp_all.std()
    dbp_mean, dbp_std = y_dbp_all.mean(), y_dbp_all.std()
    y_sbp_norm = (y_sbp_all - sbp_mean) / sbp_std
    y_dbp_norm = (y_dbp_all - dbp_mean) / dbp_std

    # ---- 3. 受试者级别分割（与论文完全一致） ----
    print("\n[3/5] 受试者级别 Train/Test 分割...")
    unique_subjects = np.unique(subjects_all)
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_subjects)

    n = len(unique_subjects)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    # n_test = n - n_train - n_val

    train_subjs = set(unique_subjects[:n_train])
    val_subjs   = set(unique_subjects[n_train:n_train + n_val])
    test_subjs  = set(unique_subjects[n_train + n_val:])

    train_idx = [i for i, s in enumerate(subjects_all) if s in train_subjs]
    val_idx   = [i for i, s in enumerate(subjects_all) if s in val_subjs]
    test_idx  = [i for i, s in enumerate(subjects_all) if s in test_subjs]

    print(f"  训练受试者: {n_train} 人, {len(train_idx):,} 段")
    print(f"  验证受试者: {n_val} 人, {len(val_idx):,} 段")
    print(f"  测试受试者: {n - n_train - n_val} 人, {len(test_idx):,} 段")

    X_train = X_norm[train_idx]
    y_sbp_train = y_sbp_norm[train_idx]
    y_dbp_train = y_dbp_norm[train_idx]

    X_val = X_norm[val_idx]
    y_sbp_val = y_sbp_norm[val_idx]
    y_dbp_val = y_dbp_norm[val_idx]

    # ---- 4. 训练基础模型 ----
    print("\n[4/5] 训练 CNN1D 基础模型（仅用训练受试者）...")
    t0 = time.time()
    base_model = train_base_model(
        X_train, y_sbp_train, y_dbp_train,
        X_val, y_sbp_val, y_dbp_val)
    train_time = time.time() - t0

    # 基础模型在验证集上的表现
    val_mae_sbp_norm, val_mae_dbp_norm, _, _ = \
        evaluate(base_model, X_val, y_sbp_val, y_dbp_val)
    val_mae_sbp, val_mae_dbp, val_r2_sbp, val_r2_dbp = \
        evaluate(base_model, X_val, y_sbp_val, y_dbp_val,
                 sbp_std, dbp_std, sbp_mean, dbp_mean)

    print(f"\n  基础模型验证集 (mmHg):")
    print(f"    SBP MAE: {val_mae_sbp:.2f} mmHg  R²: {val_r2_sbp:.3f}")
    print(f"    DBP MAE: {val_mae_dbp:.2f} mmHg  R²: {val_r2_dbp:.3f}")
    print(f"  训练耗时: {train_time:.0f}s")

    # ---- 5. 对每个测试受试者做校准实验 ----
    print("\n[5/5] 个体校准实验（每个测试受试者独立进行）...")
    print(f"  测试受试者: {len(test_subjs)} 人")
    print(f"  校准分段数: {CALIBRATION_SIZES}")

    # 按受试者分组
    test_subjects_data = {}
    for idx in test_idx:
        subj = subjects_all[idx]
        if subj not in test_subjects_data:
            test_subjects_data[subj] = {"X": [], "y_sbp": [], "y_dbp": []}
        test_subjects_data[subj]["X"].append(X_norm[idx])
        test_subjects_data[subj]["y_sbp"].append(y_sbp_norm[idx])
        test_subjects_data[subj]["y_dbp"].append(y_dbp_norm[idx])

    # 只选择有足够多数据的受试者（至少能拿出最大校准段数+至少10段测试）
    min_segments = max(CALIBRATION_SIZES) + 10
    valid_test_subjects = {}
    for subj, data in test_subjects_data.items():
        if len(data["X"]) >= min_segments:
            valid_test_subjects[subj] = {
                "X": np.array(data["X"]),
                "y_sbp": np.array(data["y_sbp"]),
                "y_dbp": np.array(data["y_dbp"]),
            }

    print(f"  数据充足的测试受试者: {len(valid_test_subjects)}/{len(test_subjs)} 人"
          f" (需要 ≥{min_segments} 段)")

    # 每个校准分段数 × 每个受试者 → 独立实验
    results = {}  # results[calib_size] = [(sbp_mae_before, dbp_mae_before, ...), ...]

    for calib_n in CALIBRATION_SIZES:
        results[calib_n] = []
        print(f"\n  --- 校准段数: {calib_n} ---")

        for subj, data in valid_test_subjects.items():
            n_total = len(data["X"])
            rng_subj = np.random.RandomState(hash(subj) % (2**31))
            perm = rng_subj.permutation(n_total)

            if calib_n == 0:
                calib_idx = perm[:0]   # 空
            else:
                calib_idx = perm[:calib_n]
            test_seg_idx = perm[calib_n:]  # 剩余全部做测试

            X_calib  = data["X"][calib_idx]
            ys_calib = data["y_sbp"][calib_idx]
            yd_calib = data["y_dbp"][calib_idx]

            X_test_subj  = data["X"][test_seg_idx]
            ys_test_subj = data["y_sbp"][test_seg_idx]
            yd_test_subj = data["y_dbp"][test_seg_idx]

            # 所有数据均保持归一化空间，评估时反归一化
            (sbp_before, dbp_before, r2s_before, r2d_before,
             sbp_after, dbp_after, r2s_after, r2d_after) = \
                calibrate_and_test(base_model,
                                   X_calib, ys_calib, yd_calib,
                                   X_test_subj, ys_test_subj, yd_test_subj,
                                   sbp_std, dbp_std, sbp_mean, dbp_mean,
                                   n_epochs=CALIB_EPOCHS)

            results[calib_n].append({
                "subject": subj,
                "n_calib": len(X_calib),
                "n_test": len(X_test_subj),
                "sbp_mae_before": sbp_before,
                "dbp_mae_before": dbp_before,
                "sbp_mae_after": sbp_after,
                "dbp_mae_after": dbp_after,
                "sbp_r2_before": r2s_before,
                "dbp_r2_before": r2d_before,
                "sbp_r2_after": r2s_after,
                "dbp_r2_after": r2d_after,
            })

    # ---- 汇总 ----
    print("\n" + "=" * 65)
    print("实验结果汇总（反归一化到 mmHg）")
    print("=" * 65)
    print(f"{'校准段数':>8s} │ {'受试者':>6s} │ {'SBP MAE(前)':>12s} │ {'SBP MAE(后)':>12s} │ "
          f"{'改善':>7s} │ {'DBP MAE(前)':>12s} │ {'DBP MAE(后)':>12s} │ {'改善':>7s}")
    print("─" * 90)

    summary = {}
    for calib_n in CALIBRATION_SIZES:
        records = results[calib_n]
        n_subj = len(records)
        if n_subj == 0:
            continue

        sbp_before = [r["sbp_mae_before"] for r in records]
        sbp_after  = [r["sbp_mae_after"] for r in records]
        dbp_before = [r["dbp_mae_before"] for r in records]
        dbp_after  = [r["dbp_mae_after"] for r in records]

        sbp_b_mean = np.mean(sbp_before)
        sbp_a_mean = np.mean(sbp_after)
        dbp_b_mean = np.mean(dbp_before)
        dbp_a_mean = np.mean(dbp_after)

        sbp_improv = (sbp_b_mean - sbp_a_mean) / max(sbp_b_mean, 1e-10) * 100
        dbp_improv = (dbp_b_mean - dbp_a_mean) / max(dbp_b_mean, 1e-10) * 100

        print(f"{calib_n:>6d}段 │ {n_subj:>4d}人 │ {sbp_b_mean:>10.2f} mmHg │ "
              f"{sbp_a_mean:>10.2f} mmHg │ {sbp_improv:>5.1f}% │ "
              f"{dbp_b_mean:>10.2f} mmHg │ {dbp_a_mean:>10.2f} mmHg │ "
              f"{dbp_improv:>5.1f}%")

        summary[calib_n] = {
            "n_subjects": n_subj,
            "sbp_mae_before_mean": float(sbp_b_mean),
            "sbp_mae_after_mean": float(sbp_a_mean),
            "sbp_improvement_pct": float(sbp_improv),
            "dbp_mae_before_mean": float(dbp_b_mean),
            "dbp_mae_after_mean": float(dbp_a_mean),
            "dbp_improvement_pct": float(dbp_improv),
        }

    # 保存完整结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "config": {
            "seed": SEED,
            "base_epochs": BASE_EPOCHS,
            "calib_epochs": CALIB_EPOCHS,
            "lr_base": LR_BASE,
            "lr_calib": LR_CALIB,
            "calibration_sizes": CALIBRATION_SIZES,
        },
        "data_stats": {
            "n_train_subjects": n_train,
            "n_val_subjects": n_val,
            "n_test_subjects": len(test_subjs),
            "n_valid_calib_subjects": len(valid_test_subjects),
            "n_total_segments": int(len(y_sbp_all)),
            "sbp_mean_mmhg": float(sbp_mean),
            "sbp_std_mmhg": float(sbp_std),
            "dbp_mean_mmhg": float(dbp_mean),
            "dbp_std_mmhg": float(dbp_std),
        },
        "base_model_val": {
            "sbp_mae_mmhg": float(val_mae_sbp * sbp_std),
            "dbp_mae_mmhg": float(val_mae_dbp * dbp_std),
            "sbp_r2": float(val_r2_sbp),
            "dbp_r2": float(val_r2_dbp),
            "train_time_s": float(train_time),
        },
        "summary": summary,
        "detailed_results": {str(k): v for k, v in results.items()},
    }

    output_path = "data/processed/calibration_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 完整结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
