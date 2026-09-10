"""生成论文核心图表：校准曲线 + 对比柱状图 + 语义抽象对比"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "/home/lby/projects/bishe/docs/figures"
os.makedirs(OUT, exist_ok=True)

# ── 数据（完整 PulseDB，前 K 个样本） ───────────────────────────────────
K = [0, 1, 3, 5, 10, 20]
SBP = [17.9, 18.0, 15.8, 14.9, 10.3, 8.4]
DBP = [14.6, 16.7, 11.8, 10.2, 8.0, 6.7]

baselines = {
    "XGBoost\n(global)": (16.1, 8.7),
    "XGBoost\nbias-corr K=20": (7.0, 4.2),
    "LLM numeric\nK=20": (8.4, 6.7),
    "LLM semantic\nK=20": (6.5, 3.6),
}

semantic = {
    "Semantic\nK=1": (16.6, 12.7),
    "Semantic\nK=3": (13.0, 8.9),
    "Semantic\nK=5": (13.0, 7.8),
    "Semantic\nK=10": (8.5, 4.8),
    "Semantic\nK=20": (6.5, 3.6),
}

plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})

# ── 图 1：校准曲线（三种方法对比） ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(K, SBP, "o-", color="#1f77b4", linewidth=2, markersize=7, label="LLM few-shot SBP")
ax.plot(K, DBP, "s-", color="#d62728", linewidth=2, markersize=7, label="LLM few-shot DBP")

# 传统 ML 校准（偏差校正，完整数据）
ml_k = [1, 3, 5, 10, 20]
bc_sbp = [14.3, 12.5, 12.0, 8.7, 7.0]
bc_dbp = [7.3, 6.5, 6.5, 5.3, 4.2]
ax.plot(ml_k, bc_sbp, "o--", color="#1f77b4", linewidth=1.5, alpha=0.6, markersize=5,
        label="ML bias-corr SBP")
ax.plot(ml_k, bc_dbp, "s--", color="#d62728", linewidth=1.5, alpha=0.6, markersize=5,
        label="ML bias-corr DBP")

ax.axhline(5, color="green", linestyle=":", linewidth=1, alpha=0.7, label="BHS Grade A (≤5)")
ax.axhline(10, color="orange", linestyle=":", linewidth=1, alpha=0.5, label="BHS Grade B (SBP ≤10)")
ax.set_xlabel("Calibration samples K")
ax.set_ylabel("MAE (mmHg)")
ax.set_title("Calibration curves: LLM few-shot vs conventional ML")
ax.set_xticks(K)
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_calibration_curve.png", dpi=200)
plt.close()

# ── 图 2：对比柱状图 ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(baselines.keys())
x = np.arange(len(names))
w = 0.36
sbp_vals = [baselines[n][0] for n in names]
dbp_vals = [baselines[n][1] for n in names]
b1 = ax.bar(x - w/2, sbp_vals, w, label="SBP MAE", color="#1f77b4")
b2 = ax.bar(x + w/2, dbp_vals, w, label="DBP MAE", color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("MAE (mmHg)")
ax.set_title("Comparison with baselines and prior work")
ax.axhline(5, color="green", linestyle="--", linewidth=1, alpha=0.6, label="BHS Grade A")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for b in [b1, b2]:
    for rect in b:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.3, f"{h:.1f}",
                ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_comparison.png", dpi=200)
plt.close()

# ── 图 3：语义抽象对比 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
names = list(semantic.keys())
x = np.arange(len(names))
sbp_vals = [semantic[n][0] for n in names]
dbp_vals = [semantic[n][1] for n in names]
ax.bar(x - w/2, sbp_vals, w, label="SBP MAE", color="#1f77b4")
ax.bar(x + w/2, dbp_vals, w, label="DBP MAE", color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("MAE (mmHg)")
ax.set_title("Semantic abstraction comparison (K=5)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for i, (s, d) in enumerate(zip(sbp_vals, dbp_vals)):
    ax.text(i - w/2, s + 0.5, f"{s:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w/2, d + 0.5, f"{d:.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_semantic.png", dpi=200)
plt.close()

print("图表已生成:")
for f in sorted(os.listdir(OUT)):
    print(" ", f)
