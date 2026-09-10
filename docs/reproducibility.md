# 实验可复现性说明（Reproducibility Checklist）

## 1. 数据集

| 项目 | 值 |
|------|-----|
| 数据集 | PulseDB（MIMIC 子集 + Vital 子集） |
| 受试者数 | 5,159 |
| PPG 段数 | 约 555 万（10 秒/段，125 Hz） |
| 原始文件 | 5,704 个 .mat（Box 分卷解压，26 个分卷） |
| 存储位置（HPC） | /mnt/parscratch/users/acp25bl/pulsedb/Segment_Files/ |

## 2. 数据划分

| 项目 | 值 |
|------|-----|
| 划分策略 | 严格 subject-level split |
| 随机种子 | seed = 42 |
| 测试受试者 | 20 名（随机选取） |
| 训练/参考受试者 | 其余 5,139 名 |
| 校准样本选择 | **时间顺序前 K 个段**（非随机） |
| 测试段 | 前 K 个之后的 5 个段 |

## 3. 特征工程

| 项目 | 值 |
|------|-----|
| 特征维度 | 55 维（时域形态学 + 导数形态学 + 统计 + 频域） |
| 人口统计 | Age + Gender |
| 特征提取脚本 | build_feature_dataset.py |
| 输出文件 | pulsedb_features_full.npz（parscratch） |

## 4. 模型与推理

| 项目 | 值 |
|------|-----|
| LLM | Qwen3-8B（Qwen/Qwen3-8B，HuggingFace） |
| 硬件 | NVIDIA A100-80GB（Stanage HPC，gpu 分区） |
| 推理模式 | enable_thinking=False（禁用思考） |
| 生成参数 | do_sample=False（贪心），max_new_tokens=64 |
| 输出解析 | 优先匹配 SBP=/DBP= 标签，回退取末尾两个 50-250 数值 |

## 5. 实验协议

### LLM few-shot 消融
- K ∈ {1, 3, 5, 10, 20}
- 每个受试者：前 K 个校准样本 + 5 个测试段
- 20 测试受试者

### 传统 ML 基线
- XGBoost：300 棵树，max_depth=6，learning_rate=0.05，subsample=0.8
- 训练集：5,139 人；测试集：20 人

### 语义抽象（第七章）
- 规则语义描述：55 特征 → 三层描述（统计/结构/语义），保留定量值
- 语义瓶颈：LLM 生成定性摘要 → 仅凭摘要预测

## 6. 关键脚本（HPC 路径）

| 脚本 | 用途 |
|------|------|
| /users/acp25bl/bishe/build_feature_dataset.py | 特征提取 |
| /users/acp25bl/bishe/ml_baseline_full.py | 传统 ML |
| /users/acp25bl/bishe/llm_bp_local.py | LLM few-shot 消融 |
| /users/acp25bl/bishe/llm_bp_semantic.py | 语义描述 |
| /users/acp25bl/bishe/llm_bp_bottleneck.py | 语义瓶颈 |
| /users/acp25bl/bishe/run_all.sbatch | 总 Pipeline |

## 7. 结果文件（HPC）

| 文件 | 内容 |
|------|------|
| llm_bp_local_results.json | few-shot 消融结果 |
| llm_bp_semantic_results.json | 语义描述结果 |
| llm_bp_bottleneck_results.json | 语义瓶颈结果 |
| ml_baseline_full.json | 传统 ML 结果 |
| full_11406897.out | 完整 Pipeline 日志 |

## 8. 复现命令

```bash
# 在 HPC 上
cd /users/acp25bl/bishe
sbatch run_all.sbatch
# 作业完成后，结果在 llm_bp_*_results.json
```

## 9. 可复现性验证（准确表述）

**核心实验已通过两次及以上独立运行验证，结果完全一致：**

| 实验 | 运行次数 | 一致性 |
|------|---------|--------|
| XGBoost 全局基线 | 3 次 | ✅ 16.1/8.7 ≈ 16.08/8.74 |
| LLM zero-shot | 2 次 | ✅ 17.9/14.6 |
| LLM few-shot 消融（K=1/3/5/10/20） | 2 次 | ✅ 完全一致 |
| 语义描述 K=5 | 2 次 | ✅ 10.6/7.3 |
| 语义瓶颈 K=5 | 2 次 | ✅ 25.0/23.5 |

**补充实验为单次运行，但采用确定性设置（可复现）：**

| 实验 | 确定性保证 |
|------|-----------|
| XGBoost 偏差校正 / 微调（K=1~20） | 固定 seed |
| 语义描述 K=1/3/10/20 | 贪心解码 + 固定 seed |
| 段级 XGBoost 5-fold | 固定 seed（KFold shuffle） |

**说明**：LLM 推理的确定性来自贪心解码（do_sample=False）、固定随机种子、关闭思考模式（enable_thinking=False）三者共同保证——即使只运行一次，结果也是确定的、可重现的。两次独立运行的一致性验证了这一点。

## 10. 已知问题

- Pipeline 作业被标记 OUT_OF_MEMORY（默认 62.5G 不足），但实验实际完成。复现时建议在 run_all.sbatch 加 `--mem=128G`。
