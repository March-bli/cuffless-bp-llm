# 毕业论文大纲（v2 — 对齐导师反馈）

## 题目
**Novel Blood Pressure Monitoring Using Wearable Devices and Artificial Intelligence (AI)**

> 中文：基于可穿戴设备与人工智能的新型血压监测

---

## 第一章 绪论（~3000字）

### 1.1 研究背景与意义
- 心血管疾病是全球首要死因，高血压是核心风险因素
- 传统袖带式测量的局限：不便携、不连续
- 可穿戴 PPG 技术为无袖带连续血压监测提供可能
- LLM 的兴起为生理信号建模带来新范式

### 1.2 国内外研究现状
- **无袖带血压估计**：PTT 方法、机器学习、深度学习
- **LLM + 生理信号**：SensorLM、TimeSRL、Health-LLM
- **现有工作的核心不足**：
  - 传统 ML 在**跨受试者泛化**上失败（subject-level split 精度崩溃）
  - LLM 方法未充分利用个性化校准能力

### 1.3 本文主要工作
- （1）构建 857 人规模的 PulseDB 特征数据集（55 维 + 人口统计）
- （2）设计**跨受试者 zero-shot** LLM 血压估计基线
- （3）提出 **LLM 少样本个性化校准**方法，达到临床级精度
- （4）校准样本数消融 + 与 ML/LLM 基线系统对比

### 1.4 论文结构

---

## 第二章 相关技术与理论基础（~2500字）

### 2.1 光电容积描记术（PPG）
- PPG 原理、波形形态、一阶/二阶导数（VPG/APG）

### 2.2 无袖带血压估计
- 机制驱动（PTT）vs 数据驱动（ML/DL）
- 评估标准：BHS、AAMI

### 2.3 大语言模型与传感器数据
- LLM 基础：Transformer、预训练
- **SensorLM**：传感器-语言对齐（层次化 caption）
- **TimeSRL**：语义瓶颈 + RL 优化（跨数据集泛化）
- LLM few-shot / in-context learning

### 2.4 本章小结

---

## 第三章 PulseDB 数据集与信号处理（~3000字）

### 3.1 数据集介绍
- PulseDB：MIMIC-III 子集 + VitalDB 子集
- **本文规模：857 个受试者、148,012 段**（此前工作仅 122 人）
- 数据结构：PPG/ECG/ABP（125 Hz，10 秒段）+ SegSBP/SegDBP + Age/Gender

### 3.2 信号预处理
- 质量门控、基线漂移校正、工频噪声抑制、运动伪影检测

### 3.3 PPG 峰值与基准点检测
- 收缩峰、onset、重搏切迹、APG 基准点

### 3.4 本章小结

---

## 第四章 多维 PPG 特征工程（~3000字）

### 4.1 特征分类体系（55 维）
- 时域形态学、导数形态学（VPG/APG）、统计、频域

### 4.2 人口统计特征
- Age、Gender（导师建议：zero-shot 需用 demographic）

### 4.3 特征重要性分析
- SHAP / Gini 重要性

### 4.4 本章小结

---

## 第五章 基线方法：传统 ML 与跨受试者泛化问题（~3000字）

### 5.1 问题形式化
- subject-level split（严格跨受试者）vs segment-level split
- **核心发现：segment-level split 会"作弊"（同人数据泄漏）**

### 5.2 传统 ML 基线
- XGBoost、RandomForest（subject-level split，857 人）
- 结果：SBP 17.0 / DBP 11.6（XGBoost），跨受试者精度有限

### 5.3 泛化失败分析
- 个体间生理差异 → 数值特征分布偏移

### 5.4 本章小结

---

## 第六章 LLM 血压估计：从跨受试者 Zero-shot 到少样本校准（~4000字，核心章节）

### 6.1 设计动机
- LLM 可通过 in-context learning 适应个体差异

### 6.2 跨受试者 Zero-shot（导师要求的正确基线）
- **输入**：N 个其他受试者的（特征 + BP + 年龄性别）+ 目标受试者特征
- 与"无参考 zero-shot"的区别

### 6.3 少样本个性化校准（Few-shot Calibration，本文核心方法）
- 输入：目标受试者 K 个校准段（特征+BP）+ 新段特征
- K = 1/3/5/10/20 消融

### 6.4 实验结果与分析
- 857 人规模、20 测试受试者、Qwen3-8B 本地推理
- 核心结果（few-shot 消融）：

| K | SBP MAE | DBP MAE |
|:---:|:---:|:---:|
| 0 (zero-shot) | 20.8 | 12.6 |
| 1 | 21.8 | 18.8 |
| 3 | 14.4 | 11.6 |
| 5 | 9.3 | 6.1 |
| **10** | **8.3** | **4.1** |
| 20 | 8.4 | 4.5 |

- 与基线对比：XGBoost (20.6/11.7)、CNN1D (16.6/9.1)、LLM-BP (9.3/6.4)
- **发现 1**：K=10 时 DBP 4.1 达 BHS Grade A，SBP 8.3 达 Grade B
- **发现 2**：8B 本地模型 K=10 追平 671B 大模型 API——校准样本数比模型规模更重要

### 6.5 讨论
- 为什么 few-shot 有效：个性化 BP-特征映射
- K=1 反而劣于 zero-shot：单样本不足以校准个体差异
- 跨受试者泛化与个性化校准的权衡

### 6.6 本章小结

---

## 第七章 语义抽象增强（借鉴 TimeSRL / SensorLM）（~2500字）

### 7.1 动机：导师建议借鉴两篇论文
- SensorLM 层次化 caption：统计/结构/语义
- TimeSRL 语义瓶颈：先抽象后预测

### 7.2 层次化 PPG 描述
- 统计层：心率、脉宽、脉间期
- 结构层：APG 波形形态（b/a、c/a 比值）、偏度/峰度
- 语义层：血管僵硬度、血管阻力、心肌收缩力（规则映射）

### 7.3 实验结果
- 规则语义描述 few-shot K=5：SBP 10.8 / DBP **5.6**
- 对比纯数值 K=5：SBP 9.3 / DBP 6.1
- LLM 语义瓶颈两阶段（TimeSRL 式）K=5：SBP 30.0 / DBP 26.9
- **发现 1**：规则语义描述在 DBP 上更优——语义层的"血管阻力"信息对舒张压关键（验证 SensorLM）
- **发现 2**：纯定性 LLM 摘要导致严重退化——语义抽象必须保留定量信息

### 7.4 本章小结

---

## 第八章 总结与展望（~1500字）

### 8.1 工作总结
### 8.2 主要贡献
- （1）857 人规模 PulseDB 基准
- （2）严格的跨受试者 zero-shot 基线
- （3）LLM 少样本校准方法（临床级 DBP）
- （4）语义抽象增强探索

### 8.3 局限性与未来工作
- 完整 PulseDB 有 5,361 人，本文用 857 人
- 端侧部署、多模态融合

---

## 参考文献（核心 10 篇）

[1] Wang et al. PulseDB. Frontiers in Digital Health, 2023.
[2] Zhang et al. SensorLM. arXiv, 2025.
[3] Fan et al. TimeSRL. arXiv, 2026.
[4] Liu et al. LLM-BP. BCB, 2024.
[5] Kim et al. Health-LLM. arXiv, 2024.
[6] Reiss et al. Deep PPG. Sensors, 2019.
[7] Goda et al. pyPPG. Physiol. Meas., 2024.
[8] Mazzà et al. Mobilise-D. BMJ Open, 2021.
[9] Kirk et al. Walking Speed. Scientific Reports, 2024.
[10] Liu et al. Few-Shot Health Learners. arXiv, 2023.
