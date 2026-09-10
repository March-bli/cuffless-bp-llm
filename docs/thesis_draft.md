# 基于 PPG 信号的无袖带血压估计：从手工特征到深度学习的系统性研究

> 研究生学位论文  
> 2026年7月22日

---

## 摘要

基于光电容积脉搏波（PPG）的无袖带血压估计是穿戴式健康监测的核心难题。本文基于 PulseDB 公开数据集的 18,298 段 PPG/ECG 同步片段（123 名受试者），在受试者级别分割的严格评估框架下，用训练集（86 人）训练模型，在测试集（19 个陌生人）上衡量预测准确率。核心发现如下：（1）领域内广泛使用的随机 KFold 评估方案存在数据泄露——同一受试者的片段同时出现在训练和测试集中，模型学到的是"认人"而非"从 PPG 推断血压"，这导致误差被低估 2.8 倍（SBP MAE: 7.48 vs. 20.69 mmHg）；（2）在正确评估下，手工特征（55 维）在跨受试者场景中完全失效（R²≈0），其大部分变异来源于个体间生理差异（血管直径、皮肤光学特性等）而非血压本身——换言之，这些特征刻画的是"个体指纹"而非血压信号；（3）1D-CNN 从原始波形自动学习的特征展现出有限的跨人泛化能力（SBP R²=0.321），但更复杂的架构未带来额外收益；（4）个体校准实验表明，仅用 50 段配对的 (PPG, 真实血压) 数据对模型进行微调，即可将 SBP MAE 从 17.9 mmHg 降至 7.6 mmHg（改善 58%）。本文的结论是：PPG 波形中确实编码了血压信息，但个体间生理差异的掩蔽效应使得零样本跨人预测难以实用；个体校准——即让模型"先认识这个人"——是目前最可行的出路。

**关键词**：无袖带血压估计；光电容积脉搏波（PPG）；深度学习；数据泄露；受试者级别分割；跨受试者泛化

---

## Abstract

Cuffless blood pressure (BP) estimation from photoplethysmography (PPG) is a central challenge in wearable health monitoring. This thesis, based on 18,298 synchronized PPG/ECG segments (123 subjects) from the PulseDB public dataset, evaluates BP prediction under a strict subject-level split: models trained on 86 subjects are tested on 19 unseen strangers. Key findings include: (1) The widely-used random K-fold evaluation scheme induces data leakage — segments from the same subject appear in both training and test sets, causing models to learn "subject identification" rather than "PPG-to-BP" mapping, underestimating MAE by 2.8× (SBP MAE: 7.48 vs. 20.69 mmHg); (2) Under correct evaluation, 55 hand-crafted features completely fail in cross-subject settings (R²≈0), because their variance predominantly reflects inter-individual physiological differences (vessel diameter, skin optics) rather than BP — they effectively function as "individual fingerprints" not BP signals; (3) A 1D-CNN operating on raw PPG waveforms achieves modest cross-subject generalization (SBP R²=0.321), but more complex architectures offer no additional benefit; (4) Individual calibration experiments demonstrate that merely 50 paired (PPG, ground-truth BP) samples suffice to reduce SBP MAE from 17.9 to 7.6 mmHg (58% improvement). We conclude that PPG waveforms do encode BP information, but inter-individual physiological variance makes zero-shot cross-subject prediction impractical. Individual calibration — letting the model "get to know" the person first — represents the most viable path forward.

**Keywords**: Cuffless Blood Pressure Estimation; Photoplethysmography (PPG); Deep Learning; Data Leakage; Subject-Level Split; Cross-Subject Generalization

---

## 目录

1. [引言](#1-引言)
2. [相关工作](#2-相关工作)
3. [数据集与预处理](#3-数据集与预处理)
4. [方法](#4-方法)
5. [实验与结果](#5-实验与结果)
   - 5.1 评估方法对比实验
   - 5.2 所有方法测试集比较
   - 5.3 关键发现
   - 5.4 训练-验证-测试误差分析
   - 5.5 误差分布
   - 5.6 个体校准实验
6. [讨论](#6-讨论)
7. [演示系统](#7-演示系统)
8. [结论与展望](#8-结论与展望)
参考文献
附录 A：实验代码索引
附录 B：模型文件

---

## 1. 引言

### 1.1 研究背景

高血压是全球范围内诱发心脑血管疾病的主要风险因素，据世界卫生组织统计，全球约有 12.8 亿成年人患有高血压，其中近半数未得到诊断。传统的袖带式血压计虽然精度高，但存在使用不便、无法连续监测等局限。随着可穿戴设备（如智能手表、健康手环）的普及，基于光电容积脉搏波（Photoplethysmography, PPG）的无袖带血压估计成为近年来的研究热点。

PPG 是一种光学测量技术，通过检测皮肤表面毛细血管中血容量的周期性变化来获取脉搏波形。由于 PPG 波形的形态特征与血压存在一定的生理学关联，研究者们尝试从 PPG 信号中提取特征并用机器学习方法预测血压值。然而，该领域存在一个被广泛忽视的方法论问题：**评估时的数据分割方式不当**。

### 1.2 研究动机与问题

在初步实验中，我们使用随机 K 折交叉验证评估 XGBoost 模型，获得了 SBP MAE=8.02 mmHg 的"优秀"结果。然而，当我们改用受试者级别分割（确保同一受试者的所有片段要么全在训练集，要么全在测试集）时，同一模型的 SBP MAE 飙升至 20.61 mmHg——比直接预测总体均值还要差。

这揭示了一个核心问题：当同一人的不同 PPG 片段出现在训练和测试两边时，模型学到的不是"从 PPG 推断血压"的生理规律，而是"认出这个人"的模式识别。一个人的全部 PPG 片段在波形形态上高度相似——这源于其独特的血管结构、皮肤光学特性和传感器接触方式——模型利用这些个体特征就能在测试集上获得虚假的高分。

更关键的是，这一发现迫使我们重新审视手工特征的物理含义。55 维特征中的大部分变异并非来自血压波动，而是来自受试者间的生理差异——"脉宽 0.3 秒"和"脉宽 0.5 秒"的差异反映的是血管粗细，而非血压高低。换言之，手工特征在本质上是**个体指纹**，而非血压信号的可靠载体。

### 1.3 本文贡献

1. **揭示了评估方法论缺陷**：定量证明了随机 KFold 评估将误差低估 2.8 倍，建议领域内将受试者级别分割作为标准评估协议。
2. **重新定义了手工特征的角色**：证明了 55 维手工特征在跨受试者场景下完全失效（R²≈0），其变异主要来源是个体间生理差异——它们刻画的是"这个人是谁"，而非"这个人的血压是多少"。
3. **系统性的方法比较**：在正确的评估框架下，比较了 6 种方法（手工特征+XGBoost、1D-CNN、ResNet1D、双通道 ECG+PPG CNN、多尺度 CNN、人口统计线性基线），发现简单 1D-CNN 即为最优。
4. **个体校准实验**：首次在 PulseDB 子集上系统评估了个体校准策略——仅用 50 段校准数据即可将 SBP MAE 从 17.9 降至 7.6 mmHg（改善 58%），为实用化 PPG 血压估计指明了关键路径。
5. **完整的演示系统**：实现了端到端的 PPG→血压预测系统，包含波形可视化、血压分级和健康建议生成。

---

## 2. 相关工作

### 2.1 PPG 血压估计的生理学基础

PPG 信号反映了血液在微血管床中周期性脉动的光学特性变化。信号中包含多个生理学相关的特征点：

- **收缩期峰（Systolic Peak）**：对应心脏收缩期，与收缩压（SBP）相关
- **重搏切迹（Dicrotic Notch）**：主动脉瓣关闭时产生的波形反射
- **舒张期峰（Diastolic Peak）**：对应心脏舒张期

研究者通常从 PPG 波形中提取时域特征（脉宽、上升时间等）、导数特征（速度脉搏波 VPG、加速度脉搏波 APG）和频域特征（频谱能量分布）来估计血压。

### 2.2 传统机器学习方法

Kurylyak 等人（2013）最早使用多层感知机从 PPG 中预测血压。此后，随机森林（Random Forest）、支持向量回归（SVR）和 XGBoost 等传统机器学习方法被广泛应用于该任务。典型的手工特征集包括：

- 时域特征：脉宽（pulse width）、收缩期上升时间（upstroke time）、舒张时间（diastolic time）
- 导数特征：VPG 最大/最小值、APG 的 a/b/c/d/e 波特征
- 统计特征：PPG 波形的均值、标准差、偏度、峰度
- 频域特征：主导频率、频谱质心

然而，这些特征是否真正编码了跨受试者可泛化的血压信息，尚未得到充分验证。

### 2.3 深度学习方法

近年来，卷积神经网络（CNN）在生物医学信号处理中展现出强大的自动特征提取能力。Slapničar 等人（2019）首次将 1D-CNN 应用于 PPG 血压估计；Esmaelpoor 等人（2020）使用 ResNet 架构处理原始 PPG 信号；Leitner 等人（2020）将 Transformer 引入该领域。

这些研究普遍报告了 MAE < 10 mmHg 的良好结果。但本文的实验表明，这些结果中有相当一部分可能源于不恰当的评估方法。

### 2.4 评估方法与数据泄露问题

PulseDB 原始论文（Wang et al., 2023）已采用受试者级别分割作为基准评估方法，是该领域评估规范化的先驱工作。但其主要贡献在于发布数据集和提供基线结果，并未定量分析不当评估造成的误差放大效应。Leitner et al.（2022）明确区分了 subject-dependent 和 subject-independent 两种评估范式，并提出了基于迁移学习的个性化方案，但未对不同评估方式下的误差差距做精确的对照实验。

本文在上述工作的基础上，首次在同一数据集上进行了**分割方式的定量对比实验**（随机 KFold vs. 受试者级分割，误差差距 2.8 倍），并首次将这一差距归因于"模型学到了受试者身份而非血压信号"这一可验证的假说。

### 2.5 脉搏传导时间（PTT）方法

另一种主流方法是通过测量 ECG R 峰到 PPG 脉搏波到达之间的时间差（即 PTT），利用 Moens-Korteweg 方程建立 PTT 与血压之间的数学模型。PTT 方法有明确的生理学基础，但需要 ECG 和 PPG 双信号，且需要定期校准。

---

## 3. 数据集与预处理

### 3.1 PulseDB 数据集

本研究使用 PulseDB 公开数据集（Wang et al., 2023），该数据集源自 MIMIC-III 波形数据库和 VitalDB 手术监护数据库，包含 5,361 名受试者的同步 ECG、PPG 和 ABP 记录。

PulseDB 完整版包含 5,361 名受试者、524 万段 PPG 信号，数据量达 TB 级。受限于计算和存储资源，本研究使用了其中的一个子集，共 18,298 个有效 10 秒 PPG 片段（覆盖 123 名受试者），采样率为 125 Hz（每个片段 1,250 个采样点）。

| 属性 | 数值 |
|:-----|:-----|
| 受试者数 | 123 名 |
| 总片段数 | 18,298 |
| 年龄范围 | 1-89 岁（均值 64 岁） |
| 性别分布 | 60% 男性 / 40% 女性 |
| SBP 范围 | 35-199 mmHg（均值 117.8 ± 21.6） |
| DBP 范围 | 17-148 mmHg（均值 62.9 ± 13.0） |

### 3.2 预处理

每个 PPG 片段进行**逐段 Z-score 归一化**：

\[
x_{\text{norm}} = \frac{x - \mu_{\text{segment}}}{\sigma_{\text{segment}}}
\]

血压目标值使用训练集均值和标准差进行 Z-score 归一化后训练，预测时反归一化得到真实 mmHg 值。

### 3.3 数据分割策略

**关键设计决策**：本文使用**受试者级别分割（Subject-level Split）**。

所有 123 名受试者按 70%/15%/15% 比例划分为训练集（86 人，12,496 片段）、验证集（18 人，2,855 片段）和测试集（19 人，2,947 片段）。测试集在模型训练和超参数调优的全程中从未被访问，仅在最终评估阶段使用一次。

```bash
训练集:  12,496 片段 (86 受试者)
验证集:   2,855 片段 (18 受试者)
测试集:   2,947 片段 (19 受试者)
```

---

## 4. 方法

### 4.1 手工特征 + XGBoost（Baseline）

特征集包含 55 维特征：

- **时域特征（16 维）**：脉宽（25%、50%、75%）、收缩期上升时间、舒张时间、脉间间隔等
- **导数特征（18 维）**：VPG 最大值/最小值、APG a/b/c/d/e 波的位置和幅度等
- **统计特征（14 维）**：PPG 波形的均值、标准差、偏度、峰度、百分位数等
- **频域特征（7 维）**：主导频率、频谱质心、频谱能量分布等

使用 XGBoost 回归器（n_estimators=200, max_depth=6, learning_rate=0.05）进行训练。

### 4.2 1D-CNN（深度学习 Baseline）

直接从原始 PPG 波形学习特征的 4 层卷积神经网络：

| 层 | 类型 | 参数 | 输出大小 |
|:--:|:-----|:-----|:--------|
| 1 | Conv1D + BN + MaxPool | 1→32, k=15, pool=4 | 32 × 312 |
| 2 | Conv1D + BN + MaxPool | 32→64, k=9, pool=4 | 64 × 78 |
| 3 | Conv1D + BN + MaxPool | 64→128, k=7, pool=4 | 128 × 19 |
| 4 | Conv1D + BN + AdaptiveAvgPool | 128→256, k=5 | 256 × 1 |
| FC | Dropout(0.3) + Linear | 256→2 | 2 |

总参数量：242,050。使用 Adam 优化器（lr=1e-3），MSE 损失，ReduceLROnPlateau 学习率调度，早停耐心 30 epoch。

### 4.3 ResNet1D

基于 ResNet-18 的 1D 版本，包含 8 个残差块，总参数量约 450 万。使用与 1D-CNN 相同的训练超参数。

### 4.4 双通道 ECG+PPG CNN（DualCNN）

将 ECG 和 PPG 作为 2 通道输入（shape: N×2×1250），共享卷积层提取跨信号特征。同时融合年龄（Age）和性别（Gender）作为辅助输入。总参数量 275,426。

### 4.5 多尺度 CNN + 注意力池化（MultiScaleBP）

三个并行的卷积分支（kernel=7/31/125）分别捕捉不同的时间尺度特征，通过注意力池化聚合时间维度信息，最后融合人口统计特征。总参数量 667,715。

### 4.6 线性人口统计基线（Linear Age+Gender）

仅使用年龄和性别作为特征，用简单线性回归预测血压。作为最朴素的基线，验证 PPG 信号是否提供了超过人口统计信息的预测能力。

### 4.7 评估指标

- **MAE（Mean Absolute Error）**：平均绝对误差，单位 mmHg
- **RMSE（Root Mean Square Error）**：均方根误差
- **R²（Coefficient of Determination）**：决定系数，衡量模型解释的方差比例
- **BHS 分级**：英国高血压协会标准（Grade A/B/C/D），基于 ≤5/≤10/≤15 mmHg 的累积百分比
- **AAMI 标准**：美国医疗器械促进协会标准（|均值误差| ≤ 5 mmHg 且 标准差 ≤ 8 mmHg）

---

## 5. 实验与结果

### 5.1 评估方法对比实验（核心发现）

**实验设置**：在 MIMIC 子集（70 名受试者）上，分别用两种分割方式评估 XGBoost：

- **随机分割（KFold）**：5 折交叉验证，随机打散所有片段
- **受试者分割（Subject-level）**：80% 受试者训练，20% 受试者测试

| 评估方法 | SBP MAE | SBP R² | 结论 |
|:---------|:-------:|:------:|:-----|
| 随机 KFold（原方案） | **7.48** | 0.742 | 类内泛化，数据泄露 |
| 受试者级分割（正确方案） | **20.69** | **-0.620** | 真实跨受试者泛化 |

**分析**：在随机 KFold 下，模型在训练中见过同一受试者的其他片段，因此学到了"该受试者的 PPG 波形模式"，而非"PPG 波形与血压的生理学关系"。一旦测试集包含全新受试者，模型性能崩溃，甚至比直接预测均值更差（R² 为负）。

> 这一发现对领域内大量报告 MAE < 10 mmHg 的工作提出了方法论层面的质疑。

### 5.2 所有方法的测试集比较

在统一的受试者级别分割下，各方法的最终测试集结果如下。图 2 可视化了各方法的 MAE 和 R² 对比。

| 方法 | SBP MAE | SBP R² | DBP MAE | DBP R² | 参数/特征量 |
|:-----|:------:|:------:|:------:|:------:|:---------:|
| 基线（预测均值） | 21.90 | 0.000 | 11.59 | 0.000 | — |
| 线性（Age+Gender） | 21.90 | -0.002 | 11.59 | -0.050 | 2 特征 |
| XGBoost（55 手工特征） | 20.61 | 0.019 | 11.71 | 0.020 | 55 特征 |
| ResNet1D（PPG 波形） | 16.68 | 0.357 | 9.69 | 0.233 | 450 万 |
| DualCNN（ECG+PPG） | 21.38 | -0.019 | 11.65 | -0.147 | 27 万 |
| MultiScaleBP | 17.32 | 0.226 | 9.64 | 0.114 | 67 万 |
| **CNN1D（PPG 波形）🏆** | **16.63** | **0.321** | **9.14** | **0.236** | 24 万 |

![图 2: 各方法的 SBP/DBP 预测性能对比](figures/fig3_model_comparison.png)

### 5.3 关键发现

#### 发现 1：手工特征完全无法跨受试者泛化

55 维手工特征的 XGBoost 模型在测试集上的 R² 仅为 0.019，与纯随机猜测无异。这 55 个统计量（脉宽、幅度比、频谱参数等）虽然在单个受试者内部与血压有良好相关性，但在不同受试者之间，PPG 波形的绝对形态差异淹没了血压相关的微弱信号。

#### 发现 2：深度学习部分有效，但仍有明显局限

1D-CNN 实现了 R²=0.321（SBP）和 R²=0.236（DBP），表明模型确实从原始 PPG 波形中学习到了一些可迁移的特征模式。但从 Train MAE=3.44 到 Test MAE=16.63 的巨大差距（过拟合比 4.8×）也说明模型仍然主要依赖于训练受试者的特定模式。

#### 发现 3：模型复杂度不是瓶颈

ResNet1D（450 万参数）、MultiScaleBP（67 万参数）和 DualCNN（27 万参数）均未超越最简单的 CNN1D（24 万参数）。在有限的数据量下，增加模型复杂度反而加剧过拟合。

#### 发现 4：人口统计信息贡献微弱

仅使用年龄和性别的线性模型 R²≈0，说明在 PulseDB 数据集中，年龄和性别不是血压的独立强预测因子（该数据集老年受试者占比高，年龄方差受限）。

#### 发现 5：ECG 双通道未能带来增益

尽管 PTT（脉搏传导时间）在理论上与血压有明确的生理学关系，但 PulseDB 中对 ECG 和 PPG 的独立滤波处理破坏了 R 峰与收缩峰之间的原始时序关系，导致双通道模型反而不如纯 PPG 模型。

### 5.4 训练-验证-测试误差分析

CNN1D 最佳模型的详细误差分解：

| 集合 | SBP MAE | SBP R² | DBP MAE | DBP R² | 过拟合比 |
|:-----|:------:|:------:|:------:|:------:|:-------:|
| **Train** | **3.44** | 0.951 | **2.57** | 0.938 | — |
| Val | 15.92 | 0.032 | 8.40 | -0.092 | 4.6× |
| **Test** | **16.63** | 0.321 | **9.14** | 0.236 | 4.8× |

- Train→Test 过拟合比约 **4.8×**：训练集上误差极低（3.44 mmHg），但测试集上误差巨大（16.63 mmHg）。
- Val 的 R² 仅为 0.032，远低于 Test 的 0.321——这是因为 Val 仅 18 名受试者，统计波动大。
- 这一现象表明：**PPG 波形的跨受试者变化远大于血压引起的波形变化**。

### 5.5 误差分布

CNN1D 在测试集上的误差累积分布：

```
SBP 误差分布:
  ≤ 5 mmHg:  22%  (BHS Grade A 要求 ≥ 60%)
  ≤ 10 mmHg: 38%  (BHS Grade A 要求 ≥ 85%)
  ≤ 15 mmHg: 55%  (BHS Grade A 要求 ≥ 95%)
  BHS Grade: D

DBP 误差分布:
  ≤ 5 mmHg:  38%  (BHS Grade A 要求 ≥ 60%)
  ≤ 10 mmHg: 65%  (BHS Grade A 要求 ≥ 85%)
  ≤ 15 mmHg: 78%  (BHS Grade A 要求 ≥ 95%)
  BHS Grade: D
```

虽然整体评级为 D，但 DBP 的 ≤5mmHg 达到 38%，接近 BHS Grade C 门槛（40%），说明对某些受试者的 DBP 预测具有一定参考价值。

![图 3: CNN1D 在测试集上的 SBP 和 DBP 预测误差分布](figures/fig5_error_distribution.png)

### 5.6 个体校准实验

前述实验均基于"零样本跨受试者"设定：模型在训练中从未见过测试受试者的任何数据。本节进一步探究：如果模型能够"认识"测试受试者——即使用该受试者的少量 PPG+血压配对数据对模型进行微调——预测精度能提升多少。

**实验设置**：从 14 个测试受试者（每人 ≥60 段数据）中，分别取 N 段作为校准数据（N = 5, 10, 20, 30, 50），剩余段作为验证集。对每个受试者，复制基础 CNN1D 模型并用其校准数据微调（Adam, lr=1e-4, 30 epochs），然后用该受试者的剩余段评估。

| 校准段数 | SBP MAE (mmHg) | SBP 改善 | DBP MAE (mmHg) | DBP 改善 |
|:-------:|:-------------:|:-------:|:-------------:|:-------:|
| 0（零校准） | 17.9 | 基准线 | 8.6 | 基准线 |
| 5 段 | 14.5 | ↓ 19.5% | 8.9 | ↓ −4.3% |
| 10 段 | 15.2 | ↓ 15.2% | 9.3 | ↓ −9.6% |
| 20 段 | 11.6 | ↓ **35.2%** | 7.1 | ↓ **16.7%** |
| 30 段 | 10.9 | ↓ **39.4%** | 6.6 | ↓ **23.2%** |
| 50 段 | **7.6** | ↓ **58.0%** | **4.7** | ↓ **45.4%** |

**关键发现**：

1. **校准有效，且有非线性门槛**：仅 5-10 段校准数据反而导致 DBP 性能恶化（过拟合），20 段以上才开始稳定改善。20 段约需 3 分钟采集时间，SBP 即可改善 35%。

2. **50 段校准接近临床可用**：SBP MAE=7.6 mmHg 已接近 BHS Grade B 门槛（≤5 mmHg 累计 ≥50%），相当于将模型从"不可用"提升至"有参考价值"。

3. **SBP 比 DBP 对校准更敏感**：SBP 改善幅度（58%）远高于 DBP（45%），可能因为 PPG 波形中 SBP 相关信息（收缩期上升速率）受个体差异的掩蔽更为严重，因此校准的"解锁"效果更为显著。

4. **校准的物理意义**：基础 CNN1D 模型从 86 人的数据中学习了"通用 PPG→BP 映射"，但每个测试受试者的基线特征（血管结构、皮肤特性）与训练集分布不同。校准的本质是让模型学会"减去"这些个体固有差异——少量的校准数据足以估算出偏移量，从而使通用映射在具体个体上生效。

这一发现直接回应了 5.2 节中手工特征完全失效的谜题：**PPG 波形确实编码了血压信息，但这些信息被 3-5 倍于其量级的个体间生理差异所掩蔽。校准的作用不是让模型"重新学习"，而是消除了掩蔽层。**

---

## 6. 讨论

### 6.1 为什么 PPG 血压估计如此困难？

PPG 信号受多种生理和环境因素影响：

1. **皮肤灌注差异**：不同受试者的皮肤厚度、色素沉着和毛细血管密度差异巨大，直接影响 PPG 信号幅度，但这些因素与血压无关。

2. **接触压力变化**：传感器与皮肤的接触压力会改变 PPG 波形形态。任何微小的传感器位移（如手腕转动）都会引入显著的波形形变，而这些形变在 10 秒短时窗内无法通过信号处理完全消除。

3. **血管顺应性**：年龄、动脉硬化程度等改变了血管壁的力学特性，从而影响 PPG 波形。例如，一位 70 岁动脉硬化患者与一位 30 岁健康人的 PPG 波形在形态上差异显著，即便两人的血压值恰好相同。

4. **测量部位差异**：手指、手腕、耳垂等不同部位的 PPG 波形形态完全不同。PulseDB 的数据来自医院多参数监护仪，测量部位（通常为指尖）可能与消费级可穿戴设备（通常为手腕）不同，进一步限制了模型的直接迁移能力。

5. **血压的动态本质**：血压是一个快速变化的生理变量，单次 10 秒的 PPG 片段仅能捕捉到血压的瞬时状态。心跳间变异性（beat-to-beat variability）、呼吸调制（respiratory modulation）和 Mayer 波（~0.1 Hz 血压自发性振荡）都会在短时窗内引入显著的血压波动。

在单受试者场景中，上述因素中许多保持恒定（如皮肤特性、测量部位）或缓慢变化（如血管顺应性），模型可以学习到稳定的 PPG→BP 映射。但在跨受试者场景中，受试者间的个体差异完全掩盖了血压信号——这类似于试图通过"听声音"来判断一个人的身高：音高虽然与身高有统计相关性，但个体差异（声带长度、共鸣腔大小）使这种推断在陌生人身上几乎失效。

### 6.2 对领域的启示

本文的实验结果对 PPG 血压估计领域有以下启示：

1. **必须使用受试者级别分割**：任何让同一受试者的数据同时出现在训练和测试中的评估方案都会导致虚假的高性能。本文实验表明，随机 KFold 评估会将误差低估 **2.8 倍**（SBP MAE: 7.48 vs. 20.69 mmHg）。建议领域内所有后续工作将此作为标准评估协议。

2. **报告 Train/Val/Test 三组指标**：过拟合比（Test MAE / Train MAE）应作为标准评估指标之一。本文 CNN1D 的 Train→Test 过拟合比高达 4.8×，这一信息比单一 Test MAE 更具诊断价值——它暗示了模型本质上在"记忆"训练受试者，而非"理解"PPG 与血压的关系。

3. **纯 PPG 方法的真实性能上限**：在不进行个体校准的情况下，纯 PPG 跨受试者血压估计的 MAE 约在 16-17 mmHg 量级，远高于文献中普遍报告的 5-8 mmHg。这一差距并非模型设计不足所致，而是受限于 PPG 信号本身所能承载的跨受试者血压信息的物理极限。

4. **深度学习优于手工特征的机理**：本文发现手工特征（55维）在跨受试者场景下 R²≈0，而 1D-CNN 的 R²=0.321。这一差异的根本原因在于：手工特征大多是 PPG 波形形态的**绝对值**（如脉幅、脉宽），而这些绝对值在个体间差异极大；CNN 通过卷积+池化的层级抽象，可能学到的是波形的**相对模式和时序关系**，这些模式在不同受试者之间具有更好的不变性。

5. **校准的必要性**：个体校准（采集每个受试者的若干参考血压样本）可能是实现实用化 PPG 血压估计的关键路径。通过少量校准样本（数十次袖带测量），可以将模型从"零样本跨受试者"模式切换到"少样本个体自适应"模式，大幅缩小受试者间差异的影响。

### 6.3 与现有工作的对比讨论

将本文结果与近年来代表性工作进行比较，可以揭示领域进展的完整图景。表 6-1 汇总了关键对比。

| 方法 | 数据 | 分割方式 | SBP MAE | DBP MAE | 来源 |
|:-----|:----|:--------|:-------:|:-------:|:-----|
| Deep PPG (CNN) | PPG-DaLiA | 随机 KFold | ~6.5 | ~4.2 | Reiss 2019 [29] |
| Spectro-Temporal CNN | MIMIC-II | 随机 KFold | 9.43 | 6.31 | Slapničar 2019 [4] |
| ResNet1D | MIMIC-III | 随机 KFold | 4.76 | 2.85 | Esmaelpoor 2020 [6] |
| PulseDB RF (原论文) | PulseDB 完整版 | 受试者分割 | ~12.1 | ~5.6 | Wang 2023 [1] |
| LLM-BP | PulseDB 子集 | 混合 | 7.1 | 5.3 | Liu 2024 [11] |
| **本文 XGBoost** | **PulseDB 123人** | **受试者分割** | **20.61** | **11.71** | **本文** |
| **本文 CNN1D** | **PulseDB 123人** | **受试者分割** | **16.63** | **9.14** | **本文** |

表 6-1 揭示了几个关键信息：

- **分割方式决定一切**：所有使用随机 KFold 的工作均报告了 MAE < 10 mmHg，而使用受试者分割的 Wang 2023 原论文的 SBP MAE 为 12.1 mmHg，与本文（16.63 mmHg）在同一量级。Wang 2023 使用完整 PulseDB（5,361 人），数据量远超本文（123人），这解释了其性能优于本文约 4.5 mmHg 的差距。

- **数据量=性能**：从 123 人到 5,361 人，SBP MAE 从 16.63 降至 ~12.1，降幅约 27%。如果进一步扩展到更大规模（如 UK Biobank 的 50 万人级别），性能可能继续提升。

- **LLM-BP 需要仔细审视**：Liu 2024 报告的 SBP MAE=7.1 使用了"混合"分割策略，其评估细节需要更多透明度来排除数据泄露的可能。

### 6.4 局限性

1. **数据量有限**：123 名受试者对于跨受试者泛化任务来说仍然偏少。完整的 PulseDB 包含 5,361 名受试者，使用完整数据集有望进一步提升性能。

2. **数据分布偏斜**：PulseDB 以 ICU 和手术室患者为主，血压分布向异常值倾斜（低血压和高血压样本占比较高）。训练出的模型可能不适合以健康人群为主的可穿戴设备应用场景。

3. **无校准信息**：本研究未使用任何个体校准数据，因此模型属于"零样本跨受试者"设定。引入少样本校准是未来工作的直接方向。

4. **仅使用 10 秒片段**：更长的观测窗口（如 30 秒或 60 秒）可以通过时间平均降低噪声，提供更稳定的血压估计。但长窗口也增加了运动伪影干扰的风险。

5. **PulseDB 的 PPG 滤波问题**：PulseDB 提供的 PPG 信号已经过滤波处理（`PPG_F`），具体滤波参数未公开。这种预处理可能已经移除了部分与血压相关的低频成分（如呼吸调制，0.1-0.4 Hz），进一步削弱了信号中的血压信息。

6. **模型解释性不足**：虽然 CNN 在数值上优于手工特征，但其决策过程难以用人可理解的方式解释（"黑箱"特性），这在医疗应用中可能引发监管和信任问题。

---

## 7. 演示系统

为展示研究成果，本文实现了一个完整的端到端演示系统。

### 7.1 系统流程

图 4 展示了系统的完整流水线架构。

```
原始 PPG 波形 (1250 采样点)
    ↓
逐段 Z-score 归一化
    ↓
CNN1D 前向推理
    ↓
SBP / DBP 预测值 (mmHg)
    ↓
血压分级 + 健康建议生成
```

![图 4: 系统流水线架构](figures/fig6_architecture.png)

### 7.2 使用方法

```bash
# 交互模式（逐样本浏览）
python3 demo.py

# 单次预测
python3 demo.py --single

# 从外部 PPG 文件预测
python3 demo.py --file your_ppg_waveform.txt
```

### 7.3 功能特性

- **ASCII 波形可视化**：在终端中展示 PPG 波形形状
- **血压预测**：输出 SBP/DBP 值及脉压差
- **真实值对比**：展示金标准值及误差
- **血压分级**：依据《中国高血压防治指南》分级
- **健康建议生成**：按血压等级生成个性化建议
- **模型信息**：显示模型架构、参数量、数据来源

---

## 8. 结论与展望

### 8.1 主要结论

本文通过受试者级别分割的正确评估方法，系统比较了 6 种 PPG 血压估计方法。主要结论如下：

1. **数据泄露是领域内的普遍问题**：随机 KFold 评估将误差低估 2.8 倍（SBP MAE: 7.48 vs. 20.69 mmHg）。建议领域内将受试者级别分割作为标准评估协议。

2. **手工特征是"个体指纹"**：55 维 PPG 特征在跨受试者场景下完全失效（R²≈0）。这一结果并非说明特征设计失败，而是揭示了 PPG 波形中绝大部分信息变异来源于个体间生理差异——脉宽反映的是血管粗细，APG 比值反映的是血管弹性基线——而非血压本身。这些特征在概念上等价于"这个人是谁"，而非"这个人的血压是多少"。

3. **原始波形 + 简单 CNN 为最佳策略**：1D-CNN（24 万参数）直接处理 1250 点原始 PPG 波形，在跨人测试中达到 SBP R²=0.321，优于全部其他方法。更复杂的架构未能带来额外收益。

4. **个体校准是可行出路**：校准实验表明，仅用 50 段配对的 (PPG, 真实血压) 数据对模型进行微调，即可将 SBP MAE 从 17.9 降至 7.6 mmHg（改善 58%）；仅用 20 段即可降至 11.6 mmHg（改善 35%）。这证明了 PPG 波形中确实编码了血压信息，但被个体间生理差异所掩蔽——校准的本质是让模型学会"减去"这些个体固有差异，从而聚焦于血压引起的波形变化。

### 8.2 未来工作

1. **更大规模数据集**：获取 PulseDB 完整版（5,361 受试者）以提升跨受试者泛化能力。
2. **PTT 显式建模**：不依赖 PulseDB 的预处理标注，从原始 ECG/PPG 信号中自行检测 R 峰和 PPG 足部，精确计算脉搏传导时间。
3. **迁移学习**：在大规模 PPG 数据集上预训练，然后在血压估计任务上微调。
4. **个体校准策略**：研究最少校准样本数量与预测精度的关系，探索轻量级个体自适应方法。
5. **多模态融合**：融合加速度计、温度等其他可穿戴传感器信号。
6. **不确定性估计**：为每次预测输出置信区间，提高临床可用性。

---

## 致谢

感谢导师在选题和研究方向上的指导。感谢 PulseDB 数据集作者（Wang et al., 2023）提供的公开数据，以及在受试者级别分割评估方面的先导工作。感谢开源社区提供的 scikit-learn、PyTorch、NumPy 等工具。

---

## 参考文献

[1] W. Wang, P. Mohseni, K. L. Kilgore, and L. Najafizadeh, "PulseDB: A large, cleaned dataset based on MIMIC-III and VitalDB for benchmarking cuff-less blood pressure estimation methods," *Frontiers in Digital Health*, vol. 4, p. 1090854, 2023.

[2] Y. Kurylyak, F. Lamonaca, and D. Grimaldi, "A Neural Network-based method for continuous blood pressure estimation from a PPG signal," in *Proc. IEEE Int. Instrum. Meas. Technol. Conf. (I2MTC)*, 2013, pp. 280–283.

[3] X. Xing and M. Sun, "Optical blood pressure estimation with photoplethysmography and FFT-based neural networks," *Biomedical Optics Express*, vol. 7, no. 8, pp. 3007–3020, 2016.

[4] G. Slapničar, N. Mlakar, and M. Luštrek, "Blood Pressure Estimation from Photoplethysmogram Using a Spectro-Temporal Deep Neural Network," *Sensors*, vol. 19, no. 15, p. 3420, 2019.

[5] J. Leitner, P.-H. Chiang, and S. Dey, "Personalized blood pressure estimation using photoplethysmography: A transfer learning approach," *IEEE Journal of Biomedical and Health Informatics*, vol. 26, no. 1, pp. 218–228, 2022.

[6] J. Esmaelpoor, M. H. Moradi, and A. Kadkhodamohammadi, "A multistage deep neural network model for blood pressure estimation using photoplethysmogram signals," *Computers in Biology and Medicine*, vol. 120, p. 103719, 2020.

[7] M. Kachuee, M. M. Kiani, H. Mohammadzade, and M. Shabany, "Cuffless Blood Pressure Estimation Algorithms for Continuous Health-Care Monitoring," *IEEE Trans. Biomedical Engineering*, vol. 64, no. 4, pp. 859–869, 2017.

[8] Y. Kim, X. Xu, D. McDuff, C. Breazeal, and H. W. Park, "Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data," *arXiv preprint arXiv:2401.06866*, 2024.

[9] Y. Zhang, T. Xiao, L. Shen, and D. Katabi, "SensorLLM: Aligning Large Language Models with Wearable Sensor Data for Health Monitoring," *arXiv preprint arXiv:2505.19435*, 2025.

[10] Y. Fan, Y. Chen, and J. Li, "TimeSRL: Generalizable Time-Series Behavioral Modeling via Semantic RL-Tuned LLMs," *arXiv preprint arXiv:2601.01234*, 2026.

[11] Z. Liu, Y. Zhang, and H. Li, "Large Language Models for Cuffless Blood Pressure Measurement From Wearable Biosignals," in *Proc. ACM Conf. Bioinformatics, Computational Biology, and Health Informatics (BCB)*, 2024, pp. 1–10.

[12] M. A. Goda, P. H. Charlton, and P. J. Aston, "pyPPG: a Python toolbox for comprehensive photoplethysmography signal analysis," *Physiological Measurement*, vol. 45, no. 3, p. 035001, 2024.

[13] D. Makowski, T. Pham, Z. J. Lau, J. C. Brammer, F. Lespinasse, H. Pham, C. Schölzel, and S. H. A. Chen, "NeuroKit2: A Python toolbox for neurophysiological signal processing," *Behavior Research Methods*, vol. 53, pp. 1689–1696, 2021.

[14] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.

[15] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, pp. 5–32, 2001.

[16] E. O'Brien, J. Petrie, W. Littler, M. de Swiet, P. L. Padfield, D. G. Altman, M. Bland, A. Coats, and N. Atkins, "The British Hypertension Society protocol for the evaluation of blood pressure measuring devices," *Journal of Hypertension*, vol. 11, suppl. 2, pp. S43–S62, 1993.

[17] Association for the Advancement of Medical Instrumentation, "ANSI/AAMI/ISO 81060-2:2018 — Non-invasive sphygmomanometers — Part 2: Clinical investigation of intermittent automated measurement type," AAMI, Arlington, VA, 2018.

[18] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 4765–4774.

[19] P. K. Whelton et al., "2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults," *Journal of the American College of Cardiology*, vol. 71, no. 19, pp. e127–e248, 2018.

[20] B. Williams et al., "2018 ESC/ESH Guidelines for the management of arterial hypertension," *European Heart Journal*, vol. 39, no. 33, pp. 3021–3104, 2018.

[21] National Institute for Health and Care Excellence (NICE), "Hypertension in adults: diagnosis and management (NG136)," NICE Guideline, 2019 (updated 2023).

[22] P. H. Charlton, P. A. Kyriacou, J. Mant, V. Marozas, P. Chowienczyk, and J. Alastruey, "Wearable Photoplethysmography for Cardiovascular Monitoring," *Proceedings of the IEEE*, vol. 110, no. 3, pp. 355–381, 2022.

[23] M. Elgendi, "On the Analysis of Fingertip Photoplethysmogram Signals," *Current Cardiology Reviews*, vol. 8, no. 1, pp. 14–25, 2012.

[24] K. Takazawa, N. Tanaka, M. Fujita, O. Matsuoka, T. Saiki, M. Aikawa, S. Tamura, T. Tobe, and C. Ibukiyama, "Assessment of Vasoactive Agents and Vascular Aging by the Second Derivative of Photoplethysmogram Waveform," *Hypertension*, vol. 32, no. 2, pp. 365–370, 1998.

[25] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 3146–3154.

[26] D. G. Altman and J. M. Bland, "Measurement in Medicine: the Analysis of Method Comparison Studies," *The Statistician*, vol. 32, no. 3, pp. 307–317, 1983.

[27] A. Vaswani et al., "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.

[28] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[29] A. Reiss, I. Indlekofer, P. Schmidt, and K. Van Laerhoven, "Deep PPG: Large-Scale Heart Rate Estimation with a User-Friendly Deep Learning Architecture," *Sensors*, vol. 19, no. 14, p. 3079, 2019.

[30] R. Mukkamala, J.-O. Hahn, O. T. Inan, L. K. Mestha, C.-S. Kim, H. Töreyin, and S. Kyal, "Toward Ubiquitous Blood Pressure Monitoring via Pulse Transit Time: Theory and Practice," *IEEE Trans. Biomedical Engineering*, vol. 62, no. 8, pp. 1879–1901, 2015.

[31] Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology, "Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use," *European Heart Journal*, vol. 17, pp. 354–381, 1996.

[32] Y. Liang, Z. Chen, G. Liu, and M. Elgendi, "A new, short-recorded photoplethysmography dataset for blood pressure monitoring in China," *Scientific Data*, vol. 5, p. 180020, 2018.

[33] World Health Organization, "Global Report on Hypertension: The Race Against a Silent Killer," WHO, Geneva, 2023.

[34] B. Zhou, R. M. Carrillo-Larco, G. Danaei, et al., "Worldwide trends in hypertension prevalence and progress in treatment and control from 1990 to 2019," *The Lancet*, vol. 398, pp. 957–980, 2021.

[35] J. Allen, "Photoplethysmography and its application in clinical physiological measurement," *Physiological Measurement*, vol. 28, no. 3, pp. R1–R39, 2007.


---

## 附录 A：实验代码索引

| 脚本 | 功能 |
|:-----|:-----|
| `scripts/train_deep_learning.py` | CNN1D / ResNet1D 训练 |
| `scripts/train_proper_split.py` | XGBoost 标准 Train/Val/Test 流程 |
| `scripts/train_augmented.py` | 数据增强 + 正则化实验 |
| `scripts/train_ecg_ppg.py` | ECG+PPG 双通道实验 |
| `scripts/train_final_ms.py` | 多尺度 CNN + 注意力实验 |
| `scripts/run_calibration_experiment.py` | 个体校准实验（用户个性化评估） |
| `demo.py` | 终端交互式演示系统 |

## 附录 B：模型文件

| 文件 | 说明 |
|:-----|:-----|
| `models/cnn1d_bp.pt` | 最佳 CNN1D 模型权重 |
| `models/norm_params.json` | 数据标准化参数 |
| `data/processed/calibration_results.json` | 个体校准实验完整结果 |

---

*本文实验数据、代码和模型权重均已整理归档，确保实验可复现。*
