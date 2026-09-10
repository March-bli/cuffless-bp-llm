# 摘要

无袖带血压监测是可穿戴健康领域的重要目标，而光电容积描记（PPG）信号因其便于持续采集被视为关键技术路线。然而，PPG 血压估计长期受困于**跨受试者泛化难题**：由于血管弹性、外周阻力等生理因素的个体差异，在严格按受试者划分的数据上，传统机器学习模型的精度严重退化（XGBoost 的 SBP 平均绝对误差为 17.0 mmHg）。

针对这一难题，本文提出**基于大语言模型的少样本个性化校准方法**。其核心思想是：利用大语言模型的上下文学习能力，向模型提供目标个体自身的少量带标签校准样本（特征与血压配对），使模型在推理时适应其个性化生理特性，而非依赖从人群数据学习到的通用映射。本文在 857 名受试者、148,012 个 PPG 段的数据集上进行了系统实验。结果表明，当校准样本数为 10 时，方法达到 SBP 平均绝对误差 8.3 mmHg、DBP 4.1 mmHg，其中舒张压满足 BHS Grade A 临床标准，较 XGBoost 基线分别提升 51% 与 65%。消融实验揭示了完整的校准曲线规律，并发现一个重要现象：**8B 本地模型配合 10 个校准样本即可追平 671B 大模型**，表明校准样本数量对精度的影响大于模型规模。

此外，本文借鉴 SensorLM 与 TimeSRL 探索了语义抽象的作用。实验表明，保留定量锚点的层次化语义描述可进一步提升舒张压预测精度（5.6 mmHg），而纯定性摘要则导致精度崩溃，从而划定了语义抽象在生理信号建模中的有效边界。

本文的贡献在于：（1）构建了 857 人规模的严格跨受试者 PPG 血压估计基准；（2）提出了 LLM 少样本个性化校准方法，达到临床级舒张压精度；（3）揭示了个性化校准优于通用建模的范式转变；（4）为生理信号的语义语言化建模提供了设计准则。

**关键词**：无袖带血压估计；光电容积描记；大语言模型；少样本学习；个性化校准；跨受试者泛化

---

# Abstract

Cuffless blood pressure (BP) monitoring is a key goal in wearable health, with photoplethysmography (PPG) as a promising sensing modality. However, PPG-based BP estimation has long suffered from the **cross-subject generalisation challenge**: due to individual differences in vascular elasticity and peripheral resistance, traditional machine learning models degrade severely under strict subject-level splits (XGBoost SBP MAE = 17.0 mmHg).

To address this, we propose a **few-shot personalised calibration method based on large language models (LLMs)**. The key idea is to exploit the in-context learning ability of LLMs: by providing the model with a small number of labelled calibration samples from the target subject, it adapts to that individual's physiological characteristics at inference time. On a dataset of 857 subjects and 148,012 PPG segments, our method achieves SBP MAE of 8.3 mmHg and DBP MAE of 4.1 mmHg with 10 calibration samples, with DBP meeting the BHS Grade A clinical standard — a 51% and 65% improvement over XGBoost, respectively. Ablation studies reveal a complete calibration curve and a notable finding: **an 8B local model with 10 calibration samples matches a 671B model**, indicating that calibration sample size matters more than model scale.

We further investigate semantic abstraction, drawing on SensorLM and TimeSRL. Experiments show that rule-based hierarchical descriptions retaining quantitative anchors further improve DBP (5.6 mmHg), while purely qualitative summaries collapse, delineating the effective boundary of semantic abstraction in physiological modelling.

**Keywords**: cuffless blood pressure; PPG; large language model; few-shot learning; personalised calibration; cross-subject generalisation
