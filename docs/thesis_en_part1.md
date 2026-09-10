# Novel Blood Pressure Monitoring Using Wearable Devices and Artificial Intelligence (AI)

## Abstract

Cuffless blood pressure (BP) monitoring is a key goal in wearable health, with photoplethysmography (PPG) as a promising sensing modality. However, PPG-based BP estimation has long suffered from the **cross-subject generalisation challenge**: due to individual differences in vascular elasticity and peripheral resistance, traditional machine learning models degrade severely under strict subject-level splits (XGBoost SBP MAE = 17.0 mmHg).

To address this, we propose a **few-shot personalised calibration method based on large language models (LLMs)**. The key idea is to exploit the in-context learning ability of LLMs: by providing the model with a small number of labelled calibration samples from the target subject, it adapts to that individual's physiological characteristics at inference time. On a dataset of 857 subjects and 148,012 PPG segments, our method achieves SBP MAE of 8.3 mmHg and DBP MAE of 4.1 mmHg with 10 calibration samples, with DBP meeting the BHS Grade A clinical standard — a 51% and 65% improvement over XGBoost, respectively. Ablation studies reveal a complete calibration curve and a notable finding: **an 8B local model with 10 calibration samples matches a 671B model**, indicating that calibration sample size matters more than model scale.

We further investigate semantic abstraction, drawing on SensorLM and TimeSRL. Experiments show that rule-based hierarchical descriptions retaining quantitative anchors further improve DBP (5.6 mmHg), while purely qualitative summaries collapse, delineating the effective boundary of semantic abstraction in physiological modelling.

**Keywords**: cuffless blood pressure; PPG; large language model; few-shot learning; personalised calibration; cross-subject generalisation

---

## Chapter 1 Introduction

### 1.1 Background and Significance

Cardiovascular disease remains the leading cause of death worldwide, with hypertension as its primary modifiable risk factor. Traditional cuff-based BP measurement, while clinically accurate, only provides intermittent snapshots and is inconvenient for continuous monitoring. Wearable devices equipped with PPG sensors — the optical technique already used in smartwatches for heart-rate monitoring — offer the possibility of continuous, cuffless BP estimation.

Despite more than a decade of research, cuffless BP estimation has yet to achieve clinical adoption. The fundamental obstacle, as this thesis demonstrates, is not signal quality or feature engineering, but **cross-subject generalisation**: the relationship between PPG morphology and BP varies substantially across individuals.

### 1.2 Related Work

Cuffless BP estimation approaches fall into two broad categories:

- **Physiology-driven methods** based on pulse transit time (PTT), which exploit the relationship between pulse wave velocity and arterial stiffness. While theoretically grounded, PTT methods require multiple sensors and frequent individual calibration.
- **Data-driven methods** using machine learning and deep learning, which learn feature-to-BP mappings from large datasets. Despite strong within-subject performance, these methods consistently degrade under cross-subject evaluation.

More recently, large language models have been applied to physiological signal modelling. SensorLM learns sensor–language alignment through hierarchical captions, while TimeSRL proposes a semantic bottleneck to improve cross-population generalisation. Our work draws on both, applying LLMs to the personalised BP estimation problem.

### 1.3 Main Contributions

**(1) A 857-subject benchmark for cross-subject PPG-BP estimation.**

We construct a feature dataset of 857 subjects and 148,012 PPG segments from PulseDB, a seven-fold increase over prior work. All experiments use strict subject-level splits to faithfully evaluate cross-subject generalisation.

**(2) A rigorous cross-subject zero-shot baseline.**

We design a zero-shot setting in which the LLM receives features and BP from *other* subjects (plus demographic information) and predicts for an unseen subject. The LLM's zero-shot accuracy (SBP 20.8 mmHg) is comparable to XGBoost (17.0 mmHg), confirming that the bottleneck lies in individual physiological differences rather than model capacity.

**(3) A few-shot personalised calibration method (core contribution).**

By providing the LLM with K labelled samples from the target subject, the model adapts to the individual's feature-to-BP mapping at inference time. With K=10, we achieve SBP MAE 8.3 mmHg and DBP MAE 4.1 mmHg, with DBP meeting BHS Grade A. Notably, an 8B local model with K=10 matches a 671B model, revealing that calibration sample size matters more than model scale.

**(4) An empirical boundary for semantic abstraction.**

Rule-based hierarchical descriptions (SensorLM-style) improve DBP (5.6 vs 6.1 mmHg), while purely qualitative LLM summaries (naive TimeSRL-style) collapse (30.0/26.9 mmHg). This delineates the design principle: semantic abstraction must retain quantitative anchors.

### 1.4 Thesis Structure

The remainder of this thesis is organised as follows. Chapter 2 reviews the technical background. Chapter 3 describes the PulseDB dataset and signal processing pipeline. Chapter 4 presents the 55-dimensional feature engineering. Chapter 5 establishes traditional ML baselines and exposes the cross-subject generalisation problem. Chapter 6 proposes the LLM few-shot calibration method. Chapter 7 investigates semantic abstraction. Chapter 8 concludes.

---

## Chapter 2 Technical Background

### 2.1 Photoplethysmography (PPG)

PPG is an optical technique that measures blood volume changes in the microvascular bed. A PPG waveform contains a systolic peak, a dicrotic notch, and a diastolic phase, whose morphology reflects cardiovascular properties including arterial stiffness, vascular resistance, and cardiac contractility.

The first and second derivatives of the PPG signal (velocity plethysmogram, VPG, and acceleration plethysmogram, APG) provide additional morphological features. In particular, the APG b/a ratio correlates with vascular resistance, and the augmentation index (AIx) correlates with arterial stiffness — both are key physiological determinants of blood pressure.

### 2.2 Cuffless Blood Pressure Estimation

PTT-based methods estimate BP from the time delay between the ECG R-peak and the PPG systolic peak. While clinically motivated, they require chest-worn ECG electrodes and per-user calibration, limiting practical wearability.

Data-driven methods learn BP mappings from PPG features using regression models or neural networks. Evaluation is typically performed with the British Hypertension Society (BHS) protocol: Grade A requires MAE ≤5 mmHg, Grade B ≤10 mmHg for SBP (≤8 for DBP).

### 2.3 Large Language Models in Health

LLMs have demonstrated strong in-context learning ability: given a few examples, they can adapt to a task at inference time without parameter updates. This property is central to our personalised calibration approach.

Recent work has applied LLMs to physiological signals. LLM-BP (Liu et al., 2024) used a GPT-level API for BP estimation with limited personalisation. Health-LLM (Kim et al., 2024) explored LLMs for wearable health prediction. Our work differs by systematically studying few-shot personalisation and its interaction with model scale.

### 2.4 Sensor–Language Alignment: SensorLM

SensorLM proposes a sensor-language foundation model that translates wearable data into natural language through **hierarchical captions** at three levels:

1. **Statistical level**: numerical statistics such as mean and standard deviation;
2. **Structural level**: temporal structure such as trends and peak timing;
3. **Semantic level**: domain-knowledge inferences such as activity or physiological state.

Trained on 59.7M hours of data from 103K participants using a CLIP/CoCa architecture, SensorLM supports zero-shot recognition, few-shot learning, and cross-modal retrieval. Its central insight — **sensor data can be translated into language, and language provides a bridge for LLMs to understand physiological signals** — motivates our semantic description experiments in Chapter 7.

### 2.5 Semantic Bottleneck and RL: TimeSRL

TimeSRL addresses cross-population generalisation in time-series behavioural modelling through a **semantic bottleneck** framework:

1. **Abstraction stage**: abstract raw time series into natural-language behavioural descriptions;
2. **Inference stage**: predict downstream outcomes from the semantic descriptions alone.

TimeSRL further optimises the abstraction process end-to-end with GRPO + RLVR reinforcement learning. Its key finding is that the semantic bottleneck forces the model to discard noise and retain essence, substantially improving cross-dataset generalisation (MAE reduced by 3.1–57.6%). However, our Chapter 7 reveals a boundary condition: in precise regression tasks such as BP estimation, purely qualitative abstraction loses critical quantitative information.

### 2.6 Implications for This Thesis

SensorLM and TimeSRL together illuminate the potential and boundary of sensor "verbalisation": language representations can highlight physiological essence, but the degree and manner of abstraction are crucial. These two works form the theoretical foundation of Chapter 7.
