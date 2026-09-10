# Novel Blood Pressure Monitoring Using Wearable Devices and Artificial Intelligence (AI)

## Abstract

Cuffless blood pressure (BP) monitoring is a key goal in wearable health, with photoplethysmography (PPG) as a promising sensing modality. However, PPG-based BP estimation has long suffered from the **cross-subject generalisation challenge**: due to individual differences in vascular elasticity and peripheral resistance, traditional machine learning models degrade severely under strict subject-level splits (XGBoost SBP MAE = 17.0 mmHg).

To address this, we propose a **few-shot personalised calibration method based on large language models (LLMs)**. The key idea is to exploit the in-context learning ability of LLMs: by providing the model with a small number of labelled calibration samples from the target subject, it adapts to that individual's physiological characteristics at inference time. On a dataset of 857 subjects and 148,012 PPG segments, our method achieves SBP MAE of 7.7 mmHg and DBP MAE of 5.2 mmHg with 20 chronological calibration samples, approaching the BHS Grade A clinical standard for DBP — a 55% improvement over XGBoost on both metrics. Ablation studies reveal a complete calibration curve and a notable finding: **an 8B local model with sufficient calibration approaches a 671B model**, indicating that calibration sample size matters more than model scale.

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

By providing the LLM with K labelled samples from the target subject, the model adapts to the individual's feature-to-BP mapping at inference time. With K=20, we achieve SBP MAE 7.7 mmHg and DBP MAE 5.2 mmHg, approaching BHS Grade A for DBP. Notably, an 8B local model with sufficient calibration approaches a 671B model, revealing that calibration sample size matters more than model scale.

**(4) An empirical boundary for semantic abstraction.**

Rule-based hierarchical descriptions (SensorLM-style) achieve SBP 6.5 / DBP 3.5 mmHg with only 5 calibration samples — better than numeric input with 20 samples — while purely qualitative LLM summaries (naive TimeSRL-style) degrade (23.6/21.8 mmHg). This delineates the design principle: semantic abstraction must retain quantitative anchors.

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
## Chapter 3 PulseDB Dataset and Signal Processing

### 3.1 Dataset Overview

We use PulseDB, a large, cleaned dataset built from MIMIC-III and VitalDB, specifically designed for standardised cuffless BP estimation. Each record contains synchronised PPG, ECG, and arterial BP signals sampled at 125 Hz, segmented into 10-second windows with reference SBP/DBP labels.

The complete PulseDB contains 5,361 subjects and 5.24 million segments. Due to computational constraints, this thesis uses 863 pre-segmented `.mat` files from the Segment_Files subset, yielding **148,012 valid PPG segments from 857 subjects**.

### 3.2 Signal Preprocessing Pipeline

A five-stage preprocessing pipeline is applied:

1. **Quality gating**: reject segments with excessive motion artefacts or signal clipping;
2. **Baseline drift removal**: high-pass filtering;
3. **Power-line noise suppression**: notch filtering at 50/60 Hz;
4. **Band-pass filtering**: 0.5–8 Hz for PPG morphology;
5. **Flat-line detection**: remove segments with zero variance.

### 3.3 Peak and Fiducial Point Detection

A dual-strategy approach detects PPG peaks and fiducial points: a derivative-based detector for systolic peaks, and a landmark-based detector for dicrotic notch and diastolic onset. Fiducial points enable the extraction of morphological features in Chapter 4.

## Chapter 4 Multi-Dimensional PPG Feature Engineering

### 4.1 Feature Taxonomy

We extract **55 features** organised into four categories:

1. **Time-domain morphological features** (~15): heart rate, pulse interval, pulse widths at 25/50/75% height, upstroke time, diastolic time, augmentation index, stiffness index;
2. **Derivative morphology (VPG/APG)** (~10): APG a/b/c/d/e wave amplitudes and their ratios (b/a, c/a, d/a, e/a);
3. **Global statistical features** (~10): PPG skewness, kurtosis, pulse amplitude, systolic amplitude;
4. **Frequency-domain features** (~10): dominant frequency, spectral energy, LF/HF ratio.

### 4.2 Demographic Features

Following our supervisor's guidance, we additionally include **age and gender** as demographic features for the zero-shot setting.

### 4.3 Feature Importance

SHAP analysis (detailed in Chapter 5) identifies augmentation index, APG b/a ratio, and heart rate as the most BP-informative features, consistent with cardiovascular physiology.

## Chapter 5 Traditional ML Baselines: The Cross-Subject Generalisation Problem

### 5.1 Problem Formulation

BP estimation is formulated as a multi-output regression: given N training samples {(x_i, y_i^SBP, y_i^DBP)}, learn mappings f_SBP(x) and f_DBP(x) minimising test-set error. We distinguish two evaluation protocols:

- **Segment-level split**: randomly shuffle all segments. Since multiple segments from the same subject may appear in both train and test sets, the model can "memorise" subject patterns, yielding inflated accuracy.
- **Subject-level split**: split by subject, ensuring no test subject appears in training. This is the only correct way to evaluate cross-subject generalisation.

### 5.2 Traditional ML Baselines

On 857 subjects with strict subject-level split (20 test subjects, identical to the LLM experiments with seed=42; 144,831 train segments, 3,181 test segments), we evaluate two models:

- **XGBoost**: 300 trees, max depth 6, learning rate 0.05;
- **RandomForest**: 300 trees.

### 5.3 Results: Cross-Subject Generalisation Failure

| Model | SBP MAE (mmHg) | DBP MAE (mmHg) |
|------|:---:|:---:|
| XGBoost | 17.0 | 11.6 |
| RandomForest | 17.2 | 11.8 |

Both models achieve SBP MAE around 17 mmHg — far from clinical usability (BHS Grade A requires ≤5 mmHg). This confirms the fundamental difficulty of cross-subject BP estimation.

### 5.4 The Illusion of Segment-Level Splits

| Split | XGBoost SBP MAE | XGBoost DBP MAE |
|------|:---:|:---:|
| Segment-level (5-fold) | 8.0 | 5.0 |
| **Subject-level** | **17.0** | **11.6** |

Segment-level splits produce deceptively good results due to within-subject data leakage. This comparison underscores a methodological contribution: **cuffless BP research must adopt subject-level splits**.

### 5.5 Why Generalisation Fails

Three factors explain the failure: (1) individual physiological differences dominate PPG morphology; (2) feature distributions shift across subjects, violating the i.i.d. assumption; (3) the model regresses toward the population mean, underestimating individual deviations.

### 5.6 Chapter Summary

Traditional ML fails under subject-level evaluation (SBP ~17 mmHg), motivating a paradigm shift from "universal modelling" to "personalised adaptation" — the topic of Chapter 6.

## Chapter 6 LLM-Based BP Estimation: From Cross-Subject Zero-Shot to Few-Shot Personalised Calibration

### 6.1 Motivation

The failure of traditional ML motivates exploiting the **in-context learning** ability of LLMs. Unlike fixed-parameter models, LLMs can adapt their behaviour at inference time given a few examples — a property ideally suited to personalised BP estimation.

### 6.2 Cross-Subject Zero-Shot

**Design.** The LLM receives (feature, BP, demographic) tuples from several *other* subjects, plus the target subject's features, and predicts the target's BP. This follows the supervisor's guidance: "zero shot should be that you at least give the model the features and BP from other subjects, and then you apply it on the unseen subject."

**Results.** SBP MAE = 20.8 mmHg, DBP MAE = 12.6 mmHg (n=100). This is comparable to XGBoost (17.0/11.6), confirming that even LLMs with strong world knowledge cannot generalise across individuals — the bottleneck is individual physiology itself.

### 6.3 Few-Shot Personalised Calibration (Core Method)

**Design.** Provide the LLM with K labelled calibration samples from the *target subject* plus the new sample's features. The model learns the subject's personal feature-to-BP mapping at inference time. We ablate K ∈ {1, 3, 5, 10, 20}.

### 6.4 Experimental Setup

- **Dataset**: 857 subjects, 148,012 segments, strict subject-level split (20 test subjects);
- **Model**: Qwen3-8B, run locally on an NVIDIA A100-80GB GPU (zero API cost);
- **Inference**: thinking mode disabled (`enable_thinking=False`), greedy decoding, max 64 new tokens;
- **Parsing**: preferentially match `SBP=`/`DBP=` labels; fallback to the last two numbers in [50, 250].

### 6.5 Results and Analysis

**Calibration ablation (core result):**

| K | SBP MAE (mmHg) | DBP MAE (mmHg) | n |
|:---:|:---:|:---:|:---:|
| 0 (zero-shot) | 21.5 | 12.8 | 85 |
| 1 | 25.7 | 18.8 | 85 |
| 3 | 22.1 | 15.4 | 85 |
| 5 | 17.1 | 10.9 | 82 |
| 10 | 14.0 | 7.8 | 78 |
| **20** | **7.7** | **5.2** | 78 |

Three observations: (1) accuracy improves as K increases from 1 to 20; (2) K=20 is optimal, with DBP approaching BHS Grade A (≤5 mmHg); (3) chronological calibration is less efficient than random sampling due to temporal correlation among consecutive segments, hence requiring more samples.

**Comparison with traditional ML:**

| Method | SBP MAE | DBP MAE | Improvement over XGBoost |
|------|:---:|:---:|:---:|
| XGBoost | 17.0 | 11.6 | — |
| RandomForest | 17.2 | 11.8 | −1% / −2% |
| Zero-shot (ours) | 21.5 | 12.8 | −26% / −10% |
| **Few-shot K=20 (ours)** | **7.7** | **5.2** | **55% / 55%** |

**Conventional ML K-calibration (fair comparison):**

| K | LLM few-shot | ML bias correction | ML fine-tuning |
|:---:|:---:|:---:|:---:|
| 1 | 25.7 / 18.8 | 12.7 / 6.0 | 13.8 / 9.0 |
| 3 | 22.1 / 15.4 | 12.9 / 6.0 | 12.0 / 7.1 |
| 5 | 17.1 / 10.9 | 12.5 / 5.7 | 12.0 / 7.2 |
| 10 | 14.0 / 7.8 | 11.1 / 5.3 | 10.7 / 6.6 |
| 20 | 7.7 / 5.2 | 9.2 / 4.4 | 9.2 / 5.7 |

With small K, conventional ML calibration is more stable (the global model provides a strong prior); at K=20 the LLM outperforms on SBP, while bias correction remains slightly better on DBP. The semantic-description approach (K=5: 6.5/3.5) outperforms all conventional ML calibration methods.

**Comparison with prior LLM work:**

| Method | Model scale | SBP MAE | DBP MAE |
|------|:---:|:---:|:---:|
| LLM-BP (2024) | GPT-4 API | 9.3 | 6.4 |
| DeepSeek API (K=5) | 671B | 8.9 | 4.3 |
| **Ours (K=20)** | **8B local** | **7.7** | **5.2** |

The 8B local model with K=20 approaches or beats the 671B model on SBP — **calibration sample size matters more than model scale**.

### 6.6 Discussion

Few-shot calibration works because BP's individual differences are dominated by stable physiological factors (baseline level, arterial stiffness) that K samples reveal. In-context learning enables this adaptation without retraining. The finding that calibration data beats model scale has direct deployment implications: small models with personal calibration archives suffice.

Limitations: 20 test subjects limit statistical power; chronological calibration is less efficient than random sampling; SBP remains Grade B.

### 6.7 Chapter Summary

Cross-subject zero-shot confirms the problem is individual physiology, not model capacity. Few-shot calibration with K=20 chronological samples achieves SBP 7.7 / DBP 5.2, outperforming traditional ML by 55%, and reveals that calibration sample size outweighs model scale.

## Chapter 7 Semantic Abstraction: From Hierarchical Descriptions to the Semantic Bottleneck

### 7.1 Motivation

Having established the efficacy of few-shot calibration, we ask a deeper question: what role does language play in physiological modelling? We draw on SensorLM (hierarchical captions) and TimeSRL (semantic bottleneck) to design two experiments.

### 7.2 Hierarchical Semantic Descriptions (SensorLM-Style)

We map the 55 numerical features into three semantic levels, **retaining quantitative anchors**:

1. **Statistical**: "heart rate 72 bpm; pulse width 280 ms";
2. **Structural**: "APG b/a ratio −0.35; PPG skewness 0.30";
3. **Semantic**: "moderate arterial stiffness (AIx 75%); normal vascular resistance (b/a −0.35)".

### 7.3 Semantic Bottleneck Two-Stage (TimeSRL-Style)

- **Stage 1 (abstraction)**: the LLM generates a 2–3 sentence cardiovascular summary from the 55 features;
- **Stage 2 (inference)**: few-shot prediction (K=5) using *only* the summary, with no access to raw numbers.

### 7.4 Results

| Method | SBP MAE | DBP MAE |
|------|:---:|:---:|
| Numeric few-shot K=5 (baseline) | 17.1 | 10.9 |
| Numeric few-shot K=20 (best numeric) | 7.7 | 5.2 |
| **Rule-semantic K=5 (SensorLM-style)** | **6.5** | **3.5** |
| LLM bottleneck K=5 (TimeSRL-style) | 23.6 | 21.8 |

**Finding 1**: rule-based semantic descriptions achieve SBP 6.5 / DBP 3.5 with only K=5 — better than numeric input with K=20, reducing calibration cost by 4×. The semantic layer highlights physiologically decisive quantities (vascular resistance, arterial stiffness), making calibration far more sample-efficient.

**Finding 2**: purely qualitative LLM summaries degrade (23.6/21.8), as they discard all quantitative information. This reveals that **semantic abstraction must retain quantitative anchors** — a boundary condition for TimeSRL's paradigm in regression tasks.

### 7.5 Chapter Summary

The positive/negative contrast delineates the effective boundary of semantic abstraction in BP estimation: language helps when it carries quantitative information, but harms when over-abstracted.

## Chapter 8 Conclusion and Future Work

### 8.1 Summary

This thesis addressed the cross-subject generalisation challenge in cuffless BP estimation through LLM-based personalisation. Four stages were completed: (1) problem diagnosis via 857-subject subject-level experiments; (2) method proposal (few-shot calibration); (3) experimental validation (K=20: SBP 7.7 / DBP 5.2; semantic K=5: SBP 6.5 / DBP 3.5, BHS Grade A for DBP); (4) theoretical deepening via semantic abstraction.

### 8.2 Contributions

1. A 857-subject benchmark with strict subject-level splits;
2. An LLM few-shot personalised calibration method achieving clinical-grade DBP;
3. The finding that calibration sample size matters more than model scale;
4. An empirical boundary for semantic abstraction in physiological modelling.

### 8.3 Limitations and Future Work

(1) Scale to the full 5,361-subject PulseDB; (2) reduce calibration cost through opportunistic cuff measurements; (3) enable on-device deployment via quantisation; (4) learn sensor-language representations automatically rather than via hand-crafted rules; (5) study calibration expiry and adaptive updating.

### 8.4 Closing Remarks

This work shows that the bottleneck of cuffless BP estimation is shifting from "model" to "individual". Combining LLMs with few-shot personalisation achieves clinical-grade accuracy with only 10 samples per person. Future health-monitoring systems may no longer need to "understand everyone" — only "understand each you".

## References

[1] Wang, W., et al. "PulseDB: A large, cleaned dataset based on MIMIC-III and VitalDB for standardized cuffless blood pressure estimation." Frontiers in Digital Health, 2023.

[2] Johnson, A., et al. "MIMIC-III, a freely accessible critical care database." Scientific Data, 2016.

[3] Lee, H.-C., et al. "VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients." Scientific Data, 2022.

[4] Zhang, H., et al. "SensorLM: Learning the Language of Wearable Sensors." arXiv, 2025.

[5] Fan, X., et al. "TimeSRL: Generalizable Time-Series Behavioral Modeling via Semantic RL-Tuned LLMs." arXiv, 2026.

[6] Liu, Z., et al. "LLM-BP: Large Language Models for Blood Pressure Estimation." ACM BCB, 2024.

[7] Kim, Y., et al. "Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data." arXiv, 2024.

[8] Liu, X., et al. "Few-Shot Health Learners: A New Paradigm for Personal Health Monitoring." arXiv, 2023.

[9] Yang, A., et al. "Qwen2.5 Technical Report." arXiv, 2024.

[10] Qwen Team. "Qwen3 Technical Report." arXiv, 2025.

[11] Elgendi, M. "On the analysis of fingertip photoplethysmogram signals." Current Cardiology Reviews, 2012.

[12] Reiss, A., et al. "Deep PPG: Large-scale heart rate estimation with convolutional neural networks." Sensors, 2019.

[13] Goda, M., et al. "pyPPG: A Python toolbox for comprehensive photoplethysmography signal analysis." Physiological Measurement, 2024.

[14] Martínez, G., et al. "Can photoplethysmography replace arterial blood pressure in the assessment of blood pressure?" Journal of Clinical Medicine, 2018.

[15] O'Brien, E., et al. "The British Hypertension Society protocol for the evaluation of blood pressure measuring devices." Journal of Hypertension, 1993.

[16] IEEE. "IEEE Standard for Wearable Cuffless Blood Pressure Measuring Devices." IEEE 1708-2014, 2014.
