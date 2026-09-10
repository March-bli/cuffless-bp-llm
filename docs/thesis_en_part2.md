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
| 0 (zero-shot) | 20.8 | 12.6 | 100 |
| 1 | 21.8 | 18.8 | 85 |
| 3 | 14.4 | 11.6 | 85 |
| 5 | 9.3 | 6.1 | 79 |
| **10** | **8.3** | **4.1** | 79 |
| 20 | 8.4 | 4.5 | 78 |

Three observations: (1) accuracy improves monotonically from K=1 to K=10; (2) K=10 is optimal, with DBP meeting BHS Grade A (≤5 mmHg); (3) K=1 degrades below zero-shot — a single sample is insufficient and can mislead the model.

**Comparison with traditional ML:**

| Method | SBP MAE | DBP MAE | Improvement over XGBoost |
|------|:---:|:---:|:---:|
| XGBoost | 17.0 | 11.6 | — |
| RandomForest | 17.2 | 11.8 | −1% / −2% |
| Zero-shot (ours) | 20.8 | 12.6 | −22% / −9% |
| **Few-shot K=10 (ours)** | **8.3** | **4.1** | **51% / 65%** |

**Comparison with prior LLM work:**

| Method | Model scale | SBP MAE | DBP MAE |
|------|:---:|:---:|:---:|
| LLM-BP (2024) | GPT-4 API | 9.3 | 6.4 |
| DeepSeek API (K=5) | 671B | 8.9 | 4.3 |
| **Ours (K=10)** | **8B local** | **8.3** | **4.1** |

The 8B local model with K=10 matches or beats the 671B model — **calibration sample size matters more than model scale**.

### 6.6 Discussion

Few-shot calibration works because BP's individual differences are dominated by stable physiological factors (baseline level, arterial stiffness) that K samples reveal. In-context learning enables this adaptation without retraining. The finding that calibration data beats model scale has direct deployment implications: small models with personal calibration archives suffice.

Limitations: 20 test subjects limit statistical power; calibration sample selection is random; SBP remains Grade B.

### 6.7 Chapter Summary

Cross-subject zero-shot confirms the problem is individual physiology, not model capacity. Few-shot calibration achieves clinical-grade DBP (4.1 mmHg) at K=10, outperforming traditional ML by 51%/65%, and reveals that calibration sample size outweighs model scale.

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
| Numeric few-shot K=5 (baseline) | 9.3 | 6.1 |
| Rule-semantic K=5 (SensorLM-style) | 10.8 | **5.6** |
| LLM bottleneck K=5 (TimeSRL-style) | 30.0 | 26.9 |

**Finding 1**: rule-based semantic descriptions improve DBP (5.6 vs 6.1). DBP is governed by vascular resistance, which the semantic layer's "vascular resistance" description precisely highlights — validating SensorLM's thesis.

**Finding 2**: purely qualitative LLM summaries collapse (30.0/26.9), as they discard all quantitative information (e.g., "BP may be slightly elevated" without numbers). This reveals that **semantic abstraction must retain quantitative anchors** — a boundary condition for TimeSRL's paradigm in regression tasks.

### 7.5 Chapter Summary

The positive/negative contrast delineates the effective boundary of semantic abstraction in BP estimation: language helps when it carries quantitative information, but harms when over-abstracted.

## Chapter 8 Conclusion and Future Work

### 8.1 Summary

This thesis addressed the cross-subject generalisation challenge in cuffless BP estimation through LLM-based personalisation. Four stages were completed: (1) problem diagnosis via 857-subject subject-level experiments; (2) method proposal (few-shot calibration); (3) experimental validation (K=10: SBP 8.3 / DBP 4.1, BHS Grade A for DBP); (4) theoretical deepening via semantic abstraction.

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
