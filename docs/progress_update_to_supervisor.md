# Progress Update & Questions for Supervisor

**Project**: Novel Blood Pressure Monitoring using Wearable Devices and Artificial Intelligence (AI)
**Date**: July 2026

---

## 1. Current Progress

### 1.1 Data Pipeline (PulseDB)
- Loaded **PulseDB** dataset (Wang et al., 2023) — 70 `.mat` files, 3,591 10-second segments from MIMIC-III + VitalDB, sampled at 125 Hz.
- Built a **multi-stage signal cleaning pipeline**: baseline wander removal (0.5 Hz high-pass), power-line notch filter (50/60 Hz), motion artifact detection (Z-score sliding window), spike removal (median filter).

### 1.2 Feature Engineering
- Extracted **55 PPG features** across four categories:
  - **Time-domain**: systolic/diastolic amplitude, pulse interval, upstroke time, augmentation index (AIx), stiffness index (SI), pulse width at 25%/50%/75%.
  - **Derivative (VPG + APG)**: a-b-c-d-e wave amplitudes, standard deviations, and ratios (b/a, c/a, d/a, e/a).
  - **Statistical**: mean, std, skewness, kurtosis, percentiles, entropy.
  - **Frequency-domain**: FFT spectral energy, HR band power ratio, dominant frequency, LF/HF ratio.

### 1.3 ML Models for BP Estimation (5-fold CV)

| Model | SBP MAE | DBP MAE | SBP R² | DBP R² | DBP BHS Grade |
|-------|:------:|:------:|:-----:|:-----:|:------------:|
| Random Forest | 7.9 | 4.9 | 0.684 | 0.620 | **A** |
| **XGBoost (tuned)** | **7.3** | **4.6** | **0.721** | **0.660** | **A** |
| LightGBM | 8.1 | 5.0 | 0.672 | 0.605 | B |
| SVR | 9.8 | 5.8 | 0.583 | 0.512 | C |

- **DBP achieves BHS Grade A** — meets clinical-grade accuracy.
- **SBP at BHS Grade C** — needs improvement; currently limited by the small subset of PulseDB (70 subjects out of 5,361 available).
- Deep learning (CNN1D / ResNet1D) was attempted end-to-end on raw waveforms but underperformed due to limited data volume.

### 1.4 Clinical Knowledge Layer + NL Report Generation
- Built a **clinical classifier** based on AHA/ESC/NICE guidelines — 5 risk levels (healthy → caution → warning → danger).
- **Dual-mode NL health report generation**:
  - **Template mode**: rule-based structured report.
  - **LLM mode**: DeepSeek / xAI Grok / OpenAI-compatible API integration for natural, empathetic health summaries. Falls back to template mode if API unavailable.

### 1.5 Literature Review (8 papers studied)

| # | Paper | Relevance |
|---|-------|-----------|
| 1 | Mazzà et al., *BMJ Open* 2021 — Mobilise-D validation protocol | Methodology for wearable device technical validation |
| 2 | Kirk et al., *Sci Rep* 2024 — Mobilise-D walking speed results | Real-world vs. lab validation framework |
| 3 | Reiss et al., *Sensors* 2019 — Deep PPG (CNN for HR) | Time-frequency spectrum approach for PPG |
| 4 | Wang et al., *Frontiers* 2023 — PulseDB dataset | Core dataset we use |
| 5 | Fan et al., *arXiv* 2026 — TimeSRL (semantic bottleneck + LLM) | Theoretical support for signal→NL abstraction |
| 6 | **Liu et al., *BCB* 2024 — LLM for Cuffless BP** | **Directly comparable work**: first LLM-based BP estimation (SBP: ~9.3, DBP: ~6.4 mmHg MAE) |
| 7 | Zhang et al., *arXiv* 2025 — SensorLM (Google) | Sensor-language foundation model scaling |
| 8 | Goda et al., *Physiol Meas* 2024 — pyPPG toolbox | Standardized PPG fiducial point detection (74 biomarkers) |

---

## 2. Key Questions for Discussion

### Q1: Is this research direction viable?

The core idea is a **two-output pipeline**: (1) numerical BP prediction + (2) natural language health report via LLM. The closest published work is Liu et al. (BCB 2024), which uses LLMs *only* for BP regression. My approach differs by:

- Building a **full clinical knowledge layer** (AHA/ESC guidelines) between signal processing and NL generation.
- Supporting a **hardware integration pathway** (ESP32-C3 wearable device).

Our current XGBoost results (DBP MAE = 4.6, BHS Grade A) already **outperform** Liu et al.'s LLM-BP on DBP, and are competitive on SBP. With the full PulseDB dataset (5,361 subjects), I expect further improvement.

**I would appreciate your feedback on whether this direction — combining physiological signal processing, ML-based BP estimation, and LLM-driven health reporting — constitutes a sufficient contribution for the dissertation.**

### Q2: Is building the hardware prototype necessary?

My original plan included an **ESP32-C3 + MAX30102 PPG sensor** wearable board for real data collection. However:

- **Pros**: Demonstrates end-to-end system feasibility, provides real-world validation data, strengthens the engineering contribution.
- **Cons**: Significant time investment; hardware debugging is unpredictable; PulseDB already provides high-quality labeled data for algorithm development.

I am leaning toward focusing on the **algorithm and system integration** aspects, using PulseDB for model training/evaluation, and treating the hardware design as a *future work* section rather than a must-deliver. Would you agree with this scope?

---

## 3. Next Steps (Pending Your Input)

1. **Scale up**: Download and process the full PulseDB dataset (5,361 subjects) to improve SBP accuracy.
2. **Enhance SBP prediction**: Experiment with stacking ensembles, feature selection via SHAP, and calibration-based approaches.
3. **LLM generation quality evaluation**: Systematic comparison of template vs. LLM health reports (fluency, clinical accuracy, empathy).
4. **Thesis writing**: Outline is complete (8 chapters), draft is ~92 KB. Ready to refine based on supervisor feedback.

---

Looking forward to your guidance.

Best regards,
LBY
