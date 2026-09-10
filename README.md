# Novel Blood Pressure Monitoring Using Wearable Devices and AI

MSc dissertation project (University of Sheffield) on cuffless blood pressure estimation from photoplethysmography (PPG) using large language models (LLMs).

## Overview

Cross-subject generalisation is the central obstacle in PPG-based cuffless blood pressure (BP) estimation: a global XGBoost trained on 3,611 subjects fails every BHS grade (SBP 15.7 / DBP 8.89 mmHg). This work proposes a **few-shot personalised calibration method based on LLMs**: the model adapts to an individual at inference time from a small number of chronological calibration samples, without any fine-tuning.

Two prompt designs are compared:
- **Numeric few-shot** — raw numerical features as in-context examples.
- **Semantic description** — the features are translated into a three-level clinical narrative (statistical / structural / semantic) that retains quantitative anchors.

## Key results (70/20/10 split, 517 test subjects, K=20)

| Method | SBP MAE | DBP MAE | BHS grade |
|--------|---------|---------|-----------|
| XGBoost (global) | 15.7 | 8.89 | — / — |
| XGBoost bias correction | 7.79 | 4.18 | C / A |
| LLM numeric few-shot | 5.99 | 3.32 | B / A |
| **LLM semantic description** | **5.52** | **2.93** | **B / A** |

Semantic description achieves DBP BHS Grade A (84.0% ≤5 mmHg) and SBP Grade B (63.2% ≤5 mmHg), and its advantage over numeric few-shot is largest when calibration data is scarce (4.5 mmHg at K=3).

## Repository structure

```
scripts/
  build_feature_dataset.py    # PulseDB → 55-dim feature dataset
  semantic_describe.py        # numeric features → 3-level clinical description
  xgboost_702010.py           # XGBoost 70/20/10 baseline (global + bias correction)
  xgboost_702010_ablation.py  # XGBoost K-ablation
  xgboost_702010_tune.py      # validation-set hyperparameter grid search
  llm_702010.py               # LLM few-shot K=20 (numeric + semantic)
  llm_702010_ablation.py      # LLM K-ablation (numeric + semantic)
  compute_bhs.py              # BHS cumulative error grading
src/ml/features.py            # 55-dim PPG feature extraction
```

## Environment

- Python 3.9+
- PyTorch, transformers, xgboost, numpy, scipy, scikit-learn
- Qwen3-8B (HuggingFace), NVIDIA A100-80GB

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducibility

1. Download PulseDB (MIMIC + Vital subsets) from the official Box link and unpack to `data/raw/`.
2. Build the feature dataset:
   ```bash
   python scripts/build_feature_dataset.py
   ```
3. Run XGBoost baselines:
   ```bash
   python scripts/xgboost_702010.py
   python scripts/xgboost_702010_ablation.py
   python scripts/xgboost_702010_tune.py
   ```
4. Download Qwen3-8B and run the LLM experiments:
   ```bash
   python scripts/llm_702010.py
   python scripts/llm_702010_ablation.py
   ```
5. Grade results with the BHS protocol:
   ```bash
   python scripts/compute_bhs.py
   ```

All experiments use greedy decoding (`do_sample=False`), disabled thinking mode, and a fixed seed (42), ensuring deterministic outputs.
