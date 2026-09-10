Subject: Full PulseDB Results — Experiments Completed

Dear Dr Sun,

Following your advice, I have re-run all experiments on the complete PulseDB dataset (5,361 subjects). Please find the updated results below.

## Main results (Qwen3-8B, chronological K calibration)

| Method | SBP MAE | DBP MAE |
|--------|---------|---------|
| XGBoost (global) | 16.1 | 8.7 |
| LLM zero-shot | 17.9 | 14.6 |
| LLM few-shot K=1 | 18.0 | 16.7 |
| LLM few-shot K=3 | 15.8 | 11.8 |
| LLM few-shot K=5 | 14.9 | 10.2 |
| LLM few-shot K=10 | 10.3 | 8.0 |
| **LLM few-shot K=20** | **8.4** | **6.7** |
| Semantic description K=5 | 10.6 | 7.3 |
| Semantic bottleneck K=5 | 25.0 | 23.5 |

## Key observations

1. **Your concern about the subset was valid.** On the full dataset, XGBoost improved notably (DBP from 11.6 to 8.7), confirming that the earlier subset overestimated some of our advantages.

2. **The core story holds.** LLM few-shot calibration with K=20 achieves SBP 8.4 / DBP 6.7, outperforming XGBoost by 48% (SBP) and 23% (DBP). The personalised-calibration advantage is real.

3. **Semantic description helps at equal K.** At K=5, semantic descriptions (10.6/7.3) outperform numeric input (14.9/10.2) by ~29%, showing improved calibration efficiency — though they do not surpass numeric K=20. I have adjusted the thesis narrative accordingly to present this honestly.

4. **Calibration sample size still outweighs model scale** — our 8B local model with K=20 approaches the 671B API baseline.

The dissertation chapters have been updated with these full-dataset results, and I am continuing to refine the writing. I would be grateful for any further feedback.

Best regards,
Boyuan Li
