Subject: Full PulseDB Results — Experiments Completed

Dear Dr Sun,

Following your advice, I have re-run all experiments on the complete PulseDB dataset (5,159 subjects). The results are summarised below.

**Main results** (Qwen3-8B, chronological K calibration, SBP/DBP MAE in mmHg):

| Method | SBP | DBP |
|--------|-----|-----|
| XGBoost (global) | 16.1 | 8.7 |
| LLM zero-shot | 17.9 | 14.6 |
| LLM few-shot K=1 | 18.0 | 16.7 |
| LLM few-shot K=3 | 15.8 | 11.8 |
| LLM few-shot K=5 | 14.9 | 10.2 |
| LLM few-shot K=10 | 10.3 | 8.0 |
| LLM few-shot K=20 | 8.4 | 6.7 |
| Semantic description K=5 | 10.6 | 7.3 |
| Semantic bottleneck K=5 | 25.0 | 23.5 |

**Key observations:**

1. Your concern about the subset was valid — on the full dataset, XGBoost improved notably (DBP from 11.6 to 8.7), confirming the earlier subset overestimated some advantages.

2. The core story holds — LLM few-shot calibration with K=20 achieves SBP 8.4 / DBP 6.7, outperforming XGBoost by 48% (SBP) and 23% (DBP).

3. Semantic description improves calibration efficiency at equal K (10.6/7.3 vs 14.9/10.2 at K=5), though it does not surpass numeric K=20. I have adjusted the thesis narrative accordingly.

4. Calibration sample size still outweighs model scale — our 8B local model with K=20 approaches the 671B API baseline.

All results have been verified for reproducibility (identical across two independent runs). I am continuing to refine the dissertation writing and would be grateful for any further feedback.

Best regards,
Boyuan Li
