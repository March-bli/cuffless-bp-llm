Subject: Re: Supplementary Results — XGBoost Calibration & Semantic K-Ablation

Dear Dr Sun,

Thank you for the helpful suggestions. I have completed the supplementary experiments you requested. The full results are below.

**1. XGBoost few-shot calibration (full PulseDB):**

| K | Bias correction SBP/DBP | Fine-tuning SBP/DBP |
|---|------------------------|---------------------|
| 1 | 14.3 / 7.3 | 12.7 / 9.4 |
| 3 | 12.5 / 6.5 | 11.8 / 7.4 |
| 5 | 12.0 / 6.5 | 10.8 / 6.8 |
| 10 | 8.7 / 5.3 | 9.3 / 4.9 |
| 20 | 7.0 / 4.2 | 7.8 / 5.0 |

**2. Semantic description K-ablation (full PulseDB):**

| K | Semantic description SBP/DBP |
|---|------------------------------|
| 1 | 16.6 / 12.7 |
| 3 | 13.0 / 8.9 |
| 5 | 13.0 / 7.8 |
| 10 | 8.5 / 4.8 |
| 20 | 6.5 / 3.6 |

**3. Combined comparison at K=20:**

| Method | SBP MAE | DBP MAE |
|--------|---------|---------|
| XGBoost (global) | 16.1 | 8.7 |
| XGBoost bias correction | 7.0 | 4.2 |
| LLM numeric few-shot | 8.4 | 6.7 |
| LLM semantic description | 6.5 | 3.6 |

**Key observations:**

1. XGBoost bias correction is a strong baseline — at K=20 it outperforms LLM numeric few-shot (7.0/4.2 vs 8.4/6.7). This was an important addition.

2. The LLM's advantage emerges when combined with semantic description — K=20 achieves SBP 6.5 / DBP 3.6 (DBP meets BHS Grade A), outperforming all other methods. This suggests the value of the LLM lies in semantic abstraction rather than raw numeric few-shot learning.

**4. Difference between semantic description and bottleneck:**

- **Semantic description** converts numerical features into language that **retains quantitative values** (e.g. "moderate arterial stiffness (AIx 75%)"), making physiologically decisive quantities more salient.
- **Semantic bottleneck** first generates a purely qualitative summary without numbers, then predicts from that summary alone — testing whether pure abstraction retains enough information.

The contrast shows that retaining quantitative anchors is crucial: semantic description improves calibration efficiency, while the bottleneck collapses (25.0/23.5).

I am updating the dissertation chapters with these results. I would be grateful for any further feedback.

Best regards,
Boyuan Li
