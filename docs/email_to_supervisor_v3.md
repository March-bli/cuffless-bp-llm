Subject: Progress Update: LLM-Based Blood Pressure Estimation

Dear Dr Sun,

I hope this email finds you well. I'm writing to update you on my project progress following your previous feedback.

## Following your suggestions

I have addressed all three points from your last email:

1. **Zero-shot baseline** — I redesigned it as you suggested: the model now receives features and BP from *other* subjects as references (plus demographic information), then predicts for an unseen subject.

2. **Dataset** — I scaled up from 122 subjects to **857 subjects (148,012 segments)** from PulseDB (MIMIC + Vital subsets), with strict subject-level splits.

3. **The two papers you shared** — I drew inspiration from both SensorLM (hierarchical captions) and TimeSRL (semantic bottleneck), and ran corresponding experiments.

## Main results (Qwen3-8B, run locally on Stanage A100)

Few-shot calibration ablation (calibration samples K from the same test subject):

| K | SBP MAE | DBP MAE |
|---|---------|---------|
| 0 (zero-shot) | 20.8 | 12.6 |
| 1 | 21.8 | 18.8 |
| 3 | 14.4 | 11.6 |
| 5 | 9.3 | 6.1 |
| **10** | **8.3** | **4.1** |
| 20 | 8.4 | 4.5 |

**Three findings worth highlighting:**

1. **K=10 achieves BHS Grade A for DBP (4.1 mmHg)** and Grade B for SBP (8.3 mmHg), substantially outperforming traditional ML baselines (XGBoost: 20.6/11.7; CNN1D: 16.6/9.1).

2. **Calibration samples matter more than model size** — my local 8B model with K=10 matches or beats the 671B DeepSeek API with K=5 (8.9/4.3). This suggests few-shot personalisation is the key factor, not model capacity.

3. **Semantic abstraction**: rule-based hierarchical descriptions (SensorLM-style) improved DBP (5.6 vs 6.1), while a purely qualitative LLM summary (naive TimeSRL-style bottleneck) collapsed (30.0/26.9) — indicating semantic abstraction must retain quantitative anchors.

## Next steps

1. Writing the dissertation chapters (I have all experiments completed).
2. Preparing the presentation for **4 September**.

I would be very grateful for any feedback on these results, particularly on whether the K=10 personalisation story is sufficiently strong for the dissertation.

Best regards,
Boyuan Li
