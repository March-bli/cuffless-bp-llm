Subject: Re: Supplementary Experiments — XGBoost Calibration & Semantic K-Ablation

Dear Dr Sun,

Thank you for the helpful suggestions. I have submitted two supplementary experiments to the HPC cluster and will share the results as soon as they complete (expected within a few hours):

1. **XGBoost few-shot calibration** — bias correction and fine-tuning on the first K chronological samples, for K ∈ {1, 3, 5, 10, 20}, under the same protocol as the LLM few-shot method.

2. **Semantic description K-ablation** — the rule-based semantic description method evaluated across K ∈ {1, 3, 5, 10, 20} to compare its calibration efficiency with numeric input.

In the meantime, let me clarify the difference between semantic description and bottleneck:

- **Semantic description (SensorLM-style)**: numerical features are converted into language descriptions that **retain quantitative values**, e.g. "moderate arterial stiffness (AIx 75%); normal vascular resistance (b/a −0.35)". The LLM still sees the numbers, but organised into clinically meaningful semantic layers. The idea is to make the physiologically decisive quantities more salient.

- **Semantic bottleneck (TimeSRL-style)**: the LLM first generates a purely qualitative summary from the features — e.g. "the subject has a normal heart rate and slightly elevated BP" — and then predicts BP from **only this summary, without access to any numbers**. This tests whether pure abstraction retains enough information for regression.

The key difference is **whether quantitative anchors are retained**. Our experiments so far show that retaining them (semantic description) improves calibration efficiency at equal K, whereas discarding them (bottleneck) causes a severe accuracy drop — suggesting that semantic abstraction must preserve quantitative information for precise BP regression.

I will send the full results once the experiments finish.

Best regards,
Boyuan Li
