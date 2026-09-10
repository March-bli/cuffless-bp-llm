Subject: Re: Progress Update — Slides and K clarification

Dear Dr Sun,

Thank you very much for your feedback. Please find the slides attached with all technical details and findings.

Regarding your question about K: yes, K refers to the number of calibration samples from the target subject. Each calibration sample consists of the subject's 55 PPG features paired with their true SBP/DBP. For K=10, the LLM is given 10 such (features, BP) pairs as in-context examples, then predicts the BP for a new segment from the same subject.

One clarification: the K samples are **randomly selected** from the subject's available segments, not necessarily the "first K" measurements in chronological order. This random selection was used to avoid any temporal bias.

The key results are summarised in the slides (pages 6–9):
- Few-shot calibration with K=10 achieves SBP MAE 8.3 / DBP MAE 4.1 mmHg, with DBP meeting BHS Grade A.
- An 8B local model with K=10 matches a 671B model — calibration sample size matters more than model scale.

I would be grateful for any further feedback.

Best regards,
Boyuan Li
