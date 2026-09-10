Subject: Re: Conventional ML K-Calibration Results Added

Dear Dr Sun,

Thank you for the suggestion — this was very helpful. I have now run the K-calibration experiments using conventional ML methods (XGBoost) under the same protocol (first K chronological samples, 20 test subjects).

Two conventional approaches were evaluated:

1. **Bias correction**: global XGBoost predictions corrected by the mean residual on the K calibration samples;
2. **Fine-tuning**: global XGBoost further trained on the K calibration samples.

The results, compared with the LLM few-shot method, are as follows (SBP/DBP MAE in mmHg):

| K | LLM few-shot | ML bias correction | ML fine-tuning |
|---|-------------|--------------------|----------------|
| 1 | 25.7 / 18.8 | 12.7 / 6.0 | 13.8 / 9.0 |
| 3 | 22.1 / 15.4 | 12.9 / 6.0 | 12.0 / 7.1 |
| 5 | 17.1 / 10.9 | 12.5 / 5.7 | 12.0 / 7.2 |
| 10 | 14.0 / 7.8 | 11.1 / 5.3 | 10.7 / 6.6 |
| 20 | 7.7 / 5.2 | 9.2 / 4.4 | 9.2 / 5.7 |

Two observations from this comparison:

1. With small K (1–10), conventional ML calibration is more stable, since the global model already captures population patterns and only needs a small residual correction. The LLM requires more samples to learn the individual's mapping from scratch.

2. With K=20, the LLM outperforms conventional ML on SBP (7.7 vs 9.2), although conventional bias correction remains slightly better on DBP (4.4 vs 5.2). Notably, the semantic-description approach (K=5) achieves 6.5 / 3.5, outperforming all conventional ML calibration methods while requiring only a quarter of the calibration samples.

I think this comparison makes the dissertation's argument more balanced and credible. I am updating the relevant chapters accordingly.

Thank you again for the guidance.

Best regards,
Boyuan Li
