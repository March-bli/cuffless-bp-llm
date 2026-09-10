Subject: Re: Progress update & a couple of questions

Hi Shaoxiong,

Thanks for the feedback last time — that was really helpful. I've
pivoted as you suggested and got some initial results. Wanted to
run them by you before going further.

So instead of using LLM just to generate health reports, I redesigned
the pipeline so the LLM directly predicts blood pressure from PPG
features. Did three approaches:

1. Zero-shot: just give features to DeepSeek, ask for BP
   → basically useless, the model ignores the features and guesses
     the population average regardless of input

2. Few-shot calibration: for each test subject, I give the LLM 5
   example segments from that same person (features + actual BP),
   then ask it to predict the remaining segments
   → SBP MAE = 8.9, DBP MAE = 4.3 mmHg (BHS Grade A)

3. Fine-tuned Qwen2.5-0.5B locally with QLoRA
   → SBP 18.5, DBP 14.2 — too small a model, but it does learn
     something (better than zero-shot at least)

For reference:
  - XGBoost (subject-level split): SBP 20.6, DBP 11.7
  - CNN1D (subject-level split):  SBP 16.6, DBP 9.1
  - Liu et al. 2024 (LLM-BP):     SBP 9.3,  DBP 6.4

Quick note on data — these are just preliminary tests on a small
subset (20 test subjects, 5 calibration + 5 test segments each).
The full PulseDB dataset has ~5,300 subjects. Once the approach
is validated I'll scale up to the complete dataset and do proper
k-fold cross-validation.

The few-shot approach seems to work — the LLM actually learns the
person's BP pattern from just 5 calibration segments, and the DBP
result beats the existing LLM-BP paper. The cross-subject
generalization problem doesn't go away, but few-shot calibration
is a practical way around it.

Does this direction look reasonable for the dissertation? I'm
thinking the story would be: traditional ML fails at cross-subject
generalization for PPG-based BP → LLM with few-shot calibration
achieves clinical-grade accuracy by adapting to each individual.

If this looks good, I'll:
- Run ablation experiments (different shot counts, feature subsets)
- Try chain-of-thought reasoning
- Expand the comparative analysis

Let me know what you think.

Cheers,
Boyuan
