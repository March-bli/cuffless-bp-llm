Subject: Progress update & a couple of questions

Hi Shaoxiong,

Just wanted to update you on where I'm at and ask a couple things
before I go too far in the wrong direction.

So I've built a full pipeline that takes raw PPG signals, cleans them
up, extracts features, and uses ML to predict blood pressure. Then it
feeds the results through clinical guidelines (AHA/ESC) and an LLM to
generate a readable health summary — so instead of just numbers like
"120/80", you get something like "your heart rate is a bit elevated,
might want to cut back on caffeine".

Now here's the interesting part — I found a pretty significant problem
and I think it's actually the real contribution of the project:

When I first trained XGBoost with just random segment-level 5-fold CV,
the results looked great: SBP MAE ~7.2, DBP MAE ~4.5, DBP BHS Grade A.
But then I realised this was mixing data from the same person across
train and test folds, so the model was essentially cheating.

When I switched to a strict subject-level split (train on person A, test
on person B), the performance collapsed: SBP MAE jumped to ~20.6, DBP
to ~11.7 — basically random guessing. The model just couldn't generalize
across different people.

So I pivoted to a calibration-based approach:
- Train a base model on many subjects
- For each new person, fine-tune the model with just a small amount of
  their own data (5-50 segments)
- Then predict on the rest of their unseen data

The calibration helps a lot. With just 5 calibration segments, SBP MAE
drops from ~17.9 to ~14.5. I'm still running more experiments on this
but the direction seems solid.

Also tried a few other things — stacking ensemble brings DBP MAE to 4.25
(BHS A), and I added age/gender as input features which helps a bit for
the deep learning side. The LLM health report module is working too.

**Two things I'm not sure about:**

1. Does this direction make sense? Like, the core story being "existing
   ML approaches fail at cross-subject generalization for PPG-based BP
   estimation, and a simple calibration fine-tuning approach solves it"
   — is that enough of a contribution for the dissertation?

2. About the hardware — I originally planned to build an ESP32-C3 + PPG
   sensor wearable board, but honestly the dataset is good enough for the
   algorithm work. Feels like hardware might eat up time without adding
   much. You think it's okay to leave that as future work?

Oh and I've got the thesis outline done (8 chapters) and a draft going,
so the writing side isn't starting from zero. The literature review
covers 8 papers including Liu et al. (BCB 2024) which is the only other
work on LLM-based cuffless BP, and PulseDB which is our dataset.

Let me know what you think whenever you get a chance.

Cheers,
Boyuan
