# 15-Minute Presentation Script (5 Slides)

**Total: ~11 min presentation + 4 min Q&A**

---

## Slide 1 — Title (30 seconds)

> "Good morning. My name is Boyuan Li. My project is **novel blood pressure monitoring using wearable devices and artificial intelligence**, supervised by Dr. Shaoxiong Sun."

*(Pause 2–3 seconds. Let the marker read the title. Don't rush.)*

---

## Slide 2 — Background: The Cross-Subject Generalisation Problem (2.5 min)

> "Cuffless blood pressure from wrist PPG has been a goal for over a decade, but it has a fundamental problem: **cross-subject generalisation**. The same PPG waveform can mean a different blood pressure in different people."

*(Point to the right box.)*

> "Look at the number. Under a strict **subject-level split** — where the test person is completely unseen — XGBoost gets an SBP error of **16.1 mmHg**. The BHS Grade A standard for clinical use requires **no more than 5 mmHg**. That's over three times worse."

*(Pause.)*

> "The problem is not signal quality, and not model size — it is **individual physiology**. Each person's PPG-to-BP mapping is different."

> "So my research question is this: **can a large language model adapt to an individual using just a few of that person's own calibration samples?**"

> "All experiments use the complete PulseDB dataset — 5,159 subjects, 5.5 million PPG segments, with strict subject-level splits."

---

## Slide 3 — Method: LLM Few-Shot Personalised Calibration (2.5 min)

> "Here is the pipeline. PPG signal at 125 Hz → 55 hand-crafted morphological features, plus age and gender → into a Qwen3-8B model."

*(Point to the left box.)*

> "I compare two calibration approaches. **Left, numeric few-shot**: I give the model the **first K samples** of the target person — their features and true BP — as in-context examples. The model then predicts for a new sample from the same person. K goes from 1 to 20."

> "Why the first K, not random? Because in reality, calibration data arrives chronologically — you can only collect it in time order."

*(Point to the right box.)*

> "**Right, my method — semantic description.** Instead of raw numbers, I translate the features into **clinical language that keeps the numbers**. For example, rather than 'AIx = 75%', the model sees 'moderate arterial stiffness, AIx 75%'. The numbers are still there, but organised in the way a clinician would describe them."

> "I also tested a semantic bottleneck — a purely qualitative summary with no numbers — to understand what happens when you over-abstract."

---

## Slide 4 — Results: Semantic Description Wins (4 min, the core)

*(Point to the left figure.)*

> "First, the calibration curves. More calibration samples help — the error comes down monotonically from K=1 to K=20."

*(Point to the right table. Slow down here.)*

> "Now the key comparison at K=20. XGBoost global is 16.1 for SBP. **Bias correction** — a strong conventional baseline that simply adds the subject's mean residual — brings it down to 7.0."

*(Pause 1 second.)*

> "**LLM numeric few-shot is 8.4 — actually worse than bias correction.** This is a surprising and important negative result. Just showing the LLM numbers is not enough."

*(Pause 2 seconds. Let this sink in.)*

> "But when I combine the LLM with **semantic description**, it achieves **SBP 6.5 and DBP 3.6 — the best of all methods, and DBP meets the BHS Grade A standard of ≤5 mmHg.**"

*(Pause.)*

> "So what does this tell us? **The LLM's advantage is not in seeing numbers — it's in understanding semantics.** Mapping features into clinical language lets the model focus on the physiological essence: arterial stiffness, vascular resistance. That is where the LLM adds value."

> "The semantic bottleneck — the purely qualitative version — collapses to 25/23, which confirms that the quantitative anchors matter. Over-abstraction loses the information needed for precise regression."

---

## Slide 5 — Conclusions (1.5 min)

> "Three findings. **First**, cross-subject generalisation is the real bottleneck — not signal quality, not model size. **Second**, conventional bias correction is a strong baseline that beats naive LLM numeric few-shot. **Third, and most important: the LLM's value lies in semantic abstraction — understanding signals, not just seeing numbers.**"

> "Future work: reduce the calibration cost, enable on-device deployment, and learn the semantic rules automatically instead of hand-crafting them."

> "Thank you — I'm happy to answer any questions."

---

## Q&A Preparation (4 min)

### Likely questions and your answers:

**Q: "Why is numeric few-shot worse than bias correction?"**
> "Because the LLM is learning the individual mapping from scratch with only a few examples, while bias correction only needs to learn a single offset from a strong global model. The global model already captures population patterns, so the correction is sample-efficient."

**Q: "What exactly is semantic description?"**
> "It's a rule-based mapping from numerical features to clinical language that retains the numbers. For example, AIx above 80% becomes 'high arterial stiffness (AIx 80%)'. The semantic layer tells the model *which* features matter physiologically."

**Q: "Why only 20 test subjects?"**
> "Twenty subjects is a standard held-out set, randomly selected with a fixed seed, completely disjoint from training. LLM inference is expensive, so a focused test set was used; results were verified across two independent runs with identical outcomes."

**Q: "Is 6.5/3.6 clinically meaningful?"**
> "DBP 3.6 meets BHS Grade A (≤5 mmHg); SBP 6.5 meets Grade B (≤10 mmHg). It's the best result in our comparison and close to clinical usability for DBP."

**Q: "Why Qwen3-8B and not a bigger model?"**
> "It runs locally on an A100 at zero API cost, and it's reproducible. Interestingly, our earlier experiments showed an 8B model with more calibration samples can approach a 671B API model — calibration data matters more than model scale."

**Q: "What would you do differently?"**
> "Learn the semantic rules rather than hand-craft them, reduce calibration cost through opportunistic measurements, and evaluate on more subjects."

---

## Delivery Tips

1. **Slide 4 is the climax** — spend most of your energy there; the two pauses around "8.4 worse than bias correction" are deliberate.
2. **Slow down on numbers** — 16.1, 7.0, 8.4, 6.5, 3.6. These are your story.
3. **Point at the slide** when referring to figures/tables.
4. **Don't read** — the script above is to be spoken naturally, not memorised word-for-word.
5. If you forget a number, the table is on the slide — glance at it.
