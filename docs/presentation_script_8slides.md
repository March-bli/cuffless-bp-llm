# 8 页 PPT 展示讲稿（15 分钟：约 11 分钟讲 + 4 分钟问答）

---

## 第 1 页：标题（30 秒）

> "Good morning. I'm Boyuan Li, and my project is **novel blood pressure monitoring using wearable devices and artificial intelligence**, supervised by Dr. Shaoxiong Sun."

---

## 第 2 页：为什么需要无袖带 BP（1 分钟）

> "Hypertension is the leading modifiable risk factor for cardiovascular disease. The problem is that cuff-based measurement is **intermittent and inconvenient** — you only get occasional snapshots, and you miss night-time and daily fluctuations."

> "The opportunity is PPG — the optical signal already in smartwatches for heart rate. If we can estimate BP from PPG, we get **continuous, cuffless, everyday monitoring**. The goal is clinical-grade accuracy."

---

## 第 3 页：现有方案 + LLM 动机（1.5 分钟）

> "Existing approaches have limits. PTT-based methods need both ECG and PPG, and frequent recalibration. Traditional ML works within a subject but **degrades across subjects**."

> "Why LLMs? Because they have **in-context learning** — they adapt from a few examples without fine-tuning. They're pre-trained with broad reasoning ability. Recent work like SensorLM and TimeSRL shows LLMs can handle physiological signals."

> "My idea is to use this in-context learning for **personalised calibration** — adapt the model to an individual with just a few of their own samples."

---

## 第 4 页：数据集（1 分钟）

> "I use PulseDB — a large, cleaned dataset built from MIMIC-III and VitalDB. **5,159 subjects, 5.55 million ten-second PPG segments at 125 Hz**, with ECG and arterial BP as reference."

> "From each segment I extract **55 hand-crafted features** plus age and gender. Everything uses a **strict subject-level split** — 20 test subjects, the rest for training. Calibration uses the **first K chronological segments** of the target subject."

---

## 第 5 页：瓶颈论证（2 分钟，重要）

> "Here's the core evidence. XGBoost, trained on **5,139 subjects**, still gets an SBP error of 14.9 mmHg. So the problem is **not data, and not model capacity**."

> "But look — **bias correction**, which only adds the individual's mean offset using K=20 of their own samples, brings the error down to **7.0**. This tells us the key factor is the **individual offset** — the bottleneck is individual physiology."

> "And against the BHS standard, XGBoost global **fails all grades** — only 30% of SBP errors are within 5 mmHg, far below even Grade C's 40%."

---

## 第 6 页：方法（2 分钟）

> "So I compare two LLM approaches. Both follow the same pipeline: PPG → 55 features → Qwen3-8B → BP prediction."

> "**Approach one, numeric few-shot**: feed raw numbers as in-context examples — the LLM learns the individual mapping from the first K samples."

> "**Approach two, my method, semantic description**: translate the features into **clinical language that keeps the numbers** — 'moderate arterial stiffness, AIx 75%' instead of 'AIx = 75%'. The idea is to let the LLM *understand* the physiology, not just see numbers."

---

## 第 7 页：结果（2.5 分钟，核心）

> "Here are the results at K=20. XGBoost global is 14.9. Bias correction is 7.0 — a strong baseline. **LLM numeric is 8.4 — actually worse than bias correction.**"

*(停顿 1 秒)*

> "But **semantic description achieves 6.5 for SBP and 3.6 for DBP — the best of all methods.**"

> "Against the full BHS protocol: **DBP meets Grade A** — 76% of errors within 5 mmHg, 90% within 10, 98.5% within 15. SBP meets Grade B."

> "The key message: **the LLM's value is not in seeing numbers, but in understanding semantics.**"

---

## 第 8 页：结论（1 分钟）

> "Three findings. One — cross-subject generalisation is the real bottleneck, and the individual offset is the key. Two — **the LLM needs no fine-tuning**: it's pre-trained and ready to use, unlike XGBoost trained on five thousand people. Three — **semantic description is interpretable and accurate**: transparent clinical language, not a black box, achieving DBP Grade A."

> "Future work: reduce calibration cost, enable on-device deployment, and learn the semantic rules automatically. Thank you — happy to take questions."

---

## 问答准备（4 分钟）

**Q: "Why is numeric few-shot worse than bias correction?"**
> "Bias correction only learns a single offset from a strong global model, so K samples are enough. Numeric few-shot asks the LLM to learn the entire individual mapping from scratch — with few samples it underfits."

**Q: "What exactly is semantic description?"**
> "A rule-based translation of numerical features into clinical language that retains the numbers. For example, AIx above 80% becomes 'high arterial stiffness (AIx 80%)'. It tells the model *which* features matter physiologically."

**Q: "Why only 20 test subjects?"**
> "Twenty is a standard held-out set with a fixed seed, fully disjoint from training. LLM inference is expensive. Core experiments were run twice with identical results, confirming reproducibility."

**Q: "Why Qwen3-8B and not bigger?"**
> "It runs locally at zero cost, is reproducible, and our finding shows calibration data matters more than model scale. A larger 32B model helps numeric few-shot but not semantic — supporting the semantic-abstraction argument."

**Q: "Is 6.5/3.6 clinically meaningful?"**
> "DBP meets the full BHS Grade A cumulative criteria (76/90/98.5% within 5/10/15 mmHg); SBP meets Grade B. It's the only method in our comparison that reaches a BHS grade for SBP."
