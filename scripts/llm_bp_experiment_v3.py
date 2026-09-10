"""
LLM-BP Experiment v3 — Aligned with supervisor feedback
========================================================
Key changes vs v2:
  1. Zero-shot now uses CROSS-SUBJECT in-context references
     (features + BP from OTHER subjects) + demographic info.
  2. Few-shot calibration ablation over K = 1/3/5/10/20.
  3. Uses full dataset with subject-level split.

Supervisor guidance:
  - "Zero shot should be that you at least give the model the features
     and BP from other subjects, and then you apply it on the unseen subject."
  - "But you better use demographic information if you can."
"""

import sys, os, json, time, re
import numpy as np
from datetime import datetime
import requests

sys.path.insert(0, "/home/lby/projects/bishe")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_BASE = "https://api.deepseek.com/v1"
MODEL = "deepseek-v4-pro"
SEED = 42

# ── experiment config ────────────────────────────────────────────────────
N_REF_SUBJECTS = 10          # zero-shot: how many OTHER subjects to show
N_REF_SAMPLES_PER = 2        # zero-shot: samples per reference subject
N_TEST_SUBJECTS = 50         # number of unseen test subjects
N_SEGMENTS_PER_SUBJECT = 20  # max segments per subject
FEWSHOT_KS = [1, 3, 5, 10, 20]  # calibration-size ablation
N_TEST = 5                   # test segments per subject

FEATURE_DESCRIPTIONS = {
    "heart_rate": "HR(bpm)", "pulse_interval_mean": "interval(ms)",
    "aix": "augmentation_idx(%)", "stiffness_index": "stiffness",
    "upstroke_time_mean": "upstroke(ms)", "diastolic_time_mean": "diastole(ms)",
    "pulse_width_75pct": "PW75%(ms)", "pulse_width_50pct": "PW50%(ms)",
    "pulse_width_25pct": "PW25%(ms)",
    "apg_a_mean": "APG_a", "apg_b_mean": "APG_b", "apg_c_mean": "APG_c",
    "apg_d_mean": "APG_d", "apg_e_mean": "APG_e",
    "apg_b_a_ratio": "b/a", "apg_c_a_ratio": "c/a", "apg_d_a_ratio": "d/a",
    "apg_e_a_ratio": "e/a",
    "sys_amp_mean": "sys_amp", "pulse_amp_mean": "pulse_amp",
    "vpg_mean_abs": "VPG_amp", "ppg_skew": "skew", "ppg_kurtosis": "kurtosis",
    "hr_band_power": "HR_power", "dominant_freq": "dom_freq(Hz)",
    "spectral_energy": "spec_energy", "lf_hf_ratio": "LF/HF",
}

CLINICAL_KNOWLEDGE = """
Clinical knowledge (AHA/ESC):
- SBP <120 normal, 120-129 elevated, 130-139 stage1 HTN, >=140 stage2 HTN
- DBP <80 normal, 80-89 elevated, >=90 HTN
- Augmentation index reflects arterial stiffness — higher = stiffer arteries = higher SBP
- APG b/a ratio reflects vascular resistance — higher = more resistance = higher DBP
- Upstroke time correlates with cardiac contractility
"""


def format_features(feats, feature_names):
    lines = []
    for f in FEATURE_DESCRIPTIONS:
        if f in feats:
            lines.append(f"  {FEATURE_DESCRIPTIONS[f]}: {feats[f]:.3f}")
    return "\n".join(lines)


def format_demographics(age, gender):
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return ""
    g = "male" if str(gender).upper().startswith("M") else "female"
    return f"(age {age:.0f}, {g})"


def build_zeroshot_prompt(feats, refs, feature_names):
    """Cross-subject zero-shot: references are OTHER subjects."""
    ref_text = ""
    for i, (rf, rs, rd, ra, rg) in enumerate(refs):
        demo = format_demographics(ra, rg)
        ref_text += (f"\nReference subject {i+1} {demo}:\n"
                     f"{format_features(rf, feature_names)}\n"
                     f"  BP: {rs:.0f},{rd:.0f}\n")

    demo = format_demographics(feats.get("_age"), feats.get("_gender"))
    return f"""{CLINICAL_KNOWLEDGE}

You are given PPG features from several DIFFERENT people (reference subjects) with their measured blood pressure. Use these to learn the general relationship between PPG features and blood pressure. Then predict the BP of a NEW, unseen subject.

{ref_text}
New subject {demo}:
{format_features(feats, feature_names)}

Based on the patterns across the reference subjects, predict this new subject's BP.
Output ONLY: SBP,DBP"""


def build_fewshot_prompt(feats, calib, feature_names):
    """Same-subject few-shot calibration."""
    ex = ""
    for i, (ef, es, ed) in enumerate(calib):
        ex += (f"\nSample {i+1}:\n{format_features(ef, feature_names)}\n"
               f"  BP: {es:.0f},{ed:.0f}\n")
    demo = format_demographics(feats.get("_age"), feats.get("_gender"))
    return f"""{CLINICAL_KNOWLEDGE}

These are PPG measurements from the SAME person {demo}. Predict their BP for the last sample.

Calibration samples:{ex}
New sample:
{format_features(feats, feature_names)}

Output ONLY: SBP,DBP"""


def call_llm(prompt, max_retries=3):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
               "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{API_BASE}/chat/completions", headers=headers,
                json={"model": MODEL, "temperature": 0.1, "max_tokens": 4000,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60)
            msg = resp.json()["choices"][0]["message"]
            text = msg.get("content") or ""
            if not text.strip():
                text = msg.get("reasoning_content") or ""
            nums = re.findall(r'\d{2,3}', text)
            valid = [int(n) for n in nums if 50 < int(n) < 250]
            if len(valid) >= 2:
                return valid[-2], valid[-1]
            elif len(valid) == 1:
                return valid[0], None
            return None, None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None, None


def main():
    print("=" * 70)
    print("LLM-BP EXPERIMENT v3 — Supervisor-aligned")
    print("=" * 70)

    # ── load data ────────────────────────────────────────────────────────
    print("\n[1/5] Loading feature dataset...")
    d = np.load("data/processed/pulsedb_features_proper.npz", allow_pickle=True)
    X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
    feature_names = list(d["feature_names"])
    subjects = d["subjects"] if "subjects" in d else np.arange(len(X))
    age = d["age"] if "age" in d else np.full(len(X), np.nan)
    gender = d["gender"] if "gender" in d else np.full(len(X), "unknown")

    print(f"  {len(X)} segments, {len(np.unique(subjects))} subjects")

    # ── subject-level split ──────────────────────────────────────────────
    rng = np.random.RandomState(SEED)
    unique_subjects = np.unique(subjects)
    rng.shuffle(unique_subjects)
    n_test = min(N_TEST_SUBJECTS, len(unique_subjects) // 5)
    test_subjs = unique_subjects[:n_test]
    ref_subjs = unique_subjects[n_test:n_test + N_REF_SUBJECTS]
    print(f"  {n_test} test subjects, {N_REF_SUBJECTS} reference subjects")

    # ── build reference pool (for zero-shot) ─────────────────────────────
    ref_pool = []
    for subj in ref_subjs:
        idx = np.where(subjects == subj)[0]
        if len(idx) < N_REF_SAMPLES_PER:
            continue
        chosen = rng.choice(idx, N_REF_SAMPLES_PER, replace=False)
        for ci in chosen:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            feats["_age"] = float(age[ci]) if ci < len(age) else np.nan
            feats["_gender"] = str(gender[ci])
            ref_pool.append((feats, float(y_sbp[ci]), float(y_dbp[ci]),
                             float(age[ci]) if ci < len(age) else np.nan,
                             str(gender[ci])))
    print(f"  reference pool: {len(ref_pool)} samples")

    # ── run experiments ──────────────────────────────────────────────────
    print("\n[2/5] Running zero-shot (cross-subject)...")
    zs_results = []
    for subj in test_subjs:
        idx = np.where(subjects == subj)[0]
        if len(idx) < N_TEST + 1:
            continue
        chosen = rng.choice(idx, N_TEST, replace=False)
        for ci in chosen:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            feats["_age"] = float(age[ci]) if ci < len(age) else np.nan
            feats["_gender"] = str(gender[ci])
            # sample references from the pool (other subjects only)
            refs = rng.choice(len(ref_pool), min(len(ref_pool), N_REF_SUBJECTS * N_REF_SAMPLES_PER),
                              replace=False)
            sel_refs = [ref_pool[r] for r in refs]
            prompt = build_zeroshot_prompt(feats, sel_refs, feature_names)
            ps, pd = call_llm(prompt)
            zs_results.append({"subj": subj, "tsbp": float(y_sbp[ci]),
                               "tdbp": float(y_dbp[ci]), "psbp": ps, "pdbp": pd})
            time.sleep(0.2)

    zs_valid = [r for r in zs_results if r["psbp"] is not None and r["pdbp"] is not None]
    if zs_valid:
        print(f"  zero-shot: n={len(zs_valid)}, "
              f"SBP MAE={np.mean([abs(r['psbp']-r['tsbp']) for r in zs_valid]):.1f}, "
              f"DBP MAE={np.mean([abs(r['pdbp']-r['tdbp']) for r in zs_valid]):.1f}")

    print("\n[3/5] Running few-shot ablation K =", FEWSHOT_KS)
    fs_results = {k: [] for k in FEWSHOT_KS}
    for subj in test_subjs:
        idx = np.where(subjects == subj)[0]
        if len(idx) < max(FEWSHOT_KS) + N_TEST:
            continue
        perm = rng.permutation(len(idx))
        calib_pool_idx = idx[perm[:max(FEWSHOT_KS)]]
        test_idx = idx[perm[max(FEWSHOT_KS):max(FEWSHOT_KS) + N_TEST]]

        for ci in test_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            feats["_age"] = float(age[ci]) if ci < len(age) else np.nan
            feats["_gender"] = str(gender[ci])

            for K in FEWSHOT_KS:
                calib = []
                for cci in calib_pool_idx[:K]:
                    cf = {feature_names[i]: float(X[cci, i]) for i in range(len(feature_names))}
                    calib.append((cf, float(y_sbp[cci]), float(y_dbp[cci])))
                prompt = build_fewshot_prompt(feats, calib, feature_names)
                ps, pd = call_llm(prompt)
                fs_results[K].append({"subj": subj, "tsbp": float(y_sbp[ci]),
                                      "tdbp": float(y_dbp[ci]), "psbp": ps, "pdbp": pd})
                time.sleep(0.2)
        print(f"  subject {subj} done", flush=True)

    print("\n[4/5] Computing metrics...")
    summary = {"zero_shot": None, "few_shot": {}}
    if zs_valid:
        summary["zero_shot"] = {
            "n": len(zs_valid),
            "sbp_mae": round(float(np.mean([abs(r['psbp'] - r['tsbp']) for r in zs_valid])), 2),
            "dbp_mae": round(float(np.mean([abs(r['pdbp'] - r['tdbp']) for r in zs_valid])), 2),
        }
    for K in FEWSHOT_KS:
        valid = [r for r in fs_results[K] if r["psbp"] is not None and r["pdbp"] is not None]
        if valid:
            summary["few_shot"][str(K)] = {
                "n": len(valid),
                "sbp_mae": round(float(np.mean([abs(r['psbp'] - r['tsbp']) for r in valid])), 2),
                "dbp_mae": round(float(np.mean([abs(r['pdbp'] - r['tdbp']) for r in valid])), 2),
            }

    print("\n  ── Results ──")
    if summary["zero_shot"]:
        z = summary["zero_shot"]
        print(f"  zero-shot (cross-subject):  SBP MAE {z['sbp_mae']:.1f}, DBP MAE {z['dbp_mae']:.1f} (n={z['n']})")
    for K in FEWSHOT_KS:
        if str(K) in summary["few_shot"]:
            m = summary["few_shot"][str(K)]
            print(f"  few-shot K={K:<2d}:             SBP MAE {m['sbp_mae']:.1f}, DBP MAE {m['dbp_mae']:.1f} (n={m['n']})")

    print("\n[5/5] Saving...")
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "config": {"n_test_subjects": n_test, "n_ref_subjects": N_REF_SUBJECTS,
                   "fewshot_ks": FEWSHOT_KS, "seed": SEED},
        "summary": summary,
        "zero_shot_results": zs_results,
        "few_shot_results": {str(k): v for k, v in fs_results.items()},
    }
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/llm_bp_results_v3.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("  Saved to data/processed/llm_bp_results_v3.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
