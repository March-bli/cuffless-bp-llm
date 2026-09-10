"""
LLM-BP Experiment v2 — Full Scale
==================================
4 prompt variants × 20 subjects × few-shot calibration
"""

import sys; sys.path.insert(0, ".")
import os, json, time, re
import numpy as np
from datetime import datetime
import requests

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_BASE = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"
SEED = 42
N_TEST_SUBJECTS = 20
N_SEGMENTS = 30
N_FEWSHOT = 5
N_TEST = 5

np.random.seed(SEED)

FEATURE_DESCRIPTIONS = {
    "heart_rate":"HR(bpm)","pulse_interval_mean":"interval(ms)","aix":"augmentation_idx(%)",
    "stiffness_index":"stiffness","upstroke_time_mean":"upstroke(ms)","diastolic_time_mean":"diastole(ms)",
    "pulse_width_75pct":"PW75%(ms)","pulse_width_50pct":"PW50%(ms)","pulse_width_25pct":"PW25%(ms)",
    "apg_a_mean":"APG_a","apg_b_mean":"APG_b","apg_c_mean":"APG_c","apg_d_mean":"APG_d","apg_e_mean":"APG_e",
    "apg_b_a_ratio":"b/a","apg_c_a_ratio":"c/a","apg_d_a_ratio":"d/a","apg_e_a_ratio":"e/a",
    "sys_amp_mean":"sys_amp","pulse_amp_mean":"pulse_amp","vpg_mean_abs":"VPG_amp",
    "ppg_skew":"skew","ppg_kurtosis":"kurtosis","hr_band_power":"HR_power",
    "dominant_freq":"dom_freq(Hz)","spectral_energy":"spec_energy","lf_hf_ratio":"LF/HF",
}

TOP_FEATURES = list(FEATURE_DESCRIPTIONS.keys())

print("=" * 70)
print("LLM-BP EXPERIMENT v2 — Full Scale")
print("=" * 70)

print("\n[1/4] Loading data...")
d = np.load("data/processed/pulsedb_features_proper.npz", allow_pickle=True)
X_all = d["X"]; y_sbp_all = d["y_sbp"]; y_dbp_all = d["y_dbp"]
subjects_all = d["subjects"]; feature_names = list(d["feature_names"])
feat_idx = {name: i for i, name in enumerate(feature_names)}

unique_subjects = np.unique(subjects_all)
rng = np.random.RandomState(SEED); rng.shuffle(unique_subjects)
test_subjs = set(unique_subjects[-N_TEST_SUBJECTS:])
print(f"  {len(y_sbp_all)} segments, {len(test_subjs)} test subjects")

# ─── Build prompts ─────────────────────────────────
CLINICAL_KNOWLEDGE = """
Clinical knowledge (AHA/ESC):
- SBP <120 normal, 120-129 elevated, 130-139 stage1 HTN, >=140 stage2 HTN
- DBP <80 normal, 80-89 elevated, >=90 HTN
- HR 60-100 normal
- Augmentation index reflects arterial stiffness — higher = stiffer arteries = higher SBP
- APG b/a ratio reflects vascular resistance — higher = more resistance = higher DBP
- Upstroke time correlates with cardiac contractility
"""

def format_features(feats):
    lines = []
    for f in TOP_FEATURES:
        if f in feats:
            lines.append(f"  {FEATURE_DESCRIPTIONS[f]}: {feats[f]:.3f}")
    return "\n".join(lines)

def build_prompt(mode, feats, calib=None):
    ftext = format_features(feats)
    
    if mode == "baseline":
        return f"""Given these PPG features from a person, predict their systolic (SBP) and diastolic (DBP) blood pressure in mmHg.

{ftext}

Output ONLY: SBP,DBP"""

    elif mode == "clinical":
        return f"""{CLINICAL_KNOWLEDGE}

Given these PPG signal features from a person, apply the clinical knowledge above to predict their blood pressure.

Features:
{ftext}

Analyze each feature against the reference ranges, then predict.
Output ONLY: SBP,DBP"""

    elif mode == "fewshot":
        ex = ""
        for i, (ef, es, ed) in enumerate(calib):
            ex += f"\nSample {i+1}:\n{format_features(ef)}\n  BP: {es:.0f},{ed:.0f}\n"
        return f"""These are PPG measurements from the SAME person. Predict their BP for the last sample.

Calibration samples:{ex}
New sample:
{ftext}

Output ONLY: SBP,DBP"""

    elif mode == "fewshot_clinical":
        ex = ""
        for i, (ef, es, ed) in enumerate(calib):
            ex += f"\nSample {i+1}:\n{format_features(ef)}\n  BP: {es:.0f},{ed:.0f}\n"
        return f"""{CLINICAL_KNOWLEDGE}

These are PPG measurements from the SAME person. Use clinical knowledge and calibration samples to predict BP.

Calibration samples:{ex}
New sample:
{ftext}

Output ONLY: SBP,DBP"""

    elif mode == "cot":
        return f"""{CLINICAL_KNOWLEDGE}

Given PPG features, reason through the cardiovascular indicators step by step, then predict BP.

Features:
{ftext}

Step 1: Heart rate assessment.
Step 2: Arterial stiffness (AIx, upstroke time, stiffness index).
Step 3: Vascular resistance (APG ratios, pulse width).
Step 4: Overall BP estimation.

End with FINAL: SBP=X, DBP=Y"""

def call_llm(prompt, max_retries=3):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json={
                "model": MODEL, "temperature": 0.1, "max_tokens": 80,
                "messages": [{"role": "user", "content": prompt}],
            }, timeout=30)
            text = resp.json()["choices"][0]["message"]["content"]
            nums = re.findall(r'\d{2,3}', text)
            # Return last two numbers that look like BP values
            valid = [int(n) for n in nums if 50 < int(n) < 250]
            if len(valid) >= 2:
                return valid[-2], valid[-1], text
            elif len(valid) == 1:
                return valid[0], None, text
            return None, None, text
        except Exception as e:
            if attempt == max_retries - 1:
                return None, None, str(e)
            time.sleep(2**attempt)

# ─── Run experiment ───────────────────────────────
print(f"\n[2/4] Running 4 variants on {N_TEST_SUBJECTS} subjects...")
print("      (baseline, clinical, fewshot, fewshot_clinical)")

all_results = {mode: [] for mode in ["baseline", "clinical", "fewshot", "fewshot_clinical"]}

for subj in sorted(test_subjs):
    mask = subjects_all == subj
    indices = np.where(mask)[0]
    if len(indices) > N_SEGMENTS:
        indices = rng.choice(indices, N_SEGMENTS, replace=False)
    
    print(f"\n  [{subj}] {len(indices)} segments")
    
    # Get calibration set and test set
    perm = rng.permutation(len(indices))
    calib_idx = indices[perm[:N_FEWSHOT]]
    test_idx = indices[perm[N_FEWSHOT:N_FEWSHOT + N_TEST]]
    
    # Build calibration examples (shared across fewshot variants)
    calib_examples = []
    for ci in calib_idx:
        feats = {feature_names[i]: float(X_all[ci, i]) for i in range(len(feature_names))}
        calib_examples.append((feats, float(y_sbp_all[ci]), float(y_dbp_all[ci])))
    
    for ti in test_idx:
        feats = {feature_names[i]: float(X_all[ti, i]) for i in range(len(feature_names))}
        tsbp, tdbp = float(y_sbp_all[ti]), float(y_dbp_all[ti])
        
        # Baseline
        if len(all_results["baseline"]) < N_TEST_SUBJECTS * N_TEST:
            time.sleep(0.3)
            ps, pd, _ = call_llm(build_prompt("baseline", feats))
            all_results["baseline"].append({"subj": subj, "tsbp": tsbp, "tdbp": tdbp, "psbp": ps, "pdbp": pd})
        
        # Clinical
        time.sleep(0.3)
        ps, pd, _ = call_llm(build_prompt("clinical", feats))
        all_results["clinical"].append({"subj": subj, "tsbp": tsbp, "tdbp": tdbp, "psbp": ps, "pdbp": pd})
        
        # Fewshot
        time.sleep(0.3)
        ps, pd, _ = call_llm(build_prompt("fewshot", feats, calib_examples))
        all_results["fewshot"].append({"subj": subj, "tsbp": tsbp, "tdbp": tdbp, "psbp": ps, "pdbp": pd, "n_calib": N_FEWSHOT})
        
        # Fewshot + Clinical
        time.sleep(0.3)
        ps, pd, _ = call_llm(build_prompt("fewshot_clinical", feats, calib_examples))
        all_results["fewshot_clinical"].append({"subj": subj, "tsbp": tsbp, "tdbp": tdbp, "psbp": ps, "pdbp": pd, "n_calib": N_FEWSHOT})

# ─── Evaluate ──────────────────────────────────────
print("\n[3/4] Computing metrics...")

def metrics(results, label):
    valid = [r for r in results if r["psbp"] is not None and r["pdbp"] is not None]
    if not valid:
        return None
    se = [abs(r["psbp"] - r["tsbp"]) for r in valid]
    de = [abs(r["pdbp"] - r["tdbp"]) for r in valid]
    m = {"n": len(valid), "sbp_mae": np.mean(se), "dbp_mae": np.mean(de)}
    return m

print("\n  ── Results Summary ──")
print(f"  {'Method':<20s} {'n':>4s} {'SBP MAE':>8s} {'DBP MAE':>8s}")
print("  " + "-" * 42)
for mode in ["baseline", "clinical", "fewshot", "fewshot_clinical"]:
    m = metrics(all_results[mode], mode)
    if m:
        print(f"  {mode:<20s} {m['n']:>4d} {m['sbp_mae']:>8.1f} {m['dbp_mae']:>8.1f}")

print(f"\n  ── Baselines ──")
print(f"  XGBoost (subject-split): SBP 20.6, DBP 11.7")
print(f"  CNN1D  (subject-split):  SBP 16.6, DBP 9.1")
print(f"  Liu et al. 2024:          SBP 9.3,  DBP 6.4")

# ─── Save ─────────────────────────────────────────
print("\n[4/4] Saving...")
summary = {}
for mode in ["baseline", "clinical", "fewshot", "fewshot_clinical"]:
    m = metrics(all_results[mode], mode)
    if m:
        summary[mode] = {"n": m["n"], "sbp_mae": round(m["sbp_mae"], 1), "dbp_mae": round(m["dbp_mae"], 1)}

output = {"timestamp": datetime.now().isoformat(), "model": MODEL,"summary": summary,"results": all_results}
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/llm_bp_results_v2.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"  Saved to data/processed/llm_bp_results_v2.json")
print("\nDone!")
