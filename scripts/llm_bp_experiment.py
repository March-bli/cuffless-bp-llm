"""
LLM-based Blood Pressure Estimation Experiment
===============================================
实验设计:
  1. LLM 基线: 55维特征 → prompt → LLM → SBP/DBP
  2. LLM + 临床知识: 加入 AHA/ESC 参考范围
  3. LLM + few-shot 校准: 同一个人的前N段数据作为示例
  4. 对比基线: XGBoost (subject-level split)

对应导师建议: "explore how to use LLM as the core ML to infer BP"
"""

import sys; sys.path.insert(0, ".")
import os, json, time, re
import numpy as np
from datetime import datetime
import requests

# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_BASE = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"
SEED = 42

if not DEEPSEEK_API_KEY:
    raise RuntimeError("Set DEEPSEEK_API_KEY environment variable")
N_TEST_SUBJECTS = 20        # 测试多少受试者
N_SEGMENTS_PER_SUBJECT = 30  # 每个受试者最多取多少段
N_FEWSHOT = 5               # few-shot 示例数量
MAX_TOKENS = 50             # 输出 SBP,DBP 两个数字足够

np.random.seed(SEED)

# 临床参考范围 (AHA/ESC 简化版)
CLINICAL_RANGES_TEXT = """
Clinical Reference Ranges (AHA/ESC):
- Systolic BP: <120 normal, 120-129 elevated, 130-139 stage1, >=140 stage2
- Diastolic BP: <80 normal, 80-89 elevated, 90-99 stage1, >=100 stage2
- Heart Rate: 60-100 bpm normal, <60 bradycardia, >100 tachycardia
"""

# 特征说明 (帮助 LLM 理解每个特征的含义)
FEATURE_DESCRIPTIONS = {
    "heart_rate": "heart rate (bpm)",
    "pulse_interval_mean": "mean pulse interval (ms)",
    "pulse_interval_std": "SD of pulse intervals (ms)",
    "sys_amp_mean": "mean systolic amplitude",
    "pulse_amp_mean": "mean pulse amplitude",
    "aix": "augmentation index (%)",
    "stiffness_index": "large artery stiffness index",
    "upstroke_time_mean": "mean systolic upstroke time (ms)",
    "upstroke_time_std": "SD of upstroke time",
    "diastolic_time_mean": "mean diastolic time (ms)",
    "pulse_width_75pct": "pulse width at 75% height (ms)",
    "pulse_width_50pct": "pulse width at 50% height (ms)",
    "pulse_width_25pct": "pulse width at 25% height (ms)",
    "apg_a_mean": "APG a-wave mean (early systolic)",
    "apg_b_mean": "APG b-wave mean (early systolic negative)",
    "apg_c_mean": "APG c-wave mean (late systolic)",
    "apg_d_mean": "APG d-wave mean (early diastolic)",
    "apg_e_mean": "APG e-wave mean (late diastolic)",
    "apg_b_a_ratio": "APG b/a ratio",
    "apg_c_a_ratio": "APG c/a ratio",
    "apg_d_a_ratio": "APG d/a ratio",
    "apg_e_a_ratio": "APG e/a ratio",
    "ppg_mean": "PPG signal mean",
    "ppg_std": "PPG signal std",
    "ppg_skew": "PPG signal skewness",
    "ppg_kurtosis": "PPG signal kurtosis",
    "ppg_entropy": "PPG signal entropy",
    "spectral_energy": "total spectral energy",
    "hr_band_power": "power in 0.5-5 Hz band",
    "dominant_freq": "dominant frequency (Hz)",
    "lf_hf_ratio": "low/high frequency ratio",
}

# ═══════════════════════════════════════════════════
# 1. Load Data
# ═══════════════════════════════════════════════════
print("=" * 70)
print("LLM-BP EXPERIMENT")
print("=" * 70)

print("\n[1/5] Loading feature data...")
d = np.load("data/processed/pulsedb_features_proper.npz", allow_pickle=True)
X_all = d["X"]
y_sbp_all = d["y_sbp"]
y_dbp_all = d["y_dbp"]
subjects_all = d["subjects"]
feature_names = list(d["feature_names"])

# Build feature lookup dict
feat_idx = {name: i for i, name in enumerate(feature_names)}

print(f"  {len(y_sbp_all)} segments from {len(set(subjects_all))} subjects")
print(f"  {len(feature_names)} features")

# ═══════════════════════════════════════════════════
# 2. Subject-level Split
# ═══════════════════════════════════════════════════
print(f"\n[2/5] Subject-level split...")

unique_subjects = np.unique(subjects_all)
rng = np.random.RandomState(SEED)
rng.shuffle(unique_subjects)

n_train = int(len(unique_subjects) * 0.75)
train_subjs = set(unique_subjects[:n_train])
test_subjs = set(unique_subjects[n_train:n_train + N_TEST_SUBJECTS])

print(f"  Train subjects: {len(train_subjs)}, Test subjects: {N_TEST_SUBJECTS}")

# ═══════════════════════════════════════════════════
# 3. Build Prompts
# ═══════════════════════════════════════════════════
print(f"\n[3/5] Building prompts...")

def format_features(feat_dict, top_n=20):
    """Format top features into a readable text block."""
    # Sort by importance (use SHAP importance for better selection)
    # For now, take a fixed set of top features
    top_features = [
        "heart_rate", "pulse_interval_mean", "apg_a_mean", "apg_c_mean",
        "apg_d_mean", "aix", "stiffness_index", "pulse_width_75pct",
        "upstroke_time_mean", "diastolic_time_mean", "ppg_skew",
        "ppg_kurtosis", "hr_band_power", "dominant_freq", "spectral_energy",
        "sys_amp_mean", "pulse_amp_mean", "apg_b_a_ratio", "apg_c_a_ratio",
        "vpg_mean_abs"
    ]
    
    lines = []
    for feat in top_features:
        if feat in feat_dict:
            desc = FEATURE_DESCRIPTIONS.get(feat, feat)
            lines.append(f"  {desc}: {feat_dict[feat]:.3f}")
    return "\n".join(lines)


def build_prompt_baseline(feat_dict):
    """Baseline: just features → BP"""
    features_text = format_features(feat_dict)
    return f"""Given these PPG signal features, predict the systolic (SBP) and diastolic (DBP) blood pressure in mmHg.

Features:
{features_text}

Output ONLY two numbers separated by a comma: SBP,DBP
Example: 120,80

Prediction:"""


def build_prompt_clinical(feat_dict):
    """With clinical knowledge"""
    features_text = format_features(feat_dict)
    return f"""{CLINICAL_RANGES_TEXT}

Given these PPG signal features, predict the systolic (SBP) and diastolic (DBP) blood pressure in mmHg.

Features:
{features_text}

Output ONLY two numbers separated by a comma: SBP,DBP

Prediction:"""


def build_prompt_fewshot(feat_dict, calibration_examples):
    """Few-shot: include same person's other segments as examples"""
    example_text = ""
    for i, (ex_feats, ex_sbp, ex_dbp) in enumerate(calibration_examples):
        example_text += f"\nExample {i+1}:\n{format_features(ex_feats)}\n  → SBP={ex_sbp:.0f}, DBP={ex_dbp:.0f}\n"
    
    features_text = format_features(feat_dict)
    return f"""Given PPG signal features from the SAME PERSON as the examples above, predict their blood pressure.

Examples from this person:
{example_text}

Now predict for this new measurement from the same person:
{features_text}

Output ONLY two numbers: SBP,DBP

Prediction:"""


def build_prompt_cot(feat_dict):
    """Chain-of-thought: reason first, then predict"""
    features_text = format_features(feat_dict)
    return f"""{CLINICAL_RANGES_TEXT}

Given these PPG signal features, analyze the cardiovascular indicators step by step,
then predict systolic (SBP) and diastolic (DBP) blood pressure.

Features:
{features_text}

Step 1: Analyze heart rate and arterial stiffness indicators.
Step 2: Assess pulse morphology (amplitudes, widths, ratios).
Step 3: Consider frequency-domain characteristics.
Step 4: Based on the above, estimate SBP and DBP.

Output your analysis then END with: SBP=X, DBP=Y"""


# ═══════════════════════════════════════════════════
# 4. Call LLM API
# ═══════════════════════════════════════════════════
print(f"\n[4/5] Calling LLM API...")

def call_llm(prompt, max_tokens=MAX_TOKENS, max_retries=3):
    """Call DeepSeek API and extract SBP/DBP numbers."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise blood pressure estimation model. "
             "Respond ONLY with numbers. Never add explanations unless asked."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,  # low temp for regression
        "max_tokens": max_tokens,
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{API_BASE}/chat/completions",
                headers=headers, json=payload, timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            
            # Extract SBP and DBP from response
            # Accept formats: "120,80" or "SBP=120, DBP=80" or "120/80"
            numbers = re.findall(r'(\d{2,3})', text)
            if len(numbers) >= 2:
                # Take first two numbers that look like BP values
                candidates = [int(n) for n in numbers[:4] if 50 < int(n) < 250]
                if len(candidates) >= 2:
                    return candidates[0], candidates[1], text
            
            # Last resort: take any two numbers
            if len(numbers) >= 2:
                return int(numbers[0]), int(numbers[1]), text
                
            return None, None, text
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                return None, None, str(e)


# ═══════════════════════════════════════════════════
# 5. Run Experiment
# ═══════════════════════════════════════════════════
print(f"\n[5/5] Running experiments...")

def get_subject_segments(subject, max_n=N_SEGMENTS_PER_SUBJECT):
    """Get all segments for a specific subject."""
    mask = subjects_all == subject
    indices = np.where(mask)[0]
    if len(indices) > max_n:
        indices = rng.choice(indices, max_n, replace=False)
    return indices


def experiment_baseline(test_subj, n_per_subj=5):
    """Test LLM baseline on random segments from test subjects."""
    results = []
    for subj in list(test_subjs)[:test_subj]:
        seg_idx = get_subject_segments(subj, n_per_subj)
        for idx in seg_idx[:n_per_subj]:
            feats = {feature_names[i]: float(X_all[idx, i]) for i in range(len(feature_names))}
            true_sbp = float(y_sbp_all[idx])
            true_dbp = float(y_dbp_all[idx])
            
            prompt = build_prompt_baseline(feats)
            pred_sbp, pred_dbp, raw = call_llm(prompt)
            
            results.append({
                "subject": subj,
                "true_sbp": true_sbp,
                "true_dbp": true_dbp,
                "pred_sbp": pred_sbp,
                "pred_dbp": pred_dbp,
            })
            
            if pred_sbp:
                print(f"  {subj}: true=({true_sbp:.0f},{true_dbp:.0f}) "
                      f"pred=({pred_sbp},{pred_dbp}) "
                      f"err=({abs(pred_sbp-true_sbp):.0f},{abs(pred_dbp-true_dbp):.0f})")
    
    return results


def experiment_fewshot(test_subj, n_calib=N_FEWSHOT, n_test=5):
    """Test LLM with few-shot calibration on test subjects."""
    results = []
    for subj in list(test_subjs)[:test_subj]:
        seg_idx = get_subject_segments(subj, n_calib + n_test)
        
        # First n_calib segments = calibration set
        calib_examples = []
        for idx in seg_idx[:n_calib]:
            feats = {feature_names[i]: float(X_all[idx, i]) for i in range(len(feature_names))}
            calib_examples.append((feats, float(y_sbp_all[idx]), float(y_dbp_all[idx])))
        
        # Remaining segments = test
        for idx in seg_idx[n_calib:n_calib + n_test]:
            feats = {feature_names[i]: float(X_all[idx, i]) for i in range(len(feature_names))}
            true_sbp = float(y_sbp_all[idx])
            true_dbp = float(y_dbp_all[idx])
            
            prompt = build_prompt_fewshot(feats, calib_examples)
            pred_sbp, pred_dbp, raw = call_llm(prompt, max_tokens=50)
            
            results.append({
                "subject": subj,
                "n_calib": n_calib,
                "true_sbp": true_sbp,
                "true_dbp": true_dbp,
                "pred_sbp": pred_sbp,
                "pred_dbp": pred_dbp,
            })
            
            if pred_sbp:
                print(f"  {subj} (calib={n_calib}): true=({true_sbp:.0f},{true_dbp:.0f}) "
                      f"pred=({pred_sbp},{pred_dbp}) "
                      f"err=({abs(pred_sbp-true_sbp):.0f},{abs(pred_dbp-true_dbp):.0f})")
    
    return results

# Run both experiments
print("\n--- BASELINE: Zero-shot LLM ---")
t0 = time.time()
baseline_results = experiment_baseline(5, n_per_subj=5)
print(f"  Time: {time.time()-t0:.0f}s, {len(baseline_results)} predictions")

print("\n--- FEW-SHOT: 5 calibration segments ---")
t0 = time.time()
fewshot_results = experiment_fewshot(5, n_calib=5, n_test=5)
print(f"  Time: {time.time()-t0:.0f}s, {len(fewshot_results)} predictions")

# ═══════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

def compute_metrics(results, label):
    valid = [r for r in results if r["pred_sbp"] is not None]
    if not valid:
        print(f"\n  {label}: NO VALID PREDICTIONS")
        return None
    
    sbp_errs = [abs(r["pred_sbp"] - r["true_sbp"]) for r in valid]
    dbp_errs = [abs(r["pred_dbp"] - r["true_dbp"]) for r in valid]
    
    m = {
        "n": len(valid),
        "sbp_mae": np.mean(sbp_errs),
        "dbp_mae": np.mean(dbp_errs),
        "sbp_rmse": np.sqrt(np.mean(np.array(sbp_errs)**2)),
        "dbp_rmse": np.sqrt(np.mean(np.array(dbp_errs)**2)),
    }
    
    print(f"\n  {label} (n={m['n']}):")
    print(f"    SBP MAE: {m['sbp_mae']:.1f} mmHg, RMSE: {m['sbp_rmse']:.1f} mmHg")
    print(f"    DBP MAE: {m['dbp_mae']:.1f} mmHg, RMSE: {m['dbp_rmse']:.1f} mmHg")
    
    return m

baseline_metrics = compute_metrics(baseline_results, "LLM Zero-shot")
fewshot_metrics = compute_metrics(fewshot_results, "LLM Few-shot (5 calib)")

# Reference: XGBoost
print(f"\n  --- Reference Baselines ---")
print(f"  XGBoost (5-fold CV mixed):   SBP=7.2, DBP=4.5  (data leakage, NOT reliable)")
print(f"  XGBoost (subject-level split): SBP=20.6, DBP=11.7  (true generalization)")
print(f"  CNN1D (subject-level):         SBP=16.6, DBP=9.1")
print(f"  Liu et al. (LLM-BP, 2024):    SBP=9.3, DBP=6.4  (LLaMA3-8B, 31 features)")

# Save
output = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL,
    "config": {
        "n_test_subjects": N_TEST_SUBJECTS,
        "n_segments_per_subject": N_SEGMENTS_PER_SUBJECT,
        "n_fewshot": N_FEWSHOT,
    },
    "baseline": baseline_results,
    "fewshot": fewshot_results,
    "baseline_metrics": {"sbp_mae": baseline_metrics["sbp_mae"] if baseline_metrics else None,
                          "dbp_mae": baseline_metrics["dbp_mae"] if baseline_metrics else None},
    "fewshot_metrics": {"sbp_mae": fewshot_metrics["sbp_mae"] if fewshot_metrics else None,
                         "dbp_mae": fewshot_metrics["dbp_mae"] if fewshot_metrics else None},
}

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/llm_bp_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to data/processed/llm_bp_results.json")
