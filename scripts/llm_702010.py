"""LLM 70/20/10：516 测试人的 K=20（数值 + 语义），Qwen3-8B"""
import sys, os, json, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/users/acp25bl/bishe")
from semantic_describe import build_semantic_description

MODEL_PATH = "/users/acp25bl/models/Qwen3-8B"
DATA_PATH = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
OUTPUT_PATH = "/users/acp25bl/bishe/llm_702010_results.json"

SEED = 42
K = 20
N_TEST = 5

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

CLINICAL_KNOWLEDGE = (
    "Clinical knowledge (AHA/ESC):\n"
    "- SBP <120 normal, 120-129 elevated, 130-139 stage1 HTN, >=140 stage2 HTN\n"
    "- DBP <80 normal, 80-89 elevated, >=90 HTN\n"
    "- Higher augmentation index -> stiffer arteries -> higher SBP\n"
    "- Higher APG b/a ratio -> more vascular resistance -> higher DBP\n"
)


def format_features(feats):
    lines = []
    for f in FEATURE_DESCRIPTIONS:
        if f in feats:
            lines.append(f"  {FEATURE_DESCRIPTIONS[f]}: {feats[f]:.3f}")
    return "\n".join(lines)


def build_numeric_prompt(feats, calib):
    ex = ""
    for i, (ef, es, ed) in enumerate(calib):
        ex += f"\nSample {i+1}:\n{format_features(ef)}\n  BP: {es:.0f},{ed:.0f}\n"
    return (f"{CLINICAL_KNOWLEDGE}\n"
            f"These are PPG measurements from the SAME person. Predict their BP for the last sample.\n"
            f"Calibration samples:{ex}\n"
            f"New sample:\n{format_features(feats)}\n\n"
            f"Predict this person's BP. Reply ONLY: SBP=xxx, DBP=xxx")


def build_semantic_prompt(sem, calib):
    ex = ""
    for i, (es, esbp, edbp) in enumerate(calib):
        ex += f"\nSample {i+1}:\n{es}\n  BP: {esbp:.0f},{edbp:.0f}\n"
    return (f"{CLINICAL_KNOWLEDGE}\n"
            f"These are PPG descriptions from the SAME person. Predict their BP for the last sample.\n"
            f"Calibration samples:{ex}\n"
            f"New sample:\n{sem}\n\n"
            f"Predict this person's BP. Reply ONLY: SBP=xxx, DBP=xxx")


def parse_bp(text):
    m = re.search(r"SBP\s*[=:]\s*(\d{2,3})", text, re.IGNORECASE)
    d = re.search(r"DBP\s*[=:]\s*(\d{2,3})", text, re.IGNORECASE)
    if m and d:
        sbp = int(m.group(1)); dbp = int(d.group(1))
        if 50 < sbp < 250 and 50 < dbp < 250:
            return sbp, dbp
    nums = re.findall(r"\d{2,3}", text)
    valid = [int(n) for n in nums if 50 < int(n) < 250]
    if len(valid) >= 2:
        return valid[-2], valid[-1]
    return None, None


@torch.no_grad()
def generate(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    print("=" * 70)
    print("LLM 70/20/10: 516 test subjects, K=20 (numeric + semantic)")
    print("=" * 70)

    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
    feature_names = list(d["feature_names"])
    subjects = d["subjects"]
    print(f"[1/3] Data: {len(X)} segments, {len(np.unique(subjects))} subjects", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    print("[2/3] Model loaded (8B)", flush=True)

    rng = np.random.RandomState(SEED)
    unique_subjects = np.unique(subjects)
    rng.shuffle(unique_subjects)
    n = len(unique_subjects)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    test_subjs = unique_subjects[n_train + n_val:]
    print(f"[3/3] test subjects: {len(test_subjs)}", flush=True)

    numeric_results = []
    semantic_results = []

    for si, subj in enumerate(test_subjs):
        idx = np.where(subjects == subj)[0]
        if len(idx) < K + N_TEST:
            continue
        calib_idx = idx[:K]
        test_idx = idx[K:K + N_TEST]

        calib_num = []
        calib_sem = []
        for ci in calib_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            calib_num.append((feats, float(y_sbp[ci]), float(y_dbp[ci])))
            calib_sem.append((build_semantic_description(feats), float(y_sbp[ci]), float(y_dbp[ci])))

        for ci in test_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            sem = build_semantic_description(feats)

            r1 = generate(model, tokenizer, build_numeric_prompt(feats, calib_num))
            ps1, pd1 = parse_bp(r1)
            numeric_results.append({"tsbp": float(y_sbp[ci]), "tdbp": float(y_dbp[ci]),
                                   "psbp": ps1, "pdbp": pd1})

            r2 = generate(model, tokenizer, build_semantic_prompt(sem, calib_sem))
            ps2, pd2 = parse_bp(r2)
            semantic_results.append({"tsbp": float(y_sbp[ci]), "tdbp": float(y_dbp[ci]),
                                     "psbp": ps2, "pdbp": pd2})

        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{len(test_subjs)} 完成", flush=True)

    summary = {}
    for name, res in [("numeric", numeric_results), ("semantic", semantic_results)]:
        for target in ["sbp", "dbp"]:
            valid = [(r[f"p{target}"], r[f"t{target}"]) for r in res
                     if r[f"p{target}"] is not None]
            if valid:
                errs = [abs(p - t) for p, t in valid]
                errs_arr = np.array(errs)
                summary[f"{name}_{target}"] = {
                    "n": len(valid),
                    "mae": round(float(np.mean(errs_arr)), 2),
                    "pct_le5": round(float(np.sum(errs_arr <= 5) / len(valid) * 100), 1),
                    "pct_le10": round(float(np.sum(errs_arr <= 10) / len(valid) * 100), 1),
                    "pct_le15": round(float(np.sum(errs_arr <= 15) / len(valid) * 100), 1),
                }
                print(f"  {name} {target}: {summary[f'{name}_{target}']}", flush=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"split": {"test": len(test_subjs)}, "summary": summary,
                   "numeric_results": numeric_results, "semantic_results": semantic_results},
                  f, indent=2, default=str)
    print(f"Saved to {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
