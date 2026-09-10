"""
LLM-BP Semantic Bottleneck (TimeSRL-style two-stage)
=====================================================
阶段一（抽象）：LLM 输入 55 维数值特征 → 生成 PPG 语义摘要
阶段二（推断）：few-shot K=5，样本用 LLM 语义摘要（不给数值）→ 预测 BP

研究问题：语义瓶颈是否保留了足够的 BP 预测信息？
用法（GPU 节点）: python llm_bp_bottleneck.py
"""

import sys, os, json, time, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/users/acp25bl/models/Qwen3-8B"
DATA_PATH = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
OUTPUT_PATH = "/users/acp25bl/bishe/llm_bp_bottleneck_results.json"

N_TEST_SUBJECTS = 20
FEWSHOT_K = 5
N_TEST = 5
SEED = 42

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


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def build_abstract_prompt(feats):
    """阶段一：生成语义摘要。"""
    return (f"{CLINICAL_KNOWLEDGE}\n"
            f"Describe this person's cardiovascular state based on their PPG features. "
            f"Include: heart rate level, arterial stiffness, vascular resistance, and "
            f"a tentative BP range estimate. Reply in 2-3 sentences.\n\n"
            f"{format_features(feats)}")


def build_predict_prompt(abstract, calib):
    """阶段二：仅凭语义摘要做 few-shot 预测。"""
    ex = ""
    for i, (a, es, ed) in enumerate(calib):
        ex += f"\nSample {i+1} description: {a}\n  BP: {es:.0f},{ed:.0f}\n"
    return (f"{CLINICAL_KNOWLEDGE}\n"
            f"These are cardiovascular descriptions from the SAME person. "
            f"Predict their BP for the last sample.\n"
            f"Calibration samples:{ex}\n"
            f"New sample description: {abstract}\n\n"
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


def main():
    print("=" * 70)
    print("LLM-BP SEMANTIC BOTTLENECK — TimeSRL-inspired two-stage")
    print("=" * 70)

    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
    feature_names = list(d["feature_names"])
    subjects = d["subjects"]
    print(f"[1/4] Data: {len(X)} segments, {len(np.unique(subjects))} subjects")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    print("[2/4] Model loaded")

    rng = np.random.RandomState(SEED)
    unique_subjects = np.unique(subjects)
    rng.shuffle(unique_subjects)
    test_subjs = unique_subjects[:N_TEST_SUBJECTS]
    print(f"[3/4] Running two-stage bottleneck (K={FEWSHOT_K})...", flush=True)

    results = []
    for si, subj in enumerate(test_subjs):
        idx = np.where(subjects == subj)[0]
        if len(idx) < FEWSHOT_K + N_TEST:
            continue
        # 时间顺序：前 K 个作为校准，之后的 N_TEST 个作为测试
        calib_idx = idx[:FEWSHOT_K]
        test_idx = idx[FEWSHOT_K:FEWSHOT_K + N_TEST]

        # 阶段一：对校准样本和测试样本生成 LLM 语义摘要
        calib = []
        for ci in calib_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            abstract = generate(model, tokenizer, build_abstract_prompt(feats))
            calib.append((abstract, float(y_sbp[ci]), float(y_dbp[ci])))

        for ci in test_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            abstract = generate(model, tokenizer, build_abstract_prompt(feats))
            # 阶段二：仅凭摘要预测
            prompt = build_predict_prompt(abstract, calib)
            resp = generate(model, tokenizer, prompt, max_new_tokens=64)
            ps, pd = parse_bp(resp)
            results.append({"subj": subj, "tsbp": float(y_sbp[ci]),
                            "tdbp": float(y_dbp[ci]), "psbp": ps, "pdbp": pd,
                            "abstract": abstract, "raw": resp})
        print(f"  subject {si+1}/{len(test_subjs)} done", flush=True)

    valid = [r for r in results if r["psbp"] is not None]
    if valid:
        sbp_mae = np.mean([abs(r["psbp"] - r["tsbp"]) for r in valid])
        dbp_mae = np.mean([abs(r["pdbp"] - r["tdbp"]) for r in valid])
        print(f"\n  bottleneck K={FEWSHOT_K}: SBP MAE {sbp_mae:.1f}, DBP MAE {dbp_mae:.1f} (n={len(valid)}/{len(results)})")
        summary = {"n": len(valid), "sbp_mae": round(float(sbp_mae), 2),
                   "dbp_mae": round(float(dbp_mae), 2)}
    else:
        summary = {"n": 0}

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)
    print(f"  Saved to {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
