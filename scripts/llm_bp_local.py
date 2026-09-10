"""
LLM-BP Local Inference on HPC A100
===================================
使用 Qwen3-8B-Instruct 本地推理，跑跨受试者 zero-shot + few-shot 消融。
零 API 费用，输出稳定（非推理模型）。

用法（在 GPU 节点上）:
  python llm_bp_local.py
"""

import sys, os, json, time, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/users/acp25bl/models/Qwen3-8B"
DATA_PATH = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
OUTPUT_PATH = "/users/acp25bl/bishe/llm_bp_local_results.json"

# ── 实验配置 ────────────────────────────────────────────────────────────
N_REF_SUBJECTS = 10          # zero-shot 参考受试者数
N_REF_SAMPLES_PER = 2        # 每参考受试者样本数
N_TEST_SUBJECTS = 20         # 测试受试者数
FEWSHOT_KS = [1, 3, 5, 10, 20]   # few-shot 校准样本数消融
N_TEST = 5                   # 每受试者测试段数
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


def format_demo(age, gender):
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return ""
    g = "male" if str(gender).upper().startswith("M") else "female"
    return f"(age {age:.0f}, {g})"


def build_zeroshot_prompt(feats, refs):
    ref_text = ""
    for i, (rf, rs, rd, ra, rg) in enumerate(refs):
        demo = format_demo(ra, rg)
        ref_text += (f"\nReference subject {i+1} {demo}:\n"
                     f"{format_features(rf)}\n  BP: {rs:.0f},{rd:.0f}\n")
    demo = format_demo(feats.get("_age"), feats.get("_gender"))
    return (f"{CLINICAL_KNOWLEDGE}\n"
            f"You are given PPG features from several DIFFERENT people "
            f"(reference subjects) with their measured blood pressure. "
            f"Learn the general relationship between PPG features and BP. "
            f"Then predict the BP of a NEW, unseen subject.\n"
            f"{ref_text}\n"
            f"New subject {demo}:\n{format_features(feats)}\n\n"
            f"Predict this new subject's BP. Reply ONLY: SBP=xxx, DBP=xxx")


def build_fewshot_prompt(feats, calib):
    ex = ""
    for i, (ef, es, ed) in enumerate(calib):
        ex += (f"\nSample {i+1}:\n{format_features(ef)}\n  BP: {es:.0f},{ed:.0f}\n")
    demo = format_demo(feats.get("_age"), feats.get("_gender"))
    return (f"{CLINICAL_KNOWLEDGE}\n"
            f"These are PPG measurements from the SAME person {demo}. "
            f"Predict their BP for the last sample.\n"
            f"Calibration samples:{ex}\n"
            f"New sample:\n{format_features(feats)}\n\n"
            f"Predict this person's BP. Reply ONLY: SBP=xxx, DBP=xxx")


def parse_bp(text):
    """从模型输出提取 SBP/DBP，优先匹配 SBP=/DBP= 标签。"""
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
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def main():
    print("=" * 70)
    print("LLM-BP LOCAL INFERENCE — Qwen3-8B on A100")
    print("=" * 70)

    # ── 加载数据 ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
    feature_names = list(d["feature_names"])
    subjects = d["subjects"]
    age = d["age"] if "age" in d else np.full(len(X), np.nan)
    gender = d["gender"] if "gender" in d else np.full(len(X), "unknown")
    print(f"  {len(X)} segments, {len(np.unique(subjects))} subjects")

    # ── 加载模型 ─────────────────────────────────────────────────────────
    print("\n[2/4] Loading Qwen3-8B-Instruct...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16)
    model = model.to("cuda")
    model.eval()
    print(f"  模型加载完成 ({time.time()-t0:.1f}s), device: {model.device}")

    # ── 受试者划分 ───────────────────────────────────────────────────────
    rng = np.random.RandomState(SEED)
    unique_subjects = np.unique(subjects)
    rng.shuffle(unique_subjects)
    n_test = min(N_TEST_SUBJECTS, len(unique_subjects) // 5)
    test_subjs = unique_subjects[:n_test]
    ref_subjs = unique_subjects[n_test:n_test + N_REF_SUBJECTS]
    print(f"\n  {n_test} test subjects, {N_REF_SUBJECTS} reference subjects")

    # 参考池（zero-shot 用，全部来自其他受试者）
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

    # ── 跑实验 ───────────────────────────────────────────────────────────
    print("\n[3/4] Running zero-shot (cross-subject)...", flush=True)
    zs_results = []
    for subj in test_subjs:
        idx = np.where(subjects == subj)[0]
        if len(idx) < max(FEWSHOT_KS) + N_TEST:
            continue
        # 测试段：前 max(K) 个之后的 N_TEST 个段（与 few-shot 一致，时间顺序）
        test_seg_idx = idx[max(FEWSHOT_KS):max(FEWSHOT_KS) + N_TEST]
        for ci in test_seg_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            feats["_age"] = float(age[ci]) if ci < len(age) else np.nan
            feats["_gender"] = str(gender[ci])
            refs = [ref_pool[r] for r in
                    rng.choice(len(ref_pool), min(len(ref_pool), N_REF_SUBJECTS * N_REF_SAMPLES_PER), replace=False)]
            prompt = build_zeroshot_prompt(feats, refs)
            resp = generate(model, tokenizer, prompt)
            ps, pd = parse_bp(resp)
            zs_results.append({"subj": subj, "tsbp": float(y_sbp[ci]),
                               "tdbp": float(y_dbp[ci]), "psbp": ps, "pdbp": pd,
                               "raw": resp})

    zs_valid = [r for r in zs_results if r["psbp"] is not None]
    if zs_valid:
        sbp_mae = np.mean([abs(r["psbp"] - r["tsbp"]) for r in zs_valid])
        dbp_mae = np.mean([abs(r["pdbp"] - r["tdbp"]) for r in zs_valid])
        print(f"  zero-shot: n={len(zs_valid)}/{len(zs_results)}, SBP MAE={sbp_mae:.1f}, DBP MAE={dbp_mae:.1f}")

    print("\n[4/4] Running few-shot ablation K =", FEWSHOT_KS, flush=True)
    fs_results = {k: [] for k in FEWSHOT_KS}
    for si, subj in enumerate(test_subjs):
        idx = np.where(subjects == subj)[0]
        if len(idx) < max(FEWSHOT_KS) + N_TEST:
            continue
        # 时间顺序：前 max(K) 个作为校准池，之后的 N_TEST 个作为测试段
        calib_pool_idx = idx[:max(FEWSHOT_KS)]
        test_idx = idx[max(FEWSHOT_KS):max(FEWSHOT_KS) + N_TEST]

        for ci in test_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            feats["_age"] = float(age[ci]) if ci < len(age) else np.nan
            feats["_gender"] = str(gender[ci])
            for K in FEWSHOT_KS:
                calib = []
                for cci in calib_pool_idx[:K]:
                    cf = {feature_names[i]: float(X[cci, i]) for i in range(len(feature_names))}
                    calib.append((cf, float(y_sbp[cci]), float(y_dbp[cci])))
                prompt = build_fewshot_prompt(feats, calib)
                resp = generate(model, tokenizer, prompt)
                ps, pd = parse_bp(resp)
                fs_results[K].append({"subj": subj, "tsbp": float(y_sbp[ci]),
                                      "tdbp": float(y_dbp[ci]), "psbp": ps, "pdbp": pd,
                                      "raw": resp})
        print(f"  subject {si+1}/{len(test_subjs)} ({subj}) done", flush=True)

    # ── 汇总 ─────────────────────────────────────────────────────────────
    summary = {"zero_shot": None, "few_shot": {}}
    if zs_valid:
        summary["zero_shot"] = {
            "n": len(zs_valid),
            "sbp_mae": round(float(np.mean([abs(r["psbp"] - r["tsbp"]) for r in zs_valid])), 2),
            "dbp_mae": round(float(np.mean([abs(r["pdbp"] - r["tdbp"]) for r in zs_valid])), 2),
        }
    for K in FEWSHOT_KS:
        valid = [r for r in fs_results[K] if r["psbp"] is not None]
        if valid:
            summary["few_shot"][str(K)] = {
                "n": len(valid),
                "sbp_mae": round(float(np.mean([abs(r["psbp"] - r["tsbp"]) for r in valid])), 2),
                "dbp_mae": round(float(np.mean([abs(r["pdbp"] - r["tdbp"]) for r in valid])), 2),
            }

    print("\n  ── Results ──")
    if summary["zero_shot"]:
        z = summary["zero_shot"]
        print(f"  zero-shot:  SBP MAE {z['sbp_mae']:.1f}, DBP MAE {z['dbp_mae']:.1f} (n={z['n']})")
    for K in FEWSHOT_KS:
        if str(K) in summary["few_shot"]:
            m = summary["few_shot"][str(K)]
            print(f"  few-shot K={K:<2d}: SBP MAE {m['sbp_mae']:.1f}, DBP MAE {m['dbp_mae']:.1f} (n={m['n']})")

    output = {
        "model": "Qwen/Qwen3-8B-Instruct",
        "config": {"n_test_subjects": n_test, "n_ref_subjects": N_REF_SUBJECTS,
                   "fewshot_ks": FEWSHOT_KS, "seed": SEED},
        "summary": summary,
        "zero_shot_results": zs_results,
        "few_shot_results": {str(k): v for k, v in fs_results.items()},
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
