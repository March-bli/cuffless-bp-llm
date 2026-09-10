"""
LLM-BP Semantic Abstraction Experiment
======================================
借鉴 SensorLM：把 PPG 数值特征替换为层次化语义描述，跑 few-shot 对比。
对比：纯数值 few-shot vs 语义描述 few-shot。

用法（GPU 节点）: python llm_bp_semantic.py
"""

import sys, os, json, time, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/users/acp25bl/bishe")
from semantic_describe import build_semantic_description

MODEL_PATH = "/users/acp25bl/models/Qwen3-8B"
DATA_PATH = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
OUTPUT_PATH = "/users/acp25bl/bishe/llm_bp_semantic_results.json"

N_TEST_SUBJECTS = 20
FEWSHOT_K = 5
N_TEST = 5
SEED = 42

CLINICAL_KNOWLEDGE = (
    "Clinical knowledge (AHA/ESC):\n"
    "- SBP <120 normal, 120-129 elevated, 130-139 stage1 HTN, >=140 stage2 HTN\n"
    "- DBP <80 normal, 80-89 elevated, >=90 HTN\n"
    "- Higher augmentation index -> stiffer arteries -> higher SBP\n"
    "- Higher APG b/a ratio -> more vascular resistance -> higher DBP\n"
)


def build_semantic_fewshot_prompt(sem_desc, calib):
    ex = ""
    for i, (es, esbp, edbp) in enumerate(calib):
        ex += f"\nSample {i+1}:\n{es}\n  BP: {esbp:.0f},{edbp:.0f}\n"
    return (f"{CLINICAL_KNOWLEDGE}\n"
            f"These are PPG descriptions from the SAME person. "
            f"Predict their BP for the last sample.\n"
            f"Calibration samples:{ex}\n"
            f"New sample:\n{sem_desc}\n\n"
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
    print("LLM-BP SEMANTIC ABSTRACTION — SensorLM-inspired")
    print("=" * 70)

    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
    feature_names = list(d["feature_names"])
    subjects = d["subjects"]
    print(f"[1/3] Data: {len(X)} segments, {len(np.unique(subjects))} subjects")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    print("[2/3] Model loaded")

    rng = np.random.RandomState(SEED)
    unique_subjects = np.unique(subjects)
    rng.shuffle(unique_subjects)
    test_subjs = unique_subjects[:N_TEST_SUBJECTS]
    print(f"[3/3] Running semantic few-shot K={FEWSHOT_K}...", flush=True)

    results = []
    for si, subj in enumerate(test_subjs):
        idx = np.where(subjects == subj)[0]
        if len(idx) < FEWSHOT_K + N_TEST:
            continue
        # 时间顺序：前 K 个作为校准，之后的 N_TEST 个作为测试
        calib_idx = idx[:FEWSHOT_K]
        test_idx = idx[FEWSHOT_K:FEWSHOT_K + N_TEST]

        # 生成语义描述（校准 + 测试）
        calib = []
        for ci in calib_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            sem = build_semantic_description(feats)
            calib.append((sem, float(y_sbp[ci]), float(y_dbp[ci])))

        for ci in test_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            sem = build_semantic_description(feats)
            prompt = build_semantic_fewshot_prompt(sem, calib)
            resp = generate(model, tokenizer, prompt)
            ps, pd = parse_bp(resp)
            results.append({"subj": subj, "tsbp": float(y_sbp[ci]),
                            "tdbp": float(y_dbp[ci]), "psbp": ps, "pdbp": pd,
                            "raw": resp})
        print(f"  subject {si+1}/{len(test_subjs)} done", flush=True)

    valid = [r for r in results if r["psbp"] is not None]
    if valid:
        sbp_mae = np.mean([abs(r["psbp"] - r["tsbp"]) for r in valid])
        dbp_mae = np.mean([abs(r["pdbp"] - r["tdbp"]) for r in valid])
        print(f"\n  semantic few-shot K={FEWSHOT_K}: SBP MAE {sbp_mae:.1f}, DBP MAE {dbp_mae:.1f} (n={len(valid)}/{len(results)})")
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
