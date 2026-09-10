"""语义描述 K 消融实验（完整数据，SensorLM 式）
K ∈ {1, 3, 5, 10, 20}，与数值 few-shot 对比校准效率。
"""
import sys, os, json, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/users/acp25bl/bishe")
from semantic_describe import build_semantic_description

MODEL_PATH = "/users/acp25bl/models/Qwen3-8B"
DATA_PATH = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
OUTPUT_PATH = "/users/acp25bl/bishe/semantic_ablation_results.json"

N_TEST_SUBJECTS = 20
FEWSHOT_KS = [1, 3, 5, 10, 20]
N_TEST = 5
SEED = 42

CLINICAL_KNOWLEDGE = (
    "Clinical knowledge (AHA/ESC):\n"
    "- SBP <120 normal, 120-129 elevated, 130-139 stage1 HTN, >=140 stage2 HTN\n"
    "- DBP <80 normal, 80-89 elevated, >=90 HTN\n"
    "- Higher augmentation index -> stiffer arteries -> higher SBP\n"
    "- Higher APG b/a ratio -> more vascular resistance -> higher DBP\n"
)


def build_prompt(sem_desc, calib):
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
    print("SEMANTIC DESCRIPTION K-ABLATION (full PulseDB)")
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
    print(f"[3/3] Running K-ablation {FEWSHOT_KS}...", flush=True)

    results = {k: [] for k in FEWSHOT_KS}
    MAX_K = max(FEWSHOT_KS)

    for si, subj in enumerate(test_subjs):
        idx = np.where(subjects == subj)[0]
        if len(idx) < MAX_K + N_TEST:
            continue
        calib_idx = idx[:MAX_K]      # 前 MAX_K 个（时间顺序）
        test_idx = idx[MAX_K:MAX_K + N_TEST]

        # 预生成所有校准样本的语义描述
        sem_calib_all = []
        for ci in calib_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            sem_calib_all.append((build_semantic_description(feats),
                                  float(y_sbp[ci]), float(y_dbp[ci])))

        for ci in test_idx:
            feats = {feature_names[i]: float(X[ci, i]) for i in range(len(feature_names))}
            sem_test = build_semantic_description(feats)
            for K in FEWSHOT_KS:
                prompt = build_prompt(sem_test, sem_calib_all[:K])
                resp = generate(model, tokenizer, prompt)
                ps, pd = parse_bp(resp)
                results[K].append({"subj": subj, "tsbp": float(y_sbp[ci]),
                                   "tdbp": float(y_dbp[ci]), "psbp": ps, "pdbp": pd})
        print(f"  subject {si+1}/{len(test_subjs)} done", flush=True)

    summary = {}
    for K in FEWSHOT_KS:
        valid = [r for r in results[K] if r["psbp"] is not None]
        if valid:
            summary[str(K)] = {
                "n": len(valid),
                "sbp_mae": round(float(np.mean([abs(r["psbp"] - r["tsbp"]) for r in valid])), 2),
                "dbp_mae": round(float(np.mean([abs(r["pdbp"] - r["tdbp"]) for r in valid])), 2),
            }
            print(f"  semantic K={K:<2d}: SBP {summary[str(K)]['sbp_mae']:.1f}, "
                  f"DBP {summary[str(K)]['dbp_mae']:.1f} (n={len(valid)})", flush=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)
    print(f"Saved to {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
