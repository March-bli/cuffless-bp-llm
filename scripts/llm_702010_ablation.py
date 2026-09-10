"""LLM 70/20/10 K 消融：K=1/3/5/10（K=20 已有），数值+语义，Qwen3-8B"""
import sys, os, json, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/users/acp25bl/bishe")
from semantic_describe import build_semantic_description

MODEL_PATH = "/users/acp25bl/models/Qwen3-8B"
DATA_PATH = "/mnt/parscratch/users/acp25bl/processed/pulsedb_features_full.npz"
OUTPUT_PATH = "/users/acp25bl/bishe/llm_702010_ablation.json"

SEED = 42
KS = [1, 3, 5, 10]
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
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def mae(preds, trues):
    p = np.array(preds, dtype=float)
    t = np.array(trues, dtype=float)
    return float(np.mean(np.abs(p - t)))


def main():
    d = np.load(DATA_PATH, allow_pickle=True)
    X = d["X"]; y_sbp = d["y_sbp"]; y_dbp = d["y_dbp"]
    subjects = d["subjects"]
    feat_names = list(d["feature_names"])
    print(f"数据: {len(X)} 段, {len(np.unique(subjects))} 受试者", flush=True)

    rng = np.random.RandomState(SEED)
    us = np.unique(subjects)
    rng.shuffle(us)
    n = len(us)
    n_train = int(n * 0.7); n_val = int(n * 0.2)
    test_s = us[n_train + n_val:]
    print(f"测试受试者: {len(test_s)}", flush=True)

    # 特征字典缓存（按行）
    feat_rows = {}
    for i, name in enumerate(feat_names):
        feat_rows[name] = X[:, i]

    print("加载模型...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    print("模型加载完成", flush=True)

    out = {}
    for K in KS:
        print(f"\n===== K={K} =====", flush=True)
        num_true_sbp, num_pred_sbp = [], []
        num_true_dbp, num_pred_dbp = [], []
        sem_true_sbp, sem_pred_sbp = [], []
        sem_true_dbp, sem_pred_dbp = [], []
        done = 0
        for subj in test_s:
            idx = np.where(subjects == subj)[0]
            if len(idx) < K + N_TEST:
                continue
            calib_idx = idx[:K]
            test_idx = idx[K:K + N_TEST]
            # 构造校准样本
            calib_feats = [{n: float(feat_rows[n][j]) for n in feat_names} for j in calib_idx]
            calib_num = [(f, float(y_sbp[j]), float(y_dbp[j])) for f, j in zip(calib_feats, calib_idx)]
            calib_sem = [(build_semantic_description(f), float(y_sbp[j]), float(y_dbp[j]))
                         for f, j in zip(calib_feats, calib_idx)]
            for j in test_idx:
                f = {n: float(feat_rows[n][j]) for n in feat_names}
                sem = build_semantic_description(f)
                pn = generate(model, tokenizer, build_numeric_prompt(f, calib_num))
                ps = generate(model, tokenizer, build_semantic_prompt(sem, calib_sem))
                sbp_n, dbp_n = parse_bp(pn)
                sbp_s, dbp_s = parse_bp(ps)
                if sbp_n is not None:
                    num_true_sbp.append(float(y_sbp[j])); num_pred_sbp.append(sbp_n)
                    num_true_dbp.append(float(y_dbp[j])); num_pred_dbp.append(dbp_n)
                if sbp_s is not None:
                    sem_true_sbp.append(float(y_sbp[j])); sem_pred_sbp.append(sbp_s)
                    sem_true_dbp.append(float(y_dbp[j])); sem_pred_dbp.append(dbp_s)
            done += 1
            if done % 100 == 0:
                print(f"  K={K} 处理 {done} 个被试", flush=True)
        out[str(K)] = {
            "numeric": {"sbp_mae": mae(num_pred_sbp, num_true_sbp),
                        "dbp_mae": mae(num_pred_dbp, num_true_dbp),
                        "n": len(num_pred_sbp)},
            "semantic": {"sbp_mae": mae(sem_pred_sbp, sem_true_sbp),
                         "dbp_mae": mae(sem_pred_dbp, sem_true_dbp),
                         "n": len(sem_pred_sbp)},
        }
        print(f"  K={K} numeric: {out[str(K)]['numeric']}", flush=True)
        print(f"  K={K} semantic: {out[str(K)]['semantic']}", flush=True)

    with open(OUTPUT_PATH, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n已保存 {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
