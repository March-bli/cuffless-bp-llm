"""
LLM Fine-tuning for BP Estimation
=================================
用 Qwen2.5-0.5B-Instruct + QLoRA 做指令微调，让 LLM 学会从 55 维 PPG 特征预测血压。

对应导师建议: "explore how to use LLM as the core ML to infer BP"
"""

import sys; sys.path.insert(0, ".")
import os, json, time
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # 本地缓存的 base model
OUTPUT_DIR = "models/qwen2.5-bp"
SEED = 42
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
EPOCHS = 3
LR = 2e-4
MAX_LENGTH = 1024
USE_LORA = True  # QLoRA for 8GB GPU

torch.manual_seed(SEED)
np.random.seed(SEED)

FEATURE_DESCRIPTIONS = {
    "heart_rate":"HR(bpm)","pulse_interval_mean":"pulse_interval(ms)","aix":"augmentation_idx(%)",
    "stiffness_index":"arterial_stiffness","upstroke_time_mean":"upstroke_time(ms)","diastolic_time_mean":"diastolic_time(ms)",
    "pulse_width_75pct":"PW75%(ms)","pulse_width_50pct":"PW50%(ms)","pulse_width_25pct":"PW25%(ms)",
    "apg_a_mean":"APG_a-wave","apg_b_mean":"APG_b-wave","apg_c_mean":"APG_c-wave","apg_d_mean":"APG_d-wave","apg_e_mean":"APG_e-wave",
    "apg_b_a_ratio":"APG_ba_ratio","apg_c_a_ratio":"APG_ca_ratio","apg_d_a_ratio":"APG_da_ratio","apg_e_a_ratio":"APG_ea_ratio",
    "sys_amp_mean":"systolic_amplitude","pulse_amp_mean":"pulse_amplitude","vpg_mean_abs":"|VPG|_mean",
    "ppg_skew":"PPG_skewness","ppg_kurtosis":"PPG_kurtosis","hr_band_power":"HR_band_power",
    "dominant_freq":"dominant_frequency(Hz)","spectral_energy":"spectral_energy","lf_hf_ratio":"LF/HF_ratio",
    "dia_amp_mean":"diastolic_amplitude","pulse_interval_std":"pulse_interval_SD(ms)",
    "upstroke_time_std":"upstroke_time_SD(ms)","ppg_entropy":"PPG_entropy",
}

TOP_FEATURES = list(FEATURE_DESCRIPTIONS.keys())[:30]

def format_prompt(feats, mode="train"):
    """Build the instruction prompt for training/inference."""
    lines = [f"{FEATURE_DESCRIPTIONS.get(f,'?')}={feats[f]:.3f}" for f in TOP_FEATURES if f in feats]
    features_text = "; ".join(lines)
    return f"""Given PPG signal features, predict systolic (SBP) and diastolic (DBP) blood pressure in mmHg.

Features: {features_text}

Blood pressure:"""

def format_response(sbp, dbp):
    """Expected model output format."""
    return f"SBP={sbp:.0f}, DBP={dbp:.0f}"

# ═══════════════════════════════════════════════════
# 1. Load & Split Data
# ═══════════════════════════════════════════════════
print("=" * 70)
print("QWEN2.5-0.5B FINE-TUNING FOR BP ESTIMATION")
print("=" * 70)

print("\n[1/5] Loading data...")
d = np.load("data/processed/pulsedb_features_proper.npz", allow_pickle=True)
X_all = d["X"]; y_sbp_all = d["y_sbp"]; y_dbp_all = d["y_dbp"]
subjects_all = d["subjects"]; feature_names = list(d["feature_names"])
feat_idx = {name: i for i, name in enumerate(feature_names)}

unique_subjects = np.unique(subjects_all)
rng = np.random.RandomState(SEED); rng.shuffle(unique_subjects)
n = len(unique_subjects)
n_train = int(n * 0.75); n_val = int(n * 0.10)
train_subjs = set(unique_subjects[:n_train])
val_subjs = set(unique_subjects[n_train:n_train+n_val])
test_subjs = set(unique_subjects[n_train+n_val:])

def split_data(subj_set, max_per_subj=50):
    indices = []
    for s in subj_set:
        idx = np.where(subjects_all == s)[0]
        if len(idx) > max_per_subj:
            idx = np.random.choice(idx, max_per_subj, replace=False)
        indices.extend(idx.tolist())
    return sorted(indices)

train_idx = split_data(train_subjs, 40)
val_idx = split_data(val_subjs, 20)
test_idx = split_data(test_subjs, 20)

print(f"  Train: {len(train_idx)} seg from {len(train_subjs)} subjects")
print(f"  Val:   {len(val_idx)} seg from {len(val_subjs)} subjects")
print(f"  Test:  {len(test_idx)} seg from {len(test_subjs)} subjects")

# ═══════════════════════════════════════════════════
# 2. Prepare Data
# ═══════════════════════════════════════════════════
print("\n[2/5] Preparing text data...")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class BPDataset(Dataset):
    def __init__(self, indices, X, y_sbp, y_dbp, feature_names, tokenizer, max_length=512):
        self.data = []
        for idx in indices:
            feats = {feature_names[i]: float(X[idx, i]) for i in range(len(feature_names))}
            prompt = format_prompt(feats)
            response = format_response(float(y_sbp[idx]), float(y_dbp[idx]))
            # Base model: simple text format, no chat template
            text = f"{prompt}\n{response}"
            encoded = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            self.data.append({
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "labels": encoded["input_ids"][0].clone(),
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

print("  Building datasets...")
train_ds = BPDataset(train_idx, X_all, y_sbp_all, y_dbp_all, feature_names, tokenizer)
val_ds = BPDataset(val_idx, X_all, y_sbp_all, y_dbp_all, feature_names, tokenizer)

# Save a sample prompt for inspection
sample_feats = {feature_names[i]: float(X_all[train_idx[0], i]) for i in range(len(feature_names))}
sample_prompt = format_prompt(sample_feats)
sample_response = format_response(float(y_sbp_all[train_idx[0]]), float(y_dbp_all[train_idx[0]]))
print(f"\n  Sample prompt:\n{sample_prompt}")
print(f"\n  Sample response: {sample_response}")
print(f"  Train samples: {len(train_ds)}")

# ═══════════════════════════════════════════════════
# 3. Load Model with QLoRA
# ═══════════════════════════════════════════════════
print("\n[3/5] Loading Qwen2.5-0.5B with QLoRA...")

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ═══════════════════════════════════════════════════
# 4. Train
# ═══════════════════════════════════════════════════
print("\n[4/5] Training...")

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_ratio=0.1,
    learning_rate=LR,
    fp16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=400,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

t0 = time.time()
trainer.train()
train_time = time.time() - t0
print(f"\n  Training completed in {train_time:.0f}s")

# Save model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"  Model saved to {OUTPUT_DIR}")

# ═══════════════════════════════════════════════════
# 5. Evaluate
# ═══════════════════════════════════════════════════
print("\n[5/5] Evaluating on test set...")

def predict_bp(feats, model, tokenizer, max_new_tokens=20):
    prompt = format_prompt(feats)
    # Base model: simple text, add a space for generation
    text = prompt + "\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.1,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    import re
    nums = re.findall(r'\d{2,3}', response)
    valid = [int(n) for n in nums if 50 < int(n) < 250]
    if len(valid) >= 2:
        return valid[0], valid[1], response
    elif len(valid) == 1:
        return valid[0], None, response
    return None, None, response

results = []
for idx in test_idx[:50]:  # First 50 test samples
    feats = {feature_names[i]: float(X_all[idx, i]) for i in range(len(feature_names))}
    true_sbp = float(y_sbp_all[idx])
    true_dbp = float(y_dbp_all[idx])
    pred_sbp, pred_dbp, raw = predict_bp(feats, model, tokenizer)
    results.append({"true_sbp": true_sbp, "true_dbp": true_dbp, "pred_sbp": pred_sbp, "pred_dbp": pred_dbp})

valid = [r for r in results if r["pred_sbp"] is not None and r["pred_dbp"] is not None]
sbp_mae = np.mean([abs(r["pred_sbp"] - r["true_sbp"]) for r in valid])
dbp_mae = np.mean([abs(r["pred_dbp"] - r["true_dbp"]) for r in valid])

print(f"\n  Valid predictions: {len(valid)}/{len(results)}")
print(f"  SBP MAE: {sbp_mae:.1f} mmHg")
print(f"  DBP MAE: {dbp_mae:.1f} mmHg")

# Save results
output = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "config": {"batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR, "lora_r": 16},
    "test_sbp_mae": round(sbp_mae, 1),
    "test_dbp_mae": round(dbp_mae, 1),
    "n_test": len(results),
    "train_time_s": int(train_time),
}
with open("data/processed/qwen_bp_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nDone! Results saved to data/processed/qwen_bp_results.json")
