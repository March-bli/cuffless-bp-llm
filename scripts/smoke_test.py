import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("Loading model...")
tok = AutoTokenizer.from_pretrained("/users/acp25bl/models/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "/users/acp25bl/models/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
print("Model loaded, device:", model.device)

msg = [{"role": "user", "content": "Given HR 72, predict BP. Reply: SBP=xxx, DBP=xxx"}]
text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("模型输出:", repr(resp))
print("SMOKE_TEST_OK")
