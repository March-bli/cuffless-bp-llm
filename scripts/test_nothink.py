import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("/users/acp25bl/models/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("/users/acp25bl/models/Qwen3-8B", torch_dtype=torch.bfloat16).to("cuda")
model.eval()

msg = [{"role": "user", "content": "Given HR 72, predict BP. Reply: SBP=xxx, DBP=xxx"}]

# 关闭思考模式
text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(text, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("NO-THINK output:", repr(resp))
print("TEST_OK" if "SBP" in resp or "DBP" in resp else "TEST_FAIL")
