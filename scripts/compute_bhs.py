"""计算 BHS 累积误差分布（≤5/≤10/≤15 mmHg 占比），评估 Grade A/B/C"""
import json, os
import numpy as np

BASE = "/users/acp25bl/bishe"

def bhs_grade(errors):
    """根据误差列表计算 BHS 等级"""
    errors = np.array(errors)
    n = len(errors)
    p5 = np.sum(errors <= 5) / n * 100
    p10 = np.sum(errors <= 10) / n * 100
    p15 = np.sum(errors <= 15) / n * 100
    grade = None
    if p5 >= 60 and p10 >= 85 and p15 >= 95:
        grade = "A"
    elif p5 >= 50 and p10 >= 75 and p15 >= 90:
        grade = "B"
    elif p5 >= 40 and p10 >= 65 and p15 >= 85:
        grade = "C"
    return {"n": n, "pct<=5": round(p5, 1), "pct<=10": round(p10, 1),
            "pct<=15": round(p15, 1), "grade": grade, "mae": round(float(np.mean(errors)), 2)}


def eval_file(path, keys):
    """从结果 JSON 计算 BHS"""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    results = {}
    # 支持 {"few_shot_results": {"K": [...]}} 或 {"results": [...]} 或直接 {"summary": {...}}
    for container in [d.get("few_shot_results"), d.get("results")]:
        if container is None:
            continue
        if isinstance(container, dict):
            for k, arr in container.items():
                if isinstance(arr, list) and arr:
                    for target in ["sbp", "dbp"]:
                        preds = [r.get(f"p{target}") for r in arr if r.get(f"p{target}") is not None]
                        trues = [r.get(f"t{target}") for r in arr if r.get(f"p{target}") is not None]
                        if preds:
                            errs = [abs(p - t) for p, t in zip(preds, trues)]
                            results[f"{target}_K{k}"] = bhs_grade(errs)
        elif isinstance(container, list):
            for target in ["sbp", "dbp"]:
                preds = [r.get(f"p{target}") for r in container if r.get(f"p{target}") is not None]
                trues = [r.get(f"t{target}") for r in container if r.get(f"p{target}") is not None]
                if preds:
                    errs = [abs(p - t) for p, t in zip(preds, trues)]
                    results[f"{target}"] = bhs_grade(errs)
    return results


files = {
    "LLM_numeric": f"{BASE}/llm_bp_local_results.json",
    "semantic_ablation": f"{BASE}/semantic_ablation_results.json",
    "semantic_k5": f"{BASE}/llm_bp_semantic_results.json",
    "bottleneck": f"{BASE}/llm_bp_bottleneck_results.json",
}

print("=== BHS 累积误差分布评估 ===\n")
for name, path in files.items():
    r = eval_file(path, None)
    if r:
        print(f"[{name}]")
        for key, g in r.items():
            print(f"  {key}: MAE={g['mae']}, ≤5={g['pct<=5']}%, ≤10={g['pct<=10']}%, ≤15={g['pct<=15']}% → Grade {g['grade']}")
    else:
        print(f"[{name}] 文件不存在或无法解析")

print("\nBHS 标准: Grade A (60/85/95), B (50/75/90), C (40/65/85)")
