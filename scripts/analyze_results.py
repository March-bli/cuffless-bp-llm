import json

d = json.load(open('/users/acp25bl/bishe/llm_bp_local_results.json'))
print('summary:', json.dumps(d['summary'], indent=2))
print()
print('=== zero-shot first 5 ===')
for r in d['zero_shot_results'][:5]:
    print(f"true: {r['tsbp']}/{r['tdbp']} | pred: {r['psbp']}/{r['pdbp']}")
    print('raw:', repr(r['raw'][:150]))
    print()
print('=== few-shot K=5 first 5 ===')
for r in d['few_shot_results'].get('5', [])[:5]:
    print(f"true: {r['tsbp']}/{r['tdbp']} | pred: {r['psbp']}/{r['pdbp']}")
    print('raw:', repr(r['raw'][:150]))
    print()
