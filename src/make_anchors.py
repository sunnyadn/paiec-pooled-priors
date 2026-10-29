"""Rebuild the eight calibration examples of the subject-prior prompt from public measurement-db (needs HF access).
Run src/prior_eval_build.py first (writes data/llm/prior_eval.json). Writes submission/anchors.json."""
import json
E = json.load(open('data/llm/prior_eval.json')); sim = set(json.load(open('data/llm/items_to_rate.json')))
bad = {'alignment_faking', 'emoji_attack', 'harmbench'}
pool = sorted([e for e in E if e['bench'] not in sim | bad], key=lambda e: e['acc']); seen = set(); pick = []
for q in [0.03, 0.15, 0.3, 0.45, 0.6, 0.75, 0.88, 0.97]:
    c = sorted([e for e in pool if e['bench'] not in seen], key=lambda e: abs(e['acc'] - q))[0]; seen.add(c['bench']); pick.append(c)
A = [{'items': [t[:300] for t in e['texts'][:2]], 'model': e['subject']['normalized_name'], 'pct': round(100 * e['acc'])} for e in pick]
json.dump(A, open('submission/anchors.json', 'w'), indent=1); print('wrote submission/anchors.json', [(a['model'], a['pct']) for a in A])
