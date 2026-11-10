"""Zero-call check on the new benchmarks, step 1 (win): the pairs with a prior estimate and >= 80 distinct scored items, their
0/1 outcomes (first trial per item), item texts, and fixed draws (7 benchmarks, subjects 2,2,1,1,1,1,1, 80 items per pair split
40 acquisition / 40 evaluation), 30 draws for each of seeds 5, 6, 7. -> data/llm/extra/sim/{pairs,items,draws_s*}.json(l)"""
import json, os
import numpy as np, pandas as pd
OUT = 'data/llm/extra/sim'; os.makedirs(OUT, exist_ok=True)
E = {f"{e['bench']}|{e['subject_id']}": e for e in json.load(open('data/llm/extra/eligible.json'))}
RES = json.load(open('data/llm/extra/results.json'))['results']
pairs, items = [], {}
for k, r in RES.items():
    if r['status'] == 'untried': continue
    if r.get('estimate') is None: raise SystemExit(f'pair {k} has no estimate')   # every simulated pair must carry its prior
    e = E[k]; d = f"data/mdb_extra/{e['bench']}/formatted_tables"
    R = pd.read_parquet(d + '/responses.parquet', columns=['subject_id', 'item_id', 'trial', 'response'])
    R = R[R.subject_id.astype(str) == e['subject_id']].assign(y=lambda x: pd.to_numeric(x.response, errors='coerce')).dropna(subset=['y'])
    R = R.sort_values(['item_id', 'trial']).drop_duplicates('item_id')   # first trial per item, as one competition response
    R = R[R.y.isin([0, 1])]
    if R.item_id.nunique() < 80: continue
    pairs.append(dict(key=k, bench=e['bench'], subject_id=e['subject_id'], subject=e['subject'], estimate=r['estimate'],
                      outcomes=[[str(i), int(y)] for i, y in zip(R.item_id, R.y)]))
    if e['bench'] not in items:
        I = pd.read_parquet(d + '/items.parquet', columns=['item_id', 'content', 'item_features'])
        items[e['bench']] = {str(i): [str(c), '' if f is None or f != f else str(f)] for i, c, f in zip(I.item_id, I.content, I.item_features)}   # text and features, as the platform sends them
need = {p['bench']: set() for p in pairs}
for p in pairs: need[p['bench']] |= {i for i, _ in p['outcomes']}
items = {b: {i: t for i, t in v.items() if i in need[b]} for b, v in items.items()}
json.dump(pairs, open(f'{OUT}/pairs.json', 'w')); json.dump(items, open(f'{OUT}/items.json', 'w'))
by = {}
for j, p in enumerate(pairs): by.setdefault(p['bench'], []).append(j)
print('pairs', len(pairs), 'benchmarks', len(by), '| benchmarks with >= 2 pairs', sum(len(v) >= 2 for v in by.values()))
NSUB = [2, 2, 1, 1, 1, 1, 1]
for seed in (5, 6, 7):
    rng = np.random.default_rng(seed); multi = sorted(b for b, v in by.items() if len(v) >= 2); single = sorted(by)
    with open(f'{OUT}/draws_s{seed}.jsonl', 'w') as f:
        for _ in range(30):
            two = list(rng.choice(multi, 2, replace=False)); rest = [b for b in single if b not in two]
            bs = two + list(rng.choice(rest, 5, replace=False)); draw = []
            for b, k in zip(bs, NSUB):
                for j in rng.choice(by[b], k, replace=False):
                    ids = [i for i, _ in pairs[j]['outcomes']]; ids = list(rng.choice(ids, 80, replace=False))
                    assert len(ids[:40]) >= 31
                    draw.append(dict(pair=pairs[j]['key'], stream=ids[:40], ev=ids[40:]))
            f.write(json.dumps(draw) + '\n')
print('draws written for seeds 5, 6, 7')
