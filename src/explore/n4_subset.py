"""N4 explanation only: v28 - v26 response-weighted Brier on the records where v26 used the single-model path without a prior (flags from
the v26 PREDLOG: every sp None, single == calls, budget > 0), per set and budget, each record weighted 1 / (pairs in its draw) as in the main result. A conditional mean on this subset: it does not
decompose the whole-population result. usage: n4_subset.py V26DIR V28DIR"""
import json, sys, numpy as np, pandas as pd
V26, V28 = sys.argv[1], sys.argv[2]
L = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y']); L['item_id'] = L.item_id.astype(str); L['subject_id'] = L.subject_id.astype(str)
G = L.groupby(['bench', 'subject_id', 'item_id']).y.agg(['sum', 'count'])
for S in ('ablate', 'multi'):
    d = {}
    for s in (5, 6, 7):
        D = [json.loads(l) for l in open(f'data/draws/{S}_s{s}.jsonl')]
        A = {(r['draw'], r['pair'], r['n']): r for r in json.load(open(f'{V26}/{S}_s{s}.json'))}
        for r in json.load(open(f'{V28}/{S}_s{s}.json')):
            a = A[(r['draw'], r['pair'], r['n'])]
            if r['n'] == 0 or (a['sp'] and any(v is not None for v in a['sp'])) or not (a['single'] == a['calls'] > 0): continue
            x = D[r['draw']][r['pair']]; g = G.loc[[(x['b'], str(x['s']), str(i)) for i, _ in x['ev']]]; c, n = g['sum'].values.astype(float), g['count'].values.astype(float)
            br = lambda p: float(np.sum(c * (1 - p) ** 2 + (n - c) * p ** 2) / n.sum())
            d.setdefault(r['n'], []).append((br(np.array(r['p'])) - br(np.array(a['p'])), 1.0 / len(D[r['draw']])))
    print(S, '(conditional mean on the v26 no-prior single-path records, draw weights)', ' | '.join(f'b{n}: {len(v)} recs, v28-v26 {sum(x * w for x, w in v) / sum(w for _, w in v):+.5f}' for n, v in sorted(d.items())))
