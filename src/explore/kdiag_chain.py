"""K flip attribution along the item-set chain (zero calls, public data): q_all (all trials, as prior_eval_build_all) -> q_first (first trial
per item) -> coverage pool of rate_prep.py -> the 300-item slice -> the draw's 120 sampled items -> the scored half. Writes one row per
record (a4, a8, each mean, ids) to data/kdiag/chain_records.csv and prints 2 mean[(a8-a4) * step] per step. usage: kdiag_chain.py"""
import json, numpy as np, pandas as pd
E = {(e['bench'], e['subject_id']): e['acc'] for e in json.load(open('data/llm/prior_eval_all.json'))}
IT = json.load(open('data/llm/items_to_rate.json'))
L = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'trial', 'y']); L['item_id'] = L.item_id.astype(str); L['subject_id'] = L.subject_id.astype(str)
F = L.sort_values('trial').drop_duplicates(['bench', 'subject_id', 'item_id'])
M = {}   # (bench, subject) -> dict of means
for b, v in IT.items():
    d = L[L.bench == b]; cnt = d.groupby('subject_id').item_id.nunique(); subs = cnt[cnt >= 80].index   # as rate_prep.py
    dd = d[d.subject_id.isin(subs)]; ic = dd.groupby('item_id').subject_id.nunique(); pool = set(ic[ic >= max(2, ic.quantile(0.5))].index)
    sl = {str(x['item_id']) for x in v}; f = F[F.bench == b]
    qf = f.groupby('subject_id').y.mean(); qp = f[f.item_id.isin(pool)].groupby('subject_id').y.mean(); qs = f[f.item_id.isin(sl)].groupby('subject_id').y.mean()
    for s in qs.index: M[(b, s)] = dict(q_first=qf.get(s), q_pool=qp.get(s), q_slice=qs.get(s))
rows = []
for S in ('ablate', 'multi'):
    for seed in (5, 6, 7):
        D = [json.loads(l) for l in open(f'data/draws/{S}_s{seed}.jsonl')]
        for r in json.load(open(f'kdiag/{S}_s{seed}_k4.json')):
            sp = [v for v in r['sp'] if v is not None]
            if r['n'] not in (1, 3, 7) or not sp or len(set(sp)) != 1 or r['single'] != r['calls']: continue
            key = (r['b'], str(D[r['draw']][r['pair']]['s']))
            if key not in E or key not in M or None in M[key].values(): continue
            rows.append(dict(set=S, seed=seed, draw=r['draw'], pair=r['pair'], n=r['n'], bench=key[0], subject_id=key[1], prior=sp[0], got=r['got'], k=r['k'],
                             a4=(r['k'] + 4 * sp[0]) / (r['got'] + 4), a8=(r['k'] + 8 * sp[0]) / (r['got'] + 8), q_all=E[key], **M[key], q_sample=r['acc'], r_eval=float(np.mean(r['y']))))
T = pd.DataFrame(rows); T.to_csv('data/kdiag/chain_records.csv', index=False)
steps = [('q_all', 'q_first'), ('q_first', 'q_pool'), ('q_pool', 'q_slice'), ('q_slice', 'q_sample'), ('q_sample', 'r_eval')]
for (S, n), g in T.groupby(['set', 'n']):
    d = g.a8 - g.a4; tot = 2 * np.mean(d * (g.r_eval - g.q_all))
    print(f'{S} n={n} recs {len(g)} | shift toward K4 when the target moves from full-benchmark to scored half: {tot:+.5f} = ' +
          ' + '.join(f'[{a}->{b}] {2 * np.mean(d * (g[b] - g[a])):+.5f}' for a, b in steps))
