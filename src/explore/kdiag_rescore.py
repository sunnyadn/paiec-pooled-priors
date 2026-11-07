"""Response-level rescoring (zero calls): keep the K4/K8 predictions and revealed labels of kdiag, rescore each pair's evaluation items
against all recorded responses (one probability per item reused for every response, response-weighted Brier per pair, as the
platform scores); ALC per draw, then mean over draws; K8 - K4 paired. Also splits q_all -> q_first into response-weighted ->
item-equal trial mean -> first trial for the kdiag records. usage: kdiag_rescore.py"""
import json, numpy as np, pandas as pd
B = [0, 1, 3, 7, 15, 31]; W = [.1, .2, .2, .2, .2, .1]; rng = np.random.default_rng(0)
L = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'trial', 'y']); L['item_id'] = L.item_id.astype(str); L['subject_id'] = L.subject_id.astype(str)
G = L.groupby(['bench', 'subject_id', 'item_id']).y.agg(['sum', 'count'])
for S in ('ablate', 'multi'):
    alc = {}
    for K in (4, 8):
        for seed in (5, 6, 7):
            D = [json.loads(l) for l in open(f'data/draws/{S}_s{seed}.jsonl')]
            for r in json.load(open(f'kdiag/{S}_s{seed}_k{K}.json')):
                x = D[r['draw']][r['pair']]; ids = [str(i) for i, _ in x['ev']]; assert len(ids) == len(r['p'])
                g = G.loc[[(x['b'], str(x['s']), i) for i in ids]]; c, n = g['sum'].values.astype(float), g['count'].values.astype(float); p = np.array(r['p'])
                br = float(np.sum(c * (1 - p) ** 2 + (n - c) * p ** 2) / n.sum())   # every response of an item scored with the item's probability
                alc.setdefault(K, {}).setdefault((seed, r['draw'], r['pair']), {})[r['n']] = (br, float(np.mean((np.array(r['y']) - p) ** 2)))
    out = {}
    for K in (4, 8):
        per = {k: (sum(w * v[n][0] for n, w in zip(B, W)), sum(w * v[n][1] for n, w in zip(B, W))) for k, v in alc[K].items()}
        dr = {}
        for k, v in per.items(): dr.setdefault(k[:2], []).append(v)
        out[K] = {d: (np.mean([a for a, _ in v]), np.mean([b for _, b in v])) for d, v in dr.items()}
    ks = sorted(out[4]); assert ks == sorted(out[8])
    for j, lab in ((1, 'first-trial scoring (as the sim)'), (0, 'response-level scoring (as the platform)')):
        d = np.array([out[8][k][j] - out[4][k][j] for k in ks])
        print(f'{S} {lab}: ALC K4 {np.mean([out[4][k][j] for k in ks]):.4f} K8 {np.mean([out[8][k][j] for k in ks]):.4f} | K8 - K4 {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}, K8 better in {np.mean(d < 0):.0%} of {len(d)} draws')
# q_all -> q_first split for the chain records
T = pd.read_csv('data/kdiag/chain_records.csv', dtype={'subject_id': str})
F = L.sort_values('trial').drop_duplicates(['bench', 'subject_id', 'item_id']).groupby(['bench', 'subject_id']).y.mean()
IM = L.groupby(['bench', 'subject_id', 'item_id']).y.mean().groupby(['bench', 'subject_id']).mean()
T['q_resp'] = [L[(L.bench == b) & (L.subject_id == s)].y.mean() for b, s in zip(T.bench, T.subject_id)]
T['q_item'] = [IM[(b, s)] for b, s in zip(T.bench, T.subject_id)]; T['q_first2'] = [F[(b, s)] for b, s in zip(T.bench, T.subject_id)]
for (S, n), g in T.groupby(['set', 'n']):
    d = g.a8 - g.a4
    print(f'{S} n={n}: q_all vs response-weighted mean max|diff| {np.max(np.abs(g.q_all - g.q_resp)):.4f} | [response-weighted -> item-equal trial mean] {2 * np.mean(d * (g.q_item - g.q_resp)):+.5f} '
          f'| [item-equal trial mean -> first trial] {2 * np.mean(d * (g.q_first2 - g.q_item)):+.5f} | total q_all->q_first {2 * np.mean(d * (g.q_first - g.q_all)):+.5f}')
