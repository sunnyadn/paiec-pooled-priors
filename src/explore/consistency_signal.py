import json, glob, numpy as np, pandas as pd
C = {k: v['agree'] for k, v in json.load(open('data/llm/consistency.json')).items() if v}
G = {k: v for k, v in json.load(open('data/llm/opus_item/ratings.json')).items() if v is not None}
rows = []
for f in glob.glob('data/llm/subj_opus/*.json'):
    b = f.split('/')[-1][:-5]
    for i, s, n, p, y in json.load(open(f)):
        i = str(i)
        if i in C and i in G: rows.append((b, s, i, C[i], G[i], y))
D = pd.DataFrame(rows, columns=['b', 's', 'i', 'agree', 'pg', 'y'])
lg = lambda p: np.log(np.clip(p, 2, 98) / (100 - np.clip(p, 2, 98))); sg = lambda x: 1 / (1 + np.exp(-x))
D['zg'] = lg(D.pg); D['za'] = D.agree
for c in ['zg', 'za']: D[c] = D.groupby('b')[c].transform(lambda v: (v - v.mean()) / (v.std() or 1))
print('rows', len(D), 'items', D.i.nunique())
im = D.groupby(['b', 'i']).agg(acc=('y', 'mean'), za=('za', 'first'), zg=('zg', 'first')).reset_index()
print('corr with item accuracy, per bench:'); print(im.groupby('b').apply(lambda g: pd.Series({'agree': g.za.corr(g.acc), 'opus': g.zg.corr(g.acc)})).round(2))
def ev(cols, bs):
    tot = []
    for _, g in D.groupby(['b', 's']):
        y = g.y.values; x = sum(b_ * g[c].values for b_, c in zip(bs, cols)) if cols else 0 * y; a = 0.
        for _ in range(30):
            p = sg(a + x); a = np.clip(a + (y - p).sum() / max((p * (1 - p)).sum(), 1e-6), -8, 8)
        tot.append(np.mean((y - sg(a + x)) ** 2))
    return np.mean(tot)
print('none %.4f' % ev([], []))
print('opus', min((round(ev(['zg'], [b]), 4), b) for b in [0.3, 0.5, 0.7]))
print('agree', min((round(ev(['za'], [b]), 4), b) for b in [0.3, 0.5, 0.7]))
print('both', min((round(ev(['zg', 'za'], [a, b]), 4), a, b) for a in [0.3, 0.5] for b in [0.1, 0.3, 0.5]))
# leave-one-benchmark-out: choose weights on 4 benchmarks, score the 5th
full = D.copy(); res = {'opus': [], 'both': []}
for hb in sorted(full.b.unique()):
    D = full[full.b != hb]
    bo = min([0.3, 0.5, 0.7], key=lambda b: ev(['zg'], [b]))
    bb = min([(a, b) for a in [0.3, 0.5] for b in [0.1, 0.3, 0.5]], key=lambda ab: ev(['zg', 'za'], list(ab)))
    D = full[full.b == hb]
    res['opus'].append(ev(['zg'], [bo])); res['both'].append(ev(['zg', 'za'], list(bb)))
    print(hb, 'opus %.4f both %.4f' % (res['opus'][-1], res['both'][-1]), bb)
print('LOBO mean opus %.4f both %.4f' % (np.mean(res['opus']), np.mean(res['both'])))
