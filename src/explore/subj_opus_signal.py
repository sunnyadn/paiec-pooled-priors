"""Within-pair signal (oracle intercept): generic Opus rating vs subject-aware Opus rating vs both."""
import json, glob, numpy as np, pandas as pd
G = {k: v for k, v in json.load(open('data/llm/opus_item/ratings.json')).items() if v is not None}
rows = []
for f in glob.glob('data/llm/subj_opus/*.json'):
    b = f.split('/')[-1][:-5]
    for i, s, n, p, y in json.load(open(f)):
        if p is not None and str(i) in G: rows.append((b, str(i), s, p, G[str(i)], y))
D = pd.DataFrame(rows, columns=['b', 'i', 's', 'ps', 'pg', 'y'])
lg = lambda p: np.log(np.clip(p, 2, 98) / (100 - np.clip(p, 2, 98))); sg = lambda x: 1 / (1 + np.exp(-x))
D['zs'] = lg(D.ps); D['zg'] = lg(D.pg)
D['zsr'] = D.zs - D.groupby(['b', 's']).zs.transform('mean')   # subject-specific item pattern
for c in ['zs', 'zg', 'zsr']: D[c] = D.groupby('b')[c].transform(lambda v: (v - v.mean()) / (v.std() or 1))
print('rows', len(D), 'pairs', D.groupby(['b', 's']).ngroups)
def ev(cols, bs):
    tot = []
    for _, g in D.groupby(['b', 's']):
        y = g.y.values; x = sum(b_ * g[c].values for b_, c in zip(bs, cols)) if cols else 0 * y; a = 0.
        for _ in range(30):
            p = sg(a + x); a = np.clip(a + (y - p).sum() / max((p * (1 - p)).sum(), 1e-6), -8, 8)
        tot.append(np.mean((y - sg(a + x)) ** 2))
    return np.mean(tot)
print('none %.4f' % ev([], []))
for c in ['zg', 'zs']:
    print(c, min((round(ev([c], [b]), 4), b) for b in [0.3, 0.5, 0.7, 0.9]))
print('both', min((round(ev(['zg', 'zs'], [a, b]), 4), a, b) for a in [0, 0.3, 0.5] for b in [0, 0.3, 0.5]))
# budget-0 usefulness: subject-aware mean rating as accuracy estimate
r = D.groupby(['b', 's']).agg(acc=('y', 'mean'), m=('ps', 'mean'))
print('pair-level: corr(mean subject-aware p, acc) = %.2f' % r.acc.corr(r.m / 100))
