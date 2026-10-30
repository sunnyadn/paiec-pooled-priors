"""Does a subject-aware LLM rating add signal over the generic rating? (data/llm/subj_haiku)"""
import json, glob, numpy as np, pandas as pd
from scipy.optimize import minimize
gen = {}
for f in glob.glob('data/llm/api_haiku/*.json'):
    for x in json.load(open(f))['ratings']: gen[str(x['id'])] = float(x['p'])
rows = []
for f in glob.glob('data/llm/subj_haiku/*.json'):
    b = f.split('/')[-1][:-5]
    for i, s, n, p, y in json.load(open(f)):
        if p is not None and i in gen: rows.append((b, i, s, n, p, gen[i], y))
D = pd.DataFrame(rows, columns=['b', 'i', 's', 'n', 'ps', 'pg', 'y'])
lg = lambda p: np.log(np.clip(p, 2, 98) / (100 - np.clip(p, 2, 98)))
D['zs'] = lg(D.ps); D['zg'] = lg(D.pg)
for c in ['zs', 'zg']: D[c + 'n'] = D.groupby('b')[c].transform(lambda v: (v - v.mean()) / (v.std() or 1))
D['pair'] = D.b + '|' + D.s
sg = lambda x: 1 / (1 + np.exp(-x))
print(D.groupby('b').agg(n=('y', 'size'), acc=('y', 'mean')).T)
# 1) budget 0, no labels: calibrated rating -> probability (a + c*z per benchmark-free, fit leave-one-benchmark-out)
def b0(col):
    out = []
    for b in D.b.unique():
        tr, te = D[D.b != b], D[D.b == b]
        f = lambda w: np.mean((tr.y - sg(w[0] + w[1] * tr[col])) ** 2)
        w = minimize(f, [0, 0.5]).x
        out.append(np.mean((te.y - sg(w[0] + w[1] * te[col])) ** 2))
    return np.mean(out)
print('budget0 Brier: const .5 = .25 | generic', round(b0('zg'), 4), '| subject-aware', round(b0('zs'), 4))
# 2) within-pair discrimination with oracle intercept per pair, common slope(s)
def within(cols):
    best = None
    for bs in ([0.0, 0.3, 0.5, 0.7, 1.0] if len(cols) == 1 else [(a, c) for a in [0, .3, .5, .7] for c in [0, .3, .5, .7]]):
        bs = np.atleast_1d(bs); tot = []
        for _, g in D.groupby('pair'):
            x = sum(bb * g[c].values for bb, c in zip(bs, cols)); y = g.y.values; a = 0.
            for _ in range(30):
                p = sg(a + x); a = np.clip(a + (y - p).sum() / max((p * (1 - p)).sum(), 1e-6), -8, 8)
            tot.append(np.mean((y - sg(a + x)) ** 2))
        m = np.mean(tot)
        if best is None or m < best[0]: best = (m, tuple(bs))
    return best
base = np.mean([g.y.mean() * (1 - g.y.mean()) for _, g in D.groupby('pair')])
print('within-pair (oracle intercept): none', round(base, 4), '| generic', within(['zgn']), '| subject', within(['zsn']), '| both', within(['zgn', 'zsn']))
# 3) does the subject-aware rating rank subjects? per benchmark, Spearman of mean rating vs accuracy
r = D.groupby(['b', 's']).agg(acc=('y', 'mean'), m=('zs', 'mean'))
print('subject ranking rho per bench:', r.groupby('b').apply(lambda g: g.acc.corr(g.m, method='spearman')).round(2).to_dict())
