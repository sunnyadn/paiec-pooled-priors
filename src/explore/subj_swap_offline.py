import json, glob, numpy as np, pandas as pd
G = {k: v for k, v in json.load(open('data/llm/opus_item/ratings.json')).items() if v is not None}
rows = []
for f in glob.glob('data/llm/subj_opus/*.json'):
    b = f.split('/')[-1][:-5]
    for i, s, n, p, y in json.load(open(f)):
        if str(i) in G and p is not None: rows.append((b, s, G[str(i)], p, y))
D = pd.DataFrame(rows, columns=['b', 's', 'pg', 'ps', 'y'])
lg = lambda p: np.log(np.clip(p, 2, 98) / (100 - np.clip(p, 2, 98))); sg = lambda x: 1 / (1 + np.exp(-x))
D['zg'] = lg(D.pg); D['zs'] = lg(D.ps)
D['zg'] = D.groupby('b').zg.transform(lambda v: (v - v.mean()) / (v.std() or 1))
D['zs'] = D.groupby(['b', 's']).zs.transform(lambda v: (v - v.mean()) / (v.std() or 1))   # within pair: no double count of strength
B = [0, 1, 3, 7, 15, 31]; W = [.1, .2, .2, .2, .2, .1]; K = 4
def run(col, beta, reps=40, seed=0):
    rng = np.random.default_rng(seed); out = []
    for (b, s), g in D.groupby(['b', 's']):
        y = g.y.values; z = g[col].values if col else 0 * y; n = len(y); acc = y.mean()
        for r in range(reps):
            m = float(np.clip(acc + rng.normal(0, 0.15), 0.03, 0.97)); perm = rng.permutation(n); h = n // 2
            acq, ev = perm[:h], perm[h:]; alc = 0
            for nb, w in zip(B, W):
                rv = acq[:min(nb, len(acq))]; sv = y[rv].sum(); k = len(rv)
                N = min(9 * nb, 60); be = 0 if nb == 0 else beta * N / (N + 25.0)
                p = np.clip((sv + K * m) / (k + K) + be * z[ev], .01, .99); alc += w * np.mean((y[ev] - p) ** 2)
            out.append(alc)
    return np.array(out)
base = run(None, 0)
for col in ['zg', 'zs']:
    o = run(col, 0.10); d = o - base; print(col, 'v26-like adaptive slope: %+.4f ± %.4f' % (d.mean(), d.std(ddof=1) / np.sqrt(len(d))))
d = run('zs', 0.10) - run('zg', 0.10); print('subject-aware minus generic: %+.4f ± %.4f' % (d.mean(), d.std(ddof=1) / np.sqrt(len(d))))
for b in sorted(D.b.unique()):
    sub = D.b == b
