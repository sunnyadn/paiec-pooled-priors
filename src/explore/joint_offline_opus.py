"""Joint Rasch update vs additive item term, using Opus generic per-item ratings on 25 pairs (data/llm/subj_opus).
Prior per pair simulated as true accuracy + N(0, 0.15) (the measured online error), K=4; budgets 0..31 from a 50/50 split."""
import json, glob, numpy as np, pandas as pd
G = {k: v for k, v in json.load(open('data/llm/opus_item/ratings.json')).items() if v is not None}
rows = []
for f in glob.glob('data/llm/subj_opus/*.json'):
    b = f.split('/')[-1][:-5]
    for i, s, n, p, y in json.load(open(f)):
        if str(i) in G: rows.append((b, s, G[str(i)], y))
D = pd.DataFrame(rows, columns=['b', 's', 'pg', 'y'])
lg = lambda p: np.log(np.clip(p, 2, 98) / (100 - np.clip(p, 2, 98))); sg = lambda x: 1 / (1 + np.exp(-x))
D['z'] = lg(D.pg); D['z'] = D.groupby('b').z.transform(lambda v: (v - v.mean()) / (v.std() or 1))
B = [0, 1, 3, 7, 15, 31]; W = [.1, .2, .2, .2, .2, .1]; K = 4
def run(method, beta, reps=40, seed=0):
    rng = np.random.default_rng(seed); out = []
    for (b, s), g in D.groupby(['b', 's']):
        y = g.y.values; z = g.z.values; n = len(y); acc = y.mean()
        for r in range(reps):
            m = float(np.clip(acc + rng.normal(0, 0.15), 0.03, 0.97)); perm = rng.permutation(n); h = n // 2
            acq, ev = perm[:h], perm[h:]; alc = 0
            for nb, w in zip(B, W):
                rv = acq[:min(nb, len(acq))]; sv = y[rv].sum(); k = len(rv)
                if method == 'add':
                    p = (sv + K * m) / (k + K) + beta * z[ev]
                elif method == 'v26like':
                    N = min(9 * nb, 60); be = 0 if nb == 0 else beta * N / (N + 25.0)
                    p = (sv + K * m) / (k + K) + be * z[ev]
                else:
                    th0 = lg(100 * m); var0 = 1 / ((K + 1) * m * (1 - m)); th = th0
                    for _ in range(20):
                        q = sg(th + beta * z[rv]); gr = -(th - th0) / var0 + (y[rv] - q).sum(); hh = -1 / var0 - (q * (1 - q)).sum(); th -= gr / hh
                    p = sg(th + beta * z[ev])
                p = np.clip(p, .01, .99); alc += w * np.mean((y[ev] - p) ** 2)
            out.append(alc)
    return np.array(out)
base = run('add', 0.0)
for meth, betas in [('v26like', [0.10]), ('add', [0.10]), ('joint', [0.5])]:
    for be in betas:
        o = run(meth, be); d = o - base
        print(f'{meth} beta={be}: ALC {o.mean():.4f}  vs no-item {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f}')
print('no-item additive ALC %.4f' % base.mean())
