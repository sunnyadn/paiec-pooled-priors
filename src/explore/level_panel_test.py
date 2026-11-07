"""Level-update test (zero calls, exploratory), kdiag K=4 records (budgets 1/3/7, _single path, one stable prior): A Beta; B ridge on prior and
k-rate; C B + mean rating of revealed items; Cdiag + scored-item mean (diagnostic); D fixed Rasch MAP. Missing ratings fall back to A; score
= response-weighted Brier per record. usage: level_panel_test.py  ->  data/kdiag/level_panel.txt, level_panel_probs.csv"""
import glob, json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98))); sg = lambda z: 1 / (1 + np.exp(-z)); rng = np.random.default_rng(0)
rat = {}
for f in glob.glob('data/llm/batches_opus_fixed/*.json'):
    for x in json.load(open(f))['ratings']: rat[str(x['id'])] = float(x['p'])
Z = lambda i: lg(rat[i] / 100.0) if i in rat else None
L = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'trial', 'y']); L['item_id'] = L.item_id.astype(str); L['subject_id'] = L.subject_id.astype(str)
G = L.groupby(['bench', 'subject_id', 'item_id']).y.agg(['sum', 'count'])
rows = []
for S in ('ablate', 'multi'):
    for seed in (5, 6, 7):
        D = [json.loads(l) for l in open(f'data/draws/{S}_s{seed}.jsonl')]
        for r in json.load(open(f'kdiag/{S}_s{seed}_k4.json')):
            sp = [v for v in r['sp'] if v is not None]
            if r['n'] not in (1, 3, 7) or not sp or None in r['sp'] or len(set(sp)) != 1 or r['single'] != r['calls']: continue   # frozen population
            x = D[r['draw']][r['pair']]; ev = [str(i) for i, _ in x['ev']]; rev = r['revealed']
            ys = {str(i): int(L.y.iat[L.index.get_loc(row)]) for i, row in x['stream']}   # the sim reveals the first trial
            g = G.loc[[(x['b'], str(x['s']), i) for i in ev]]
            rows.append(dict(set=S, seed=seed, draw=r['draw'], pair=r['pair'], n=r['n'], bench=x['b'], m=sp[0], k=r['k'], got=r['got'],
                             c=g['sum'].values.astype(float), nr=g['count'].values.astype(float), rev_y=[ys[i] for i in rev],
                             rev_z=[Z(i) for i in rev], ev_z=[Z(i) for i in ev]))
T = pd.DataFrame(rows); assert all(T.k == T.rev_y.map(sum)), 'revealed outcomes do not match k'
T['target'] = T.c.map(sum) / T.nr.map(sum); T['rev_ok'] = T.rev_z.map(lambda v: len(v) > 0 and None not in v); T['ev_ok'] = T.ev_z.map(lambda v: None not in v)
T['zr'] = [np.mean(v) if ok else np.nan for v, ok in zip(T.rev_z, T.rev_ok)]; T['ze'] = [np.mean(v) if ok else np.nan for v, ok in zip(T.ev_z, T.ev_ok)]
print('records', len(T), '| revealed ratings complete', int(T.rev_ok.sum()), '| scored ratings complete', int(T.ev_ok.sum()))
A_ = (T.k + 4 * T.m) / (T.got + 4); P = {'A': [np.full(len(c), a) for c, a in zip(T.c, A_)]}
def rasch(r, K=4):   # fixed: scale 1, centred on revealed items, prior precision (K+1) m (1-m); unrated scored items get the centre
    zc = np.mean(r.rev_z); er = np.array(r.rev_z) - zc; ee = np.array([0.0 if z is None else z - zc for z in r.ev_z]); y = np.array(r.rev_y)
    mu = lg(r.m); prec = (K + 1) * r.m * (1 - r.m); th = mu
    for _ in range(25):
        p = sg(th + er); th += (np.sum(y - p) - prec * (th - mu)) / (np.sum(p * (1 - p)) + prec)
    return sg(th + ee)
P['D'] = [rasch(r) if r.rev_ok else np.full(len(r.c), a) for (_, r), a in zip(T.iterrows(), A_)]
for col in ('B', 'C', 'Cdiag'): T[col] = A_.values   # fallback to A where the method's inputs are missing
fb = lambda d: np.column_stack([lg(d.m), lg((d.k + .5) / (d.got + 1))]); fc = lambda d: np.column_stack([fb(d), d.zr]); fd = lambda d: np.column_stack([fc(d), d.ze])
for n in (1, 3, 7):
    for hb in T.bench.unique():   # leave the benchmark out of training in both sets; every training row weighs 1
        trn, tst = (T.n == n) & (T.bench != hb), (T.n == n) & (T.bench == hb)
        for col, f, need in (('B', fb, lambda d: d.n == d.n), ('C', fc, lambda d: d.rev_ok), ('Cdiag', fd, lambda d: d.rev_ok & d.ev_ok)):
            tr = T[trn & need(T)]; te = T[tst & need(T)]
            if len(te) and len(tr): T.loc[te.index, col] = sg(Ridge(alpha=1.0).fit(f(tr), lg(tr.target)).predict(f(te)))
for col in ('B', 'C', 'Cdiag'): P[col] = [np.full(len(c), a) for c, a in zip(T.c, T[col])]
brier = lambda c, nr, p: float(np.sum(c * (1 - p) ** 2 + (nr - c) * p ** 2) / np.sum(nr))   # every response scored, as the platform does
for k, v in P.items(): T['br_' + k] = [brier(c, nr, p) for c, nr, p in zip(T.c, T.nr, v)]; T['lv_' + k] = [(np.sum(p * nr) / np.sum(nr) - t) ** 2 for p, nr, t in zip(v, T.nr, T.target)]
out = []; summ = {}
pop = {}   # share of each set's budget records that fall in the frozen population (all kdiag records counted per budget)
for S in ('ablate', 'multi'):
    allr = [r for seed in (5, 6, 7) for r in json.load(open(f'kdiag/{S}_s{seed}_k4.json')) if r['n'] in (1, 3, 7)]
    for n in (1, 3, 7): pop[(S, n)] = (int(((T.set == S) & (T.n == n)).sum()), sum(r['n'] == n for r in allr))
for S in ('ablate', 'multi'):
    for n in (1, 3, 7):
        g = T[(T.set == S) & (T.n == n)]; k_, tot = pop[(S, n)]
        if not len(g): out.append(f'{S} n={n} recs 0/{tot} in the population | N/A'); continue
        b = g.bench.values; ub = np.unique(b); line = f'{S} n={n} recs {k_}/{tot} in the population | Brier A {g.br_A.mean():.4f} (level {g.lv_A.mean():.4f})'
        for c in ('B', 'C', 'D', 'Cdiag'):
            d = (g['br_' + c] - g.br_A).values; summ[(S, n, c)] = d.mean()
            bs = [np.mean(np.concatenate([d[b == u] for u in rng.choice(ub, len(ub))])) for _ in range(2000)]
            line += f' | {c} {g["br_" + c].mean():.4f} ({c}-A {d.mean():+.5f} [{np.percentile(bs, 5):+.5f}, {np.percentile(bs, 95):+.5f}], better on {sum(np.mean(d[b == u]) < 0 for u in ub)}/{len(ub)})'
        out.append(line)
for c in ('B', 'C', 'D', 'Cdiag'):   # common summary: budgets 1 and 3 (both sets have records); budget 7 reported separately
    out.append(f'summary {c}-A, budgets 1+3 equal weight: ablate {np.mean([summ[("ablate", n, c)] for n in (1, 3)]):+.5f} | multi {np.mean([summ[("multi", n, c)] for n in (1, 3)]):+.5f} | '
               f'both sets equal weight {np.mean([summ[(S, n, c)] for S in ("ablate", "multi") for n in (1, 3)]):+.5f} | budget 7: ablate '
               + (f'{summ[("ablate", 7, c)]:+.5f}' if ("ablate", 7, c) in summ else 'N/A') + ', multi ' + (f'{summ[("multi", 7, c)]:+.5f}' if ("multi", 7, c) in summ else 'N/A'))
print('\n'.join(out)); open('data/kdiag/level_panel.txt', 'w').write('\n'.join(out) + '\n')
pd.DataFrame([dict(set=r.set, seed=r.seed, draw=r.draw, pair=r.pair, n=r.n, bench=r.bench, item=j, **{k: float(P[k][i][j]) for k in P})
              for i, r in enumerate(T.itertuples()) for j in range(len(r.c))]).to_csv('data/kdiag/level_panel_probs.csv', index=False)
