"""M1 (zero calls): does the early loss of the fixed Rasch update come from its difficulty centre? Same population and labels as
level_panel_test.py; A Beta, D centre = mean revealed z, D_b centre shrunk toward the training-benchmark mean (c0 = 5), D0 slope 0;
invariance check (c+d, mu+d -> same predictions). usage: m1_center_test.py  ->  data/kdiag/m1_center.txt"""
import glob, json, numpy as np, pandas as pd
lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98))); sg = lambda z: 1 / (1 + np.exp(-z)); rng = np.random.default_rng(0)
rat = {}
for f in glob.glob('data/llm/batches_opus_fixed/*.json'):
    for x in json.load(open(f))['ratings']: rat[str(x['id'])] = float(x['p'])
Z = lambda i: lg(rat[i] / 100.0) if i in rat else None
L = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'trial', 'y']); L['item_id'] = L.item_id.astype(str); L['subject_id'] = L.subject_id.astype(str)
G = L.groupby(['bench', 'subject_id', 'item_id']).y.agg(['sum', 'count'])
ITB = {b: [Z(str(x['item_id'])) for x in v] for b, v in json.load(open('data/llm/items_to_rate.json')).items()}   # rated items per benchmark
rows = []
for S in ('ablate', 'multi'):
    for seed in (5, 6, 7):
        D = [json.loads(l) for l in open(f'data/draws/{S}_s{seed}.jsonl')]
        for r in json.load(open(f'kdiag/{S}_s{seed}_k4.json')):
            sp = [v for v in r['sp'] if v is not None]
            if r['n'] not in (1, 3, 7) or not sp or None in r['sp'] or len(set(sp)) != 1 or r['single'] != r['calls']: continue   # as level_panel_test.py
            x = D[r['draw']][r['pair']]; ev = [str(i) for i, _ in x['ev']]; rev = r['revealed']
            ys = {str(i): int(L.y.iat[L.index.get_loc(row)]) for i, row in x['stream']}
            g = G.loc[[(x['b'], str(x['s']), i) for i in ev]]
            rows.append(dict(set=S, n=r['n'], bench=x['b'], m=sp[0], k=r['k'], got=r['got'], c=g['sum'].values.astype(float), nr=g['count'].values.astype(float),
                             rev_y=[ys[i] for i in rev], rev_z=[Z(i) for i in rev], ev_z=[Z(i) for i in ev]))
T = pd.DataFrame(rows); assert all(T.k == T.rev_y.map(sum)); T['rev_ok'] = T.rev_z.map(lambda v: len(v) > 0 and None not in v)
GM = {b: np.mean([z for bb, v in ITB.items() if bb != b for z in v if z is not None]) for b in T.bench.unique()}   # training-benchmark mean z, held-out bench excluded
def rasch(r, centre, slope=1.0, shift=0.0, K=4):
    y = np.array(r.rev_y); er = slope * (np.array(r.rev_z) - centre); ee = slope * np.array([0.0 if z is None else z - centre for z in r.ev_z])
    mu = lg(r.m) + shift; prec = (K + 1) * r.m * (1 - r.m); th = mu
    for _ in range(25):   # as level_panel_test.py
        p = sg(th + er); th += (np.sum(y - p) - prec * (th - mu)) / (np.sum(p * (1 - p)) + prec)
    return sg(th + ee)
cD = lambda r: np.mean(r.rev_z); cB = lambda r: (np.sum(r.rev_z) + 5 * GM[r.bench]) / (len(r.rev_z) + 5)
worst = 0.0
for _, r in T[T.rev_ok].head(200).iterrows():   # invariance: centre + d with prior mean + d leaves predictions unchanged (slope 1)
    for d in (-1.3, 0.7):
        rated = np.array([z is not None for z in r.ev_z])   # unrated scored items take the current centre (imputation rule), so they are excluded here
        worst = max(worst, np.max(np.abs(rasch(r, cB(r))[rated] - rasch(r, cB(r) + d, shift=d)[rated])) if rated.any() else 0.0)
print('invariance max |diff| over 200 records x 2 shifts:', worst); assert worst < 1e-9
A_ = (T.k + 4 * T.m) / (T.got + 4); P = {'A': [np.full(len(c), a) for c, a in zip(T.c, A_)]}
for name, f in (('D', lambda r: rasch(r, cD(r))), ('D_b', lambda r: rasch(r, cB(r))), ('D0', lambda r: rasch(r, 0.0, slope=0.0))):
    P[name] = [f(r) if r.rev_ok else np.full(len(r.c), a) for (_, r), a in zip(T.iterrows(), A_)]   # missing ratings -> A
brier = lambda c, nr, p: float(np.sum(c * (1 - p) ** 2 + (nr - c) * p ** 2) / np.sum(nr))
for k, v in P.items(): T['br_' + k] = [brier(c, nr, p) for c, nr, p in zip(T.c, T.nr, v)]
import re   # A and D must reproduce level_panel_test.py on the same records
for line in open('data/kdiag/level_panel_run.txt'):
    m = re.match(r'(ablate|multi) n=(\d+) recs .*?Brier A ([\d.]+) .*? D ([\d.]+) \(D-A', line)
    if m:
        g = T[(T.set == m.group(1)) & (T.n == int(m.group(2)))]
        assert abs(g.br_A.mean() - float(m.group(3))) < 6e-5 and abs(g.br_D.mean() - float(m.group(4))) < 6e-5, f'A/D differ from level_panel: {line[:60]}'
print('A and D reproduce level_panel_test.py')
out = [f'records {len(T)} | revealed ratings complete {int(T.rev_ok.sum())} | invariance max diff {worst:.2e}']
for S in ('ablate', 'multi'):
    for n in (1, 3, 7):
        g = T[(T.set == S) & (T.n == n)]
        if not len(g): out.append(f'{S} n={n}: N/A'); continue
        b = g.bench.values; ub = np.unique(b); line = f'{S} n={n} recs {len(g)} | A {g.br_A.mean():.4f}'
        for c1, c2 in (('D', 'A'), ('D_b', 'A'), ('D0', 'A'), ('D_b', 'D')):
            d = (g['br_' + c1] - g['br_' + c2]).values; bs = [np.mean(np.concatenate([d[b == u] for u in rng.choice(ub, len(ub))])) for _ in range(2000)]
            line += f' | {c1}-{c2} {d.mean():+.5f} [{np.percentile(bs, 5):+.5f}, {np.percentile(bs, 95):+.5f}]'
        out.append(line)
print('\n'.join(out)); open('data/kdiag/m1_center.txt', 'w').write('\n'.join(out) + '\n')
