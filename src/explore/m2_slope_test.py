"""M2 (zero calls): early loss of the Rasch update comes from a too-large rating slope. Same population, labels and smoothed centre (c0 = 5)
as m1_center_test.py; slope chosen per held-out benchmark from {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} by Brier on the other benchmarks (budgets 1/3/7
pooled, rows weight 1), then scored once on the held-out benchmark. usage: m2_slope_test.py  ->  data/kdiag/m2_slope.txt"""
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
def rasch(r, centre, slope=1.0, shift=0.0, K=4):
    y = np.array(r.rev_y); er = slope * (np.array(r.rev_z) - centre); ee = slope * np.array([0.0 if z is None else z - centre for z in r.ev_z])
    mu = lg(r.m) + shift; prec = (K + 1) * r.m * (1 - r.m); th = mu
    for _ in range(25):   # as level_panel_test.py
        p = sg(th + er); th += (np.sum(y - p) - prec * (th - mu)) / (np.sum(p * (1 - p)) + prec)
    return sg(th + ee)
GMX = lambda excl: np.mean([z for bb, v in ITB.items() if bb not in excl for z in v if z is not None])   # mean rated z over benchmarks not in excl
cB = lambda r, excl: (np.sum(r.rev_z) + 5 * GMX(excl)) / (len(r.rev_z) + 5)
brier = lambda c, nr, p: float(np.sum(c * (1 - p) ** 2 + (nr - c) * p ** 2) / np.sum(nr))
A_ = (T.k + 4 * T.m) / (T.got + 4); T['br_A'] = [brier(c, nr, np.full(len(c), a)) for c, nr, a in zip(T.c, T.nr, A_)]
SL = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]; T['br_M2'] = np.nan; T['br_Db'] = np.nan; CH = {}
for hb in T.bench.unique():   # within the fold, no quantity uses the held-out benchmark: training centres exclude hb as well as their own bench
    tr = T[(T.bench != hb) & T.rev_ok]
    score = {s_: np.mean([brier(r.c, r.nr, rasch(r, cB(r, {hb, r.bench}), slope=s_)) for _, r in tr.iterrows()] + list(T[(T.bench != hb) & ~T.rev_ok].br_A)) for s_ in SL}
    best = min(SL, key=score.get); CH[hb] = dict(slope=best, train_brier={s_: round(v, 5) for s_, v in score.items()})
    te = T[T.bench == hb]
    T.loc[te.index, 'br_M2'] = [brier(r.c, r.nr, rasch(r, cB(r, {hb}), slope=best)) if r.rev_ok else r.br_A for _, r in te.iterrows()]
    T.loc[te.index, 'br_Db'] = [brier(r.c, r.nr, rasch(r, cB(r, {hb}), slope=1.0)) if r.rev_ok else r.br_A for _, r in te.iterrows()]   # fixed control: D_b at slope 1
out = [f'records {len(T)} | revealed ratings complete {int(T.rev_ok.sum())} | slope chosen per held-out benchmark {CH}']
for S in ('ablate', 'multi'):
    for n in (1, 3, 7):
        g = T[(T.set == S) & (T.n == n)]
        if not len(g): out.append(f'{S} n={n}: N/A'); continue
        b = g.bench.values; ub = np.unique(b); line = f'{S} n={n} recs {len(g)} | A {g.br_A.mean():.4f} | M2 {g.br_M2.mean():.4f} | D_b {g.br_Db.mean():.4f}'
        for c2 in ('br_A', 'br_Db'):
            d = (g.br_M2 - g[c2]).values; bs = [np.mean(np.concatenate([d[b == u] for u in rng.choice(ub, len(ub))])) for _ in range(2000)]
            line += f' | M2-{c2[3:]} {d.mean():+.5f} [{np.percentile(bs, 5):+.5f}, {np.percentile(bs, 95):+.5f}], better on {sum(np.mean(d[b == u]) < 0 for u in ub)}/{len(ub)}'
        out.append(line)
print('\n'.join(out)); open('data/kdiag/m2_slope.txt', 'w').write('\n'.join(out) + '\n')
