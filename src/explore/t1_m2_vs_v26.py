"""T1 (zero calls, round 3): M2 against the full v26 on the same records. Population as m2_slope_test.py, extended to budgets 15/31; v26 = the
predictions logged by sim_sh.py PREDLOG (kdiag K4 = v26 defaults, fixed rating cache). PRIMARY = the original M2: slope per held-out benchmark chosen
exactly as in M2 (budgets 1/3/7 of the other benchmarks). SECONDARY, reported apart = the H6 spec's fixed slope 0.4 (H6s04). Each difference is split into a level part
(response-weighted mean p vs mean outcome, squared) and the centred residual (the rest; not by itself evidence of better item ordering).
H6s04 keeps M2's fallback (any revealed rating missing -> Beta A), so it is a fixed-0.4 diagnostic, not a replication of the H6 deployment spec.
Solver unchanged (25 undamped Newton steps, as M1/M2); the gradient left after the last step is logged per solve and summarised per method and cell. usage: t1_m2_vs_v26.py  ->  data/kdiag/t1_m2_vs_v26.txt"""
import glob, json, re, numpy as np, pandas as pd
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
            if r['n'] not in (1, 3, 7, 15, 31) or not sp or None in r['sp'] or len(set(sp)) != 1 or r['single'] != r['calls']: continue   # as level_panel_test.py
            x = D[r['draw']][r['pair']]; ev = [str(i) for i, _ in x['ev']]; rev = r['revealed']
            ys = {str(i): int(L.y.iat[L.index.get_loc(row)]) for i, row in x['stream']}
            g = G.loc[[(x['b'], str(x['s']), i) for i in ev]]
            yf = [int(L.y.iat[L.index.get_loc(row)]) for _, row in x['ev']]; assert yf == [int(v) for v in r['y']], 'logged y not in ev order'   # the row the sim scored (draw file)
            pv = np.array(r['p'], float); assert len(pv) == len(r['y']) == len(ev) and np.all(np.isfinite(pv)) and np.all((pv >= 0) & (pv <= 1)), 'logged p malformed'
            rows.append(dict(set=S, n=r['n'], p26=np.array(r['p'], float), bench=x['b'], m=sp[0], k=r['k'], got=r['got'], c=g['sum'].values.astype(float), nr=g['count'].values.astype(float),
                             rev_y=[ys[i] for i in rev], rev_z=[Z(i) for i in rev], ev_z=[Z(i) for i in ev]))
T = pd.DataFrame(rows); assert all(T.k == T.rev_y.map(sum)); T['rev_ok'] = T.rev_z.map(lambda v: len(v) > 0 and None not in v)
def rasch(r, centre, slope=1.0, shift=0.0, K=4):
    y = np.array(r.rev_y); er = slope * (np.array(r.rev_z) - centre); ee = slope * np.array([0.0 if z is None else z - centre for z in r.ev_z])
    mu = lg(r.m) + shift; prec = (K + 1) * r.m * (1 - r.m); th = mu
    for _ in range(25):   # as level_panel_test.py
        p = sg(th + er); th += (np.sum(y - p) - prec * (th - mu)) / (np.sum(p * (1 - p)) + prec)
    DIAG.append((TAG[0], abs(np.sum(y - sg(th + er)) - prec * (th - mu)) if np.isfinite(th) else np.inf))   # gradient left after the 25 steps (solver unchanged)
    return sg(th + ee)
DIAG, TAG = [], ['train']
GMX = lambda excl: np.mean([z for bb, v in ITB.items() if bb not in excl for z in v if z is not None])   # mean rated z over benchmarks not in excl
cB = lambda r, excl: (np.sum(r.rev_z) + 5 * GMX(excl)) / (len(r.rev_z) + 5)
brier = lambda c, nr, p: float(np.sum(c * (1 - p) ** 2 + (nr - c) * p ** 2) / np.sum(nr))
A_ = (T.k + 4 * T.m) / (T.got + 4)
lvl = lambda c, nr, p: float((np.sum(nr * p) / np.sum(nr) - np.sum(c) / np.sum(nr)) ** 2)
P = {'A': [np.full(len(r.c), a) for (_, r), a in zip(T.iterrows(), A_)], 'v26': list(T.p26)}
SL = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]; CH = {}; P['M2'] = [None] * len(T); P['H6s04'] = [None] * len(T)
pos = {ix: j for j, ix in enumerate(T.index)}
for hb in T.bench.unique():   # within the fold, no quantity uses the held-out benchmark: training centres exclude hb as well as their own bench
    tr = T[(T.bench != hb) & T.rev_ok & T.n.isin([1, 3, 7])]
    score = {s_: np.mean([brier(r.c, r.nr, rasch(r, cB(r, {hb, r.bench}), slope=s_)) for _, r in tr.iterrows()] + [brier(r.c, r.nr, np.full(len(r.c), a)) for (_, r), a in zip(T.iterrows(), A_) if r.bench != hb and not r.rev_ok and r.n in (1, 3, 7)]) for s_ in SL}
    best = min(SL, key=score.get); CH[hb] = best
    for ix, r in T[T.bench == hb].iterrows():
        j = pos[ix]
        TAG[0] = ('M2', r.set, r.n); P['M2'][j] = rasch(r, cB(r, {hb}), slope=best) if r.rev_ok else P['A'][j]
        TAG[0] = ('H6s04', r.set, r.n); P['H6s04'][j] = rasch(r, cB(r, {hb}), slope=0.4) if r.rev_ok else P['A'][j]
        TAG[0] = 'train'
for k in P:
    T['br_' + k] = [brier(r.c, r.nr, p) for (_, r), p in zip(T.iterrows(), P[k])]; T['lv_' + k] = [lvl(r.c, r.nr, p) for (_, r), p in zip(T.iterrows(), P[k])]
pop = {}
for S in ('ablate', 'multi'):
    allr = [r for seed in (5, 6, 7) for r in json.load(open(f'kdiag/{S}_s{seed}_k4.json'))]
    for n in (1, 3, 7, 15, 31): pop[(S, n)] = (int(((T.set == S) & (T.n == n)).sum()), sum(r['n'] == n for r in allr), int(((T.set == S) & (T.n == n) & ~T.rev_ok).sum()))
CELLS = [('ablate', 1), ('ablate', 3), ('ablate', 7), ('multi', 1), ('multi', 3)]   # pre-registered primary cells; 15/31 reported apart
out = ['population (in / all records, of which revealed ratings incomplete -> Beta A fallback): ' + ', '.join(f'{S} n={n} {a}/{b} ({c})' for (S, n), (a, b, c) in pop.items())]
out += [f'records {len(T)} | revealed ratings complete {int(T.rev_ok.sum())} | y alignment checked | slope per held-out benchmark {CH}']
for S in ('ablate', 'multi'):
    for n in (1, 3, 7, 15, 31):
        g = T[(T.set == S) & (T.n == n)]
        if not len(g): out.append(f'{S} n={n}: N/A'); continue
        b = g.bench.values; ub = np.unique(b); line = f'{S} n={n} recs {len(g)} | A {g.br_A.mean():.4f} v26 {g.br_v26.mean():.4f} M2 {g.br_M2.mean():.4f} H6s04 {g.br_H6s04.mean():.4f}'
        for c1, c2 in (('M2', 'v26'), ('H6s04', 'v26'), ('A', 'v26')):
            d = (g['br_' + c1] - g['br_' + c2]).values; dl = (g['lv_' + c1] - g['lv_' + c2]).values
            bs = [np.mean(np.concatenate([d[b == u] for u in rng.choice(ub, len(ub))])) for _ in range(2000)]
            line += f' | {c1}-{c2} {d.mean():+.5f} [{np.percentile(bs, 5):+.5f}, {np.percentile(bs, 95):+.5f}] (level {dl.mean():+.5f}, centred residual {d.mean() - dl.mean():+.5f}), better on {sum(np.mean(d[b == u]) < 0 for u in ub)}/{len(ub)}'
        out.append(line)
# reproduction check against the earlier M2 output: per-fold slopes and M2 - A in budgets 1/3/7
old = open('data/kdiag/m2_slope.txt').read()
oldCH = {b: float(v) for b, v in re.findall(r"'(\w+)': \{'slope': ([\d.]+)", old.splitlines()[0])}
assert oldCH == CH, f'slopes differ from m2_slope.txt: {oldCH} vs {CH}'
for S, n, v in re.findall(r'^(ablate|multi) n=(\d+) recs .*?M2-A ([+-][\d.]+)', old, re.M):
    g = T[(T.set == S) & (T.n == int(n))]; new_ = (g.br_M2 - g.br_A).mean()
    assert abs(new_ - float(v)) < 1e-5, f'M2-A differs from m2_slope.txt in {S} n={n}: {new_:+.5f} vs {v}'
out.append(f'reproduces m2_slope.txt: slopes {CH == oldCH}, M2-A in every earlier cell within 1e-5')
dg = pd.DataFrame(DIAG, columns=['tag', 'grad']); dg['bad'] = dg.grad > 1e-6   # solver diagnostic (SecondEye f7f9571e): not converged = |gradient| > 1e-6 after 25 steps
out.append(f'solver, training-fold solves: {len(dg[dg.tag == "train"])}, not converged {int(dg[dg.tag == "train"].bad.sum())}, max |grad| {dg[dg.tag == "train"].grad.max():.2e}')
for t, g in dg[dg.tag != 'train'].groupby(dg.tag.astype(str)):
    out.append(f'solver {t}: solves {len(g)}, not converged {int(g.bad.sum())}, max |grad| {g.grad.max():.2e}')
missing = [c for c in CELLS if not ((T.set == c[0]) & (T.n == c[1])).any()]
if missing:
    out.append(f'PRIMARY rule NOT APPLIED: cells missing {missing}'); print('\n'.join(out)); open('data/kdiag/t1_m2_vs_v26.txt', 'w').write('\n'.join(out) + '\n'); raise SystemExit(1)
dm = {c: (T[(T.set == c[0]) & (T.n == c[1])].br_M2 - T[(T.set == c[0]) & (T.n == c[1])].br_v26).mean() for c in CELLS if ((T.set == c[0]) & (T.n == c[1])).any()}
sup = sum(v <= -0.002 for v in dm.values()) >= 3 and all(v <= 0.002 for v in dm.values())
drop = all(abs(v) < 0.001 for v in dm.values()) or sum(v > 0 for v in dm.values()) >= 3
out.append(f'PRIMARY rule (M2 - v26, five cells {dict((a + str(b), round(v, 5)) for (a, b), v in dm.items())}): ' + ('SUPPORT H6' if sup else 'DROP H6' if drop else 'INCONCLUSIVE'))
print('\n'.join(out)); open('data/kdiag/t1_m2_vs_v26.txt', 'w').write('\n'.join(out) + '\n')
