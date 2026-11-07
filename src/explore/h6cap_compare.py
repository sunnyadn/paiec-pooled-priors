"""Rating-availability check (zero calls), cold scenario (COLD_DISK=1): v26 and v27 with the simulated rating cap (300 per draw, round 4
runs) and without it (PP_RATEMAX=10**9; cache misses still give no rating, no API). Scored as h6_compare.py (every response, pairs equal
within a draw, draws equal, weight 1 / pairs in the draw; bootstrap over benchmarks for 90% intervals).
Pre-registered: (1) rating availability matters for v26 if v26 uncapped - capped ALC <= -0.002 in both sets;
(2) H6 dilution supported if the interaction (v27 - v26 uncapped) - (v27 - v26 capped) <= -0.0005 in ablate with its 90% interval below 0,
not supported if the ablate interaction >= 0, otherwise inconclusive; (3) H6 uncapped judged by the round-4 rule as well (both sets
v27 - v26 <= -0.001, no set-budget mean > +0.002), reported, not a deployment decision.
usage: h6cap_compare.py CAPDIR NOCAPDIR  (each holds v26/ and v27/ with {set}_s{seed}.json, plus DONE)  ->  data/kdiag/h6cap_compare.txt"""
import json, os, sys, numpy as np, pandas as pd
CAP, NOCAP = sys.argv[1], sys.argv[2]
for d in (CAP, NOCAP):
    if os.path.exists(os.path.join(d, 'FAILED')) or not os.path.exists(os.path.join(d, 'DONE')): sys.exit(f'{d}: run not finished or FAILED')
B = [0, 1, 3, 7, 15, 31]; W = dict(zip(B, [.1, .2, .2, .2, .2, .1])); rng = np.random.default_rng(0)
L = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y']); L['item_id'] = L.item_id.astype(str); L['subject_id'] = L.subject_id.astype(str)
G = L.groupby(['bench', 'subject_id', 'item_id']).y.agg(['sum', 'count'])
ARMS = {'v26cap': f'{CAP}/v26', 'v27cap': f'{CAP}/v27', 'v26nocap': f'{NOCAP}/v26', 'v27nocap': f'{NOCAP}/v27'}
out, res = [], {}
for S in ('ablate', 'multi'):
    want = sorted((s, d, i) for s in (5, 6, 7) for d, dr in enumerate(json.loads(l) for l in open(f'data/draws/{S}_s{s}.jsonl')) for i in range(len(dr)))
    R = {}
    for arm, base in ARMS.items():
        for s in (5, 6, 7):
            D = [json.loads(l) for l in open(f'data/draws/{S}_s{s}.jsonl')]
            for r in json.load(open(f'{base}/{S}_s{s}.json')):
                x = D[r['draw']][r['pair']]; ids = [str(i) for i, _ in x['ev']]; p = np.array(r['p'], float)
                assert len(ids) == len(p) and np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
                g = G.loc[[(x['b'], str(x['s']), i) for i in ids]]; c, n = g['sum'].values.astype(float), g['count'].values.astype(float)
                rec = R.setdefault(arm, {}).setdefault((s, r['draw'], r['pair']), {'b': x['b']})
                assert r['n'] not in rec, f'duplicate budget in {arm} {S}'
                rec[r['n']] = (float(np.sum(c * (1 - p) ** 2 + (n - c) * p ** 2) / n.sum()), (r['revealed'], r['k'], r['got'], r['y']))
        assert sorted(R[arm]) == want, f'{arm} {S}: pairs differ from the frozen draw lists'
    for k in want:
        assert all(set(R[a][k]) - {'b'} == set(B) for a in R), f'budgets of {k}'
        assert all(R[a][k][n][1] == R['v26cap'][k][n][1] for a in R for n in B), f'labels differ between arms in {k}'
    bench = np.array([R['v26cap'][k]['b'] for k in want]); ub = np.unique(bench)
    npd = {}
    for k in want: npd[k[:2]] = npd.get(k[:2], 0) + 1
    w = np.array([1.0 / npd[k[:2]] for k in want]); wm = lambda v: float(np.sum(w * v) / np.sum(w))
    alc = {a: np.array([sum(W[n] * R[a][k][n][0] for n in B) for k in want]) for a in R}
    pb = {a: {n: np.array([R[a][k][n][0] for k in want]) for n in B} for a in R}
    def ci(v):
        bs = []
        for _ in range(2000):
            idx = np.concatenate([np.where(bench == u)[0] for u in rng.choice(ub, len(ub))]); bs.append(np.sum(w[idx] * v[idx]) / np.sum(w[idx]))
        return np.percentile(bs, 5), np.percentile(bs, 95)
    con = {'v26 uncapped - capped': alc['v26nocap'] - alc['v26cap'], 'v27 uncapped - capped': alc['v27nocap'] - alc['v27cap'],
           'v27 - v26, capped': alc['v27cap'] - alc['v26cap'], 'v27 - v26, uncapped': alc['v27nocap'] - alc['v26nocap'],
           'interaction (uncapped diff - capped diff)': (alc['v27nocap'] - alc['v26nocap']) - (alc['v27cap'] - alc['v26cap'])}
    out.append(f'{S}: {len(npd)} draws, {len(want)} pairs, {len(ub)} benchmarks | ALC ' + ' '.join(f'{a} {wm(v):.4f}' for a, v in alc.items()))
    for name, v in con.items():
        lo, hi = ci(v); res[(S, name)] = (wm(v), lo, hi)
        out.append(f'  {name}: {wm(v):+.5f} [{lo:+.5f}, {hi:+.5f}], better on {sum(np.sum(w[bench == u] * v[bench == u]) < 0 for u in ub)}/{len(ub)}')
    for name, (a1, a0) in (('v26 uncapped - capped', ('v26nocap', 'v26cap')), ('v27 - v26, uncapped', ('v27nocap', 'v26nocap'))):
        dd = {n: wm(pb[a1][n] - pb[a0][n]) for n in B}; res[(S, name, 'pb')] = dd
        out.append(f'  per budget {name}: ' + ', '.join(f'b{n} {v:+.5f}' for n, v in dd.items()))
r1 = all(res[(S, 'v26 uncapped - capped')][0] <= -0.002 for S in ('ablate', 'multi'))
ia = res[('ablate', 'interaction (uncapped diff - capped diff)')]
r2 = 'SUPPORTED' if ia[0] <= -0.0005 and ia[2] < 0 else 'NOT SUPPORTED' if ia[0] >= 0 else 'INCONCLUSIVE'
r3 = all(res[(S, 'v27 - v26, uncapped')][0] <= -0.001 for S in ('ablate', 'multi')) and all(v <= 0.002 for S in ('ablate', 'multi') for v in res[(S, 'v27 - v26, uncapped', 'pb')].values())
out += [f'RULE 1 rating availability matters for v26 (both sets <= -0.002): {"MET" if r1 else "NOT MET"}', f'RULE 2 H6 dilution by missing ratings: {r2}',
        f'RULE 3 H6 uncapped by the round-4 rule: {"MET" if r3 else "NOT MET"}']
print('\n'.join(out)); open('data/kdiag/h6cap_compare.txt', 'w').write('\n'.join(out) + '\n')
