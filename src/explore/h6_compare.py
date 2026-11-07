"""H6 full-path check (zero calls): v27 (model_v27.py, H6 slope 0.4) against v26 run in the same scenario on identical cached draws, both with the
fixed rating cache, ACQ31=1 and the submitted labeling; scenario COLD_DISK=1: every draw is one independent evaluation starting from an
empty rating cache, reused across its six budgets (a rating-cap scenario, not the platform's full multi-process behaviour). Every evaluation item is scored against all its recorded responses (as the platform),
pairs equal within a draw, draws equal: every pair carries weight 1 / (pairs in its draw) in every mean, per budget, per benchmark and in
the bootstrap over benchmarks (90% interval). Pre-registered rule: SUPPORT if v27 - v26 ALC <= -0.001 in both sets and no (set, budget) mean difference > +0.002.
usage: h6_compare.py V26DIR V27DIR [OUT [NAME [BASE]]]  ->  OUT (default data/kdiag/h6_compare.txt); DIRs hold {set}_s{seed}.json PREDLOG files"""
import json, sys, numpy as np, pandas as pd
V26, V27 = sys.argv[1], sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else 'data/kdiag/h6_compare.txt'; NAME = sys.argv[4] if len(sys.argv) > 4 else 'v27'; BASE = sys.argv[5] if len(sys.argv) > 5 else 'v26'   # arm labels in the output (difference = NAME minus BASE)
import os
for d in {os.path.dirname(V26.rstrip('/')), os.path.dirname(V27.rstrip('/'))}:   # only finished, unfailed run directories
    if os.path.exists(os.path.join(d, 'FAILED')) or not os.path.exists(os.path.join(d, 'DONE')): sys.exit(f'{d}: run not finished or FAILED')
B = [0, 1, 3, 7, 15, 31]; W = dict(zip(B, [.1, .2, .2, .2, .2, .1])); rng = np.random.default_rng(0)
L = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y']); L['item_id'] = L.item_id.astype(str); L['subject_id'] = L.subject_id.astype(str)
G = L.groupby(['bench', 'subject_id', 'item_id']).y.agg(['sum', 'count'])
out, ok, DUMP = [], {}, {}
for S in ('ablate', 'multi'):
    R = {}   # arm -> (seed, draw, pair) -> {budget: brier}
    for arm, path in (('v26', V26 + '/{S}_s{s}.json'), ('v27', V27 + '/{S}_s{s}.json')):
        for s in (5, 6, 7):
            D = [json.loads(l) for l in open(f'data/draws/{S}_s{s}.jsonl')]
            for r in json.load(open(path.format(S=S, s=s))):
                x = D[r['draw']][r['pair']]; ids = [str(i) for i, _ in x['ev']]; p = np.array(r['p'], float)
                assert len(ids) == len(p) and np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
                g = G.loc[[(x['b'], str(x['s']), i) for i in ids]]; c, n = g['sum'].values.astype(float), g['count'].values.astype(float)
                rec = R.setdefault(arm, {}).setdefault((s, r['draw'], r['pair']), {'b': x['b']})
                assert r['n'] not in rec, f'duplicate budget {r["n"]} in {arm} {S} s{s} draw {r["draw"]} pair {r["pair"]}'
                rec[r['n']] = (float(np.sum(c * (1 - p) ** 2 + (n - c) * p ** 2) / n.sum()), p, (r['revealed'], r['k'], r['got'], r['y']))
    want = sorted((s, d, i) for s in (5, 6, 7) for d, dr in enumerate(json.loads(l) for l in open(f'data/draws/{S}_s{s}.jsonl')) for i in range(len(dr)))
    for a in R: assert sorted(R[a]) == want, f'{a} {S}: pairs differ from the frozen draw lists'   # every pair of every draw, nothing extra
    keys = want
    for k in keys:   # identical labels in both arms (the submitted labeling ignores predictions), exactly the six budgets
        assert all(set(R[a][k]) - {'b'} == set(B) for a in R), f'budgets of {k} are not exactly {B}'
        assert all(R['v26'][k][n][2] == R['v27'][k][n][2] for n in B), f'revealed/k/got/y differ between arms in {k}'
    alc = {a: np.array([sum(W[n] * R[a][k][n][0] for n in B) for k in keys]) for a in R}
    d = alc['v27'] - alc['v26']; bench = np.array([R['v26'][k]['b'] for k in keys]); ub = np.unique(bench)
    draws = sorted({k[:2] for k in keys}); npd = {q: sum(k[:2] == q for k in keys) for q in draws}
    w = np.array([1.0 / npd[k[:2]] for k in keys]); wm = lambda v, m=None: float(np.sum((w * v)[m]) / np.sum(w[m])) if m is not None else float(np.sum(w * v) / np.sum(w))
    dd = np.array([d[[k[:2] == q for k in keys]].mean() for q in draws])   # per draw, pairs equal
    bs = []
    for _ in range(2000):
        pick = rng.choice(ub, len(ub)); idx = np.concatenate([np.where(bench == u)[0] for u in pick]); bs.append(np.sum(w[idx] * d[idx]) / np.sum(w[idx]))
    changed = {n: wm(np.array([float(not np.array_equal(R['v26'][k][n][1], R['v27'][k][n][1])) for k in keys])) for n in B}
    pb = {n: wm(np.array([R['v27'][k][n][0] - R['v26'][k][n][0] for k in keys])) for n in B}
    ok[S] = (dd.mean(), pb)
    DUMP[S] = dict(alc_diff=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))), ci_lo=float(np.percentile(bs, 5)), ci_hi=float(np.percentile(bs, 95)), per_budget={str(n): float(v) for n, v in pb.items()})
    out.append(f'{S}: {len(draws)} draws, {len(keys)} pairs, {len(ub)} benchmarks | ALC v26 {wm(alc["v26"]):.4f} v27 {wm(alc["v27"]):.4f} | v27-v26 {dd.mean():+.5f} +- {dd.std(ddof=1) / np.sqrt(len(dd)):.5f} (draws), '
               f'v27 better in {np.mean(dd < 0):.0%} of draws | 90% by benchmark [{np.percentile(bs, 5):+.5f}, {np.percentile(bs, 95):+.5f}], better on {sum(wm(d, bench == u) < 0 for u in ub)}/{len(ub)}')
    out.append('  per budget v27-v26: ' + ', '.join(f'b{n} {v:+.5f}' for n, v in pb.items()) + ' | share of pair-budgets with changed predictions: ' + ', '.join(f'b{n} {v:.0%}' for n, v in changed.items()))
    out.append('  per benchmark ALC v27-v26: ' + ', '.join(f'{u} {wm(d, bench == u):+.4f}' for u in ub))
sup = all(v[0] <= -0.001 for v in ok.values()) and all(x <= 0.002 for v in ok.values() for x in v[1].values())
out.append('RULE (both sets ALC <= -0.001, no set-budget mean > +0.002): ' + ('SUPPORT' if sup else 'NOT MET'))
bad = any(x > 0.002 for v in ok.values() for x in v[1].values()) or all(v[0] >= 0 for v in ok.values())
out.append('THREE-WAY (round 5+): ' + ('SUPPORT' if sup else 'NOT PROMOTED' if bad else 'INCONCLUSIVE') + ' (NOT PROMOTED = some set-budget mean > +0.002 or both sets ALC >= 0)')
if os.environ.get('CMP_JSON'):   # full-precision results for a downstream verdict (no rounding)
    DUMP['three_way'] = 'SUPPORT' if sup else 'NOT PROMOTED' if bad else 'INCONCLUSIVE'
    json.dump(DUMP, open(os.environ['CMP_JSON'], 'w'), indent=1)
out = [l.replace('v26', '\x00').replace('v27', NAME).replace('\x00', BASE) for l in out]
print('\n'.join(out)); open(OUT, 'w').write('\n'.join(out) + '\n')
