"""K diagnostic (zero calls), from kdiag/{set}_s{seed}_k{4,8}.json prediction logs: per budget, (1) the prior each prediction actually
received, (2) the full model's level term (mean p - mean y)^2 for K 4 vs 8, and (3) on records that went only through _single with one
prior value, the simplified update (k + K * prior) / (got + K) on the same labels. usage: kdiag_analysis.py DIR"""
import json, sys, numpy as np
D = sys.argv[1]; B = [0, 1, 3, 7, 15, 31]; W = [.1, .2, .2, .2, .2, .1]; NDRAW = {'ablate': 90, 'multi': 30}
for S in ('ablate', 'multi'):
    R = {K: [dict(r, seed=s) for s in (5, 6, 7) for r in json.load(open(f'{D}/{S}_s{s}_k{K}.json'))] for K in (4, 8)}
    for K in (4, 8):   # integrity: six budgets per pair-draw, unique records, finite predictions, expected draws
        keys = [(r['seed'], r['draw'], r['pair'], r['n']) for r in R[K]]; assert len(keys) == len(set(keys)), 'duplicate records'
        per = {}
        for r in R[K]: per.setdefault((r['seed'], r['draw'], r['pair']), []).append(r['n'])
        assert all(sorted(v) == B for v in per.values()), 'a pair-draw lacks some budget'
        assert len({k[:2] for k in per}) == NDRAW[S], 'unexpected number of draws'; assert all(np.isfinite(r['p']).all() for r in R[K])
    kk = lambda r: (r['seed'], r['draw'], r['pair'], r['n']); M4, M8 = ({kk(r): r for r in R[K]} for K in (4, 8))
    assert M4.keys() == M8.keys(), 'K 4 and 8 have different pair-budget records'
    assert all((M4[k]['b'], M4[k]['y'], M4[k]['revealed'], M4[k]['got'], M4[k]['k']) == (M8[k]['b'], M8[k]['y'], M8[k]['revealed'], M8[k]['got'], M8[k]['k'])
               for k in M4), 'K 4 and 8 differ in benchmark, evaluation outcomes or revealed labels'
    for r in R[4] + R[8]:
        assert all(v is None or np.isfinite(v) for v in r['sp']) and r['calls'] >= r['single'] >= 0 and 0 <= r['k'] <= r['got'] and len(r['sp']) >= 0
    A, Bk = ([M[k] for k in sorted(M4)] for M in (M4, M8))
    print(f'== {S}: {len(A)} pair-budget records, {NDRAW[S]} draws')
    for n in B:
        a = [r for r in A if r['n'] == n]; b = [r for r in Bk if r['n'] == n]; yb = lambda r: np.mean(r['y'])
        calls = [(v - yb(r)) ** 2 for r in a for v in r['sp'] if v is not None]   # every call's prior, weighted by how often it was used
        has = [r for r in a if any(v is not None for v in r['sp'])]; multi = np.mean([len({v for v in r['sp'] if v is not None}) > 1 for r in has]) if has else np.nan
        perr = np.mean(calls) if calls else np.nan
        lv = lambda rs: np.mean([(np.mean(r['p']) - yb(r)) ** 2 for r in rs])
        one = lambda r: r['sp'] and None not in r['sp'] and len(set(r['sp'])) == 1   # every call got the same, non-empty prior
        ok = [i for i, r in enumerate(a) if r['calls'] and r['single'] == r['calls'] and one(r) and b[i]['single'] == b[i]['calls'] and one(b[i]) and b[i]['sp'][0] == r['sp'][0]]
        simp = {K: np.mean([((a[i]['k'] + K * a[i]['sp'][0]) / (a[i]['got'] + K) - yb(a[i])) ** 2 for i in ok]) if ok else np.nan for K in (4, 8)}
        full = {K: np.mean([(np.mean(rs[i]['p']) - yb(rs[i])) ** 2 for i in ok]) if ok else np.nan for K, rs in ((4, a), (8, b))}
        print(f'  n={n:2d} | with prior {len(has)}/{len(a)}, >1 prior value {multi:.0%}, prior err^2 {perr:.4f} | level term all: K4 {lv(a):.4f} K8 {lv(b):.4f} | '
              f'_single+one prior ({len(ok)} recs): full K4 {full[4]:.4f} K8 {full[8]:.4f}, simplified K4 {simp[4]:.4f} K8 {simp[8]:.4f}')
    for K, rs in ((4, A), (8, Bk)):
        pd_ = {}
        for r in rs: pd_.setdefault((r['seed'], r['draw'], r['pair']), {})[r['n']] = np.mean((np.array(r['y']) - np.array(r['p'])) ** 2)
        alc = {k: sum(w * v[n] for n, w in zip(B, W)) for k, v in pd_.items()}
        draws = {}
        for k, v in alc.items(): draws.setdefault(k[:2], []).append(v)
        print(f'  ALC K={K}: mean over draws {np.mean([np.mean(v) for v in draws.values()]):.4f} (main) | mean over pair-draws {np.mean(list(alc.values())):.4f}')
