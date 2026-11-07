"""New-benchmark check, step 3: prior vs constant 0.5 vs data-only predictor on the same pairs; answers-equivalent as in
Table 1; pair-pooled and benchmark-weighted; bootstrap over benchmarks (main) and over models (robustness); then pooled with
the 277 main pairs. Failed or unparsed estimates fall back to 0.5; untried pairs are left out and counted."""
import json, numpy as np
E = {f"{e['bench']}|{e['subject_id']}": e for e in json.load(open('data/llm/extra/eligible.json'))}
D = json.load(open('data/llm/extra/data_pred.json')); RUN = json.load(open('data/llm/extra/results.json'))
R = RUN['results']; rng = np.random.default_rng(0)
cnt = {s: sum(r['status'] == s for r in R.values()) for s in ('ok', 'unparsed', 'failed', 'untried')}
print('stop:', RUN['stop'], '| status', cnt, '| answered by', {m: sum(r.get('answered_by') == m for r in R.values()) for m in ('claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5')})
K = [k for k, r in R.items() if r['status'] != 'untried']
laplace_mse = lambda a, n: np.mean((n * a * (1 - a) + (1 - 2 * a) ** 2) / (n + 2) ** 2)
labels = lambda a, mse: next((n for n in range(500) if laplace_mse(a, n) <= mse), 500)   # 500 = more than 499 answers

def arrays(keys):
    a = np.array([E[k]['acc'] for k in keys]); est = [R[k].get('estimate') for k in keys]
    p = np.array([0.5 if v is None else 0.5 + 0.75 * (v - 0.5) for v in est]); d = np.array([D[k]['p'] for k in keys])
    return a, p, d, np.array([E[k]['bench'] for k in keys]), np.array([E[k]['subject']['normalized_name'] for k in keys])

def report(label, a, p, d, b, n):
    ub = np.unique(b); bw = lambda x: np.mean([np.mean(x[b == u]) for u in ub])
    print(f'{label}: pairs {len(a)} benchmarks {len(ub)} models {len(np.unique(n))}')
    for nm, q in (('prior', p), ('constant 0.5', np.full(len(a), 0.5)), ('data-only', d)):
        se = (q - a) ** 2; out = []
        for cl in (b, n):
            uc = np.unique(cl); L = []
            for _ in range(2000):
                ix = np.concatenate([np.where(cl == x)[0] for x in rng.choice(uc, len(uc))]); L.append(labels(a[ix], np.mean(se[ix])))
            out.append(f'[{np.percentile(L, 5):.0f}, {np.percentile(L, 95):.0f}]')
        print(f'   {nm:13s} RMSE pooled {np.sqrt(se.mean()):.3f}  benchmark-weighted {np.sqrt(bw(se)):.3f}  answers {labels(a, se.mean())}'
              f'  90% by benchmark {out[0]}  by model {out[1]}')
    for nm, q in (('prior - constant', np.full(len(a), 0.5)), ('prior - data-only', d)):   # paired MSE differences on the same pairs
        dd = (p - a) ** 2 - (q - a) ** 2; ci = {}
        for cn, cl in (('benchmark', b), ('model', n)):
            uc = np.unique(cl); P_, W_ = [], []
            for _ in range(2000):
                pick = rng.choice(uc, len(uc)); ix = np.concatenate([np.where(cl == x)[0] for x in pick]); P_.append(dd[ix].mean())
                if cn == 'benchmark': W_.append(np.mean([dd[b == x].mean() for x in pick]))   # drawn benchmarks keep their multiplicity
                else: W_.append(np.mean([dd[ix][b[ix] == u].mean() for u in np.unique(b[ix])]))   # benchmark-weighted statistic per resample
            ci[cn] = (np.percentile(P_, [5, 95]), np.percentile(W_, [5, 95]))
        f = lambda x: f'[{x[0]:+.4f}, {x[1]:+.4f}]'
        print(f'   MSE {nm:17s} pooled {dd.mean():+.4f} 90% by benchmark {f(ci["benchmark"][0])} by model {f(ci["model"][0])} | '
              f'benchmark-weighted {bw(dd):+.4f} 90% by benchmark {f(ci["benchmark"][1])} by model {f(ci["model"][1])}  (negative = prior better)')

a, p, d, b, n = arrays(K); report('new benchmarks, all tried pairs', a, p, d, b, n)
has = np.array([D[k]['other_scores'] > 0 for k in K]); report('  pairs whose model has other benchmark scores', a[has], p[has], d[has], b[has], n[has])
# pooled with the 277 main pairs (prior and constant only; data-only uses the same rule, from reproduce.py)
M = json.load(open('data/llm/prior_eval_all.json')); O = json.load(open('data/llm/prior_eval_oall.json'))
S, H = json.load(open('data/llm/prior_eval_smiss.json')), json.load(open('data/llm/prior_eval_hmiss.json'))
num = lambda x: x[0] if isinstance(x[0], (int, float)) else np.nan
est = np.array([num(o) for o in O], float); skip = np.array([bool(o[1].get('skip')) for o in O])
miss = [i for i in range(len(O)) if np.isnan(est[i]) and not skip[i]]
for j, i in enumerate(miss): est[i] = num(S[j]) if not np.isnan(num(S[j])) else num(H[j])
bench = np.array([m['bench'] for m in M]); calib = set(bench[skip]); keep = ~skip & ~np.isin(bench, list(calib))
a0 = np.array([m['acc'] for m in M])[keep]; p0 = 0.5 + 0.75 * (est[keep] - 0.5)
A, P, B = np.r_[a0, a], np.r_[p0, p], np.r_[bench[keep], b]; ub = np.unique(B); L = []
for _ in range(2000):
    ix = np.concatenate([np.where(B == x)[0] for x in rng.choice(ub, len(ub))]); L.append(labels(A[ix], np.mean((P[ix] - A[ix]) ** 2)))
print(f'pooled with the main 277: pairs {len(A)} benchmarks {len(ub)} | RMSE prior {np.sqrt(np.mean((P - A) ** 2)):.3f} constant {np.sqrt(np.mean((0.5 - A) ** 2)):.3f}'
      f' | answers {labels(A, np.mean((P - A) ** 2))} [{np.percentile(L, 5):.0f}, {np.percentile(L, 95):.0f}] (by benchmark)')
