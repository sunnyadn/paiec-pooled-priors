"""Zero-call check on the new benchmarks, step 3: ALC and per-budget Brier per arm; paired differences of the prior arm against
the other three over draws (mean, standard error, share of draws better) and a 90% interval from a bootstrap over benchmarks."""
import json, numpy as np
S = 'data/llm/extra/sim'; ARMS = ['prior', 'noprior', 'laplace', 'empirical_mean']; B = ['0', '1', '3', '7', '15', '31']
O = {a: sum((json.load(open(f'{S}/out_{a}_s{s}.json')) for s in (5, 6, 7)), []) for a in ARMS}
nd = len(O['prior']); assert all(len(O[a]) == nd for a in ARMS)
draw_alc = {a: np.array([np.mean([r['alc'] for r in d.values()]) for d in O[a]]) for a in ARMS}
print(f'draws {nd}')
for a in ARMS:
    pb = {n: np.mean([r['brier'][n] for d in O[a] for r in d.values()]) for n in B}
    print(f'{a:15s} ALC {draw_alc[a].mean():.4f} | ' + ' '.join(f'b{n} {v:.4f}' for n, v in pb.items()))
rng = np.random.default_rng(0)
rows = [(k.split('|')[0], {a: O[a][i][k]['alc'] for a in ARMS}) for i in range(nd) for k in O['prior'][i]]   # (benchmark, ALC per arm) per pair-draw
bench = np.array([b for b, _ in rows]); ub = np.unique(bench)
for a in ARMS[1:]:
    d = draw_alc['prior'] - draw_alc[a]
    pd_ = np.array([r['prior'] - r[a] for _, r in rows])
    bs = [np.mean(np.concatenate([pd_[bench == x] for x in rng.choice(ub, len(ub))])) for _ in range(2000)]
    print(f'prior - {a:15s} {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(nd):.4f} (draws)  prior better in {np.mean(d < 0):.0%} of draws'
          f' | per pair {pd_.mean():+.4f}, 90% by benchmark [{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}]')
