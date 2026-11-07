"""Diagnostic of the submitted model's error (zero calls), from sim_sh.py PREDLOG files; uses true outcomes, so diagnosis only.
Per pair and budget: Brier = (mean p - mean y)^2 [level] + rest [item], exactly, over the evaluation items; plus raw reference scores.
usage: err_breakdown.py DIR SET...   (reads DIR/{SET}_s{5,6,7}.json)"""
import json, sys, numpy as np
D, SETS = sys.argv[1], sys.argv[2:]
B, W = [0, 1, 3, 7, 15, 31], [.1, .2, .2, .2, .2, .1]
lg = lambda p: np.log(p / (1 - p)); sg = lambda z: 1 / (1 + np.exp(-z))

def shift(p, acc):   # constant c with mean(sigmoid(logit(p) + c)) = acc, by bisection
    z, acc = lg(np.clip(p, 1e-4, 1 - 1e-4)), min(max(acc, 1e-3), 1 - 1e-3); lo, hi = -30.0, 30.0
    for _ in range(80):
        c = (lo + hi) / 2; lo, hi = (c, hi) if sg(z + c).mean() < acc else (lo, c)
    return sg(z + (lo + hi) / 2)

COLS = ['model', 'level', 'item', 'noise', 'ref_logit', 'const_all', 'const_ev']
rng = np.random.default_rng(0)
for S in SETS:
    rows = {}   # (seed, draw, pair) -> [benchmark, {budget: scores}]
    for seed in (5, 6, 7):
        for r in json.load(open(f'{D}/{S}_s{seed}.json')):
            p, y = np.array(r['p']), np.array(r['y']); yb = y.mean()
            lvl = (p.mean() - yb) ** 2   # the most any per-pair constant shift in probability can remove
            rows.setdefault((seed, r['draw'], r['pair']), [r['b'], {}])[1][r['n']] = (
                np.mean((y - p) ** 2), lvl, np.mean((y - p) ** 2) - lvl,
                yb * (1 - yb) / max(1, len(y) - 1),   # part of the level term from sampling the evaluation items (approx.)
                np.mean((y - shift(p, r['acc'])) ** 2),   # shifted in logit space to the accuracy over all sampled items
                np.mean((y - r['acc']) ** 2), np.mean((y - yb) ** 2))
    keys = list(rows); bench = np.array([rows[k][0] for k in keys]); ub = np.unique(bench)
    M = np.array([[rows[k][1][n] for n in B] for k in keys])   # pair-draws x budgets x COLS
    M = np.concatenate([M, np.einsum('pbk,b->pk', M, np.array(W))[:, None]], 1)
    print(f'\n== {S}: {len(keys)} pair-draws, {len(ub)} benchmarks; level/item use the mean over evaluation items')
    print(f'{"budget":>7} ' + ' '.join(f'{c:>9}' for c in COLS) + '   level 90% by benchmark')
    for j, n in enumerate(B + ['ALC']):
        lv = M[:, j, 1]
        bs = [np.mean(np.concatenate([lv[bench == x] for x in rng.choice(ub, len(ub))])) for _ in range(2000)]
        print(f'{n!s:>7} ' + ' '.join(f'{v:9.4f}' for v in M[:, j].mean(0)) + f'   [{np.percentile(bs, 5):.4f}, {np.percentile(bs, 95):.4f}]')
