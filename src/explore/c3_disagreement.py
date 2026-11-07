"""C3 (zero calls): Sonnet estimate / Opus-Sonnet disagreement as information beyond Opus on Table 1 held-out pairs; leave-one-benchmark-out
least squares, models A0/B/C/D/E and the pre-registered rule as in ROUND3_CANDIDATES.md (C3 plan). usage: c3_disagreement.py
 -> data/kdiag/c3_disagreement.txt"""
import json, numpy as np
E_ = json.load(open('data/llm/prior_eval.json')); A = json.load(open('submission/anchors.json'))
acc = np.array([r['acc'] for r in E_]); bench = np.array([r['bench'] for r in E_])
anchor_pair = {i for a in A for i, r in enumerate(E_) if r['subject']['normalized_name'] == a['model'] and any(a['items'][0][:150] in t for t in r['texts'])}   # as reproduce.py
est = lambda tag: np.array([x[0] if isinstance(x, list) and x and isinstance(x[0], (int, float)) else np.nan for x in json.load(open(f'data/llm/prior_eval_{tag}.json'))], float)
O, S = est('of5_600'), est('sd5low')
held = np.array([i not in anchor_pair for i in range(len(E_))])
m = held & ~np.isnan(O) & ~np.isnan(S)
assert len(O) == len(S) == len(acc) == len(E_) and np.all(np.isfinite(acc)) and np.all((acc >= 0) & (acc <= 1))
assert np.all((O[m] >= 0) & (O[m] <= 1)) and np.all((S[m] >= 0) & (S[m] <= 1))
out = [f'pairs {len(E_)} | held out {int(held.sum())} | with Opus {int((held & ~np.isnan(O)).sum())} | with Sonnet {int((held & ~np.isnan(S)).sum())} | used (both) {int(m.sum())} | benchmarks {len(set(bench[m]))}']
o, s, y, b = O[m], S[m], acc[m], bench[m]; ub = np.unique(b)
X = {'B': lambda: np.column_stack([np.ones_like(o), o, np.abs(o - .5)])}
X['B5'] = lambda: np.column_stack([X['B'](), o ** 2, o ** 3])   # Opus-only control with as many columns as E (SecondEye de91bf87)
X['C'] = lambda: np.column_stack([X['B'](), s]); X['D'] = lambda: np.column_stack([X['B'](), np.abs(o - s) * (o - .5)]); X['E'] = lambda: np.column_stack([X['B'](), s, np.abs(o - s) * (o - .5)])
pred = {'A0': 0.5 + 0.75 * (o - 0.5)}; diag = {}
for k, f in X.items():
    M = f(); p = np.zeros_like(y); ntr, rk = [], []
    for u in ub:
        tr, te = b != u, b == u
        coef, *_ = np.linalg.lstsq(M[tr], y[tr], rcond=None); p[te] = M[te] @ coef
        ntr.append(int(tr.sum())); rk.append(int(np.linalg.matrix_rank(M[tr])))
    diag[k] = f'{k}: columns {M.shape[1]}, training pairs per fold min {min(ntr)}, rank-deficient folds {sum(r < M.shape[1] for r in rk)}, predictions clipped to [0,1] {int(((p < 0) | (p > 1)).sum())}'
    pred[k] = np.clip(p, 0, 1)
out += list(diag.values())
se = {k: (v - y) ** 2 for k, v in pred.items()}
rng = np.random.default_rng(0)
PICKS = [np.concatenate([np.where(b == u)[0] for u in rng.choice(ub, len(ub))]) for _ in range(4000)]   # one set of benchmark resamples, reused by every comparison
def ci(d):
    bs = [np.mean(d[ix]) for ix in PICKS]
    return np.percentile(bs, 5), np.percentile(bs, 95)
out.append('MSE: ' + ' '.join(f'{k} {v.mean():.5f}' for k, v in se.items()) + f' | corr(|o-s|, |o-acc|) {np.corrcoef(np.abs(o - s), np.abs(o - y))[0, 1]:+.3f} (shares the Opus term; not a test)')
CI = {}
for k, r in (('C', 'B'), ('D', 'B'), ('E', 'B'), ('B5', 'B'), ('E', 'B5')):
    d = se[k] - se[r]; CI[(k, r)] = ci(d); lo, hi = CI[(k, r)]
    out.append(f'{k} - {r}: {d.mean():+.5f} ({d.mean() / se[r].mean():+.1%} of {r}) [{lo:+.5f}, {hi:+.5f}], better on {sum(d[b == u].mean() < 0 for u in ub)}/{len(ub)} benchmarks')
for k in ('B', 'E'):
    d = se[k] - se['A0']; lo, hi = ci(d); out.append(f'{k} - A0 (vs deployed shrink): {d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]')
def verdict(k, r):   # intervals computed once above and reused here
    d = (se[k] - se[r]).mean(); return 'pass' if d <= -0.05 * se[r].mean() and CI[(k, r)][1] < 0 else 'not better' if d >= 0 else 'unclear'
v = {r: verdict('E', r) for r in ('B', 'B5')}
rule = 'SUPPORT' if all(x == 'pass' for x in v.values()) else 'NOT PROMOTED' if any(x == 'not better' for x in v.values()) else 'INCONCLUSIVE'
out.append(f'E vs B: {v["B"]} | E vs B5: {v["B5"]} | PRIMARY rule (Crux 023bd85c: both pass -> SUPPORT; either mean not better -> NOT PROMOTED; else INCONCLUSIVE): {rule}')
print('\n'.join(out)); open('data/kdiag/c3_disagreement.txt', 'w').write('\n'.join(out) + '\n')
