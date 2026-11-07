"""Pick SP_K x SP_SHRINK from the zero-call grid (tools/grid/run.sh): objective = mean of the ablate and multi ALC (draw means, as
sim_sh MEAN, seeds 5-7 pooled); within 0.0005 of the best counts as tied -> nearest to (4, .75) in grid steps, then smaller K, S.
usage: grid_select.py GRID_DIR"""
import ast, sys, numpy as np
D = sys.argv[1]; KS, SS = [2, 4, 6, 8, 12, 16], ['0.65', '0.75', '0.85', '1.0']; TIE = 0.0005

def alcs(f):
    lines = open(f).read().splitlines()
    assert any(l.startswith('PRIOR_CACHE_MISSES 0 API_CALLS 0') for l in lines), f'{f}: cache miss or API call'
    a = ast.literal_eval(next(l for l in lines if l.startswith('ALCS '))[5:])
    assert len(a) == (30 if '/ablate_' in f else 10) and np.isfinite(a).all(), f'{f}: {len(a)} draws or non-finite ALC'
    return a

obj = {}
for k in KS:
    for s in SS:
        per = {S: float(np.mean(sum((alcs(f'{D}/{S}_s{seed}_k{k}_s{s}.log') for seed in (5, 6, 7)), []))) for S in ('ablate', 'multi')}
        obj[(k, s)] = (np.mean(list(per.values())), per)
print(f'{"K":>3} ' + ' '.join(f'{s:>7}' for s in SS) + '   (objective; ablate / multi in the log below)')
for k in KS: print(f'{k:>3} ' + ' '.join(f'{obj[(k, s)][0]:7.4f}' for s in SS))
best = min(v[0] for v in obj.values()); cur = (KS.index(4), SS.index('0.75'))
tied = [c for c, v in obj.items() if v[0] <= best + TIE]
pick = min(tied, key=lambda c: (abs(KS.index(c[0]) - cur[0]) + abs(SS.index(c[1]) - cur[1]), c[0], float(c[1])))
for c in sorted(tied): print('tied', c, round(obj[c][0], 5), {S: round(v, 5) for S, v in obj[c][1].items()})
print('current (4, 0.75)', round(obj[(4, '0.75')][0], 5), '| best', round(best, 5), '| PICK', pick, round(obj[pick][0], 5))
