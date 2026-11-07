"""Pick RIDGE x CALIB_MAX from tools/grid/run_ridge.sh: select on seeds 5-6 (objective = mean of ablate and multi draw-mean ALC),
ties within 0.0005 -> nearest to (25, 60) in grid steps, then larger RIDGE, smaller CALIB_MAX; then report seed 7 (held out) paired.
usage: grid_select_ridge.py GRID_DIR"""
import ast, sys, numpy as np
D = sys.argv[1]; RS, CS = [5, 10, 25, 50, 100], [30, 60, 120]; TIE = 0.0005; CUR = (25, 60)

def alcs(f):
    lines = open(f).read().splitlines()
    assert any(l.startswith('PRIOR_CACHE_MISSES 0 API_CALLS 0') for l in lines), f'{f}: cache miss or API call'
    a = ast.literal_eval(next(l for l in lines if l.startswith('ALCS '))[5:])
    assert len(a) == (30 if '/ablate_' in f else 10) and np.isfinite(a).all(), f'{f}: {len(a)} draws or non-finite ALC'
    return a

def obj(r, c, seeds):
    return np.mean([np.mean(sum((alcs(f'{D}/{S}_s{s}_r{r}_c{c}.log') for s in seeds), [])) for S in ('ablate', 'multi')])

sel = {(r, c): obj(r, c, (5, 6)) for r in RS for c in CS}
print(f'{"RIDGE":>6} ' + ' '.join(f'{c:>8}' for c in CS) + '   (selection: seeds 5-6)')
for r in RS: print(f'{r:>6} ' + ' '.join(f'{sel[(r, c)]:8.4f}' for c in CS))
best = min(sel.values()); tied = [k for k, v in sel.items() if v <= best + TIE]
pick = min(tied, key=lambda k: (abs(RS.index(k[0]) - RS.index(CUR[0])) + abs(CS.index(k[1]) - CS.index(CUR[1])), -k[0], k[1]))
print('current', CUR, round(sel[CUR], 5), '| best', round(best, 5), '| tied', sorted(tied), '| PICK', pick)
if pick != CUR:   # held-out seed 7, paired by draw
    for S in ('ablate', 'multi'):
        a, b = np.array(alcs(f'{D}/{S}_s7_r{pick[0]}_c{pick[1]}.log')), np.array(alcs(f'{D}/{S}_s7_r{CUR[0]}_c{CUR[1]}.log')); d = a - b
        print(f'held out {S} s7: pick - current {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}, better in {np.mean(d < 0):.0%} of {len(d)} draws')
