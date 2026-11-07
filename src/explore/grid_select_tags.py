"""Select among the configurations of a zero-call grid (logs DIR/{ablate,multi}_s{seed}_{TAG}.log): objective = mean of the ablate and
multi draw-mean ALC on seeds 5-6; anything within 0.0005 of the best ties with it, and a tie with CURRENT keeps CURRENT; then seed 7
(fixed-draw stability check) paired against CURRENT. usage: TAGS=t1,t2,... grid_select_tags.py DIR CURRENT_TAG"""
import ast, glob, os, sys, numpy as np
D, CUR = sys.argv[1], sys.argv[2]; TIE = 0.0005

def alcs(f):
    lines = open(f).read().splitlines()
    assert any(l.startswith('PRIOR_CACHE_MISSES 0 API_CALLS 0') for l in lines), f'{f}: cache miss or API call'
    a = ast.literal_eval(next(l for l in lines if l.startswith('ALCS '))[5:])
    assert len(a) == (30 if '/ablate_' in f else 10) and np.isfinite(a).all(), f'{f}: {len(a)} draws or non-finite ALC'
    return a

tags = sorted(os.environ['TAGS'].split(','))   # the frozen candidate list, so a candidate whose logs are all missing cannot drop out
found = {os.path.basename(f)[len('ablate_s5_'):-4] for f in glob.glob(f'{D}/ablate_s5_*.log')}
assert found == set(tags), f'logs on disk {sorted(found ^ set(tags))} differ from the frozen list'
assert CUR in tags and all(os.path.exists(f'{D}/{S}_s{s}_{t}.log') for t in tags for S in ('ablate', 'multi') for s in (5, 6, 7)), 'incomplete grid'
obj = {t: np.mean([np.mean(sum((alcs(f'{D}/{S}_s{s}_{t}.log') for s in (5, 6)), [])) for S in ('ablate', 'multi')]) for t in tags}
for t in sorted(tags, key=obj.get): print(f'{t:28s} {obj[t]:.5f}' + ('  <- current' if t == CUR else ''))
best = min(obj, key=obj.get); pick = CUR if obj[CUR] <= obj[best] + TIE else best
print('best', best, round(obj[best], 5), '| current', round(obj[CUR], 5), '| PICK', pick)
if pick != CUR:
    for S in ('ablate', 'multi'):
        d = np.array(alcs(f'{D}/{S}_s7_{pick}.log')) - np.array(alcs(f'{D}/{S}_s7_{CUR}.log'))
        print(f'seed 7 {S}: pick - current {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}, better in {np.mean(d < 0):.0%} of {len(d)} draws')
