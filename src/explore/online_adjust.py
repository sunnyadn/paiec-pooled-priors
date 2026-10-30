"""Compare online runs after adjusting for draw difficulty: per-pair ALC regressed on budget-31 Brier (baseline = v6 runs).
usage: online_adjust.py RUN_ID [RUN_ID ...]   (scoring_<id>.txt in data/online)"""
import sys; sys.path.insert(0, 'src')
from online_pairs import load
import numpy as np
BASE = ('1016627', '1016628', '1016651')
P = load(); base = [p for p in P if p['run'] in BASE]
x = np.array([p['pb'][5] for p in base]); y = np.array([p['alc'] for p in base])
c = np.linalg.lstsq(np.vstack([np.ones_like(x), x, x ** 2]).T, y, rcond=None)[0]
f = lambda v: c[0] + c[1] * v + c[2] * v ** 2
sd = (y - f(x)).std()
for rid in sys.argv[1:]:
    new = [p for p in P if p['run'] == rid]
    xn = np.array([p['pb'][5] for p in new]); yn = np.array([p['alc'] for p in new])
    d = yn - f(xn)
    print(f'{rid}: ALC {yn.mean():.3f}  draw b31 {xn.mean():.3f}  adjusted diff vs v6 {d.mean():+.4f} ± {sd/np.sqrt(len(d)):.4f}  ({len(d)} pairs)')
