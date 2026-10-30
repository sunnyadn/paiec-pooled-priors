"""Table of online formative runs: raw ALC, draw-adjusted ALC and excess over each pair's budget-31 level,
with standard errors over runs (the draw is the unit of variation). usage: python src/online_table.py"""
import sys; sys.path.insert(0, 'src')
from online_pairs import load
import numpy as np
W = np.array([.1, .2, .2, .2, .2, .1])
GROUPS = [('Pooled baseline (v6)', ['1016627', '1016628', '1016651']),
          ('+ subject prior (v19, v23, v25, v27)', ['1017498', '1017499', '1017545', '1017562', '1017629']),
          ('+ Opus item rater (final, 1017682)', ['1017607', '1017628', '1017682'])]
def main():
    P = load()
    base = [p for p in P if p['run'] in GROUPS[0][1]]
    x = np.array([p['pb'][5] for p in base]); y = np.array([p['alc'] for p in base])
    c = np.linalg.lstsq(np.vstack([np.ones_like(x), x, x ** 2]).T, y, rcond=None)[0]
    f = lambda v: c[0] + c[1] * v + c[2] * v ** 2
    for name, runs in GROUPS:
        per = []
        for r in runs:   # one row per run: the draw, not the pair, is the unit of variation
            R = [p for p in P if p['run'] == r]
            pb = np.array([p['pb'] for p in R]); alc = np.array([p['alc'] for p in R])
            per.append(((alc - f(pb[:, 5])).mean(), ((pb - pb[:, 5:6]) * W).sum(1).mean(), alc.mean(), len(R)))
        per = np.array(per); se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
        print(f'{name:38s} runs {len(runs)} pairs {int(per[:, 3].sum()):2d}  raw {per[:, 2].mean():.3f}  '
              f'adjusted {per[:, 0].mean():+.4f} ± {se(per[:, 0]):.4f}  excess {per[:, 1].mean():.4f} ± {se(per[:, 1]):.4f}  runs={",".join(runs)}')


if __name__ == '__main__':
    main()
