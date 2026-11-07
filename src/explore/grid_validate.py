"""One-time validation of the picked constants against (4, .75): new-benchmark component sim (prior arm, sandbox outputs) and the
disjoint replay; paired difference of draw ALC, and a 90% interval from a bootstrap over benchmarks of the per pair-draw differences.
usage: grid_validate.py EXTRA_CUR_DIR EXTRA_NEW_DIR DISJ_CUR_PREFIX DISJ_NEW_PREFIX   (prefix + _s{5,6,7}.json = PREDLOG files)"""
import json, sys, numpy as np
XC, XN, DC, DN = sys.argv[1:5]; B, W = [0, 1, 3, 7, 15, 31], [.1, .2, .2, .2, .2, .1]; rng = np.random.default_rng(0)

def report(name, cur, new):   # cur/new: list of draws, each {pair key: alc}; same draws and keys in both
    assert len(cur) == len(new) == 90, f'{name}: {len(cur)} vs {len(new)} draws'
    assert all(c.keys() == n.keys() for c, n in zip(cur, new)), f'{name}: pair sets differ'
    d = np.array([np.mean(list(n.values())) - np.mean(list(c.values())) for c, n in zip(cur, new)])
    rows = [(k.split('|')[0], n[k] - c[k]) for c, n in zip(cur, new) for k in c]
    bench = np.array([b for b, _ in rows]); x = np.array([v for _, v in rows]); ub = np.unique(bench)
    bs = [np.mean(np.concatenate([x[bench == b] for b in rng.choice(ub, len(ub))])) for _ in range(2000)]
    print(f'{name}: {len(cur)} draws, {len(ub)} benchmarks | new - current {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f} (draws),'
          f' new better in {np.mean(d < 0):.0%} | per pair-draw {x.mean():+.4f}, 90% by benchmark [{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}]')

load_x = lambda D: sum((json.load(open(f'{D}/data/llm/extra/sim/out_prior_s{s}.json')) for s in (5, 6, 7)), [])   # draws in seed order
report('new benchmarks (component sim)', *[[{k: r['alc'] for k, r in d.items()} for d in load_x(D)] for D in (XC, XN)])

def load_d(P):   # PREDLOG -> per draw {benchmark|pair: ALC}, plus the outcomes and budgets seen, for the pairing check
    out, ys = {}, {}
    for s in (5, 6, 7):
        for r in json.load(open(f'{P}_s{s}.json')):
            dd = out.setdefault((s, r['draw']), {}); k = f"{r['b']}|{r['pair']}"
            dd[k] = dd.get(k, 0.0) + W[B.index(r['n'])] * float(np.mean((np.array(r['y']) - np.array(r['p'])) ** 2))
            ys[(s, r['draw'], k, r['n'])] = r['y']
    return [out[k] for k in sorted(out)], ys
(dc, yc), (dn, yn) = load_d(DC), load_d(DN)
assert yc == yn and len({k[:3] for k in yc}) * 6 == len(yc), 'disjoint: draws, pairs, budgets or outcomes differ between arms'
report('disjoint (replay, ratings off)', dc, dn)
