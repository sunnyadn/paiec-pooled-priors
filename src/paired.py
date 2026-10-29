"""Paired comparison of online-sim logs: python src/paired.py BASE.log OTHER.log ..."""
import sys, ast, numpy as np
def load(f):
    d = {}
    for l in open(f):
        if l.startswith('ALCS'): d['a'] = np.array(ast.literal_eval(l[5:]))
        if l.startswith('PER_BUDGET'): d['pb'] = l.split(' ', 1)[1].strip()
        if l.startswith('DRAWS'): d['m'] = l.strip()
    return d
b = load(sys.argv[1])
print(sys.argv[1], b['m'])
for f in sys.argv[2:]:
    o = load(f); dlt = o['a'] - b['a']
    print(f"{f}: mean {o['a'].mean():.4f}  diff {dlt.mean():+.4f} ± {dlt.std(ddof=1)/np.sqrt(len(dlt)):.4f} (paired SE)  better in {np.mean(dlt<0)*100:.0f}% of draws")
    print('   ', o['pb'])
