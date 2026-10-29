"""Pair accuracies implied by online formative feedback: with a constant 0.5 at budget 0, ECE = |acc - 0.5|."""
import re, glob
def load():
    out = []
    for f in sorted(glob.glob('data/online/scoring_*.txt')):
        txt = open(f).read(); tab = {}
        for s in re.split(r'\n(?=ALC summary|Label budget: \d+)', txt):
            h = s.strip().split('\n')[0]
            tab[h] = {(a, b): (int(n), float(x), float(e)) for a, b, n, x, e in re.findall(r'(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)', s)}
        for k, (n, alc, _) in tab['ALC summary'].items():
            out.append(dict(run=f[-11:-4], subj=k[0], bench=k[1], n=n, dev=tab['Label budget: 0'][k][2], alc=alc,
                            pb=[tab[f'Label budget: {b}'][k][1] for b in [0, 1, 3, 7, 15, 31]]))
    return out
