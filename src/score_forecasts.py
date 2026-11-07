"""Score forecasts against true accuracies from the gated build files (data/llm/prior_eval_all.json via src/prior_eval_build_all.py,
data/llm/extra/eligible.json via src/prior_extra_build.py). The scored pairs are fixed by data/forecasts/forecasts.csv: the 677 rows
whose benchmark has no calibration example. usage: score_forecasts.py [CSV with set, bench, subject_id, estimate]"""
import csv, json, sys, numpy as np
REF = 'data/forecasts/forecasts.csv'; F = sys.argv[1] if len(sys.argv) > 1 else REF
ACC = {('main', e['bench'], e['subject_id']): e['acc'] for e in json.load(open('data/llm/prior_eval_all.json'))}
ACC.update({('new', e['bench'], e['subject_id']): e['acc'] for e in json.load(open('data/llm/extra/eligible.json'))})
key = lambda r: (r['set'], r['bench'], r['subject_id'])
PAIRS = [key(r) for r in csv.DictReader(open(REF)) if r['benchmark_has_calibration_example'] == 'False']
EST = {}
for r in csv.DictReader(open(F)):
    if key(r) in EST: sys.exit(f'duplicate pair {key(r)}')
    EST[key(r)] = r['estimate']
miss = [k for k in PAIRS if k not in EST]
if miss: sys.exit(f'{len(miss)} scored pairs missing, e.g. {miss[0]}')
try: Q = {k: float(EST[k]) for k in PAIRS}
except ValueError as x: sys.exit(f'non-numeric estimate: {x}')
if not all(0 <= v <= 1 for v in Q.values()): sys.exit('estimates must be probabilities in [0, 1]')
for name, keep in (('main', {'main'}), ('new', {'new'}), ('pooled', {'main', 'new'})):
    ks = [k for k in PAIRS if k[0] in keep]; a = np.array([ACC[k] for k in ks]); q = np.array([Q[k] for k in ks])
    rm = lambda p: np.sqrt(np.mean((p - a) ** 2))
    print(f'{name:7s} pairs {len(ks):3d}  benchmarks {len({k[1] for k in ks}):2d}  RMSE raw {rm(q):.3f}  shrunk 0.75 {rm(0.5 + 0.75 * (q - 0.5)):.3f}  constant 0.5 {rm(np.full(len(a), 0.5)):.3f}')
