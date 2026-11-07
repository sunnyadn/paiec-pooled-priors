"""Zero-call check on the new benchmarks (win): replay fixed draws for one arm: prior / noprior (v26, ratings off),
laplace, empirical_mean (official baseline); the v26 arms stop on a missing prior or an attempted API call.
usage: sim_extra.py ARM SEED [NDRAWS]  ->  data/llm/extra/sim/out_{ARM}_s{SEED}.json"""
import functools, importlib.util, json, os, sys, tempfile
import numpy as np
ARM, SEED = sys.argv[1], int(sys.argv[2]); ND = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
S = 'data/llm/extra/sim'
P = {p['key']: p for p in json.load(open(f'{S}/pairs.json'))}; ITEMS = json.load(open(f'{S}/items.json'))
DRAWS = [json.loads(l) for l in open(f'{S}/draws_s{SEED}.jsonl')][:ND]
B, W = [0, 1, 3, 7, 15, 31], [.1, .2, .2, .2, .2, .1]
load = lambda path, name: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(name, path))
_V26 = 'submission/model_v26.py' if os.path.exists('submission/model_v26.py') else 'submission/model.py'   # the release ships v26 as model.py (anchors from submission/anchors.json)
MODEL = {'prior': _V26, 'noprior': _V26, 'laplace': 'src/baselines/laplace.py',
         'empirical_mean': 'src/baselines/empirical_mean.py'}[ARM]   # the organisers' baseline (the copy first run lacked only a __future__ import)
M = load(MODEL, 'm'); LAB = load('submission/labeling.py', 'lab')
if ARM in ('prior', 'noprior'):
    M._cfg = lambda: {'provider': 'cache', 'model': 'x', 'key': 'x'}   # language-model terms on, served from the fixed table below
    M._call = lambda text: None   # no item ratings on these benchmarks: off in both arms
    M._disk_path = lambda: '/dev/null'
    def _no_call(*a, **k): raise SystemExit('the model tried to ask for a prior: not allowed in this zero-call check')
    M._call_prior = _no_call if ARM == 'prior' else (lambda *a, **k: None)
item = lambda b, i: {'item_content': ITEMS[b][i][0], 'item_features': ITEMS[b][i][1], 'interactors': '', 'benchmark_id': 'b_' + b}
out = []
for d, draw in enumerate(DRAWS):
    pairs = []
    for x in draw:
        p = P[x['pair']]; y = dict(p['outcomes'])
        pairs.append(dict(key=x['pair'], b=p['bench'], subj=p['subject'], stream=[(i, y[i]) for i in x['stream']],
                          ev=[(item(p['bench'], i), y[i]) for i in x['ev']], pos=0, got=[]))
    if ARM in ('prior', 'noprior'):   # a fresh evaluation run
        for a in ('_cache', '_SPRIOR', '_SEEN_TXT', '_RATE', '_SK2DESC'):
            if hasattr(M, a): getattr(M, a).clear()
        for a in ('_PROC_CALLS', '_RATE_CALLS'): getattr(M, a)[0] = 0
        M._PDIRS = [tempfile.mkdtemp(prefix='pp_extra_')]
        if ARM == 'prior':
            for q in pairs:   # the pair's one estimate, shrunk as when deployed, available from the first answer on
                desc, bench = M._subject_desc(q['subj']), 'b_' + q['b']; v = 0.5 + M.SP_SHRINK * (P[q['key']]['estimate'] - 0.5)
                M._SPRIOR[('full', desc, bench)] = v; M._SPRIOR[(desc, bench)] = v; M._SK2DESC[M._key(q['subj'])] = desc
    labeled = []; rec = {q['key']: {} for q in pairs}
    for n, w in zip(B, W):
        for q in pairs:
            while len(q['got']) < n and q['pos'] < len(q['stream']):
                i, y = q['stream'][q['pos']]; q['pos'] += 1
                ctx = {'labels_remaining': 31 - len(q['got']), 'items_remaining': len(q['stream']) - q['pos'] + 1}   # allowance out of 31, as on the platform
                if LAB.acquisition_function([q['subj'], item(q['b'], i)], labeled=labeled, context=ctx):
                    q['got'].append(i); labeled.append([[q['subj'], item(q['b'], i)], int(y)])
            assert n < 31 or len(q['got']) == 31, 'a pair got fewer than 31 labels'
        for q in pairs:
            pr = np.array([M.predict([q['subj'], it], labeled) for it, _ in q['ev']]); yy = np.array([y for _, y in q['ev']])
            rec[q['key']][n] = float(np.mean((yy - pr) ** 2))
    out.append({k: dict(brier=v, alc=sum(w * v[n] for n, w in zip(B, W))) for k, v in rec.items()})
    print(d, round(float(np.mean([r['alc'] for r in out[-1].values()])), 4), flush=True)
json.dump(out, open(f'{S}/out_{ARM}_s{SEED}.json', 'w'))
