"""The language-model accuracy forecasts as one table, without item text or true accuracies: every named (benchmark, subject) pair
of the main evaluation (Table 1) and of the new benchmarks (Section 4), with the subject's name, provider, release date and the raw estimate.
usage: make_forecasts_csv.py  ->  data/forecasts/forecasts.csv (needs the gated build files only to read pair keys and metadata)"""
import csv, json, os
OUT = 'data/forecasts'; os.makedirs(OUT, exist_ok=True)
META = ['normalized_name', 'provider', 'release_date']   # the other prompt fields (harness, effort) join from measurement-db by subject_id
num = lambda x: x[0] if isinstance(x[0], (int, float)) else None
rows = []

E = json.load(open('data/llm/prior_eval_all.json')); O = json.load(open('data/llm/prior_eval_oall.json'))
S, H = json.load(open('data/llm/prior_eval_smiss.json')), json.load(open('data/llm/prior_eval_hmiss.json'))
skip = [bool(o[1].get('skip')) for o in O]; calib = {e['bench'] for e, s in zip(E, skip) if s}
miss = [i for i, o in enumerate(O) if num(o) is None and not skip[i]]   # Opus gave no number: Sonnet, then Haiku (as in reproduce.py)
fill = {i: (num(S[j]), 'claude-sonnet-5-5') if num(S[j]) is not None else (num(H[j]), 'claude-haiku-5-5') for j, i in enumerate(miss)}
for i, (e, o) in enumerate(zip(E, O)):
    if skip[i]: continue   # the pair is one of the eight calibration examples in the prompt
    est, by = fill.get(i, (num(o), 'claude-opus-5-5'))
    rows.append(dict(set='main', bench=e['bench'], subject_id=e['subject_id'], **{k: e['subject'].get(k, '') for k in META},
                     estimate=est, answered_by=by if est is not None else '', benchmark_has_calibration_example=e['bench'] in calib))

EL = {(e['bench'], e['subject_id']): e for e in json.load(open('data/llm/extra/eligible.json'))}
for k, r in json.load(open('data/llm/extra/results.json'))['results'].items():
    if r['status'] == 'untried': continue
    e = EL[tuple(k.split('|', 1))]
    rows.append(dict(set='new', bench=e['bench'], subject_id=e['subject_id'], **{m: e['subject'].get(m, '') for m in META},
                     estimate=r.get('estimate'), answered_by=r.get('answered_by') or '', benchmark_has_calibration_example=False))

with open(f'{OUT}/forecasts.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for s in ('main', 'new'):
    r = [x for x in rows if x['set'] == s]
    print(s, 'pairs', len(r), 'benchmarks', len({x['bench'] for x in r}), 'with estimate', sum(x['estimate'] is not None for x in r),
          '| without a calibration example', sum(not x['benchmark_has_calibration_example'] for x in r))
