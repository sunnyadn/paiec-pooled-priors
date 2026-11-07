"""New-benchmark check, step 0 (runs on win): benchmarks outside the main evaluation that are 0/1-scored and have a named
configuration with at least 50 scored items; their item tables are then fetched. -> data/llm/extra/candidates.json"""
import glob, json, os
import pandas as pd
os.makedirs('data/llm/extra', exist_ok=True)
MAIN = {x['bench'] for x in json.load(open('data/llm/prior_eval_all.json'))}
out = []
for p in sorted(glob.glob('data/mdb_extra/*/formatted_tables/responses.parquet')):
    b = p.split('/')[-3]; d = os.path.dirname(p)
    if b in MAIN: continue
    S = pd.read_parquet(d + '/subjects.parquet'); R = pd.read_parquet(p, columns=['subject_id', 'item_id', 'response'])
    y = pd.to_numeric(R.response, errors='coerce'); R = R.assign(y=y).dropna(subset=['y'])
    if not len(R) or not R.y.between(0, 1).all() or R.y.isin([0, 1]).mean() < 0.95 or 'normalized_name' not in S: continue
    named = S[S.normalized_name.notna()]
    cnt = R[R.subject_id.isin(named.subject_id)].groupby('subject_id').item_id.nunique()
    k = named[named.subject_id.isin(cnt[cnt >= 50].index)].normalized_name.nunique()
    if k: out.append(dict(bench=b, names=int(k)))
json.dump(out, open('data/llm/extra/candidates.json', 'w'), indent=0)
print(len(out), 'candidate benchmarks,', sum(o['names'] for o in out), 'named pairs')
