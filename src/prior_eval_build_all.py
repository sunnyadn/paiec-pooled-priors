"""Every (public benchmark, named subject) pair for the subject-prior evaluation, not only four per benchmark:
same filters and item sample as prior_eval_build.py. -> data/llm/prior_eval_all.json"""
import json, pandas as pd, numpy as np
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y'])
out = []
L = pd.read_csv('data/llm/prior_estimates_all.csv')   # the 765 pairs behind the cached estimates; the build must reproduce exactly these
for b, g in R.groupby('bench'):
    if b not in set(L.benchmark): continue   # benchmarks outside the evaluation set (their items tables were not downloaded when it was drawn)
    try:
        S = pd.read_parquet(f'data/mdb/{b}/formatted_tables/subjects.parquet')
        I = pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet', columns=['item_id', 'content', 'item_features'])
    except Exception:
        continue
    S = S[S.normalized_name.astype(str).str.len().gt(0) & S.normalized_name.notna()] if 'normalized_name' in S else S.iloc[:0]
    acc = g.groupby('subject_id').y.agg(['mean', 'size']); acc = acc[acc['size'] >= 50]
    S = S[S.subject_id.isin(acc.index)].drop_duplicates('normalized_name')
    if not len(S): continue
    I = I[I.item_id.isin(set(g.item_id))]
    samp = I.sample(min(10, len(I)), random_state=0)
    texts = [(str(c)[:600] + (' ' + str(f) if f is not None and str(f) not in ('', 'None', '{}', 'nan') else '')) for c, f in zip(samp.content, samp.item_features)]
    for _, s in S.iterrows():
        subj = {k: ('' if s.get(k) is None or str(s.get(k)) in ('None', 'nan', 'NaT') else str(s.get(k))) for k in
                ['normalized_name', 'provider', 'release_date', 'access_date', 'harness', 'harness_version', 'reasoning_effort', 'subject_features_extra']}
        out.append(dict(bench=b, subject=subj, subject_id=str(s.subject_id), acc=float(acc.loc[s.subject_id, 'mean']), n=int(acc.loc[s.subject_id, 'size']), texts=texts))
got = [(o['bench'], o['subject']['normalized_name']) for o in out]; want = list(zip(L.benchmark, L.normalized_name.astype(str)))
if got != want: raise SystemExit(f'rebuilt {len(got)} pairs do not match the {len(want)} listed in data/llm/prior_estimates_all.csv; the cached estimates would not line up')
json.dump(out, open('data/llm/prior_eval_all.json', 'w'))
a = np.array([o['acc'] for o in out]); print(len(out), 'pairs from', len({o['bench'] for o in out}), 'benches; acc mean', a.mean().round(3))
