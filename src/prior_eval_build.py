"""Build an evaluation set for the subject-accuracy prior: (benchmark, named subject) pairs from the public training
benchmarks, with true accuracy, subject attributes and 10 sample item texts. -> data/llm/prior_eval.json"""
import json, random, pandas as pd, numpy as np
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y'])
rng = random.Random(0); out = []
L = pd.read_csv('data/llm/prior_estimates.csv')   # the 87 pairs behind the cached estimates; the build must reproduce exactly these
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
    pick = S.sample(min(4, len(S)), random_state=0)
    I = I[I.item_id.isin(set(g.item_id))]
    samp = I.sample(min(10, len(I)), random_state=0)
    texts = [(str(c)[:600] + (' ' + str(f) if f is not None and str(f) not in ('', 'None', '{}', 'nan') else '')) for c, f in zip(samp.content, samp.item_features)]
    for _, s in pick.iterrows():
        subj = {k: ('' if s.get(k) is None or str(s.get(k)) in ('None', 'nan', 'NaT') else str(s.get(k))) for k in
                ['normalized_name', 'provider', 'release_date', 'access_date', 'harness', 'harness_version', 'reasoning_effort', 'subject_features_extra']}
        out.append(dict(bench=b, subject=subj, acc=float(acc.loc[s.subject_id, 'mean']), n=int(acc.loc[s.subject_id, 'size']), texts=texts))
got = [(o['bench'], o['subject']['normalized_name']) for o in out]; want = list(zip(L.benchmark, L.model.astype(str)))
if got != want: raise SystemExit(f'rebuilt {len(got)} pairs do not match the {len(want)} listed in data/llm/prior_estimates.csv; the cached estimates would not line up')
json.dump(out, open('data/llm/prior_eval.json', 'w'))
a = np.array([o['acc'] for o in out]); print(len(out), 'pairs from', len({o['bench'] for o in out}), 'benches; acc mean', a.mean().round(3), 'share >.9', (a > .9).mean().round(2), '<.1', (a < .1).mean().round(2))
print(out[0]['subject'])
