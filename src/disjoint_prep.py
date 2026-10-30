"""Item files for benchmarks used in no development step (not simulation, not prior evaluation): data/llm/items_disjoint.json."""
import pandas as pd, numpy as np, json, os
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id'])
used = set(json.load(open('data/llm/items_to_rate.json'))) | {r['bench'] for r in json.load(open('data/llm/prior_eval.json'))}
rng = np.random.default_rng(11); out = {}
IDS = json.load(open('data/llm/item_ids_disjoint.json')) if os.path.exists('data/llm/item_ids_disjoint.json') else None   # the released sample
for b, d in R.groupby('bench'):
    if b in used or not os.path.exists(f'data/mdb/{b}/formatted_tables/items.parquet'):
        continue
    cnt = d.groupby('subject_id').item_id.nunique()
    ic = d.groupby('item_id').subject_id.nunique()
    pool = ic[ic >= max(2, ic.quantile(0.5))].index.values
    its = rng.choice(pool, min(300, len(pool)), replace=False)
    if IDS is not None:
        if b not in IDS: continue
        its = np.array(IDS[b], dtype=object)
    good = (d[d.item_id.isin(its)].groupby('subject_id').item_id.nunique() >= int(0.7 * min(120, len(its)))).sum()
    if good < 2:
        continue
    items = pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet', columns=['item_id', 'content', 'item_features']).drop_duplicates('item_id').set_index('item_id')
    its = [i for i in its if i in items.index]
    out[b] = [dict(item_id=i, text=str(items.loc[i, 'content'])[:1500] + ' || features: ' + str(items.loc[i, 'item_features'])[:200]) for i in its]
    print(b, len(out[b]), 'subjects with enough items', good)
json.dump(out, open('data/llm/items_disjoint.json', 'w'), ensure_ascii=False)
