"""Round-2 LoRA data: all training benchmarks with item text; system name = normalized_name or display_name."""
import json, glob, numpy as np, pandas as pd
items = json.load(open('data/llm/items_to_rate.json')); TB = set(items)
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'normalized_name', 'y']); R['item_id'] = R.item_id.astype(str)
R = R.drop_duplicates(['bench', 'subject_id', 'item_id'])
train = []
for b, g in R[~R.bench.isin(TB)].groupby('bench'):
    try:
        I = pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet', columns=['item_id', 'content'])
        S = pd.read_parquet(f'data/mdb/{b}/formatted_tables/subjects.parquet')
    except Exception: continue
    nm = {}
    for _, r in S.iterrows():
        n = r.get('normalized_name') if 'normalized_name' in S else None
        if n is None or str(n) in ('', 'None', 'nan'): n = r.get('display_name')
        if n is not None and str(n) not in ('', 'None', 'nan'): nm[str(r.subject_id)] = str(n)
    I['item_id'] = I.item_id.astype(str); tx = dict(zip(I.item_id, I.content.astype(str).str[:1200]))
    g = g[g.item_id.isin(set(tx)) & g.subject_id.astype(str).isin(set(nm))]
    if len(g) < 200: continue
    g = g.sample(min(len(g), 1500), random_state=0)
    train += [[nm[str(s)], tx[i], int(y), b] for s, i, y in zip(g.subject_id, g.item_id, g.y)]
old = json.load(open('/tmp/lora_data.json'))
json.dump({'train': train, 'test': old['test']}, open('/tmp/lora_data2.json', 'w'))
print('train', len(train), 'benches', len({t[3] for t in train}))
