"""Training/eval data for the LoRA correctness model: (model name, item text) -> correct.
Train: non-simulation training benchmarks. Eval: the 12 simulation benchmarks (never trained on)."""
import json, numpy as np, pandas as pd
items = json.load(open('data/llm/items_to_rate.json')); TB = set(items)
R = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'normalized_name', 'y'])
R = R.dropna(subset=['normalized_name']); R['item_id'] = R.item_id.astype(str)
R = R.drop_duplicates(['bench', 'subject_id', 'item_id'])
rng = np.random.default_rng(0)
def texts(b, ids):
    try: I = pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet', columns=['item_id', 'content'])
    except Exception: return {}
    I['item_id'] = I.item_id.astype(str); I = I[I.item_id.isin(ids)]
    return dict(zip(I.item_id, I.content.astype(str).str[:1200]))
train = []
for b, g in R[~R.bench.isin(TB)].groupby('bench'):
    g = g.sample(min(len(g), 4000), random_state=0); tx = texts(b, set(g.item_id))
    train += [[str(n), tx[i], int(y), b] for n, i, y in zip(g.normalized_name, g.item_id, g.y) if i in tx]
test = []
for b in TB:
    g = R[R.bench == b]; tx = {str(x['item_id']): x['text'][:1200] for x in items[b]}
    g = g[g.item_id.isin(set(tx))]
    subs = g.groupby('subject_id').filter(lambda h: len(h) >= 30).subject_id.unique()
    subs = rng.choice(subs, min(15, len(subs)), replace=False) if len(subs) else []
    for s in subs:
        h = g[g.subject_id == s]; h = h.sample(min(150, len(h)), random_state=0)
        test += [[str(n), tx[i], int(y), b, str(s), i] for n, i, y in zip(h.normalized_name, h.item_id, h.y)]
json.dump({'train': train, 'test': test}, open('/tmp/lora_data.json', 'w'))
print('train', len(train), 'benches', len({t[3] for t in train}), 'test', len(test), 'pos rate', np.mean([t[2] for t in train]).round(3))
