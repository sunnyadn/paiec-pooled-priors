"""Per evaluation item response counts for the kdiag records (zero calls): one row per (set, seed, draw, pair, item) with the number of
recorded responses and correct ones, for independent rechecks of kdiag_rescore.py. usage: kdiag_resp_table.py -> data/kdiag/resp_table.csv"""
import json, pandas as pd
L = pd.read_parquet('data/long.parquet', columns=['bench', 'subject_id', 'item_id', 'y']); L['item_id'] = L.item_id.astype(str); L['subject_id'] = L.subject_id.astype(str)
G = L.groupby(['bench', 'subject_id', 'item_id']).y.agg(['sum', 'count']); rows = []
for S in ('ablate', 'multi'):
    for seed in (5, 6, 7):
        for d, draw in enumerate(json.loads(l) for l in open(f'data/draws/{S}_s{seed}.jsonl')):
            for j, x in enumerate(draw):
                for i, _ in x['ev']:
                    g = G.loc[(x['b'], str(x['s']), str(i))]; rows.append(dict(set=S, seed=seed, draw=d, pair=j, bench=x['b'], subject_id=str(x['s']), item_id=str(i), n_resp=int(g['count']), n_correct=int(g['sum'])))
pd.DataFrame(rows).to_csv('data/kdiag/resp_table.csv', index=False); print(len(rows), 'rows')
