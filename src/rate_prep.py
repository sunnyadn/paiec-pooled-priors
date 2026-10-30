import pandas as pd, numpy as np, json, os
IDS = json.load(open('data/llm/item_ids.json')) if os.path.exists('data/llm/item_ids.json') else None   # the released sample, if present
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id'])
benches=['agi_elo','cmmlu','ceval','cruxeval','bird_sql','harmbench','editbench','financebench','helm_afr','afrimedqa','agentdojo','benger']
rng=np.random.default_rng(7); out={}
for b in benches:
    d=R[R.bench==b]; cnt=d.groupby('subject_id').item_id.nunique(); subs=cnt[cnt>=80].index
    d=d[d.subject_id.isin(subs)]
    # items answered by many subjects
    ic=d.groupby('item_id').subject_id.nunique().sort_values(ascending=False)
    pool=ic[ic>=max(2,ic.quantile(0.5))].index.values
    its=rng.choice(pool,min(300,len(pool)),replace=False)
    if IDS: its=np.array(IDS[b], dtype=type(its[0]))
    items=pd.read_parquet(f'data/mdb/{b}/formatted_tables/items.parquet',columns=['item_id','content','item_features']).drop_duplicates('item_id').set_index('item_id').loc[its]
    out[b]=[dict(item_id=i,text=(str(r.content)[:1500]+' || features: '+str(r.item_features)[:200])) for i,r in items.iterrows()]
    print(b,len(out[b]))
json.dump(out,open('data/llm/items_to_rate.json','w'),ensure_ascii=False)
# write batch prompts
P='''You will see {n} test items from an AI evaluation benchmark (text may be truncated; images are not shown).
For each item, estimate the probability (0-100) that a typical current large language model (a mid-range 2025-2026 model, single attempt, no tools beyond what the item states) answers or completes it correctly.
Judge only from the item itself. Consider difficulty, ambiguity, required knowledge or reasoning, and length.
Output ONLY JSON: {{"ratings": [{{"id": "<item id>", "p": <0-100>}}, ...]}} with one entry per item, same ids.

'''
os.makedirs('data/llm/batches',exist_ok=True)
for b,its in out.items():
    for k in range(0,len(its),50):
        chunk=its[k:k+50]
        s=P.format(n=len(chunk))+'\n\n'.join(f'### id: {x["item_id"]}\n{x["text"]}' for x in chunk)
        open(f'data/llm/batches/{b}__{k//50:02d}.prompt','w').write(s)
print(len(os.listdir('data/llm/batches')))
