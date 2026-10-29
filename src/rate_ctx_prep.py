"""In-context difficulty rating: give the LLM 25 items from this benchmark with their observed pass rate, then rate others."""
import json, numpy as np, pandas as pd, os
items_all=json.load(open('data/llm/items_to_rate.json'))
R=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','trial','y'])
R=R.sort_values('trial').drop_duplicates(['bench','subject_id','item_id'])
rng=np.random.default_rng(11); os.makedirs('data/llm/ctx_batches',exist_ok=True); meta={}
P='''You will estimate item difficulty for one AI evaluation benchmark (text may be truncated; images are not shown).
First, here are {k} example items from THIS benchmark with the observed share of evaluated AI systems that got each right:

{ctx}

Now, for each new item below, estimate the share (0-100) of the same evaluated AI systems that would get it right.
Use the examples to understand what is easy or hard in this benchmark.
Output ONLY JSON: {{"ratings": [{{"id": "<item id>", "p": <0-100>}}, ...]}} with one entry per new item.

NEW ITEMS:

'''
for b,its in items_all.items():
    ids=[x['item_id'] for x in its]; tx={x['item_id']:x['text'] for x in its}
    d=R[(R.bench==b)&R.item_id.isin(ids)]; acc=d.groupby('item_id').y.mean()
    ids=[i for i in ids if i in acc.index]; rng.shuffle(ids)
    A=ids[:len(ids)//2]; E=ids[len(ids)//2:]; C=A[:25]; rest=A[25:]+E
    meta[b]=dict(A=A,E=E,C=C)
    ctx='\n\n'.join(f'### example (share correct: {round(100*acc[i])}%)\n{tx[i][:800]}' for i in C)
    for k in range(0,len(rest),50):
        chunk=rest[k:k+50]
        s=P.format(k=len(C),ctx=ctx)+'\n\n'.join(f'### id: {i}\n{tx[i][:1200]}' for i in chunk)
        open(f'data/llm/ctx_batches/{b}__{k//50:02d}.prompt','w').write(s)
json.dump(meta,open('data/llm/ctx_meta.json','w'))
print(len(os.listdir('data/llm/ctx_batches')))
