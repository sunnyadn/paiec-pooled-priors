# exact attribution of the K-ranking flip: [(a8-q)^2-(a4-q)^2] - [(a8-r)^2-(a4-r)^2] = 2 (a8-a4)(r-q), with r-q split into
# (slice300 - q) [coverage-filtered 300-item slice vs full benchmark] + (sample120 - slice300) [draw's sample] + (r - sample120) [eval half]
import json, numpy as np, pandas as pd
E={(e['bench'],e['subject_id']):e['acc'] for e in json.load(open('data/llm/prior_eval_all.json'))}
IT=json.load(open('data/llm/items_to_rate.json')); L=pd.read_parquet('data/long.parquet',columns=['bench','subject_id','item_id','trial','y'])
L=L.sort_values('trial').drop_duplicates(['bench','subject_id','item_id']); L['item_id']=L.item_id.astype(str); L['subject_id']=L.subject_id.astype(str)
SL={}
for b,v in IT.items():
    ids={str(x['item_id']) for x in v}; g=L[(L.bench==b)&L.item_id.isin(ids)].groupby('subject_id').y.mean(); SL.update({(b,s):a for s,a in g.items()})
for S in ('ablate','multi'):
    for n in (1,3,7):
        rows=[]
        for s in (5,6,7):
            D=[json.loads(l) for l in open(f'data/draws/{S}_s{s}.jsonl')]
            for r in json.load(open(f'kdiag/{S}_s{s}_k4.json')):
                sp=[v for v in r['sp'] if v is not None]
                if r['n']!=n or not sp or len(set(sp))!=1 or r['single']!=r['calls']: continue
                key=(r['b'],D[r['draw']][r['pair']]['s'])
                if key not in E or key not in SL: continue
                a4=(r['k']+4*sp[0])/(r['got']+4); a8=(r['k']+8*sp[0])/(r['got']+8)
                rows.append((a4,a8,E[key],SL[key],r['acc'],np.mean(r['y'])))
        if not rows: print(S,n,'no records'); continue
        a4,a8,q,sl,smp,rr=np.array(rows).T; d=a8-a4
        flip=np.mean(((a8-q)**2-(a4-q)**2)-((a8-rr)**2-(a4-rr)**2))
        parts=[2*np.mean(d*(sl-q)), 2*np.mean(d*(smp-sl)), 2*np.mean(d*(rr-smp))]
        print(f'{S} n={n} recs {len(rows)} | flip {flip:+.5f} = 2 mean[(a8-a4)(r-q)] {2*np.mean(d*(rr-q)):+.5f} | slice-vs-full {parts[0]:+.5f} | draw sample-vs-slice {parts[1]:+.5f} | eval half-vs-sample {parts[2]:+.5f}')
