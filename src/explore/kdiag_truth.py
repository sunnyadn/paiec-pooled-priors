import json, csv, numpy as np
E={(e['bench'],e['subject_id']):e['acc'] for e in json.load(open('data/llm/prior_eval_all.json'))}
rng=np.random.default_rng(0)
for S in ('ablate','multi'):
    rec=[]
    for s in (5,6,7):
        D=[json.loads(l) for l in open(f'data/draws/{S}_s{s}.jsonl')]
        for r in json.load(open(f'kdiag/{S}_s{s}_k4.json')):
            sp=[v for v in r['sp'] if v is not None]
            if r['n'] not in (1,3,7) or not sp or len(set(sp))!=1 or r['single']!=r['calls']: continue
            key=(r['b'],D[r['draw']][r['pair']]['s'])
            if key not in E: continue
            rec.append((r['n'],sp[0],r['k'],r['got'],E[key],np.mean(r['y'])))
    for n in (1,3,7):
        a=np.array([x[1:] for x in rec if x[0]==n])
        if not len(a): print(f"{S} n={n}: no records"); continue
        sp,k,got,pop,ev=a.T
        act={K:(((k+K*sp)/(got+K)-pop)**2).mean() for K in (4,8)}
        kb=rng.binomial(got.astype(int)[:,None],pop[:,None],size=(len(pop),2000))
        bino={K:(((kb+K*sp[:,None])/(got[:,None]+K)-pop[:,None])**2).mean() for K in (4,8)}
        actev={K:(((k+K*sp)/(got+K)-ev)**2).mean() for K in (4,8)}
        print(f'{S} n={n} recs {len(a)} | prior err^2 vs pop {((sp-pop)**2).mean():.4f} | actual labels vs pop K4 {act[4]:.4f} K8 {act[8]:.4f} | binomial labels vs pop K4 {bino[4]:.4f} K8 {bino[8]:.4f} | actual vs eval-subset K4 {actev[4]:.4f} K8 {actev[8]:.4f} | mean k/got {np.mean(k/np.maximum(got,1)):.3f} vs pop {pop.mean():.3f} vs ev {ev.mean():.3f}')
