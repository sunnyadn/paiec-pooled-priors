import json, csv, numpy as np, collections
F={(r['bench'],r['subject_id']):r for r in csv.DictReader(open('data/forecasts/forecasts.csv'))}
E={(e['bench'],e['subject_id']):e['acc'] for e in json.load(open('data/llm/prior_eval_all.json'))}
for S in ('ablate','multi'):
    rows=[]
    for s in (5,6,7):
        D=[json.loads(l) for l in open(f'data/draws/{S}_s{s}.jsonl')]
        for r in json.load(open(f'kdiag/{S}_s{s}_k4.json')):
            if r['n']!=1: continue
            sp=[v for v in r['sp'] if v is not None]
            if not sp: continue
            subj=D[r['draw']][r['pair']]['s']; key=(r['b'],subj); f=F.get(key)
            yb=np.mean(r['y'])
            rows.append((sp[0], yb, None if f is None or f['estimate'] in ('','None') else 0.5+0.75*(float(f['estimate'])-0.5), f['benchmark_has_calibration_example'] if f else 'missing', key))
    c=collections.Counter(r[3] for r in rows); print(S, 'records', len(rows), 'match in forecasts:', dict(c))
    m=[r for r in rows if r[2] is not None]
    a=np.array([(r[0],r[1],r[2]) for r in m])
    print(f'  matched {len(m)}: sim prior err^2 {np.mean((a[:,0]-a[:,1])**2):.4f} | table (shrunk) err^2 {np.mean((a[:,2]-a[:,1])**2):.4f} | corr(sim,table) {np.corrcoef(a[:,0],a[:,2])[0,1]:.2f} | mean sim {a[:,0].mean():.3f} table {a[:,2].mean():.3f} truth {a[:,1].mean():.3f}')
    for cal in ('True','False'):
        mm=[r for r in m if r[3]==cal]
        if mm:
            b=np.array([(r[0],r[1],r[2]) for r in mm]); print(f'   calib bench={cal}: n {len(mm)} sim err^2 {np.mean((b[:,0]-b[:,1])**2):.4f} table err^2 {np.mean((b[:,2]-b[:,1])**2):.4f}')
