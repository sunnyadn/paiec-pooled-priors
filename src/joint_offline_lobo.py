import sys, numpy as np
sys.argv=['x']; exec(open('src/joint_offline_opus.py').read().split("base = run('add', 0.0)")[0])
benches=sorted(D.b.unique()); full=D.copy()
grid=[0.03,0.05,0.08,0.10,0.12,0.15]
res={'v26like':[], 'fixed_lobo':[], 'none':[]}
for hb in benches:
    D=full[full.b!=hb]; best=min(grid, key=lambda be: run('add',be,reps=20).mean())
    D=full[full.b==hb]; base=run('add',0.0); fx=run('add',best); v26=run('v26like',0.10)
    print(hb,'chosen',best,'fixed %+.4f v26like %+.4f'%((fx-base).mean(),(v26-base).mean()))
    res['fixed_lobo'].append(fx-base); res['v26like'].append(v26-base)
for k in ['fixed_lobo','v26like']:
    d=np.concatenate(res[k]); print(k,'%+.4f ± %.4f'%(d.mean(), d.std(ddof=1)/np.sqrt(len(d))))
d=np.concatenate(res['fixed_lobo'])-np.concatenate(res['v26like']); print('fixed - v26like %+.4f ± %.4f'%(d.mean(), d.std(ddof=1)/np.sqrt(len(d))))
