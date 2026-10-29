# Build one long table of binary responses across public benchmarks.
import pandas as pd, glob, os
rows=[]; meta=[]
for d in sorted(glob.glob('data/mdb/*/formatted_tables')):
    b=os.path.basename(os.path.dirname(d))
    try:
        r=pd.read_parquet(f'{d}/responses.parquet',columns=['subject_id','item_id','benchmark_id','trial','response'])
        s=pd.read_parquet(f'{d}/subjects.parquet')
        bm=pd.read_parquet(f'{d}/benchmarks.parquet')
    except Exception as e: print('skip',b,e); continue
    rel=bm['release'].iloc[0] if 'release' in bm else 'public'
    if rel!='public': continue
    vals=set(r['response'].dropna().unique())
    binary = vals<= {0.0,1.0}
    meta.append(dict(bench=b,n=len(r),binary=binary,nvals=len(vals),nsub=r.subject_id.nunique(),nitem=r.item_id.nunique(),rtype=bm.get('response_type',pd.Series([None])).iloc[0]))
    if not binary: continue
    r=r.merge(s[['subject_id','normalized_name','provider','release_date','reasoning_effort','harness']],on='subject_id',how='left')
    r['bench']=b
    rows.append(r)
M=pd.DataFrame(meta); M.to_csv('data/bench_meta.csv',index=False)
R=pd.concat(rows,ignore_index=True)
R=R[R['response'].notna()].copy(); R['y']=R['response'].astype('int8')
R.drop(columns=['response']).to_parquet('data/long.parquet')
print('benchmarks',len(M),'binary',M.binary.sum(),'rows',len(R),'subjects',R.subject_id.nunique(),'names',R.normalized_name.nunique())
print('names appearing in >=5 benchmarks:',(R.groupby('normalized_name').bench.nunique()>=5).sum())
print(R.groupby('bench').y.mean().describe())
