"""combo.py OUT DIR1 DIR2 ...: average logit ratings across rating dirs (items rated by all) into data/llm/OUT."""
import json,os,glob,re,sys,numpy as np
from scipy.special import logit, expit
def load(dirn):
    rat={}
    for f in glob.glob(f'data/llm/{dirn}/*.json'):
        b=os.path.basename(f).split('__')[0]; m=re.search(r'\{.*\}',open(f).read(),re.S)
        try:
            for x in json.loads(m.group(0))['ratings']: rat.setdefault(b,{})[str(x['id'])]=float(x['p'])
        except Exception: pass
    return rat
out=sys.argv[1]; Ds=[load(d) for d in sys.argv[2:]]; Cs=[load(d) for d in os.environ.get("COMMON","").split()]; os.makedirs(f'data/llm/{out}',exist_ok=True)
lp=lambda v: logit(np.clip(v/100,.02,.98))
for b in Ds[0]:
    ids=[i for i in Ds[0][b] if all(i in D.get(b,{}) for D in Ds+Cs)]
    json.dump({'ratings':[{'id':i,'p':float(100*expit(np.mean([lp(D[b][i]) for D in Ds])))} for i in ids]},open(f'data/llm/{out}/{b}__00.json','w'))
