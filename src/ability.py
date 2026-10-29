"""Subject ability features: relative logit accuracy of (bench, model) vs bench mean, explained by metadata."""
import numpy as np, pandas as pd, re
from scipy.special import logit
def pair_table(R):
    g=R.groupby(['bench','subject_id']).agg(acc=('y','mean'),n=('y','size'),name=('normalized_name','first'),prov=('provider','first'),rel=('release_date','first'),eff=('reasoning_effort','first')).reset_index()
    g=g[g.n>=20]
    g['lg']=logit(np.clip((g.acc*g.n+0.5)/(g.n+1),1e-3,1-1e-3))
    g['z']=g.lg-g.groupby('bench').lg.transform('mean')
    return g
TOK=re.compile(r'[a-z]+|\d+(?:\.\d+)?')
def feats(name,prov,rel,eff):
    f={}
    n=str(name).lower()
    for t in set(TOK.findall(n)): f['t:'+t]=1
    f['p:'+str(prov).lower()]=1
    try: y=float(str(rel)[:4])+ (float(str(rel)[5:7])-1)/12 if str(rel)[:4].isdigit() else None
    except: y=None
    if y: f['year']=y-2023.0; f['year2']=(y-2023.0)**2
    else: f['year_missing']=1
    f['e:'+str(eff).lower()]=1
    m=re.search(r'(\d+(?:\.\d+)?)\s*b\b',n)
    if m: f['logsize']=np.log(float(m.group(1)))
    for k in ['mini','nano','small','tiny','lite','flash','haiku','pro','opus','large','turbo','instruct','chat','reason','thinking']:
        if k in n: f['k:'+k]=1
    return f
class Ability:
    def __init__(self,alpha=3.0): self.alpha=alpha
    def fit(self,P):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import Ridge
        self.dv=DictVectorizer(); X=self.dv.fit_transform([feats(*r) for r in P[['name','prov','rel','eff']].values])
        self.m=Ridge(alpha=self.alpha).fit(X,P.z.values,sample_weight=np.sqrt(P.n.clip(upper=400).values))
        self.byname=P.groupby('name').z.mean().to_dict(); self.byname_n=P.groupby('name').size().to_dict()
        return self
    def predict(self,name,prov,rel,eff):
        x=self.dv.transform([feats(name,prov,rel,eff)]); zf=float(self.m.predict(x)[0])
        if name in self.byname:
            k=self.byname_n[name]; return (k*self.byname[name]+1*zf)/(k+1)
        return zf
