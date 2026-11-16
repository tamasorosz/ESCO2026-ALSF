"""Construct and independently validate reference; lock targets BEFORE new runs."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from benchmark_test1 import evaluate,gradient
from benchmark_test1_minmax import to_optimization
from src.test1_minmax_reference import reference_front,branches
from src.drop_metrics import indicators
from validate_alsf import perturbation_test

def prepare(root):
    root=Path(root);out=root/'reference';out.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():raise ValueError('Protocol already locked')
    front,features=reference_front(10001,4001)
    front.to_csv(out/'test1.csv',index=False)
    def metric(f,ref):
        ref=ref.copy();ref[['Obj1','Obj2']]=to_optimization(ref[['Obj1','Obj2']]);return indicators(to_optimization(f[['Obj1','Obj2']]),ref)
    fine,_=reference_front(20001,8001)
    hv=metric(front,front)['hypervolume'];finehv=metric(fine,fine)['hypervolume']
    rng=np.random.default_rng(20261116);rows=[]
    qs=np.unique(np.r_[np.linspace(.01,50,21),np.linspace(12.47,12.51,15),features['peak_local_maximum_q']])
    for q in qs:
        envelope=float(max(branches(q)))
        angles=np.linspace(-np.pi,np.pi,65537)
        x=np.sqrt(q)*np.column_stack([np.cos(angles),np.sin(angles)])
        x=np.clip(x[((x>=-5-1e-12)&(x<=10+1e-12)).all(axis=1)],-5,10)
        sampled=float(evaluate(x)[:,1].max())
        starts=np.r_[[-3*np.pi/4,np.pi/4],rng.uniform(-np.pi,np.pi,14)]
        for angle in starts:
            x0=np.sqrt(q)*np.array([np.cos(angle),np.sin(angle)])
            if (x0<-5-1e-12).any() or (x0>10+1e-12).any():continue
            x0=np.clip(x0,-5,10)
            r=minimize(lambda z:-float(evaluate(z)[1]),x0,jac=lambda z:-gradient(z)[1],
                constraints={'type':'eq','fun':lambda z:float(z@z-q),'jac':lambda z:2*z},
                bounds=[(-5,10)]*2,method='SLSQP',options={'ftol':1e-11,'maxiter':500})
            feasible=bool(abs(r.x@r.x-q)<1e-7 and (r.x>=-5-1e-9).all() and (r.x<=10+1e-9).all())
            rows.append(dict(q=q,success=bool(r.success),feasible=feasible,value=float(evaluate(r.x)[1]),envelope=envelope,sampling_max=sampled,residual=float(abs(r.x@r.x-q))))
    numerical=pd.DataFrame(rows);numerical.to_csv(out/'multistart.csv',index=False)
    violation=float((numerical.loc[numerical.feasible,'value']-numerical.loc[numerical.feasible,'envelope']).max())
    best=numerical.loc[numerical.feasible].groupby('q').agg(value=('value','max'),envelope=('envelope','first'))
    gap=float((best.envelope-best.value).max())
    assert violation<1e-6 and gap<1e-6
    points=front.loc[front.peak_branch,['X1','X2']].to_numpy()[::max(1,int(front.peak_branch.sum())//25)]
    values,bad=perturbation_test(points)
    pd.DataFrame(dict(X1=points[:,0],X2=points[:,1],R1=values[:,0],R2=values[:,1],sensitive=bad)).to_csv(out/'peak_independent_audit.csv',index=False)
    validation=dict(**features,reference_rows=len(front),refined_rows=len(fine),hv=hv,refined_hv=finehv,hv_resolution_difference=abs(hv-finehv),
        multistart_rows=len(rows),multistart_failures=int((~numerical.success).sum()),max_feasible_envelope_violation=violation,max_best_multistart_gap=gap,
        max_polar_sampling_gap=float((numerical.envelope-numerical.sampling_max).max()),polar_angles=65537,
        peak_audit_sensitive_fraction=float(bad.mean()),peak_audit_kind='21x21 finite perturbation grid; positive witnesses, no stability guarantee')
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    protocol=dict(benchmark='original Test1 min f1, max f2',seeds=[*range(1,18),42,71,97],modes=['baseline','resample','drop'],generations=200,pop_size=100,flag_start=50,
        reference_hv=hv,target_fractions=[.99,.995,.999],target_hv=[hv*v for v in [.99,.995,.999]],peak_radius=.18,
        reference_sha256=hashlib.sha256((out/'test1.csv').read_bytes()).hexdigest(),normalization_ideal=[0,-200],normalization_span=[50,150],hv_reference=[1.1,1.1],
        target_definition='Locked before new optimization runs; fractions of independent corrected reference HV',
        manuscript_blob='09d3c15ffeaa36acd4696b4d4f747f2d6d39e04e',workers=2)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(validation,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);prepare(p.parse_args().root)
