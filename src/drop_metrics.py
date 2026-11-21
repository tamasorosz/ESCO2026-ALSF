"""Fixed-reference metrics, identical to the previous convex-validation study."""
import numpy as np
import pandas as pd
def normalization(reference):
    f=reference[['Obj1','Obj2']].to_numpy(float)
    ideal=f.min(axis=0);span=np.ptp(f,axis=0)
    if not np.all(span>0):raise ValueError('Degenerate reference front')
    return ideal,span

def arc_reference(reference,n=501):
    """Uniform arclength reference for IGD+: linear interpolation, not new optima."""
    ideal,span=normalization(reference)
    f=(reference[['Obj1','Obj2']].to_numpy()-ideal)/span
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))]
    return np.column_stack([np.interp(np.linspace(0,distance[-1],n),distance,f[:,i]) for i in range(2)])

def indicators(f,reference):
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd_plus import IGDPlus
    ideal,span=normalization(reference)
    normalized=(np.asarray(f)-ideal)/span
    return dict(hypervolume=float(HV(ref_point=np.array([1.1,1.1]))(normalized)),
                igd_plus=float(IGDPlus(arc_reference(reference))(normalized)))

def polyline_distances(f,reference):
    ideal,span=normalization(reference)
    points=(np.asarray(f)-ideal)/span
    curve=(reference[['Obj1','Obj2']].to_numpy()-ideal)/span
    a=curve[:-1];edge=np.diff(curve,axis=0)
    denom=np.sum(edge**2,axis=1)
    t=np.clip(np.einsum('nki,ki->nk',points[:,None,:]-a,edge)/denom,0,1)
    return np.linalg.norm(points[:,None,:]-(a+t[:,:,None]*edge),axis=2).min(axis=1)

def exact_relative_sensitivity(model,x,floors=(.001,.001)):
    """Independent exhaustive/convex box extrema, as in the validated study.

    Does not modify the benchmark, reference solver or empirical classifications.
    """
    from scipy.optimize import lsq_linear
    from src.local_variation import relative_pairwise_variation
    x=np.asarray(x,float)
    step=model.s.independent_j_tolerance/model.s.j_scale
    lo=np.maximum(model.lower,x-step);hi=np.minimum(model.upper,x+step)
    bits=((np.arange(1<<model.n_var)[:,None]>>np.arange(model.n_var))&1)
    points=lo+bits*(hi-lo)
    maximum1=np.mean((points@model.K.T-model.b_target)**2,axis=1).max()
    result=lsq_linear(model.K,model.b_target,bounds=(lo,hi),tol=1e-12,max_iter=1000)
    if not result.success:raise RuntimeError('Independent exact minimum failed')
    nearest=np.where((lo<=0)&(hi>=0),0,np.minimum(abs(lo),abs(hi)))
    farthest=np.maximum(abs(lo),abs(hi))
    low=np.array([model.evaluate(result.x)[0],np.mean(nearest**2)])
    high=np.array([maximum1,np.mean(farthest**2)])
    center=np.array(model.evaluate(x))
    return np.maximum(relative_pairwise_variation(center,low,floors),
                      relative_pairwise_variation(center,high,floors))


def exact_audit(frame):
    from di_barba_solenoid_benchmark import DiBarbaSolenoid
    model=DiBarbaSolenoid();columns=[f'X{i+1}' for i in range(10)]
    selected=frame.sort_values(['Obj1','Obj2'])
    selected=selected.iloc[np.unique(np.linspace(0,len(selected)-1,min(25,len(selected)),dtype=int))]
    rows=[]
    for idx,row in selected.iterrows():
        r=exact_relative_sensitivity(model,row[columns].to_numpy(float))
        rows.append(dict(archive_row=int(idx),flag=row.get('flag','NEI'),Obj1=float(row.Obj1),
                         Obj2=float(row.Obj2),R1_exact=float(r[0]),R2_exact=float(r[1]),
                         exact_sensitive=bool(np.any(r>.1))))
    return pd.DataFrame(rows)
