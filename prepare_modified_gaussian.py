"""Validate and lock the independent reference before any primary runs."""
from pathlib import Path
import argparse,hashlib,json,platform,subprocess,sys
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from benchmark_test1_modified_gaussian import evaluate,to_optimization
from src.modified_gaussian_reference import reference,endpoint,stationary,sensitivity
from src.drop_metrics import indicators,normalization


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():raise ValueError('Protocol already locked; use a separate output directory')
    out=root/'reference';out.mkdir(exist_ok=False)
    front=reference();fine=reference(2001)
    front.to_csv(out/'test1.csv',index=False);fine.to_csv(out/'refined.csv',index=False)
    opt=front.copy();opt[['Obj1','Obj2']]=to_optimization(front[['Obj1','Obj2']])
    fineopt=fine.copy();fineopt[['Obj1','Obj2']]=to_optimization(fine[['Obj1','Obj2']])
    hv=indicators(opt[['Obj1','Obj2']],opt)['hypervolume']
    finehv=indicators(fineopt[['Obj1','Obj2']],fineopt)['hypervolume']
    end=endpoint();endpoint_f=evaluate([end,end])
    # Numerical cross-checks supplement the full-domain analytical proof.
    rng=np.random.default_rng(20261116)
    x=rng.uniform(-5,10,(250000,2));q=np.sum(x*x,axis=1);f=evaluate(x)
    diagonal=np.sqrt(q[q<=50]/2)
    envelope=evaluate(np.column_stack((diagonal,diagonal)))[:,1]
    gap=float((f[q<=50,1]-envelope).max())
    assert gap<1e-10
    starts=[]
    for seed in range(5):
        # Include narrow peak analytically rather than trusting stochastic discovery.
        r=differential_evolution(lambda x:-evaluate(x)[1],[(-5,10)]*2,seed=seed,
            x0=[2.5,2.5],tol=1e-11,popsize=30,polish=True)
        starts.append(dict(seed=seed,X1=float(r.x[0]),X2=float(r.x[1]),f2=float(-r.fun),success=bool(r.success)))
    pd.DataFrame(starts).to_csv(out/'global_numerical_checks.csv',index=False)
    roots=stationary();derivative=lambda t: -4*(t-5)-400/.01*(t-2.5)*np.exp(-2*(t-2.5)**2/.01)
    t=np.linspace(0,end,100001)
    assert np.all(derivative(t[:-1])>0)
    assert abs(end-2.50024998)<1e-8 and abs(endpoint_f[1]-87.50125)<3e-7
    validation=dict(endpoint_t=end,endpoint_objectives=endpoint_f.tolist(),stationary_t=(2.5+roots).tolist(),
        endpoint_derivative_residual=float(abs(derivative(end))),random_points=len(x),max_off_diagonal_improvement=gap,
        reference_rows=len(front),refined_rows=len(fine),reference_hv=hv,refined_hv=finehv,hv_resolution_difference=abs(hv-finehv),
        root_accuracy='bracketed monotone intervals, xtol=5e-15; float64, not interval certification',
        domain_proof='q<=50: strictly increasing in s=x1+x2, s<=sqrt(2q), diagonal feasible. q>50: f2<=100 exp(-(sqrt(q)-sqrt(12.5))^2/.01)<1e-540; endpoint f2>87.5 and f1<50.',
        monotonicity_proof='On 0<=t<=2.5 both derivative terms nonnegative. On [2.5,t*] complete derivative-root enumeration gives no other root. All stationary points and endpoints on [0,5] checked for global maximum. Thus t>t* dominated by t*.')
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    rows=[]
    for p in np.linspace(2.35,end,101):
        a,r=sensitivity([p,p]);rows.append(dict(X1=p,X2=p,S1=a[0],S2=a[1],R1=r[0],R2=r[1],sensitive=bool((r>.1).any())))
    pd.DataFrame(rows).to_csv(out/'peak_sensitivity.csv',index=False)
    ideal,span=normalization(opt)
    protocol=dict(benchmark='modified Gaussian min f1, max f2',width=.01,amplitude=100,center=[2.5,2.5],bounds=[-5,10],
        seeds=[*range(1,18),42,71,97],modes=['baseline','resample','drop'],generations=200,pop_size=100,n_offsprings=100,flag_start=50,
        sbx_prob=.9,sbx_eta=5,pm_prob=1.,pm_eta=5,tolerances=[.1,.1],thresholds=[.1,.1],floors=[.001,.001],min_samples=10,
        reference_hv=hv,target_fractions=[.99,.995,.999],target_hv=[hv*v for v in [.99,.995,.999]],
        radii=[.03,.10,.18],narrow_segment_t=[2.4,end],narrow_coverage_normalized_distance=.01,
        reference_sha256=hashlib.sha256((out/'test1.csv').read_bytes()).hexdigest(),
        normalization_ideal=ideal.tolist(),normalization_span=span.tolist(),hv_reference=[1.1,1.1],
        igd_reference='501 equally spaced normalized objective-arclength points',
        target_definition='Fractions of reference HV; locked before primary runs',workers=2)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(python=sys.version,platform=platform.platform(),packages={p:version(p) for p in ['numpy','scipy','pandas','pymoo','matplotlib','scikit-learn']},
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('benchmark_test1_modified_gaussian.py'),Path('run_test1_modified_gaussian.py'),Path('src/modified_gaussian_reference.py'),Path('src/modified_gaussian_observation.py'),Path('src/local_variation.py'),Path('src/preprocessing.py'),Path('src/offspring_drop.py'),Path('src/offspring_policy.py')]})
    (root/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    print(json.dumps(validation,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);prepare(p.parse_args().root)
