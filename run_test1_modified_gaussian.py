"""Modified Gaussian: separate fixed-width paired experiment.

A separate, paired 60-run experiment; historical harness and algorithms unchanged.
"""
from pathlib import Path
from time import perf_counter
import argparse,json,os,sys,subprocess
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from src.preprocessing import OffspringRepair
from src.offspring_drop import OffspringDropFilter,DropNSGA2
from src.local_variation import LocalVariationState
from src.find_robust_clusters import classify_dataframe
from src.drop_metrics import indicators,polyline_distances,exact_relative_sensitivity
from benchmark_test1_modified_gaussian import to_optimization, to_physical
from src.modified_gaussian_observation import ObservedRepair, ObservedDrop

SEEDS=[*range(1,18),42,71,97]


def run_case(root,benchmark,mode,seed,generations=200,pop_size=100,flag_start=50):
    if benchmark != 'test1': raise ValueError('This restoration runs Test 1 only')
    root=Path(root);out=root/'runs'/benchmark/mode/f'seed{seed}'
    if (out/'summary.json').exists():
        existing=json.loads((out/'summary.json').read_text())
        for key,value in dict(benchmark=benchmark,mode=mode,seed=seed,generations=generations,pop_size=pop_size,flag_start=flag_start).items():
            if existing[key]!=value:raise ValueError('Resume settings differ: '+key)
        return existing
    out.mkdir(parents=True,exist_ok=False)
    if benchmark=='test1':
        from benchmark_test1_modified_gaussian import evaluate,LOWER,UPPER
        lower,upper=LOWER,UPPER;dimension=2;model=None
    else:
        from di_barba_solenoid_benchmark import DiBarbaSolenoid
        model=DiBarbaSolenoid();evaluate=model.evaluate;lower,upper=model.lower,model.upper;dimension=10
    columns=[f'X{i+1}' for i in range(dimension)]+['Obj1','Obj2']
    reference=pd.read_csv(root/'reference'/f'{benchmark}.csv')
    reference[['Obj1','Obj2']]=to_optimization(reference[['Obj1','Obj2']])
    observation=dict(generation=1,mode=mode)
    events=[]; evaluated_peak=[]; evaluated_all=[]; snapshots=[]; peak_labels=[]
    settings=dict(benchmark=benchmark,mode=mode,seed=seed,generations=generations,pop_size=pop_size,
        offsprings=pop_size,flag_start=flag_start,thresholds=[.1,.1],floors=[.001,.001],
        tolerances=[.1]*dimension,local_radius=1.,exclusion_radius=1.,min_samples=10,
        sbx_prob=.9,sbx_eta=5,pm_prob=1.,pm_eta=5,round_decimals=8,cache=False,
        diagnostic_clustering='final only; never used by exclusion',partition=False)
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    raw=[];keys=set();hist=[];logs=[];calls=0
    times=dict(objectives=0.,classification=0.,indicators=0.)
    class Problem(ElementwiseProblem):
        def __init__(self):super().__init__(n_var=dimension,n_obj=2,xl=lower,xu=upper)
        def _evaluate(self,x,result,*args,**kwargs):
            nonlocal calls
            tick=perf_counter();x=np.round(x,8);f=evaluate(x);calls+=1
            times['objectives']+=perf_counter()-tick
            if tuple(x) not in keys:keys.add(tuple(x));raw.append([*x,*f])
            if np.linalg.norm(x-2.5)<=.18: evaluated_peak.append(dict(generation=observation['generation'],X1=x[0],X2=x[1],Obj1=f[0],Obj2=f[1]))
            evaluated_all.append((observation['generation'],*x,*f))
            result['F']=to_optimization(f)
    repair=ObservedRepair(observation,events,'memory',{},1.,tolerances=np.full(dimension,.1),rng=np.random.default_rng(seed),
                           max_resample_attempts=1000 if benchmark=='test1' else 100)
    drop=ObservedDrop(observation,events,np.full(dimension,.1));state=LocalVariationState(np.full(dimension,.1))
    classified=None
    totals=dict(proposed=0,dropped=0,accepted=0,regenerated=0)
    class Monitor(Callback):
        def notify(self,a):
            nonlocal classified
            gen=int(a.n_gen)
            batch=dict(n_proposed=pop_size,n_dropped=0,n_accepted=pop_size) if gen==1 or mode!='drop' else a.last_drop_counts.copy()
            for k in ['proposed','dropped','accepted']:totals[k]+=batch['n_'+k]
            tick=perf_counter();evidence={}
            if gen>=flag_start:
                coords=np.asarray(raw)[:,:dimension];scores=np.asarray(raw)[:,dimension:]
                counts=state.update(coords,scores);bad=state.verified_violation
                flags=np.where(bad,'non-robust',np.where(counts>=10,'robust','NEI'))
                policy=drop.policy if mode=='drop' else repair.policy
                if mode!='baseline':policy.update(coords,bad)
                for radius in [.03,.10,.18]:
                    selected=flags[np.linalg.norm(coords-2.5,axis=1)<=radius]
                    peak_labels.append(dict(generation=gen,radius=radius,robust=int((selected=='robust').sum()),sensitive=int((selected=='non-robust').sum()),NEI=int((selected=='NEI').sum())))
                evidence=dict(n_robust=int((flags=='robust').sum()),n_nei=int((flags=='NEI').sum()),
                    n_verified=int(bad.sum()),neighbor_p95=float(np.quantile(counts,.95)))
            times['classification']+=perf_counter()-tick
            diagnostics=repair.pop_diagnostics();totals['regenerated']+=diagnostics['n_regenerated']
            tick=perf_counter();metrics=indicators(a.opt.get('F'),reference)
            times['indicators']+=perf_counter()-tick
            hist.append(dict(generation=gen,evaluations=calls,**metrics))
            snapshots.append(dict(generation=gen,evaluations=calls,X=a.opt.get('X').copy(),G=a.opt.get('F').copy()))
            observation['generation']=gen+1
            logs.append(dict(generation=gen,evaluations=calls,n_population=len(a.pop),**batch,**evidence,**diagnostics))
            if len(a.pop)!=pop_size:raise RuntimeError('Population size changed')
            if gen%50==0:print(benchmark,mode,seed,'gen',gen,'calls',calls,flush=True)
    kwargs=dict(pop_size=pop_size,n_offsprings=pop_size,repair=repair,
        crossover=SBX(prob=.9,eta=5),mutation=PM(prob=1.,eta=5),eliminate_duplicates=True)
    algorithm=DropNSGA2(drop,**kwargs) if mode=='drop' else NSGA2(**kwargs)
    # Drop/baseline round via the empty Repair policy; only Resample updates it.
    start=perf_counter()
    result=minimize(Problem(),algorithm,('n_gen',generations),seed=seed,callback=Monitor(),
                    copy_algorithm=False,save_history=False,verbose=False)
    elapsed=perf_counter()-start
    diagnostic_start=perf_counter()
    classified,stats=classify_dataframe(pd.DataFrame(raw,columns=columns),.1,.1,.1,
        state=state if generations>=flag_start else None,partition_width=None)
    classified.to_csv(out/'archive.csv',index=False)
    front=pd.DataFrame(np.column_stack([result.X,to_physical(result.F)]),columns=columns)
    front.to_csv(out/'pareto.csv',index=False)
    pd.DataFrame(hist).to_csv(out/'history.csv',index=False);pd.DataFrame(logs).to_csv(out/'process.csv',index=False)
    from src.modified_gaussian_reference import sensitivity, endpoint
    chosen=front.sort_values(['Obj1','Obj2'])
    assessment=[sensitivity(x) for x in chosen[['X1','X2']].to_numpy()]
    absolute=np.array([a for a,r in assessment]);values=np.array([r for a,r in assessment]);bad=(values>.1).any(axis=1)
    pd.DataFrame(dict(front_row=chosen.index,S1=absolute[:,0],S2=absolute[:,1],R1=values[:,0],R2=values[:,1],sensitive=bad)).to_csv(out/'sensitivity.csv',index=False)
    distance=polyline_distances(to_optimization(front[['Obj1','Obj2']]),reference)
    event_frame=pd.DataFrame(events,columns=['generation','stage','X1','X2','rejected'])
    event_frame.to_csv(out/'screening.csv.gz',index=False)
    pd.DataFrame(evaluated_all,columns=['generation','X1','X2','Obj1','Obj2']).to_csv(out/'evaluated.csv.gz',index=False)
    pd.DataFrame(peak_labels).to_csv(out/'peak_classification_history.csv',index=False)
    final_population=pd.DataFrame(np.column_stack([algorithm.pop.get('X'),to_physical(algorithm.pop.get('F'))]),columns=columns)
    final_population.to_csv(out/'final_population.csv',index=False)
    np.savez_compressed(out/'front_history.npz',**{f'{i}_{key}':snap[key] for i,snap in enumerate(snapshots) for key in ['X','G']})
    peak_counts={}
    evaluated_coords=np.array(evaluated_all)[:,1:3]
    for radius in [.03,.10,.18]:
        tag=str(radius)
        near=np.linalg.norm(event_frame[['X1','X2']].to_numpy()-2.5,axis=1)<=radius
        rejected=event_frame.rejected.to_numpy(bool)
        peak_counts[tag]=dict(evaluated=int((np.linalg.norm(evaluated_coords-2.5,axis=1)<=radius).sum()),
            final_pareto=int((np.linalg.norm(front[['X1','X2']].to_numpy()-2.5,axis=1)<=radius).sum()),
            final_population=int((np.linalg.norm(final_population[['X1','X2']].to_numpy()-2.5,axis=1)<=radius).sum()),
            checked=int(near.sum()),rejected=int((near & rejected).sum()),accepted=int((near & ~rejected).sum()),
            rejected_outside=int((~near & rejected).sum()),
            original_rejected=int((near & rejected & (event_frame.stage=='original')).sum()),
            replacement_rejected=int((near & rejected & (event_frame.stage=='replacement')).sum()))
    closest_evaluated=float(np.linalg.norm(evaluated_coords-endpoint(),axis=1).min())
    closest_final=float(np.linalg.norm(front[['X1','X2']].to_numpy()-endpoint(),axis=1).min())
    # Narrow segment fixed before runs: diagonal from t=2.4 to the endpoint.
    narrow=reference[reference.X1>=2.4]
    from src.drop_metrics import normalization,arc_reference
    ideal,span=normalization(reference)
    from scipy.spatial.distance import cdist
    narrow_distance=cdist((narrow[['Obj1','Obj2']].to_numpy()-ideal)/span,
        (to_optimization(front[['Obj1','Obj2']])-ideal)/span).min(axis=1)
    # Arc-uniform coverage avoids overweighting adaptive sampling knots.
    narrow_curve=(narrow[['Obj1','Obj2']].to_numpy()-ideal)/span
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(narrow_curve,axis=0),axis=1))]
    narrow_uniform=np.column_stack([np.interp(np.linspace(0,arc[-1],501),arc,narrow_curve[:,j]) for j in range(2)])
    narrow_distance=cdist(narrow_uniform,(to_optimization(front[['Obj1','Obj2']])-ideal)/span).min(axis=1)
    summary=dict(**settings,actual_model_calls=calls,evaluator_calls=int(algorithm.evaluator.n_eval),
        unique_archive=len(raw),expected_evaluations=generations*pop_size,saved_evaluations=totals['dropped'],
        **totals,run_seconds=elapsed,timings=times,posthoc_seconds=perf_counter()-diagnostic_start,
        **hist[-1],mean_reference_distance=float(distance.mean()),
        sensitive_audit_fraction=float(bad.mean()),sensitivity_kind='complete_stationarity_enumeration_float64',
        R1_median=float(np.median(values[:,0])),R2_median=float(np.median(values[:,1])),
        peak_counts=peak_counts,closest_evaluated_endpoint=closest_evaluated,closest_final_endpoint=closest_final,
        narrow_coverage=float((narrow_distance<=.01).mean()),narrow_mean_distance=float(narrow_distance.mean()),
        peak_rejected=peak_counts['0.18']['rejected'],
        peak_labels=classified.loc[np.linalg.norm(classified[['X1','X2']].to_numpy()-2.5,axis=1)<=.18,'flag'].value_counts().to_dict(),
        objective_directions=['min','max'],final_labels=classified.flag.value_counts().to_dict(),final_clusters=stats['n_clusters'])
    if mode=='drop':
        assert totals['accepted']==calls and generations*pop_size-calls==totals['dropped']
    else:assert calls==generations*pop_size
    assert calls==algorithm.evaluator.n_eval
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--generations',type=int,default=200);p.add_argument('--pop-size',type=int,default=100)
    p.add_argument('--flag-start',type=int,default=50);p.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    args=p.parse_args();root=args.root
    if not (root/'protocol.json').exists(): raise ValueError('Run prepare_modified_gaussian.py first: lock reference and HV targets')
    protocol=json.loads((root/'protocol.json').read_text())
    for key in ['generations','pop_size','flag_start','seeds']:
        if protocol[key]!=getattr(args,key):raise ValueError('Locked protocol mismatch: '+key)
    import hashlib
    if hashlib.sha256((root/'reference/test1.csv').read_bytes()).hexdigest()!=protocol['reference_sha256']:
        raise ValueError('Reference differs from locked protocol')
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,str(root),'test1',mode,seed,args.generations,args.pop_size,args.flag_start)
                 for seed in args.seeds for mode in ['baseline','resample','drop']]
        for f in as_completed(futures):
            r=f.result();results.append(r)
            print('COMPLETED',len(results),'/',len(futures),r['mode'],r['seed'],flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results]).to_csv(root/'all_runs.csv',index=False)
if __name__=='__main__':main()
