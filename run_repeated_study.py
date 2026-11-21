"""Common paired-seed harness for Test 1 / solenoid and baseline/resample/drop.

Genetic/sensitivity rules unchanged. Pointwise evidence updates every generation;
DBSCAN/cells are reporting only, computed once at completion (cells disabled).
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
from src.study_reference import test1_reference

SEEDS=[*range(1,18),42,71,97]


def run_case(root,benchmark,mode,seed,generations=200,pop_size=100,flag_start=50):
    root=Path(root);out=root/'runs'/benchmark/mode/f'seed{seed}'
    if (out/'summary.json').exists():
        existing=json.loads((out/'summary.json').read_text())
        for key,value in dict(benchmark=benchmark,mode=mode,seed=seed,generations=generations,pop_size=pop_size,flag_start=flag_start).items():
            if existing[key]!=value:raise ValueError('Resume settings differ: '+key)
        return existing
    out.mkdir(parents=True,exist_ok=False)
    if benchmark=='test1':
        from benchmark_test1 import evaluate,LOWER,UPPER
        lower,upper=LOWER,UPPER;dimension=2;model=None
    else:
        from di_barba_solenoid_benchmark import DiBarbaSolenoid
        model=DiBarbaSolenoid();evaluate=model.evaluate;lower,upper=model.lower,model.upper;dimension=10
    columns=[f'X{i+1}' for i in range(dimension)]+['Obj1','Obj2']
    reference=pd.read_csv(root/'reference'/f'{benchmark}.csv')
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
            result['F']=f
    repair=OffspringRepair('memory',{},1.,tolerances=np.full(dimension,.1),rng=np.random.default_rng(seed),
                           max_resample_attempts=1000 if benchmark=='test1' else 100)
    drop=OffspringDropFilter(np.full(dimension,.1));state=LocalVariationState(np.full(dimension,.1))
    classified=None
    totals=dict(proposed=0,dropped=0,accepted=0,regenerated=0)
    class Monitor(Callback):
        def notify(self,a):
            nonlocal classified
            gen=int(a.n_gen)
            batch=dict(n_proposed=pop_size,n_dropped=0,n_accepted=pop_size) if gen==1 or mode!='drop' else a.last_drop_counts.copy()
            for k in ['proposed','dropped','accepted']:totals[k]+=batch['n_'+k]
            tick=perf_counter();evidence={}
            if mode!='baseline' and gen>=flag_start:
                coords=np.asarray(raw)[:,:dimension];scores=np.asarray(raw)[:,dimension:]
                counts=state.update(coords,scores);bad=state.verified_violation
                flags=np.where(bad,'non-robust',np.where(counts>=10,'robust','NEI'))
                policy=drop.policy if mode=='drop' else repair.policy
                policy.update(coords,bad)
                evidence=dict(n_robust=int((flags=='robust').sum()),n_nei=int((flags=='NEI').sum()),
                    n_verified=int(bad.sum()),neighbor_p95=float(np.quantile(counts,.95)))
            times['classification']+=perf_counter()-tick
            diagnostics=repair.pop_diagnostics();totals['regenerated']+=diagnostics['n_regenerated']
            tick=perf_counter();metrics=indicators(a.opt.get('F'),reference)
            times['indicators']+=perf_counter()-tick
            hist.append(dict(generation=gen,evaluations=calls,**metrics))
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
        state=state if mode!='baseline' and generations>=flag_start else None,partition_width=None)
    classified.to_csv(out/'archive.csv',index=False)
    front=pd.DataFrame(np.column_stack([result.X,result.F]),columns=columns)
    front.to_csv(out/'pareto.csv',index=False)
    pd.DataFrame(hist).to_csv(out/'history.csv',index=False);pd.DataFrame(logs).to_csv(out/'process.csv',index=False)
    chosen=front.sort_values(['Obj1','Obj2']).iloc[np.unique(np.linspace(0,len(front)-1,min(25,len(front)),dtype=int))]
    audit=[]
    if model is None:
        from validate_alsf import perturbation_test
        values,bad=perturbation_test(chosen[columns[:dimension]].to_numpy())
    else:
        values=np.array([exact_relative_sensitivity(model,x) for x in chosen[columns[:dimension]].to_numpy()]);bad=(values>.1).any(axis=1)
    pd.DataFrame(dict(front_row=chosen.index,R1=values[:,0],R2=values[:,1],sensitive=bad)).to_csv(out/'sensitivity.csv',index=False)
    distance=polyline_distances(front[['Obj1','Obj2']],reference)
    summary=dict(**settings,actual_model_calls=calls,evaluator_calls=int(algorithm.evaluator.n_eval),
        unique_archive=len(raw),expected_evaluations=generations*pop_size,saved_evaluations=totals['dropped'],
        **totals,run_seconds=elapsed,timings=times,posthoc_seconds=perf_counter()-diagnostic_start,
        **hist[-1],mean_reference_distance=float(distance.mean()),
        sensitive_audit_fraction=float(bad.mean()),sensitivity_kind='finite_grid' if model is None else 'exact_convex_box',
        R1_median=float(np.median(values[:,0])),R2_median=float(np.median(values[:,1])),
        final_labels=classified.flag.value_counts().to_dict(),final_clusters=stats['n_clusters'])
    if mode=='drop':
        assert totals['accepted']==calls and generations*pop_size-calls==totals['dropped']
    else:assert calls==generations*pop_size
    assert calls==algorithm.evaluator.n_eval
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--modes',nargs='+',choices=['baseline','resample','drop'],default=['baseline','resample','drop'])
    p.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    p.add_argument('--benchmarks',nargs='+',choices=['test1','solenoid'],default=['test1','solenoid'])
    p.add_argument('--generations',type=int,default=200);p.add_argument('--pop-size',type=int,default=100)
    p.add_argument('--flag-start',type=int,default=50)
    args=p.parse_args();root=args.root
    if not (root/'reference').exists():
        (root/'reference').mkdir(parents=True)
        test1_reference().to_csv(root/'reference/test1.csv',index=False)
        # Reuse the validated convex reference byte-for-byte.
        import shutil
        shutil.copy2(Path('docs/alsf_drop/reference/reference.csv'),root/'reference/solenoid.csv')
        protocol=dict(**vars(args),primary_target_fraction=.995,secondary_target_fractions=[.99,.999],
            target_hv={b:.995*indicators(pd.read_csv(root/'reference'/f'{b}.csv')[['Obj1','Obj2']],pd.read_csv(root/'reference'/f'{b}.csv'))['hypervolume'] for b in ['test1','solenoid']},
            analysis='paired by seed, median/IQR, paired bootstrap CI, two-sided Wilcoxon with Holm correction; censored target attainment never imputed',
            reference_test1='analytic fixed-f1 envelope; equations unchanged',
            reference_solenoid='unchanged validated convex front',
            timing='two concurrent workers, single-threaded BLAS; descriptive wall time')
        (root/'protocol.json').write_text(json.dumps(protocol,default=str,indent=2)+'\n')
    else:
        previous=json.loads((root/'protocol.json').read_text())
        for key in ['modes','seeds','benchmarks','generations','pop_size','flag_start']:
            if previous[key]!=getattr(args,key):raise ValueError('Protocol mismatch on resume: '+key)
    tasks=[(str(root),b,m,s,args.generations,args.pop_size,args.flag_start) for b in args.benchmarks for s in args.seeds for m in args.modes]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,*t):t for t in tasks}
        for f in as_completed(futures):
            result=f.result();results.append(result)
            print('COMPLETED',len(results),'/',len(tasks),result['benchmark'],result['mode'],result['seed'],flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results]).to_csv(root/'all_runs.csv',index=False)
if __name__=='__main__':main()
