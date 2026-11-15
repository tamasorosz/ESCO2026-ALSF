"""ALSF-Drop: one normal NSGA-II batch, pre-evaluation rejection, no refill."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.core.repair import Repair
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from di_barba_solenoid_benchmark import DiBarbaSolenoid
from src.local_variation import LocalVariationState
from src.find_robust_clusters import classify_dataframe
from src.offspring_drop import OffspringDropFilter,DropNSGA2
from src.drop_metrics import indicators

ROOT=Path(__file__).resolve().parent
COLS=[f'X{i+1}' for i in range(10)]+['Obj1','Obj2']


def run(outdir,generations=200,pop_size=100,offsprings=100,flag_start=50,seed=42,reference=None):
    if generations<1 or pop_size<4 or offsprings<1 or flag_start<1:
        raise ValueError('Positive generations/offsprings/flag-start and population >=4 required')
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=False)
    settings=dict(generations=generations,pop_size=pop_size,offsprings=offsprings,
        flag_start=flag_start,seed=seed,tolerances=[.1]*10,local_radius=1.,exclusion_radius=1.,
        local_min_samples=10,objective_thresholds=[.1,.1],objective_floors=[.001,.001],
        sbx_prob=.9,sbx_eta=5,pm_prob=1.,pm_eta=5,dbscan_eps=1.,dbscan_min_samples=10,
        partition_enabled=False,cache_enabled=False,copy_algorithm=False,
        filtering_timing='classify after flag_start generation; first affected batch is flag_start+1',
        initialization='unfiltered, included in cumulative proposal/evaluation counts')
    (outdir/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    model=DiBarbaSolenoid();archive=[];known=set();process=[];history=[]
    counters=dict(cumulative_proposed=0,cumulative_dropped=0,cumulative_evaluated=0,
                  cumulative_accepted=0,cumulative_shortfall=0)
    classifier_state=LocalVariationState(np.full(10,.1));classified=None
    times=dict(archive_processing_seconds=0.,indicator_seconds=0.,objective_seconds=0.)
    class Problem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=10,n_obj=2,xl=model.lower,xu=model.upper)
            self.calls=0;self.repeated=0
        def _evaluate(self,x,out,*args,**kwargs):
            tick=perf_counter();x=np.round(np.asarray(x),8)
            values=model.evaluate(x);self.calls+=1
            times['objective_seconds']+=perf_counter()-tick
            key=tuple(x)
            if key not in known:
                known.add(key);archive.append([*x,*values])
            else:self.repeated+=1
            out['F']=values
    class RoundAndCount(Repair):
        def __init__(self):super().__init__();self.raw=0
        def _do(self,problem,X,**kwargs):
            self.raw+=len(X)
            return np.clip(np.round(np.asarray(X,dtype=float),8),problem.xl,problem.xu)
    class CountDuplicates(DefaultDuplicateElimination):
        def __init__(self):super().__init__();self.removed=0
        def do(self,pop,*args,**kwargs):
            result=super().do(pop,*args,**kwargs)
            kept=result[0] if kwargs.get('return_indices',False) else result
            self.removed+=len(pop)-len(kept)
            return result
    problem=Problem();rounder=RoundAndCount();duplicates=CountDuplicates()
    drop=OffspringDropFilter(np.full(10,.1))
    class Monitor(Callback):
        def notify(self,a):
            nonlocal classified
            generation=int(a.n_gen)
            batch=dict(n_proposed=len(a.pop),n_dropped=0,n_accepted=len(a.pop)) if generation==1 else a.last_drop_counts.copy()
            n_eval=problem.calls-counters['cumulative_evaluated']
            if n_eval!=batch['n_accepted']:raise RuntimeError('Accepted/evaluated mismatch; cache or evaluator skip')
            expected=pop_size if generation==1 else offsprings
            shortfall=expected-batch['n_proposed']
            if shortfall<0:raise RuntimeError('Too many proposals')
            for key,col in [('cumulative_proposed','n_proposed'),('cumulative_dropped','n_dropped'),('cumulative_accepted','n_accepted')]:
                counters[key]+=batch[col]
            counters['cumulative_evaluated']=problem.calls;counters['cumulative_shortfall']+=shortfall
            used_centers=drop.policy.count
            tick=perf_counter();stats={};counts={};available=False
            if generation>=flag_start:
                classified,stats=classify_dataframe(pd.DataFrame(archive,columns=COLS),.1,.1,.1,
                                                    state=classifier_state,partition_width=None)
                drop.update_flags(classified);available=True
                flags=classified.flag.value_counts()
                counts=dict(n_verified_violation=int(classified.verified_violation.sum()),
                    n_robust=int(flags.get('robust',0)),n_nei=int(flags.get('NEI',0)),
                    local_median=float(classified.local_n.median()),local_p95=float(classified.local_n.quantile(.95)))
            elapsed=perf_counter()-tick;times['archive_processing_seconds']+=elapsed
            row=dict(generation=generation,initialization=generation==1,**batch,n_evaluated=n_eval,
                **counters,n_raw_generated=rounder.raw,n_duplicate_removed=duplicates.removed,
                n_generation_shortfall=shortfall,n_population=len(a.pop),
                n_exclusion_centers_used=used_centers,n_exclusion_centers=drop.policy.count,
                evidence_available=available,n_cache_hits=0,archive_seconds=elapsed,
                **counts,**{k:stats[k] for k in ['n_clusters','n_noise','n_mixed_clusters'] if k in stats})
            rounder.raw=0;duplicates.removed=0
            if len(a.pop)!=pop_size:raise RuntimeError('Invalid parent population size')
            process.append(row)
            if reference is not None:
                tick=perf_counter();metric=indicators(a.opt.get('F'),reference)
                times['indicator_seconds']+=perf_counter()-tick
                history.append(dict(generation=generation,evaluations=problem.calls,**metric))
            if generation%50==0:print('drop',seed,'generation',generation,'calls',problem.calls,'dropped',counters['cumulative_dropped'],flush=True)
    algorithm=DropNSGA2(drop,pop_size=pop_size,n_offsprings=offsprings,repair=rounder,
        eliminate_duplicates=duplicates,crossover=SBX(prob=.9,eta=5),mutation=PM(prob=1.,eta=5))
    tick=perf_counter()
    result=minimize(problem,algorithm,('n_gen',generations),seed=seed,callback=Monitor(),
                    copy_algorithm=False,save_history=False,verbose=False)
    run_seconds=perf_counter()-tick
    raw=pd.DataFrame(archive,columns=COLS);raw.to_csv(outdir/'archive.csv',index=False)
    tick=perf_counter()
    if classified is None:
        classified,_=classify_dataframe(raw,.1,.1,.1,state=classifier_state,partition_width=None)
    posthoc_seconds=perf_counter()-tick
    classified.to_csv(outdir/'archive_classified.csv',index=False)
    pareto=pd.DataFrame(np.column_stack([result.X,result.F]),columns=COLS)
    lookup={tuple(x):flag for x,flag in zip(classified[COLS[:10]].to_numpy(),classified.flag)}
    pareto['flag']=[lookup[tuple(np.round(x,8))] for x in result.X]
    pareto.to_csv(outdir/'pareto.csv',index=False)
    p=pd.DataFrame(process)
    # No evidence before flag_start: blanks distinguish unmeasured from zero.
    for name in ['n_verified_violation','n_robust','n_nei']:
        if name not in p:p[name]=np.nan
    p.to_csv(outdir/'process.csv',index=False)
    if history:pd.DataFrame(history).to_csv(outdir/'history.csv',index=False)
    expected=pop_size+(generations-1)*offsprings
    summary=dict(total_proposed=counters['cumulative_proposed'],total_dropped=counters['cumulative_dropped'],
        total_accepted=counters['cumulative_accepted'],actual_model_calls=problem.calls,
        evaluator_evaluations=int(result.algorithm.evaluator.n_eval),baseline_expected_evaluations=expected,
        saved_evaluations=counters['cumulative_dropped'],total_missing_evaluations=expected-problem.calls,
        non_alsf_shortfall=counters['cumulative_shortfall'],cache_hits=0,
        archive_repeat_evaluations=problem.repeated,duplicate_candidates_removed=int(p.n_duplicate_removed.sum()),
        drop_fraction=counters['cumulative_dropped']/counters['cumulative_proposed'],
        final_pareto_size=len(pareto),unique_archive=len(raw),completed_generations=len(p),
        mating_batches=algorithm.mating_batches,run_seconds=run_seconds,**times,
        posthoc_classification_seconds=posthoc_seconds,
        final_classifications=classified.flag.value_counts().astype(int).to_dict())
    if not (summary['actual_model_calls']==summary['total_accepted']==summary['evaluator_evaluations'] and
            summary['total_proposed']==summary['total_accepted']+summary['total_dropped'] and
            summary['total_missing_evaluations']==summary['saved_evaluations']+summary['non_alsf_shortfall'] and
            len(p)==generations and algorithm.mating_batches==generations-1):
        raise RuntimeError('Drop accounting/lifecycle invariant failed')
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generations',type=int,default=200);p.add_argument('--pop-size',type=int,default=100)
    p.add_argument('--offsprings',type=int,default=100);p.add_argument('--flag-start',type=int,default=50)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--tag',default='full1')
    p.add_argument('--outdir',type=Path);p.add_argument('--reference',type=Path)
    args=p.parse_args(argv)
    if Path(args.tag).name!=args.tag:p.error('--tag must be a simple directory-name suffix')
    target=args.outdir or ROOT/'results'/f'di_barba_drop_seed{args.seed}_{args.tag}'
    ref=pd.read_csv(args.reference) if args.reference else None
    run(target,args.generations,args.pop_size,args.offsprings,args.flag_start,args.seed,ref)
if __name__=='__main__':main()
