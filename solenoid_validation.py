"""Phased, independent validation of the convex 10-D solenoid benchmark.

Subcommands never overwrite an existing output directory. No surrogate models.
Optimization, archive processing, indicators and exact audits have separate timers.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from di_barba_solenoid_benchmark import DiBarbaSolenoid
from src.find_robust_clusters import classify_dataframe
from src.local_variation import LocalVariationState

XCOLS=[f'X{i+1}' for i in range(10)]
COLS=XCOLS+['Obj1','Obj2']


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


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


def reference_phase(outdir):
    outdir.mkdir(parents=True,exist_ok=False)
    model=DiBarbaSolenoid();start=perf_counter()
    coarse=model.reference_pareto(241,adaptive_gap=.01,max_points=1200)
    model.reference_diagnostics.to_csv(outdir/'solver_coarse.csv',index=False)
    fine=model.reference_pareto(481,adaptive_gap=.005,max_points=1800)
    model.reference_diagnostics.to_csv(outdir/'solver_fine.csv',index=False)
    coarse.to_csv(outdir/'reference_coarse.csv',index=False)
    fine.to_csv(outdir/'reference.csv',index=False)
    # Assess geometric convergence in a common normalization from the fine set.
    coarse_on_fine=pd.DataFrame(coarse)
    distance=polyline_distances(coarse[['Obj1','Obj2']].to_numpy(),fine)
    reverse=polyline_distances(fine[['Obj1','Obj2']].to_numpy(),coarse_on_fine)
    rng=np.random.default_rng(913);gradient_error=0.
    for x in rng.uniform(-1.8,1.8,(5,10)):
        for i in range(10):
            step=np.eye(10)[i]*1e-6
            fd=(np.array(model.evaluate(x+step))-model.evaluate(x-step))/2e-6
            gradient_error=max(gradient_error,float(np.max(abs(fd-model.gradient(x)[:,i]))))
    report=dict(coarse_points=len(coarse),fine_points=len(fine),
        matrix_condition_number=float(np.linalg.cond(model.K)),
        hessian_min_eigenvalues=[float(np.linalg.eigvalsh(h).min()) for h in model.hessians()],
        gradient_max_absolute_error=gradient_error,
        max_kkt_residual=float(model.reference_diagnostics.kkt_residual.max()),
        max_bound_violation=float(model.reference_diagnostics.bound_violation.max()),
        all_solvers_successful=bool(model.reference_diagnostics.solver_success.all()),
        coarse_to_fine_max_distance=float(distance.max()),fine_to_coarse_max_distance=float(reverse.max()),
        reference_hv_coarse=indicators(coarse[['Obj1','Obj2']],fine)['hypervolume'],
        reference_hv_fine=indicators(fine[['Obj1','Obj2']],fine)['hypervolume'],
        seconds=perf_counter()-start)
    report['validated']=bool(report['all_solvers_successful'] and
        min(report['hessian_min_eigenvalues'])>-1e-10 and gradient_error<1e-7 and
        report['max_kkt_residual']<1e-7 and report['max_bound_violation']<1e-12 and
        max(distance.max(),reverse.max())<1e-3)
    write_json(outdir/'validation.json',report)
    if not report['validated']:raise RuntimeError('Reference validation gate failed; inspect validation.json')
    return fine,report


def audit_points(frame,model=None):
    """Independent exact extrema; preserve empirical labels and row IDs."""
    model=model or DiBarbaSolenoid();rows=[]
    for idx,row in frame.iterrows():
        x=row[XCOLS].to_numpy(float)
        r=model.exact_relative_box_sensitivity(x)
        record=dict(archive_row=int(idx),flag=str(row.get('flag','unclassified')),
                    Obj1=float(row.Obj1),Obj2=float(row.Obj2),R1_exact=float(r[0]),R2_exact=float(r[1]),
                    exact_sensitive=bool(np.any(r>.1)))
        rows.append(record)
    return pd.DataFrame(rows,columns=['archive_row','flag','Obj1','Obj2','R1_exact','R2_exact','exact_sensitive'])


def classification_metrics(audit):
    stable=audit.flag.eq('robust');sensitive=audit.flag.eq('non-robust');nei=audit.flag.eq('NEI')
    actual=audit.exact_sensitive.astype(bool)
    false_stable=int((stable&actual).sum());tp=int((sensitive&actual).sum())
    fp=int((sensitive&~actual).sum());fn=int((stable&actual).sum())
    # NEI abstentions stay separate; report conditional recall and coverage.
    return dict(n_audited=len(audit),n_empirical_stable=int(stable.sum()),
        n_verified_sensitive=int(sensitive.sum()),n_nei=int(nei.sum()),
        n_exact_sensitive=int(actual.sum()),n_false_stable=false_stable,
        false_stable_rate=false_stable/int(stable.sum()) if stable.any() else None,
        sensitive_precision=tp/(tp+fp) if tp+fp else None,
        sensitive_recall_classified=tp/(tp+fn) if tp+fn else None,
        sensitive_detection_all=tp/int(actual.sum()) if actual.any() else None,
        classification_coverage=float((stable|sensitive).mean()) if len(audit) else None)


def sensitivity_phase(reference,outdir):
    outdir.mkdir(parents=True,exist_ok=False)
    model=DiBarbaSolenoid();start=perf_counter()
    idx=np.unique(np.linspace(0,len(reference)-1,61,dtype=int))
    reference_audit=audit_points(reference.iloc[idx],model)
    reference_audit.to_csv(outdir/'reference_sensitivity.csv',index=False)
    # A controlled small archive checks classification semantics independently.
    rng=np.random.default_rng(713)
    x=np.vstack([rng.uniform(-2,2,(40,10)),rng.uniform(-.001,.001,(12,10)),.5+rng.uniform(-.001,.001,(12,10)),
                 np.zeros((1,10)),np.full((1,10),.05)])
    f=np.array([model.evaluate(y) for y in x])
    data=pd.DataFrame(np.column_stack([x,f]),columns=COLS)
    flagged,_=classify_dataframe(data,.1,.1,.1,partition_width=None)
    audit=audit_points(flagged,model)
    audit.to_csv(outdir/'empirical_vs_exact.csv',index=False)
    flagged.to_csv(outdir/'sample_archive.csv',index=False)
    # Independent random perturbations must lie inside computed extrema/relative maxima.
    maximum_error=0.
    from src.local_variation import relative_pairwise_variation
    for center in [np.zeros(10),np.full(10,2.),rng.uniform(-1.9,1.9,10)]:
        perturb=np.clip(center+rng.uniform(-.1,.1,(1000,10)),model.lower,model.upper)
        values=np.array([model.evaluate(y) for y in perturb])
        low,high=model.box_objective_extrema(center)
        maximum_error=max(maximum_error,float(np.max(low-values)),float(np.max(values-high)))
        observed=relative_pairwise_variation(model.evaluate(center),values,[.001]*2).max(axis=0)
        maximum_error=max(maximum_error,float(np.max(observed-model.exact_relative_box_sensitivity(center))))
    report=dict(**classification_metrics(audit),reference_points_tested=len(idx),
        reference_exact_sensitive=int(reference_audit.exact_sensitive.sum()),
        maximum_random_bound_violation=maximum_error,seconds=perf_counter()-start,
        validated=maximum_error<1e-8)
    # A verified observed endpoint must be sensitive under exactly the same box.
    if ((audit.flag=='non-robust')&~audit.exact_sensitive).any():report['validated']=False
    write_json(outdir/'validation.json',report)
    if not report['validated']:raise RuntimeError('Exact sensitivity validation gate failed')
    return report


def run_optimization(reference,outdir,mode,seed=42,generations=200,pop_size=100,
                     offsprings=100,flag_start=50):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.callback import Callback
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from src.preprocessing import OffspringRepair
    outdir.mkdir(parents=True,exist_ok=False)
    settings=dict(mode=mode,seed=seed,generations=generations,pop_size=pop_size,
        offsprings=offsprings,flag_start=flag_start,sbx_prob=.9,sbx_eta=5,pm_prob=1.,pm_eta=5,
        tolerances=[.1]*10,local_radius=1.,local_min_samples=10,dbscan_eps=1.,
        dbscan_min_samples=10,exclusion_radius=1.,objective_thresholds=[.1]*2,
        objective_floors=[.001]*2,partition_enabled=False)
    write_json(outdir/'settings.json',settings)
    model=DiBarbaSolenoid();raw=[];keys=set();history=[];process=[]
    timings=dict(objectives=0.,archive_processing=0.,indicators=0.)
    class Problem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=10,n_obj=2,xl=model.lower,xu=model.upper)
            self.model_calls=0
        def _evaluate(self,x,out,*args,**kwargs):
            tick=perf_counter();x=np.round(np.asarray(x),8)
            values=model.evaluate(x);self.model_calls+=1
            timings['objectives']+=perf_counter()-tick
            key=tuple(x)
            if key not in keys:
                keys.add(key);raw.append([*x,*values])
            out['F']=values
    problem=Problem()
    # Baseline uses the same rounding/clipping hook with permanently empty policy.
    repair=OffspringRepair('in-memory',{},1.,tolerances=np.full(10,.1),
        rng=np.random.default_rng(seed),max_resample_attempts=100)
    state=LocalVariationState(np.full(10,.1))
    classified=None
    class Monitor(Callback):
        def notify(self,algorithm):
            nonlocal classified
            start=perf_counter();stats={};point={};available=False
            if mode=='alsf' and algorithm.n_gen>=flag_start:
                classified,stats=classify_dataframe(pd.DataFrame(raw,columns=COLS),.1,.1,.1,
                                                     state=state,partition_width=None)
                repair.update_flags(classified)
                counts=classified.flag.value_counts()
                point=dict(n_robust=int(counts.get('robust',0)),n_non_robust=int(counts.get('non-robust',0)),
                    n_nei=int(counts.get('NEI',0)),neighbor_median=float(classified.local_n.median()),
                    neighbor_p95=float(classified.local_n.quantile(.95)),neighbor_max=int(classified.local_n.max()),
                    neighbor_min=int(classified.local_n.min()))
                available=True
            archive_seconds=perf_counter()-start
            timings['archive_processing']+=archive_seconds
            tick=perf_counter()
            metrics=indicators(algorithm.opt.get('F'),reference)
            timings['indicators']+=perf_counter()-tick
            history.append(dict(generation=int(algorithm.n_gen),evaluations=int(algorithm.evaluator.n_eval),
                                unique_archive=len(raw),**metrics))
            if int(algorithm.n_gen) % 50 == 0:
                print(mode,seed,'generation',algorithm.n_gen,'archive',len(raw),
                      'verified',int(classified.verified_violation.sum()) if classified is not None else 0,flush=True)
            process.append(dict(generation=int(algorithm.n_gen),evaluations=int(algorithm.evaluator.n_eval),
                n_archive=len(raw),evidence_available=available,archive_seconds=archive_seconds,
                **stats,**point,**repair.pop_diagnostics()))
    algorithm=NSGA2(pop_size=pop_size,n_offsprings=offsprings,sampling=FloatRandomSampling(),
        crossover=SBX(prob=.9,eta=5),mutation=PM(prob=1.,eta=5),repair=repair,eliminate_duplicates=True)
    start=perf_counter()
    result=minimize(problem,algorithm,('n_gen',generations),seed=seed,callback=Monitor(),
                    copy_algorithm=False,save_history=False,verbose=False)
    run_seconds=perf_counter()-start
    # Independent final diagnostic for baseline; never changes its optimization.
    start=perf_counter()
    if classified is None:
        classified,final_stats=classify_dataframe(pd.DataFrame(raw,columns=COLS),.1,.1,.1,partition_width=None)
    else:
        final_stats={k:v for k,v in process[-1].items() if k.startswith('n_') or k=='largest_cluster'}
    diagnostic_seconds=perf_counter()-start
    classified.to_csv(outdir/'archive.csv',index=False)
    front=pd.DataFrame(np.column_stack([result.X,result.F]),columns=COLS)
    labels={tuple(row):flag for row,flag in zip(classified[XCOLS].to_numpy(),classified.flag)}
    front['flag']=[labels[tuple(np.round(x,8))] for x in result.X]
    front.to_csv(outdir/'pareto.csv',index=False)
    pd.DataFrame(history).to_csv(outdir/'history.csv',index=False)
    proc=pd.DataFrame(process);proc.to_csv(outdir/'process.csv',index=False)
    classified.local_n.value_counts().sort_index().rename_axis('local_n').rename('n_samples').to_csv(outdir/'neighbor_counts.csv')
    # Same objective-ranked representative subset rule in both modes; bounded cost.
    indices=np.unique(np.linspace(0,len(front)-1,min(25,len(front)),dtype=int))
    selected=front.sort_values(['Obj1','Obj2']).iloc[indices]
    tick=perf_counter();audit=audit_points(selected,model)
    audit.to_csv(outdir/'pareto_sensitivity.csv',index=False)
    exact_seconds=perf_counter()-tick
    distance=polyline_distances(front[['Obj1','Obj2']],reference)
    summary=dict(**settings,evaluations=int(result.algorithm.evaluator.n_eval),model_calls=problem.model_calls,
        unique_archive=len(classified),pareto_size=len(front),run_seconds=run_seconds,
        objective_seconds=timings['objectives'],archive_processing_seconds=timings['archive_processing'],
        indicator_seconds=timings['indicators'],posthoc_classification_seconds=diagnostic_seconds,
        exact_audit_seconds=exact_seconds,hypervolume=history[-1]["hypervolume"],
        igd_plus=history[-1]["igd_plus"],
        reference_distance_mean=float(distance.mean()),reference_distance_max=float(distance.max()),
        exact_sensitive_fraction=float(audit.exact_sensitive.mean()),
        exact_R1_median=float(audit.R1_exact.median()),exact_R2_median=float(audit.R2_exact.median()),
        final_classifications=classified.flag.value_counts().astype(int).to_dict(),
        final_neighbor_median=float(classified.local_n.median()),final_neighbor_p95=float(classified.local_n.quantile(.95)),
        final_neighbor_max=int(classified.local_n.max()),final_clusters=int(final_stats.get('n_clusters',0)),
        final_dbscan_noise=int(final_stats.get('n_noise',0)),
        total_regenerated=int(proc.n_regenerated.sum()),total_offspring_seen=int(proc.n_offspring_seen.sum()),
        total_attempts=int(proc.n_attempts.sum()),total_forbidden=int(proc.n_forbidden_candidates.sum()),
        classification_audit=classification_metrics(audit))
    summary['regenerated_fraction']=summary['total_regenerated']/summary['total_offspring_seen']
    if summary['model_calls']!=summary['evaluations']:raise RuntimeError('Unequal evaluator/model accounting')
    if summary['evaluations']!=pop_size+(generations-1)*offsprings:raise RuntimeError('Evaluation budget mismatch')
    write_json(outdir/'summary.json',summary)
    print(mode,seed,'evaluations',summary['evaluations'],'seconds',round(run_seconds,2),
          'HV',round(summary['hypervolume'],6),'IGD+',round(summary['igd_plus'],6),
          'regenerated',summary['total_regenerated'],flush=True)
    return summary


def archive_audit_phase(study,outdir,max_per_class=25):
    """Stratified exact audit; class-conditional rates, not population recall."""
    outdir.mkdir(parents=True,exist_ok=False)
    reports=[]
    for directory in sorted((study/'full').glob('*_seed*')):
        data=pd.read_csv(directory/'archive.csv');selected=[]
        for flag in ('robust','non-robust','NEI'):
            group=data.loc[data.flag==flag].sort_values(['Obj1','Obj2'])
            rank=np.unique(np.linspace(0,len(group)-1,min(max_per_class,len(group)),dtype=int))
            selected.extend(group.iloc[rank].index.tolist())
        start=perf_counter();audit=audit_points(data.loc[selected])
        audit.to_csv(outdir/(directory.name+'.csv'),index=False)
        metrics=classification_metrics(audit)
        reports.append(dict(run=directory.name,**metrics,seconds=perf_counter()-start))
    report=dict(selection='Up to 25 objective-ranked points per empirical class; stratified descriptive audit.',
                reports=reports)
    write_json(outdir/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['reference','sensitivity','run','audit'])
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--study',type=Path)
    parser.add_argument('--mode',choices=['baseline','alsf'],default='alsf')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--generations',type=int,default=200)
    parser.add_argument('--pop-size',type=int,default=100)
    parser.add_argument('--offsprings',type=int,default=100)
    parser.add_argument('--flag-start',type=int,default=50)
    args=parser.parse_args()
    if args.phase=='audit':
        if args.study is None:parser.error('--study is required')
        print(json.dumps(archive_audit_phase(args.study,args.outdir),indent=2))
    elif args.phase=='reference':
        _,report=reference_phase(args.outdir);print(json.dumps(report,indent=2))
    else:
        if args.reference is None:parser.error('--reference is required')
        reference=pd.read_csv(args.reference)
        if args.phase=='sensitivity':print(json.dumps(sensitivity_phase(reference,args.outdir),indent=2))
        else:run_optimization(reference,args.outdir,args.mode,args.seed,args.generations,
                             args.pop_size,args.offsprings,args.flag_start)

if __name__=='__main__':main()
