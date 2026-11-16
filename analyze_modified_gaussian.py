"""Analyze all 60 paired runs, verify accounting, export statistics and figures."""
from pathlib import Path
from itertools import combinations
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.integrate import quad
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_test1_modified_gaussian import evaluate,to_optimization
from src.modified_gaussian_reference import sensitivity,reference
from src.drop_metrics import indicators,normalization,arc_reference

ORDER=['baseline','resample','drop']
LABEL={'baseline':'NSGA-II','resample':'ALSF-Resample','drop':'ALSF-Drop'}
COLOR={'baseline':'#356B9A','resample':'#C17B3A','drop':'#237C69'}
METRICS=['hypervolume','igd_plus','actual_model_calls','saved_percent','mean_reference_distance',
    'narrow_coverage','narrow_mean_distance','sensitive_audit_fraction','R1_median','R2_median','run_seconds']


def paired_effect(a,b,draws=10000):
    difference=np.asarray(b)-np.asarray(a)
    rng=np.random.default_rng(20261116)
    boot=difference[rng.integers(0,len(difference),(draws,len(difference)))].mean(axis=1)
    p=1. if np.all(difference==0) else float(wilcoxon(difference,alternative='two-sided',method='auto').pvalue)
    return dict(mean_difference=float(difference.mean()),ci95_low=float(np.quantile(boot,.025)),
        ci95_high=float(np.quantile(boot,.975)),p_raw=p)


def holm(pvalues):
    pvalues=np.asarray(pvalues);order=np.argsort(pvalues);result=np.empty(len(order));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(order)-rank)*pvalues[index]));result[index]=previous
    return result


def table(frame):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6g}'
        return str(v)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def analyze(root):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text())
    if protocol['seeds']!=[*range(1,18),42,71,97] or protocol['generations']!=200 or protocol['pop_size']!=100:
        raise ValueError('Primary report requires exact 60-run protocol')
    ref=pd.read_csv(root/'reference/test1.csv');refopt=ref.copy();refopt[['Obj1','Obj2']]=to_optimization(ref[['Obj1','Obj2']])
    rows=[];histories={};fronts={};archives={};events={};evaluated={};audits=[];rejections=[];hits=[];labels=[];sensitivity_frames=[];verification=[]
    for seed in protocol['seeds']:
        for mode in ORDER:
            directory=root/'runs/test1'/mode/f'seed{seed}'
            r=json.loads((directory/'summary.json').read_text())
            h=pd.read_csv(directory/'history.csv');process=pd.read_csv(directory/'process.csv')
            f=pd.read_csv(directory/'pareto.csv');a=pd.read_csv(directory/'archive.csv')
            e=pd.read_csv(directory/'screening.csv.gz',dtype={'rejected':bool});evaluations=pd.read_csv(directory/'evaluated.csv.gz')
            histories[mode,seed]=h;fronts[mode,seed]=f;events[mode,seed]=e;archives[mode,seed]=a;evaluated[mode,seed]=evaluations
            assert len(h)==200 and h.generation.tolist()==list(range(1,201))
            assert len(evaluations)==r['actual_model_calls']==r['evaluator_calls']==int(h.evaluations.iloc[-1])
            assert r['seed']==seed and r['mode']==mode and r['generations']==200
            assert (process.n_population==100).all()
            np.testing.assert_allclose(evaluate(f[['X1','X2']]),f[['Obj1','Obj2']],atol=1e-10)
            calculated=indicators(to_optimization(f[['Obj1','Obj2']]),refopt)
            for key,value in calculated.items():assert abs(value-r[key])<1e-11
            if mode=='drop':
                assert int(e.rejected.sum())==int(process.n_dropped.sum())==20000-len(evaluations)
                assert len(e)==int(process.n_proposed.iloc[1:].sum())
            elif mode=='resample':
                assert len(evaluations)==20000
                assert int((e.stage=='replacement').sum())==int(process.n_attempts.sum())
                assert int((e.rejected & (e.stage=='original')).sum())==int(process.n_forbidden_candidates.sum())
            else:assert len(evaluations)==20000 and len(e)==0
            verification.append(dict(mode=mode,seed=seed,generations=len(h),evaluations=len(evaluations),recomputed_hv=calculated['hypervolume'],recomputed_igd_plus=calculated['igd_plus'],accounting_valid=True))
            row={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
            row['saved_percent']=100*(20000-r['actual_model_calls'])/20000
            row['classification_seconds']=r['timings']['classification'];row['objective_seconds']=r['timings']['objectives'];row['indicator_seconds']=r['timings']['indicators']
            for radius,counts in r['peak_counts'].items():
                for key,value in counts.items():row[f'r{radius}_{key}']=value
            rows.append(row)
            sens=pd.read_csv(directory/'sensitivity.csv');sens['mode']=mode;sens['seed']=seed
            sensitivity_frames.append(sens)
            peak_labels=pd.read_csv(directory/'peak_classification_history.csv');peak_labels['mode']=mode;peak_labels['seed']=seed;labels.append(peak_labels)
            for fraction,target in zip(protocol['target_fractions'],protocol['target_hv']):
                attained=h[h.hypervolume>=target]
                hits.append(dict(mode=mode,seed=seed,target_fraction=fraction,target_hv=target,reached=bool(len(attained)),
                    first_evaluations=int(attained.evaluations.iloc[0]) if len(attained) else np.nan,censoring_evaluations=r['actual_model_calls']))
            for generation in range(1,201):
                batch=e[e.generation==generation]
                rejections.append(dict(mode=mode,seed=seed,generation=generation,checked=len(batch),rejected=int(batch.rejected.sum()),
                    original_rejected=int((batch.rejected & (batch.stage=='original')).sum()),replacement_rejected=int((batch.rejected & (batch.stage=='replacement')).sum())))
            # Predefined systematic selection across observed order, never select by sensitivity.
            for kind,source in [('evaluated',a),('rejected',e[e.rejected])]:
                near=source[np.linalg.norm(source[['X1','X2']].to_numpy()-2.5,axis=1)<=.18]
                chosen=near.iloc[np.unique(np.linspace(0,len(near)-1,min(31,len(near)),dtype=int))] if len(near) else near
                for idx,p in chosen.iterrows():
                    absolute,relative=sensitivity(p[['X1','X2']].to_numpy(float))
                    audits.append(dict(mode=mode,seed=seed,kind=kind,source_row=int(idx),X1=p.X1,X2=p.X2,
                        flag=p.get('flag','unevaluated'),S1=absolute[0],S2=absolute[1],R1=relative[0],R2=relative[1],sensitive=bool((relative>.1).any())))
    frame=pd.DataFrame(rows).sort_values(['seed','mode']);frame.to_csv(out/'all_runs.csv',index=False)
    pd.DataFrame(verification).to_csv(out/'verification.csv',index=False)
    pd.DataFrame(hits).to_csv(out/'target_hv_attainment.csv',index=False)
    pd.DataFrame(rejections).to_csv(out/'rejections_by_generation.csv',index=False)
    label_frame=pd.concat(labels,ignore_index=True);label_frame.to_csv(out/'peak_classification_history.csv',index=False)
    audit=pd.DataFrame(audits,columns=['mode','seed','kind','source_row','X1','X2','flag','S1','S2','R1','R2','sensitive']);audit.to_csv(out/'peak_sensitivity_audit.csv',index=False)
    all_sens=pd.concat(sensitivity_frames,ignore_index=True);all_sens.to_csv(out/'final_pareto_sensitivity.csv',index=False)
    descriptive=[];effects=[]
    for metric in METRICS:
        for mode in ORDER:
            values=frame.loc[frame['mode']==mode,metric].to_numpy()
            descriptive.append(dict(mode=mode,metric=metric,n=len(values),mean=float(values.mean()),median=float(np.median(values)),
                std=float(values.std(ddof=1)),q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),iqr=float(np.quantile(values,.75)-np.quantile(values,.25))))
        pivot=frame.pivot(index='seed',columns='mode',values=metric).reindex(protocol['seeds'])
        for a,b in combinations(ORDER,2):effects.append(dict(metric=metric,method_a=a,method_b=b,n_pairs=20,**paired_effect(pivot[a],pivot[b])))
    descriptive=pd.DataFrame(descriptive);descriptive.to_csv(out/'descriptive.csv',index=False)
    effects=pd.DataFrame(effects);effects['p_holm']=holm(effects.p_raw);effects.to_csv(out/'paired_effects.csv',index=False)
    shared=int(frame.actual_model_calls.min());grid=np.unique(np.r_[np.arange(100,shared+1,100),shared]);checkpoints=[]
    for (mode,seed),h in histories.items():
        for budget in grid:
            row=h[h.evaluations<=budget].iloc[-1]
            checkpoints.append(dict(mode=mode,seed=seed,budget=budget,actual_evaluations=int(row.evaluations),
                unused_budget=budget-int(row.evaluations),hypervolume=row.hypervolume,igd_plus=row.igd_plus))
    checkpoints=pd.DataFrame(checkpoints);checkpoints.to_csv(out/'common_evaluation_budgets.csv',index=False)
    # Analytic HV integral is an independent limit check; locked targets stay fixed.
    end=float(ref.X1.iloc[-1]);qend=float(ref.Obj1.iloc[-1]);f2end=float(ref.Obj2.iloc[-1])
    integral,error=quad(lambda t:4*t*(evaluate([t,t])[1]+50)/(qend*(f2end+50)),
        0,end,points=[2.2,2.4,2.48],epsabs=1e-12,epsrel=1e-12)
    limit_hv=.21+integral
    (out/'reference_accuracy.json').write_text(json.dumps(dict(quadrature_limit_hv=limit_hv,
        quadrature_error_estimate=error,locked_reference_hv=protocol['reference_hv'],
        reference_hv_discretization_gap=limit_hv-protocol['reference_hv'],
        target_policy='Original discrete-reference targets remain locked; quadrature is a posthoc accuracy diagnostic.'),indent=2)+'\n')
    # Recompute final indicators with a refined independent front as accuracy check.
    fine=pd.read_csv(root/'reference/refined.csv');fine[['Obj1','Obj2']]=to_optimization(fine[['Obj1','Obj2']]);resolution=[]
    for (mode,seed),front in fronts.items():
        metrics=indicators(to_optimization(front[['Obj1','Obj2']]),fine)
        old=frame[(frame['mode']==mode)&(frame.seed==seed)].iloc[0]
        resolution.append(dict(mode=mode,seed=seed,hv_refined=metrics['hypervolume'],igd_refined=metrics['igd_plus'],hv_delta=metrics['hypervolume']-old.hypervolume,igd_delta=metrics['igd_plus']-old.igd_plus))
    pd.DataFrame(resolution).to_csv(out/'indicator_reference_resolution.csv',index=False)
    make_figures(root,ref,frame,fronts,archives,events,evaluated,all_sens,checkpoints,pd.DataFrame(rejections),audit)
    write_report(root,protocol,frame,descriptive,effects,pd.DataFrame(hits),audit,shared,pd.DataFrame(resolution))
    return frame


def make_figures(root,ref,frame,fronts,archives,events,evaluated,all_sens,checkpoints,rejections,audit):
    out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':320,'pdf.fonttype':42})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=320,bbox_inches='tight')
        plt.close(fig)
    fig=plt.figure(figsize=(10,3.4))
    for i,(bounds,title) in enumerate([((-5,10),'Full feasible domain'),((2.25,2.75),'Gaussian peak detail')]):
        grid=np.unique(np.r_[np.linspace(*bounds,121),np.linspace(2.35,2.65,121)])
        x,y=np.meshgrid(grid,grid);f=evaluate(np.stack((x,y),axis=-1))
        for j in range(2):
            ax=fig.add_subplot(1,4,2*i+j+1,projection='3d');ax.plot_surface(x,y,f[:,:,j],cmap='viridis',rasterized=True,linewidth=0,rcount=100,ccount=100)
            ax.set(xlabel='$x_1$',ylabel='$x_2$',zlabel=f'$f_{j+1}$',title=f'{title}: $f_{j+1}$');ax.tick_params(labelsize=6)
    save(fig,'01_objective_surfaces')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    axes[0].plot(ref.Obj1,ref.Obj2,c='black');axes[0].set(xlabel='$f_1$ (min)',ylabel='$f_2$ (max)')
    axes[1].plot(ref.X1,ref.X2,c='black');axes[1].scatter(ref.X1.iloc[-1],ref.X2.iloc[-1],c='red',s=15);axes[1].set(xlabel='$x_1$',ylabel='$x_2$')
    save(fig,'02_independent_reference')
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for ax,mode in zip(axes,ORDER):
        for (m,seed),f in fronts.items():
            if m==mode:ax.scatter(f.Obj1,f.Obj2,s=2,alpha=.16,c=COLOR[mode],rasterized=True)
        ax.plot(ref.Obj1,ref.Obj2,c='black',lw=1,label='Reference');ax.set(title=LABEL[mode]+' (all 20 seeds)',xlabel='$f_1$',ylabel='$f_2$');ax.legend(fontsize=7)
    save(fig,'03_final_pareto_fronts')
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for ax,mode in zip(axes,ORDER):
        a=archives[mode,42]
        for flag,c in [('NEI','#AAAAAA'),('robust','#237C69'),('non-robust','#B04444')]:
            points=a[a.flag==flag];ax.scatter(points.X1,points.X2,s=2,c=c,label=flag,rasterized=True)
        ax.set(title=LABEL[mode]+'; seed 42',xlabel='$x_1$',ylabel='$x_2$',xlim=(-5,10),ylim=(-5,10));ax.legend(fontsize=6)
    save(fig,'04_archive_classifications_seed42')
    for number,verified in [('05',False),('06',True)]:
        fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
        for ax,mode in zip(axes,ORDER):
            for (m,seed),source in evaluated.items():
                if m!=mode:continue
                near=source[np.linalg.norm(source[['X1','X2']].to_numpy()-2.5,axis=1)<=.18]
                ax.scatter(near.X1,near.X2,s=2,c='#AAAAAA',alpha=.25,rasterized=True)
                rejected=events[m,seed];rejected=rejected[rejected.rejected]
                near=rejected[np.linalg.norm(rejected[['X1','X2']].to_numpy()-2.5,axis=1)<=.18]
                ax.scatter(near.X1,near.X2,s=8,c='#B04444',marker='x',alpha=.3,rasterized=True)
                f=fronts[m,seed];ax.scatter(f.X1,f.X2,s=4,c=COLOR[m],alpha=.25,rasterized=True)
            if verified:
                points=audit[(audit['mode']==mode)&audit.sensitive];ax.scatter(points.X1,points.X2,marker='o',s=16,facecolors='none',edgecolors='#83348B',lw=.5,label='Independent sensitive witness')
            ax.plot(ref.X1,ref.X2,c='black',lw=1,label='True Pareto set')
            for radius in [.03,.10,.18]:ax.add_patch(plt.Circle((2.5,2.5),radius,fill=False,ec='.5',lw=.5,ls=':'))
            ax.set(title=LABEL[mode],xlabel='$x_1$',ylabel='$x_2$',xlim=(2.3,2.7),ylim=(2.3,2.7));ax.set_aspect('equal')
        axes[0].scatter([],[],s=5,c='.6',label='Evaluated (all seeds)');axes[0].scatter([],[],s=10,c='#B04444',marker='x',label='Actual rejection');axes[0].scatter([],[],s=6,c=COLOR['baseline'],label='Final approximate front');axes[0].legend(fontsize=5.5)
        save(fig,number+('_verified_sensitive_and_rejected' if verified else '_peak_zoom'))
    for number,metric,ylabel in [('07','hypervolume','Hypervolume'),('08','igd_plus','IGD+')]:
        fig,ax=plt.subplots(figsize=(6.5,3.2),layout='constrained')
        for mode in ORDER:
            group=checkpoints[checkpoints['mode']==mode].groupby('budget')[metric]
            med=group.median();ax.plot(med.index,med,c=COLOR[mode],label=LABEL[mode]);ax.fill_between(med.index,group.quantile(.25),group.quantile(.75),color=COLOR[mode],alpha=.18)
        ax.set(xlabel='Allowed objective-evaluation budget (completed generations)',ylabel=ylabel);ax.legend(fontsize=7)
        if metric=='igd_plus':ax.set_yscale('log')
        save(fig,number+'_'+metric+'_vs_evaluations')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    for ax,metric in zip(axes,['hypervolume','igd_plus']):
        data=[frame.loc[frame['mode']==m,metric] for m in ORDER];ax.boxplot(data,tick_labels=[LABEL[m] for m in ORDER],showfliers=False)
        for i,(mode,values) in enumerate(zip(ORDER,data)):ax.scatter(i+1+np.linspace(-.12,.12,20),values,c=COLOR[mode],s=8)
        ax.set_ylabel(metric);ax.tick_params(axis='x',rotation=15,labelsize=7)
    save(fig,'09_final_indicator_distributions')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    for mode in ['resample','drop']:
        for ax,metric in zip(axes,['original_rejected','replacement_rejected']):
            group=rejections[rejections['mode']==mode].groupby('generation')[metric]
            med=group.median();ax.plot(med.index,med,c=COLOR[mode],label=LABEL[mode]);ax.fill_between(med.index,group.quantile(.25),group.quantile(.75),color=COLOR[mode],alpha=.18)
            ax.set(xlabel='Generation',ylabel=metric.replace('_',' '))
    axes[0].legend(fontsize=7);save(fig,'10_rejected_offspring_by_generation')
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for ax,metric in zip(axes,['sensitive_audit_fraction','R1_median','R2_median']):
        data=[frame.loc[frame['mode']==m,metric] for m in ORDER];ax.boxplot(data,tick_labels=[LABEL[m] for m in ORDER],showfliers=False)
        for i,(mode,values) in enumerate(zip(ORDER,data)):ax.scatter(i+1+np.linspace(-.12,.12,20),values,c=COLOR[mode],s=8)
        ax.set_ylabel(metric.replace('_',' '));ax.tick_params(axis='x',rotation=15,labelsize=7)
    save(fig,'11_final_pareto_sensitivity')


def write_report(root,protocol,frame,desc,effects,hits,audit,shared,resolution):
    validation=json.loads((root/'reference/validation.json').read_text())
    accuracy=json.loads((root/'analysis/reference_accuracy.json').read_text())
    agg=frame.groupby('mode').agg(n=('seed','count'),mean_HV=('hypervolume','mean'),mean_IGD_plus=('igd_plus','mean'),mean_evaluations=('actual_model_calls','mean'),mean_saved_percent=('saved_percent','mean'),mean_seconds=('run_seconds','mean'),mean_sensitive_fraction=('sensitive_audit_fraction','mean'),mean_narrow_coverage=('narrow_coverage','mean')).reindex(ORDER)
    discovery=[]
    for mode in ORDER:
        group=frame[frame['mode']==mode]
        for radius in [.03,.10,.18]:
            tag=str(radius);discovery.append(dict(mode=mode,radius=radius,runs_evaluated=int((group[f'r{tag}_evaluated']>0).sum()),evaluated=int(group[f'r{tag}_evaluated'].sum()),runs_retained=int((group[f'r{tag}_final_pareto']>0).sum()),final_pareto=int(group[f'r{tag}_final_pareto'].sum()),checked=int(group[f'r{tag}_checked'].sum()),rejected=int(group[f'r{tag}_rejected'].sum()),accepted=int(group[f'r{tag}_accepted'].sum()),original_rejected=int(group[f'r{tag}_original_rejected'].sum()),replacement_rejected=int(group[f'r{tag}_replacement_rejected'].sum()),rejected_outside=int(group[f'r{tag}_rejected_outside'].sum())))
    discovery=pd.DataFrame(discovery);discovery.to_csv(root/'analysis/peak_discovery_and_rejection.csv',index=False)
    audit_table=audit.groupby(['mode','kind','flag'],dropna=False).agg(n=('sensitive','count'),independently_sensitive_fraction=('sensitive','mean')).reset_index() if len(audit) else pd.DataFrame()
    audit_table.to_csv(root/'analysis/archive_vs_independent_sensitivity.csv',index=False)
    attained=hits.groupby(['mode','target_fraction']).agg(reached=('reached','sum'),n=('seed','count')).reset_index()
    time_table=frame.groupby('mode')[['objective_seconds','classification_seconds','indicator_seconds','run_seconds','posthoc_seconds']].mean().reindex(ORDER).reset_index()
    baseline=agg.loc['baseline'];drop=agg.loc['drop']
    tight=discovery[discovery.radius==.03].set_index('mode')
    outcome=(f"NSGA-II evaluated and retained radius-.03 peak designs in {int(tight.loc['baseline','runs_evaluated'])}/20 runs. "
        f"ALSF-Resample and ALSF-Drop did so in {int(tight.loc['resample','runs_evaluated'])}/20 and {int(tight.loc['drop','runs_evaluated'])}/20 runs. "
        f"Drop recorded {int(tight.loc['drop','rejected'])} actual radius-.03 pre-evaluation rejections. "
        f"It saved {drop.mean_saved_percent:.2f}% of optimization evaluations on average, while mean HV fell from {baseline.mean_HV:.6f} to {drop.mean_HV:.6f} "
        f"and mean narrow-segment coverage fell from {baseline.mean_narrow_coverage:.3f} to {drop.mean_narrow_coverage:.3f}. "
        f"The independently sensitive final-solution fraction fell from {baseline.mean_sensitive_fraction:.4f} to {drop.mean_sensitive_fraction:.4f}, "
        "but remained high. The experiment therefore demonstrates discovery and actual sensitivity-driven rejection, together with a measurable loss of nominal quality; it does not establish an acceptable application tradeoff or universal robustness.")
    text=f'''# Modified Gaussian benchmark: independent reference and paired 60-run study

## Problem and scope

Physical objectives: min f1=x1²+x2²; max f2=−[(x1−5)²+(x2−5)²]+100 exp(−[(x1−2.5)²+(x2−2.5)²]/0.01), with x∈[−5,10]².
All optimizer dominance, HV, IGD+ and distance calculations use g=(f1,−f2). Physical objectives are exported as Obj1,Obj2. Sensitivity uses physical values and is invariant under the sign transform. This is a separate experiment; previous implementations, results and the manuscript are untouched.

## Independent Pareto reference and complete domain

At fixed q=x1²+x2², put s=x1+x2. Then f2=−q+10s−50+100 exp(−[q−5s+12.5]/0.01), whose derivative with respect to s is strictly positive. For 0≤q≤50, s≤√(2q), attained at x1=x2=√(q/2)∈[0,5], inside the feasible bounds. Thus every off-diagonal point in this range is dominated at the same f1 by its diagonal counterpart.
For q>50, distance to the Gaussian center is at least √q−√12.5. The negative quadratic is nonpositive, so f2≤100 exp(−[√q−√12.5]²/0.01)<10^−540. Such points have f1>50 and f2 below the peak endpoint and are dominated. This covers the entire square, including negative coordinates and q up to 200.
On the diagonal, f1=2t² and f2=−2(t−5)²+100 exp(−2(t−2.5)²/0.01). Its derivative is positive on [0,2.5]. Complete stationary-root enumeration finds t={validation['stationary_t']}; checking these and both endpoints establishes the global maximum at **t*={validation['endpoint_t']:.14f}**, f1={validation['endpoint_objectives'][0]:.12f}, f2={validation['endpoint_objectives'][1]:.12f}. The derivative stays positive up to t*, and all later diagonal points are dominated by t*. The true Pareto set is exactly (t,t), 0≤t≤t*.

The independent adaptive reference has {validation['reference_rows']} rows and its refinement has {validation['refined_rows']}. Reference HV changes by {validation['hv_resolution_difference']:.8g}; final-run HV changes by at most {resolution.hv_delta.abs().max():.8g}, IGD+ by at most {resolution.igd_delta.abs().max():.8g}. Stationary residual is {validation['endpoint_derivative_residual']:.3g}. An independent one-dimensional quadrature of the continuous-front dominated area gives limiting HV {accuracy['quadrature_limit_hv']:.12f} (quadrature error estimate {accuracy['quadrature_error_estimate']:.3g}), a gap of {accuracy['reference_hv_discretization_gap']:.8g} above the locked discrete reference HV. The original target values remain fixed; they are fractions of that discrete reference, not of an exact continuous HV. Numerical cross-checks use 250,000 uniform full-domain points and five global differential-evolution searches explicitly initialized at the peak. These supplement the proof; stochastic searches alone do not certify the maximum. Root solves use float64, not interval arithmetic. Reference coordinates, objective values, all stationary roots and validation are in `reference/`.

## Locked protocol and methods

Seeds: {protocol['seeds']}. Exactly 20 paired seeds per algorithm, 60 runs, 200 generations each, population=offspring count=100. Filtering evidence is first updated at generation 50 and affects generation 51 onward, matching the inherited harness. SBX probability .9, eta 5; PM probability 1, eta 5. Decision tolerance (.1,.1), relative thresholds (.1,.1), fixed objective floors (.001,.001), at least 10 local observations including self for the archive stable label. Eight-decimal coordinate rounding and duplicate elimination match the earlier experiment. No objective caching.
The existing LocalVariationState, OffspringRepair, OffspringDropFilter and DropNSGA2 are reused without modification. Neighborhood and exclusion distances use tolerance-normalized **L∞**, as implemented on the source branch. One observed violating pair establishes empirical sensitivity of both endpoints. Resample retains the inherited uniform replacement procedure with up to 1000 attempts; Drop discards the actual post-mating proposals before evaluation, without replacement. Initial population is unfiltered. An empty accepted batch still advances the generation. The benchmark parameters and methods were not tuned after outcomes.

Normalization: ideal={protocol['normalization_ideal']}, span={protocol['normalization_span']}, HV point=(1.1,1.1). IGD+ uses 501 equally spaced points along normalized objective arclength of the independent reference, avoiding a Gaussian-knot-density weighting. Nominal front error is mean shortest normalized Euclidean distance from final front to the reference polyline. The narrow segment is fixed as diagonal t∈[2.4,t*]; coverage is the proportion of 501 arc-uniform reference points within normalized Euclidean distance .01 of a final solution.

Every objective call, including duplicates, is stored in evaluated.csv.gz; archive.csv contains unique rounded designs and empirical classifications. Every screening check is stored in screening.csv.gz, including outside the peak; stage distinguishes original inputs from repeated replacement attempts. Resample repair checks may include proposals later removed as duplicates; they must not be interpreted as scheduled objective calls saved. Drop events correspond to the actual duplicate-eliminated proposal batch. Actual savings are 20000 minus actual optimization evaluations, verified against dropped batches. Independent posthoc sensitivity calls are outside the optimizer budget.

## All 60 results

{table(frame[['mode','seed','actual_model_calls','hypervolume','igd_plus','saved_percent','narrow_coverage','sensitive_audit_fraction','closest_evaluated_endpoint','closest_final_endpoint','run_seconds']])}

Algorithm aggregates (equal weight per seed):

{table(agg.reset_index())}

## Peak discovery, retained solutions and actual rejection

Euclidean discovery radii are fixed at .03, .10 and .18 around (2.5,2.5). General-neighborhood discovery is not accurate approximation: consult closest evaluated/final endpoint distance and narrow-segment coverage per run above. Final Pareto counts are optimizer nondominated approximations, not a claim that off-diagonal designs are mathematically Pareto-optimal.

{table(discovery)}

Rejected proposals outside each radius are also shown; these radii overlap and their totals must not be added. Peak rejection events, original inputs and replacement checks remain separate. Evidence for peak filtering is limited to the actual rejection counts, and any reduction in narrow coverage represents a loss of nominally efficient peak-region solutions. See `analysis/peak_classification_history.csv` for robust/stable, sensitive and NEI counts at each radius and generation from activation onward. Observed stability means sufficient sampled neighbors without a violation; NEI means insufficient evidence. Neither establishes worst-case robustness.

## Independently assessed sensitivity

For every final nondominated solution, intersect [x−.1,x+.1] with [−5,10]² and compute both objective minima/maxima. f1 is separable convex: its minimum is at the coordinatewise closest-to-zero point and maximum at a corner. For f2 enumerate all corners, edge stationary points and interior stationary points. Interior stationarity implies x1=x2=2.5+z, 0<z<2.5. On each edge the free-coordinate derivative has the form h(z)=5−2z−200/0.01 z exp(−[kz²+d²]/0.01), with k=1 for edges and k=2 on the diagonal. Its derivative's root equation is log(2u−1)−u=log(0.01/100)+d²/0.01, u=kz²/0.01. This function increases up to u=1.5 then decreases, so all critical points can be bracketed. Splitting h at them enumerates all roots; negative z and z>2.5 have no roots. The finite list is mathematically complete; numeric roots use double precision, without interval-certified rounding bounds.
Absolute S_j=max(|fmin−f(x)|,|fmax−f(x)|) is saved alongside the inherited symmetric relative deviation |a−b|/max(|a|,|b|,.001). Relative maxima additionally consider b=−a and floor breakpoints inside the attained interval; checking just the absolute extrema is insufficient when f2 changes sign. Sensitive means either maximal relative deviation exceeds .1. All final fronts are assessed, rather than a favorable subsample.
For archive/proposal comparison, systematically select at most 31 near-peak evaluated archive points and 31 near-peak rejected events per run by observed order, without selecting by sensitivity. These are diagnostic samples and cannot represent all rejected/evaluated designs. Rejected coordinates are assessed only posthoc and were never evaluated by the optimization. The archive comparison is:

{table(audit_table)}

Independent full-box sensitivity is separate from archive-supported flags. Improved nominal metrics do not imply robustness, and removing sensitive nominal optima can reduce HV or narrow coverage. The median relative sensitivity and actual independently sensitive fractions are included in the statistical tables.

## Paired statistics

Means, medians, sample SD, quartiles and IQR for all metrics:

{table(desc)}

Paired differences are B minus A across the same 20 seeds. Confidence intervals are percentile 95% paired-bootstrap intervals from 10,000 fixed-seed resamples of seed pairs. Two-sided Wilcoxon signed-rank tests use SciPy's automatic method, retaining ties/zeros; Holm adjustment is across all {len(effects)} metric/comparison tests in this report. Bootstrap intervals are pointwise, not simultaneous. Failure to reject does not establish equivalence.

{table(effects)}

## Actual evaluation budgets, targets and overhead

`analysis/common_evaluation_budgets.csv` compares all 20 seeds at common budgets up to {shared} evaluations. Only the latest completed generation at or below each budget is used (step hold); unused budget is recorded. This does not claim interpolation or exact partial-generation trajectories. Figures show seed medians and IQR, retaining the full cohort over the shared domain. Fixed-generation final results are separately reported above.
Reference-derived HV targets (99%, 99.5%, 99.9%) were locked before optimization. Unattained targets are right-censored at each run's actual evaluation count; they are never replaced with a successful attainment time. Attainment counts:

{table(attained)}

Mean timing in seconds:

{table(time_table)}

Wall time includes instrumented optimization, archive classification from activation for all three algorithms, indicators, proposal logging and operators. Posthoc seconds include final archive classification and independent sensitivity/export work. Two worker processes share this machine; timings characterize this instrumented experiment, not isolated production speed. Fewer objective evaluations alone do not demonstrate a wall-clock advantage for this inexpensive analytical model.

## Figures and retained data

`figures/01` through `11` provide objective surfaces (full domain and peak detail), independent reference, all-seed final fronts, seed-42 archive classifications, all-seed peak zoom, independent sensitive points with actual rejected events, HV/evaluations, IGD+/evaluations, final indicator distributions, rejection histories and final-front sensitivity. Both PNG (320 dpi) and vector PDF with rasterized dense point clouds are supplied. Seed 42 is a predefined paired seed, used only for the full-domain classification illustration; metrics and the other comparisons use all 20 seeds.
Each run retains history.csv, process.csv, archive.csv, pareto.csv, final_population.csv, sensitivity.csv, evaluated.csv.gz, screening.csv.gz, front_history.npz, peak_classification_history.csv, settings.json and summary.json. `analysis/verification.csv` verifies all 60 counts, 200-generation completion and recomputed indicators. Software versions, source hashes and source base commit are recorded in environment.json.

## Scientific limits and conclusions

{outcome}

Discovery, sufficient archive evidence and actual pre-evaluation rejection are different questions. The tables quantify all three and permit negative findings. Neighborhood exploration alone is insufficient evidence of accurate peak approximation; archive robust labels are not certified robustness. This study supports only the fixed w=.01 problem and protocol. No Gaussian-width tuning or second-stage experiment was performed. A separate predeclared w∈{{.01,.04,.10}} study could investigate spatial extent, but must retain its own output and paired protocol. Nominal-quality losses and independent sensitivity fractions must be considered together when judging whether evaluation savings are acceptable; no application-specific acceptable-loss threshold was predefined, so acceptability cannot be claimed.

## Reproduction

From this branch, install requirements.txt (the exact tested package versions are in environment.json). Use a fresh directory; preparation refuses to overwrite a locked protocol:

```sh
python prepare_modified_gaussian.py --root results/test1_modified_gaussian_reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_test1_modified_gaussian.py --root results/test1_modified_gaussian_reproduction --workers 2
python analyze_modified_gaussian.py --root results/test1_modified_gaussian_reproduction
python -m pytest tests/test_modified_gaussian.py tests/test_offspring_drop.py tests/test_local_filter.py
```

Completed runs are resumed only when requested settings match. A partially written run directory is deliberately not silently overwritten; move that incomplete directory aside and rerun the identical protocol. Do not remove a completed unsuccessful run. The JCAM manuscript is unchanged; suggested_subsection.tex is a separate conservative insertion proposal.
'''
    (root/'REPORT.md').write_text(text)
    b=agg.loc['baseline'];r=agg.loc['resample'];d=agg.loc['drop']
    total_peak=int(discovery.loc[(discovery['mode']=='drop')&(discovery.radius==.03),'rejected'].iloc[0])
    subsection=rf'''% Suggested insertion only; the manuscript has not been edited.
\subsection{{Modified Gaussian benchmark: nominal quality and local sensitivity}}
We studied $\min f_1=x_1^2+x_2^2$ and $\max f_2=-[(x_1-5)^2+(x_2-5)^2]+100\exp(-[(x_1-2.5)^2+(x_2-2.5)^2]/0.01)$ on $[-5,10]^2$.
An independent reduction establishes the nominal Pareto set $(t,t)$, $0\le t\le {validation['endpoint_t']:.10f}$. All indicators use $(f_1,-f_2)$ with fixed reference normalization.
Twenty paired seeds per method were run for 200 generations with population and nominal offspring sizes of 100, retaining the inherited ALSF operators, thresholds and tolerances.
Mean final hypervolumes were {b.mean_HV:.6f}, {r.mean_HV:.6f} and {d.mean_HV:.6f} for NSGA-II, ALSF-Resample and ALSF-Drop, respectively; corresponding mean IGD+ values were {b.mean_IGD_plus:.6f}, {r.mean_IGD_plus:.6f} and {d.mean_IGD_plus:.6f}.
ALSF-Drop used {d.mean_evaluations:.1f} evaluations on average, saving {d.mean_saved_percent:.2f}\% of the nominal 20000-evaluation schedule. There were {total_peak} actual Drop rejections within radius 0.03 of the Gaussian center, aggregated across the 20 runs.
All final nondominated designs underwent independent uncertainty-box extrema analysis. Their mean per-run sensitive fractions were {b.mean_sensitive_fraction:.4f}, {r.mean_sensitive_fraction:.4f} and {d.mean_sensitive_fraction:.4f}. These full-box assessments are distinct from archive evidence labels.
Neighborhood discovery, accurate approximation of the narrow nominal Pareto segment and demonstrated rejection must be interpreted separately. Filtering can remove nominally efficient designs; lower evaluation counts alone do not establish computational advantage. Detailed per-seed results, paired bootstrap intervals, multiplicity-adjusted comparisons, censored HV targets, timing and reference accuracy are reported in the accompanying reproducibility report. No equivalence or universal superiority is inferred.
'''
    (root/'suggested_subsection.tex').write_text(subsection)
    print(agg.to_string())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    analyze(parser.parse_args().root)
