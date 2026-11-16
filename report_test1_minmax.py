"""Publication figures and evidence-based report for the restored Test 1."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_test1 import evaluate
from benchmark_test1_minmax import to_optimization
from src.drop_metrics import normalization
from validate_alsf import perturbation_test
from analyze_repeated_study import ORDER,LABEL,COLOR,paired_effect

def reference_figures(out,ref):
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=320,bbox_inches='tight')
        plt.close(fig)
    # All physical labels use the original positive f2.
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for ax,obj,limits,title in zip(axes,[0,1,1],[(-5,10),(-5,10),(2.25,2.75)],['f1 (minimize)','f2 (maximize)','f2: Gaussian peak']):
        line=np.linspace(*limits,401);u,v=np.meshgrid(line,line);z=evaluate(np.stack([u,v],axis=-1))[...,obj]
        im=ax.pcolormesh(u,v,z,shading='auto',cmap='viridis',rasterized=True);fig.colorbar(im,ax=ax,shrink=.8)
        ax.set(xlabel='x1',ylabel='x2',title=title,aspect='equal')
    save(fig,'01_objective_surfaces')
    fig=plt.figure(figsize=(10,3.4),layout='constrained')
    for j,(obj,limits,title) in enumerate([(0,(-5,10),'f1 (minimize)'),(1,(-5,10),'f2 (maximize)'),(1,(2.25,2.75),'Gaussian peak')]):
        ax=fig.add_subplot(1,3,j+1,projection='3d');line=np.linspace(*limits,301);u,v=np.meshgrid(line,line)
        z=evaluate(np.stack([u,v],axis=-1))[...,obj]
        ax.plot_surface(u,v,z,cmap='viridis',linewidth=0,rasterized=True,rcount=151,ccount=151)
        ax.set(xlabel='x1',ylabel='x2',zlabel=f'f{obj+1}',title=title);ax.view_init(28,-125)
    save(fig,'01b_objective_surfaces_3d')
    fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for ax in axes:
        curve=ref[['Obj1','Obj2']].to_numpy().copy()
        transitions=np.flatnonzero(ref.peak_branch.to_numpy()[:-1] & ~ref.peak_branch.to_numpy()[1:])+1
        curve[transitions]=np.nan
        ax.plot(curve[:,0],curve[:,1],color='.3',lw=1,label='Full reference')
        peak=ref[ref.peak_branch];ax.plot(peak.Obj1,peak.Obj2,c='#C17B3A',lw=2,label='Peak-efficient segment')
        ax.scatter([12.5],[112.5],s=15,c='r',marker='x',label='Dominated center')
        ax.set(xlabel='f1 (minimize)',ylabel='f2 (maximize)')
    axes[1].set(xlim=(12.475,12.505),ylim=(112.475,112.505));axes[0].legend(fontsize=6)
    save(fig,'02_independent_reference')


def supplement(root,out,frame,effects):
    shutil.copytree(root/'reference',out/'reference')
    shutil.copy2(root/'protocol.json',out/'protocol.json')
    ref=pd.read_csv(root/'reference/test1.csv');validation=json.loads((root/'reference/validation.json').read_text())
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=320,bbox_inches='tight')
        plt.close(fig)
    reference_figures(out,ref)
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for ax,mode in zip(axes,ORDER):
        ax.plot(ref.Obj1,ref.Obj2,c='.65',lw=1,label='Reference')
        for seed in frame.seed.unique():
            f=pd.read_csv(root/f'runs/test1/{mode}/seed{seed}/pareto.csv')
            ax.scatter(f.Obj1,f.Obj2,c=COLOR[mode],alpha=.12,s=3,rasterized=True)
        ax.set(xlabel='f1 (minimize)',ylabel='f2 (maximize)',title=LABEL[mode])
    save(fig,'03_all_final_fronts')
    selected={};peak_rows=[];audit_rows=[];peak_quality=[];evidence=[]
    for mode in ORDER:
        for seed in frame.seed.unique():
            folder=root/f'runs/test1/{mode}/seed{seed}'
            a=pd.read_csv(folder/'archive.csv');e=pd.read_csv(folder/'peak_screening.csv')
            near=np.linalg.norm(a[['X1','X2']].to_numpy()-2.5,axis=1)<=.18
            p=a[near].copy()
            if len(p):
                values,bad=perturbation_test(p[['X1','X2']].to_numpy())
                p['R1_independent']=values[:,0];p['R2_independent']=values[:,1];p['independently_sensitive']=bad
                p['seed']=seed;p['mode']=mode;audit_rows.append(p)
            for flag in ['robust','non-robust','NEI']:
                peak_rows.append(dict(mode=mode,seed=seed,flag=flag,n=int((p.flag==flag).sum())))
            # Only the actual exclusion stage counts for Drop, not its empty repair.
            checks=e[e.policy==('drop_proposal' if mode=='drop' else 'repair_check')]
            evidence.append(dict(mode=mode,seed=seed,checks=len(checks),rejected=int(checks.rejected.sum()),
                allowed_sparse=int(checks.allowed_sparse.sum()),
                evaluated=int(frame.loc[(frame['mode']==mode)&(frame.seed==seed),'peak_evaluated'].iloc[0])))
            f=pd.read_csv(folder/'pareto.csv');g=to_optimization(f[['Obj1','Obj2']]);r=to_optimization(ref.loc[ref.peak_branch,['Obj1','Obj2']])
            # Peak-restricted IGD+, normalized with the SAME full-front ranges.
            z=np.maximum((g[None,:,:]-r[:,None,:])/np.array([50,150]),0)
            local=float(np.linalg.norm(z,axis=2).min(axis=1).mean())
            peak_quality.append(dict(mode=mode,seed=seed,peak_igd_plus=local,
                final_points_near_peak=int((np.linalg.norm(f[['X1','X2']].to_numpy()-2.5,axis=1)<=.18).sum())))
            if seed==42:selected[mode]=(a,e)
    pd.DataFrame(peak_rows).to_csv(out/'peak_archive_labels.csv',index=False)
    audits=pd.concat(audit_rows,ignore_index=True) if audit_rows else pd.DataFrame()
    audits.to_csv(out/'peak_archive_independent_audit.csv',index=False)
    evidence=pd.DataFrame(evidence);evidence.to_csv(out/'peak_screening_summary.csv',index=False)
    pd.DataFrame(peak_quality).to_csv(out/'peak_front_quality.csv',index=False)
    labels={'robust':('Locally stable (observed)','#237C69'),'non-robust':('Verified archive violation','#C17B3A'),'NEI':('NEI','#9CA3AF')}
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    for ax,mode in zip(axes,ORDER):
        a,_=selected[mode]
        for flag,(label,c) in labels.items():
            group=a[a.flag==flag];ax.scatter(group.X1,group.X2,c=c,s=2,alpha=.35,rasterized=True,label=label)
        ax.set(xlabel='x1',ylabel='x2',title=LABEL[mode],xlim=(-5,10),ylim=(-5,10),aspect='equal')
    axes[0].legend(fontsize=6,markerscale=3);save(fig,'04_archive_labels_seed42')
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    t=np.linspace(2.3,2.7,301);u,v=np.meshgrid(t,t);z=evaluate(np.stack([u,v],axis=-1))[...,1]
    for ax,mode in zip(axes,ORDER):
        ax.contour(u,v,z,levels=[20,50,80,100,110],colors='.8',linewidths=.6)
        a,e=selected[mode]
        for flag,(label,c) in labels.items():
            group=a[a.flag==flag];ax.scatter(group.X1,group.X2,c=c,s=10,alpha=.7,label=label)
        checks=e[e.policy==('drop_proposal' if mode=='drop' else 'repair_check')]
        rejected=checks[checks.rejected];ax.scatter(rejected.X1,rejected.X2,marker='x',c='#A02030',s=18,label='Actually rejected')
        p=ref[ref.peak_branch];ax.plot(p.X1,p.X2,c='black',lw=2,label='Efficient segment')
        ax.text(.03,.03,f'r <= 0.18: {int((np.linalg.norm(a[["X1","X2"]].to_numpy()-2.5,axis=1)<=.18).sum())} archived; {len(rejected)} rejected',transform=ax.transAxes,fontsize=6)
        ax.set(xlim=(2.3,2.7),ylim=(2.3,2.7),xlabel='x1',ylabel='x2',title=LABEL[mode],aspect='equal')
    axes[0].legend(fontsize=5,loc='upper left');save(fig,'05_peak_zoom_seed42')
    fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for mode in ORDER:
        tables=[pd.read_csv(root/f'runs/test1/{mode}/seed{seed}/process.csv') for seed in frame.seed.unique()]
        for ax,key in zip(axes,['n_accepted','n_dropped']):
            data=np.array([t[key].to_numpy() for t in tables]);x=tables[0].generation
            ax.plot(x,np.median(data,axis=0),c=COLOR[mode],label=LABEL[mode]);ax.fill_between(x,*np.quantile(data,[.25,.75],axis=0),color=COLOR[mode],alpha=.15)
    axes[0].set(ylabel='Accepted for evaluation',xlabel='Generation');axes[1].set(ylabel='Dropped without replacement',xlabel='Generation');axes[0].legend(fontsize=6)
    save(fig,'08_offspring_evolution')
    # Compact publishable raw artifacts; preserve every original run, including classified archives.
    for mode in ORDER:
        for seed in frame.seed.unique():
            src=root/f'runs/test1/{mode}/seed{seed}';dst=out/f'runs/{mode}/seed{seed}';dst.mkdir(parents=True)
            for path in src.iterdir():
                if path.suffix=='.csv' and path.name=='archive.csv':
                    pd.read_csv(path).to_csv(dst/'archive.csv.gz',index=False,compression={'method':'gzip','mtime':0})
                else:shutil.copy2(path,dst/path.name)
    make_report(root,out,frame,effects,evidence,audits,validation)


def table(df):
    def cell(value):
        return f'{value:.6g}' if isinstance(value,(float,np.floating)) else str(value)
    rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    rows += ['| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def make_report(root,out,frame,effects,evidence,audits,v):
    desc=pd.read_csv(out/'descriptive.csv');desc['iqr']=desc.q75-desc.q25;desc.to_csv(out/'descriptive.csv',index=False)
    means=frame.groupby('mode')[['actual_model_calls','hypervolume','igd_plus','rejected_fraction','sensitive_audit_fraction']].mean().reindex(ORDER)
    savings=100*(20000-means.loc['drop','actual_model_calls'])/20000
    h=pd.read_csv(out/'target_hv_attainment.csv');hits=h.groupby(['mode','target_fraction']).reached.sum().unstack().reindex(ORDER)
    target_effects=[]
    for fraction,group in h.groupby('target_fraction'):
        p=group.pivot(index='seed',columns='mode',values='first_evaluations')
        for a,b in [('baseline','resample'),('baseline','drop'),('resample','drop')]:
            joint=p[[a,b]].dropna();row=dict(target_fraction=fraction,method_a=a,method_b=b,n_both=len(joint),n_total=20)
            if len(joint):
                effect=paired_effect(joint[a],joint[b]);row.update({k:effect[k] for k in ['mean_difference','ci95_low','ci95_high']})
            target_effects.append(row)
    target_effects=pd.DataFrame(target_effects);target_effects.to_csv(out/'target_efficiency_CI.csv',index=False)
    ev=evidence.groupby('mode')[['checks','rejected','allowed_sparse','evaluated']].sum().reindex(ORDER)
    pk=pd.read_csv(out/'peak_front_quality.csv').groupby('mode')[['peak_igd_plus','final_points_near_peak']].mean().reindex(ORDER)
    report=f'''# Restored Test 1: minimize f1, maximize f2

## Scientific result

ALSF-Drop saved **{savings:.2f}%** of actual model calls on average in this new, corrected 60-run experiment. Quality changes and uncertainty are reported below; savings alone do not establish efficiency or equivalence. These results replace no historical experiment. The earlier 25.3% saving belongs exclusively to the both-minimization study.

{table(means.reset_index())}

The 99% HV target was reached by every method after 900–1,900 evaluations (mean 1,290), before filtering starts; paired target costs are identical. Neither higher target was reached by any method. Therefore these targets provide **no evidence of faster attainment at comparable quality**. Final paired HV/IGD+ confidence intervals contain zero, which does not establish equivalence.

Across all 60 final fronts, **none contains a point within radius 0.18 of the Gaussian center**. Drop rejected **zero** candidates there; Resample rejected **three screening checks**. All 29,230 Drop discards therefore occurred outside the predefined peak neighborhood. Thus the observed savings cannot be attributed to successful filtering of the sensitive peak. The peak was sparsely explored, and most archived nearby designs remained NEI. This central scientific hypothesis is not demonstrated by this experiment.

The Gaussian has a genuine but extremely narrow efficient segment. Its center is **dominated**, not Pareto optimal. Global HV/IGD+ can obscure omission of this segment. Peak-specific evidence below is essential to interpret the headline savings.

## Exact problem and provenance

Domain [-5,10]^2. Minimize f1=x1²+x2²; maximize f2=(x1−5)²+(x2−5)²+100 exp(−((x1−2.5)²+(x2−2.5)²)/0.01). At (2.5,2.5), (f1,f2)=(12.5,112.5). The physical model and Gaussian are unchanged. The explicit adapter `benchmark_test1_minmax.py` supplies g=(f1,−f2) to pymoo. Archives, fronts and physical plots retain positive f2. Dominance, HV and IGD+ use g. Sensitivity is calculated from raw physical values; tests verify sign invariance of relative differences, cached evidence and flags.

Source: [JCAM manuscript](https://github.com/tamasorosz/Archive-Based-Local-Sensitivity-Filtering-for-Robust-Design-Optimisation-of-Electric-Machines/blob/09d3c15ffeaa36acd4696b4d4f747f2d6d39e04e/elsarticle-template-num.tex), Results and Discussion → Analytical bi-objective optimization problem: “The first objective is subject to minimization, whereas the second is subject to maximization.” No manuscript file was modified. The historical `benchmark_test1.py` documentation and min-min helpers are retained for old experiment reproducibility; the new entry point is explicit.

## Independent reference and accuracy

Put q=f1 and s=x1+x2. At fixed q, f2=q−10s+50+100 exp(−(q−5s+12.5)/0.01), which is convex in s. For 0≤q≤50, both endpoints s=±√(2q) are feasible, attained at the two diagonals. Thus their maximum is the full feasible upper envelope, not an assumption about NSGA-II output. For q>50, some coordinate must exceed 5; the quadratic background is ≤125 and the Gaussian ≤100 exp(−1250). These points are dominated by (−5,−5), with (50,200). Mixed-direction nondominance filtering is applied after constructing the envelope, including its locally decreasing portion.

The efficient positive-diagonal segment starts at q={v['peak_branch_start_q']:.12f} and ends at q={v['peak_local_maximum_q']:.12f}, where f2={v['peak_maximum_f2']:.12f}. The negative branch catches this maximum at q={v['negative_branch_catch_q']:.12f}. The endpoint slightly preceding the Gaussian center dominates that center.

The reference uses 10,001 global q values plus 4,001 adaptively located peak values and exact transition points; {v['reference_rows']} nondominated rows remain. Doubling both resolutions gives {v['refined_rows']} rows and changes reference HV by {v['hv_resolution_difference']:.3g} (absolute). There are 532 independent SLSQP starts over 37 q levels, including peak refinement, and 65,537 polar angles per level. Maximum feasible SLSQP envelope excess is {v['max_feasible_envelope_violation']:.3g}; maximum gap of the best feasible multistart value is {v['max_best_multistart_gap']:.3g}; maximum polar gap is {v['max_polar_sampling_gap']:.3g}. Two optimizer starts report failure; they remain in `reference/multistart.csv` with residuals and feasibility, rather than being silently discarded. The convex-envelope proof, not multistart success, establishes global coverage. These numerical checks validate the implementation separately.

Normalization is fixed from the corrected reference: ideal g=(0,−200), range=(50,150), HV reference=(1.1,1.1). Global IGD+ uses 501 uniformly normalized-arclength reference points, avoiding accidental weighting by peak refinement density. Linear interpolation bridges the negligible nondominated gap; reference resolution error is documented above. `protocol.json` locks HV targets at 99%, 99.5%, 99.9% of independent reference HV **before** these 60 optimization runs. No old absolute targets or old normalization are reused.

## Protocol and counting

20 paired seeds: 1–17,42,71,97. All three methods use population=100, offspring=100, 200 generations (including initialization), SBX probability .9/eta 5, PM probability 1/eta 5, tolerances (.1,.1), sensitivity thresholds (.1,.1), floors (.001,.001), scaled Euclidean neighborhood and exclusion radius 1, minimum support 10, rounding 8 decimals, no objective cache. Evidence is updated after generation 50; filtering affects the next batch. Resample retains max 1,000 replacement attempts. Drop calls mating once then discards forbidden proposals without replacement; the existing policy and algorithm files are unchanged. Two workers with single-threaded BLAS were used. Wall times include observation and indicators and are descriptive, not expensive-solver timings.

Each actual objective request counts, including repeated designs; the archive stores unique rounded designs. Each Drop run satisfies accepted=evaluator calls=actual calls and dropped+actual calls=20,000. Resample and baseline each evaluate 20,000 designs. Resample screening counts include replacement trials and mating duplicate handling; they are not model calls or unique rejected offspring. `all_runs.csv`, `runs/*/seed*/summary.json`, `process.csv` and `history.csv` contain every run, final labels, rejection counts/fractions and per-generation evidence. For Resample, `rejected_fraction` is rejected initial repair inputs divided by all initial repair inputs (including normal mating duplicate retries); total rejected replacement checks and their denominator are exported separately. These counts are not directly interchangeable with Drop's single post-mating rejection fraction. Drop's rejection fraction includes the 100 unfiltered initialization points in its denominator. `resample_forbidden_checks` counts rejected original repair inputs; replacement trial totals are separate in `process.csv`.

## Paired statistics and comparable quality

`descriptive.csv` provides mean, median, sample standard deviation, quartiles and IQR. `paired_effects.csv` provides B−A effects and 95% paired percentile bootstrap mean CIs (10,000 resamples), plus two-sided Wilcoxon tests, Holm-corrected across all nine primary comparisons (3 metrics × 3 pairs). A nonsignificant difference does not establish equivalence; no equivalence margin was specified.

{table(effects[['metric','method_a','method_b','mean_difference','ci95_low','ci95_high','p_holm']])}

Target attainment counts (out of 20):

{table(hits.reset_index())}

At the 99% target, paired evaluation differences B−A (conditional only when both attain; confidence intervals descriptive):

{table(target_effects[target_effects.target_fraction==.99])}

Unreached targets remain censored at the actual final evaluation budget. No failure is assigned a fictitious attainment time. Conditional paired target differences are descriptive only; restricting to joint successes can bias comparison. `evaluation_checkpoints.csv` and the convergence figure compare all 20 seeds at identical allowed evaluation budgets up to the minimum final budget across every run. Values are held at the last completed generation; unused budget is recorded. This supports assessment at comparable quality without changing the cohort or extrapolating beyond available data.

## Peak sensitivity, evidence and actual rejection

Near-peak means Euclidean distance≤0.18 from (2.5,2.5), fixed before the experiment. The independent 21×21 perturbation grid over the prescribed ±0.1 box finds sensitive witnesses for **100%** of sampled reference peak-efficient points (`reference/peak_independent_audit.csv`). This is a finite witness test; passing it would not certify stability. All actually archived near-peak points are independently audited in `peak_archive_independent_audit.csv` with their final observed archive flag. Nominal efficiency, archive evidence, independent sensitivity and actual rejection are separate quantities.

Aggregated screening evidence across 20 seeds (checks can repeat candidates):

{table(ev.reset_index())}

For Drop, this table counts only `drop_proposal` events, after mating. Its empty rounding repair is excluded. For Resample it counts `repair_check` events, including replacement attempts. Baseline checks use an empty policy and are unfiltered, so its sparse-support counts have no scientific classification meaning. `allowed_sparse` means fewer than10 existing archive neighbors and no exclusion evidence, **not** an assigned flag on an unevaluated candidate. A single verified violation can exclude a point even with sparse support. Allowed candidates are never called robust solely because they were allowed. Actual archived NEI, stable-observed and verified-violation counts are separate in `peak_archive_labels.csv`.

The near-peak independent audit contains {len(audits)} unique-within-run archived designs; {int(audits.independently_sensitive.sum()) if len(audits) else 0} have independent sensitive witnesses. Every audited near-peak design has a sensitive witness, including those whose archive flag remains NEI; no nearby archived design is labeled locally stable. These are posthoc audits, not evidence supplied to the filtering policy. Empty/no-rejection observations do not establish successful peak exclusion. Archive flags use observed pairs within scaled Euclidean tolerance balls; the independent audit uses a grid spanning a coordinate box, a different and larger perturbation domain.

Peak-restricted IGD+ (same global normalization, unweighted dense efficient segment) and mean number of final points within radius.18:

{table(pk.reset_index())}

A point within radius.18 need not approximate the extremely thin nominally efficient segment. Peak IGD+ is an objective-space metric and cannot certify capture of the positive decision branch when the negative branch has nearly identical objectives. The supplied fronts and peak IGD+ are required to assess its approximation; global HV alone cannot justify preservation of the narrow feature. No global metric result establishes robustness or expensive-model speedup.

## Figures

All figures have PDF and PNG versions. Physical f2 is always maximized.

1. [Objective surfaces](01_objective_surfaces.png), including a peak zoom.
2. [Independent reference](02_independent_reference.png), including the efficient segment and dominated center.
3. [All20 final fronts per method](03_all_final_fronts.png).
4. [Decision archives and final flags, seed42](04_archive_labels_seed42.png).
5. [Peak zoom and actual rejected points, seed42](05_peak_zoom_seed42.png). A prespecified illustration, not a best run.
6. [Savings versus final quality](test1_04_savings_quality_tradeoff.png).
7. [Per-seed HV/IGD+/calls](test1_02_final_distributions.png).
8. [Accepted and dropped offspring](08_offspring_evolution.png), median/IQR across 20 seeds. Resample rejects then replaces; its accepted count remains100.

Also: [3D objective surfaces](01b_objective_surfaces_3d.png), [quality vs actual evaluation budget](test1_01_convergence_median_IQR.png), [censored target attainment](test1_03_target_attainment.png).

## Limitations and interpretation

These are 20 finite stochastic runs on one analytical problem. The Gaussian-efficient segment is exceptionally small, potentially missed by all methods. Observed stability is empirical and does not certify a continuous tolerance box. Global IGD+ and HV have limited resolution for this feature. Bootstrap intervals are finite-sample uncertainty, not an equivalence test. Evaluation savings trade off against any front degradation; assess paired quality, common-budget curves, target attainment and local peak evidence jointly. This experiment does not establish transfer to expensive electromagnetic models. The solenoid was not rerun and its code/results and all previous studies remain unchanged. Existing files are verified byte-for-byte against the parent commit; all restoration changes are new files.

## Reproduce

From the repository root, using the existing Python requirements/environment:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python prepare_test1_minmax.py --root work/test1_original_minmax
python run_test1_original_minmax.py --root work/test1_original_minmax --workers 2
python analyze_test1_original_minmax.py --root work/test1_original_minmax --outdir docs/test1_original_minmax
python -m pytest -q
python verify_test1_original_minmax.py
```

Use a new root to lock a fresh protocol and an absent output directory for analysis. Completed optimization runs resume only when settings match. Raw archives are gzip-compressed under `runs/`; all other per-run CSV/JSON files are retained. The new unit and pymoo integration tests cover physical exports, mixed dominance, sign invariance, full-domain envelope, center dominance, call accounting and observational instrumentation preserving decisions and random draws. No JCAM manuscript edits are included. See `suggested_subsection.tex` for a separate insertion proposal.
'''
    (out/'REPORT.md').write_text(report)
    latex=r'''\subsection{Restored analytical Test 1 with mixed objective directions}
The original directions were restored by minimizing $(f_1,-f_2)$ internally,
while retaining physical objectives for reporting and sensitivity analysis.
An independent fixed-$f_1$ upper envelope was validated by multistart constrained
optimization and polar sampling. The Gaussian contributes an exceptionally
narrow Pareto-efficient segment; its center itself is dominated. Twenty paired
seeds were evaluated for NSGA-II, ALSF-Resample and ALSF-Drop, without changing
the sensitivity thresholds or genetic operators.
'''
    latex+=f'ALSF-Drop reduced actual objective calls by {savings:.2f}\\% on average. '
    latex+='Final mean hypervolumes (baseline, Resample, Drop) were '+', '.join(f'{means.loc[m,"hypervolume"]:.6f}' for m in ORDER)+'. '
    latex+='The corresponding IGD$^+$ values were '+', '.join(f'{means.loc[m,"igd_plus"]:.6f}' for m in ORDER)+'.\n'
    latex+=r'''No final front contained a design within radius 0.18 of the Gaussian center; Drop rejected no candidate in this neighborhood. The 99\% HV target was attained before filtering activated, providing no evidence of accelerated target attainment. These global indicators must be interpreted together with paired confidence
intervals, censored target attainment, and the peak-specific archive and rejection
audits. A nonsignificant test does not establish equivalence. The earlier 25.3\%
saving concerns a separate both-minimization formulation and is not a result of
this restored experiment. See the supplementary report for all sixty runs and
the distinction between nominal efficiency, observed sensitivity and actual
pre-evaluation rejection.
'''
    (out/'suggested_subsection.tex').write_text(latex)
