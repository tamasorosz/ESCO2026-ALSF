"""Post-process existing 20-seed solenoid fronts; never runs optimization.

Uniformly weight every stored reference row. h=0.1 is a HALF-WIDTH in normalized
currents (MA/m²). Objective rectangles use exact coordinate extrema, not an
exact joint attainable set. All objective scales come only from the reference.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.metrics.box_igd import nearest_box_distances,nearest_rectangle_distances,summarize_distances
from di_barba_solenoid_benchmark import DiBarbaSolenoid
from src.drop_metrics import normalization,indicators

METHODS=['baseline','resample','drop']
LABELS={'baseline':'NSGA-II','resample':'ALSF-Resample','drop':'ALSF-Drop'}
COLORS={'baseline':'#356B9A','resample':'#C17B3A','drop':'#237C69'}
XCOLS=[f'X{i+1}' for i in range(10)]
FCOLS=['Obj1','Obj2']
# Illustrative margins fixed before calculating these new diagnostics. They
# require engineering approval and do not convert matching into robustness.
MARGINS={'decision_box_igd':.01,'objective_box_igd':.001,
         'decision_tolerance_coverage':.05,'objective_tolerance_coverage':.05}


def hashes(root,exclude):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(exclude)}


def paired_statistics(values_a,values_b,margin,draws=20000):
    diff=np.asarray(values_b,float)-np.asarray(values_a,float)
    if len(diff)<2 or not np.isfinite(diff).all():raise ValueError('Need finite paired samples')
    bootstrap=diff[np.random.default_rng(20261116).integers(0,len(diff),(draws,len(diff)))].mean(axis=1)
    mean=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
    result=dict(mean_difference=mean,median_difference=float(np.median(diff)),
        ci95_low=float(np.quantile(bootstrap,.025)),ci95_high=float(np.quantile(bootstrap,.975)),
        proposed_margin=margin)
    if se==0:
        # A zero sample variance does not justify a t-distribution population
        # equivalence claim. Preserve identical observed values explicitly.
        result.update(tost_p=np.nan,equivalent_proposed_margin=False,
                      tost_status='not estimable: zero paired variance',t_ci90_low=np.nan,t_ci90_high=np.nan)
    else:
        low_p=float(t.sf((mean+margin)/se,len(diff)-1))
        high_p=float(t.cdf((mean-margin)/se,len(diff)-1))
        radius=float(t.ppf(.95,len(diff)-1)*se)
        result.update(tost_p=max(low_p,high_p),equivalent_proposed_margin=max(low_p,high_p)<.05,
                      tost_status='exploratory paired t-TOST; normal paired-mean assumption',
                      t_ci90_low=mean-radius,t_ci90_high=mean+radius)
    return result


def checked_objective_extrema(model,point):
    """Same convex extrema; BVLS fallback only on primary solver nonconvergence.

    The nominal point is a known feasible candidate. Including it avoids tiny
    numerical minima worse than that known design without adding an epsilon.
    """
    point=np.asarray(point,float);fallback=False;kkt=np.nan;reason=''
    try:
        low,high=model.box_objective_extrema(point)
    except RuntimeError as error:
        if 'Sensitivity minimum solve failed' not in str(error):raise
        from scipy.optimize import lsq_linear
        fallback=True;reason=str(error)
        step=model.s.independent_j_tolerance/model.s.j_scale
        lo=np.maximum(model.lower,point-step);hi=np.minimum(model.upper,point+step)
        result=lsq_linear(model.K,model.b_target,bounds=(lo,hi),method='bvls',tol=1e-12,max_iter=10000)
        if not result.success:raise RuntimeError('Both convex minimum solvers failed: '+result.message)
        gradient=2*model.K.T@(model.K@result.x-model.b_target)/model.s.n_field_samples
        kkt=float(np.max(abs(result.x-np.clip(result.x-gradient,lo,hi))))
        if kkt>1e-8:raise RuntimeError('BVLS projected-gradient check failed')
        bits=((np.arange(1<<model.n_var)[:,None]>>np.arange(model.n_var))&1)
        corners=lo+bits*(hi-lo)
        maximum=np.mean((corners@model.K.T-model.b_target)**2,axis=1).max()
        near=np.where((lo<=0)&(hi>=0),0.,np.minimum(abs(lo),abs(hi)))
        far=np.maximum(abs(lo),abs(hi))
        low=np.array([model.evaluate(result.x)[0],np.mean(near**2)])
        high=np.array([maximum,np.mean(far**2)])
    nominal=np.asarray(model.evaluate(point))
    nominal_included=bool(np.any(low>nominal) or np.any(high<nominal))
    low=np.minimum(low,nominal);high=np.maximum(high,nominal)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>high):
        raise RuntimeError('Invalid objective extrema')
    return low,high,dict(bvls_fallback=fallback,projected_kkt=kkt,
                        nominal_candidate_included=nominal_included,primary_failure=reason)


def analyze(study,outdir,*,bootstrap_draws=20000):
    study=Path(study).resolve();outdir=Path(outdir).resolve()
    if outdir.exists():raise FileExistsError(f'Refusing existing output directory: {outdir}')
    before=hashes(study,outdir)
    ref=pd.read_csv(study/'reference/solenoid.csv')
    reference_x=ref[XCOLS].to_numpy(float);reference_f=ref[FCOLS].to_numpy(float)
    if not len(ref) or not np.isfinite(reference_x).all() or not np.isfinite(reference_f).all():
        raise ValueError('Invalid reference coordinates/objectives')
    seeds=json.loads((study/'protocol.json').read_text())['seeds']
    if len(seeds)!=20 or len(set(seeds))!=20:raise ValueError('Expected 20 unique paired seeds')
    summaries=pd.read_csv(study/'analysis/all_runs.csv')
    summaries=summaries[summaries.benchmark=='solenoid'].set_index(['mode','seed'])
    expected={(m,s) for m in METHODS for s in seeds}
    if len(summaries)!=60 or summaries.index.duplicated().any() or set(summaries.index)!=expected:raise ValueError('Incomplete or unexpected solenoid run summary keys')
    inputs={}
    model=DiBarbaSolenoid()
    np.testing.assert_allclose([model.evaluate(x) for x in reference_x],reference_f,rtol=1e-10,atol=1e-12)
    for mode in METHODS:
        found={int(p.parent.name[4:]) for p in (study/'runs/solenoid'/mode).glob('seed*/pareto.csv')}
        if found!=set(seeds):raise ValueError(f'Incomplete paired fronts: {mode}')
        for seed in seeds:
            p=pd.read_csv(study/'runs/solenoid'/mode/f'seed{seed}/pareto.csv')
            if p.empty:raise ValueError(f'Empty experimental front: {mode}/{seed}')
            x=p[XCOLS].to_numpy(float);f=p[FCOLS].to_numpy(float)
            if not np.isfinite(x).all() or not np.isfinite(f).all() or np.any(x<model.lower) or np.any(x>model.upper):
                raise ValueError('Invalid experimental design/objectives')
            np.testing.assert_allclose([model.evaluate(v) for v in x],f,rtol=1e-10,atol=1e-12)
            inputs[mode,seed]=(x,f)
    outdir.mkdir(parents=True)
    ideal,scale=normalization(ref)
    settings=dict(reference_solutions=len(ref),methods=METHODS,seeds=seeds,n_paired_seeds=20,
        decision_half_widths=[.1]*10,decision_full_widths=[.2]*10,
        decision_units='normalized J/(1 MA/m²); distances numerically in MA/m²',
        objective_ideal=ideal.tolist(),objective_scales=scale.tolist(),
        objective_scale_source='coordinate ranges of the common stored reference only',
        objective_box='coordinatewise exact extrema on physical decision box intersected with [-2,2]^10; joint bounding approximation',
        reference_weighting='all stored reference rows, equally weighted; no arclength resampling',
        coverage_rule='computed distance == 0; no hidden epsilon',bootstrap_draws=bootstrap_draws,
        bootstrap_seed=20261116,paired_difference='B minus A',
        proposed_equivalence_margins=MARGINS,
        margin_status='illustrative proposals fixed before new metric evaluation; not approved engineering specifications',
        equivalence_method='exploratory paired t-TOST alpha=.05, plus Holm adjustment across estimable comparisons; zero variance not estimable')
    (outdir/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    # Model calls here are independent post-processing diagnostics, not new
    # optimizer evaluations and never inserted into any experimental archive.
    low=[];high=[];solver_diagnostics=[]
    for reference_row,point in enumerate(reference_x):
        a,b,diagnostic=checked_objective_extrema(model,point)
        solver_diagnostics.append(dict(reference_row=reference_row,**diagnostic))
        if np.any(a>model.evaluate(point)+np.array([1e-10,1e-10])) or np.any(b<model.evaluate(point)-np.array([1e-10,1e-10])):
            raise RuntimeError('Reference value not contained in computed extrema')
        low.append(a);high.append(b)
    pd.DataFrame(solver_diagnostics).to_csv(outdir/'objective_extrema_solver_diagnostics.csv',index=False)
    low=np.array(low);high=np.array(high)
    boxes=pd.DataFrame(dict(reference_row=np.arange(len(ref)),Obj1=reference_f[:,0],Obj2=reference_f[:,1],
        Obj1_min=low[:,0],Obj1_max=high[:,0],Obj2_min=low[:,1],Obj2_max=high[:,1]))
    boxes.to_csv(outdir/'objective_tolerance_rectangles.csv',index=False)
    # Save source/reference distribution diagnostics: uniform stored-row weights
    # can emphasize the densely sampled low-error endpoint.
    ref[FCOLS].describe(percentiles=[.05,.25,.5,.75,.95]).to_csv(outdir/'reference_distribution.csv')
    rows=[];decision_dist=[];objective_dist=[];run_keys=[]
    for mode in METHODS:
        for seed in seeds:
            x,f=inputs[mode,seed]
            dx=nearest_box_distances(x,reference_x,np.full(10,.1))
            df=nearest_rectangle_distances((f-ideal)/scale,(low-ideal)/scale,(high-ideal)/scale)
            metrics=indicators(f,ref);old=summaries.loc[mode,seed]
            for key in ['hypervolume','igd_plus']:
                if not np.isclose(metrics[key],old[key],rtol=1e-12,atol=1e-12):raise ValueError('Conventional indicator mismatch')
            row=dict(method=mode,seed=seed,n_obtained=len(x),n_reference=len(ref),
                     hypervolume=float(old.hypervolume),igd_plus=float(old.igd_plus),
                     actual_model_calls=int(old.actual_model_calls))
            for prefix,values in [('decision',dx),('objective',df)]:
                result=summarize_distances(values)
                row.update({prefix+'_'+k:v for k,v in result.items()})
            rows.append(row);decision_dist.append(dx);objective_dist.append(df);run_keys.append(f'{mode}/seed{seed}')
    frame=pd.DataFrame(rows);frame.to_csv(outdir/'box_igd_per_seed.csv',index=False)
    np.savez_compressed(outdir/'per_reference_distances.npz',run_keys=np.asarray(run_keys),
                        decision=np.asarray(decision_dist),objective=np.asarray(objective_dist))
    metrics=[k for k in frame if k not in ['method','seed','n_obtained','n_reference']]
    aggregate=[]
    for method,group in frame.groupby('method'):
        for metric in metrics:
            v=group[metric].to_numpy(float)
            aggregate.append(dict(method=method,metric=metric,n=20,mean=float(v.mean()),median=float(np.median(v)),
                std=float(v.std(ddof=1)),q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75)),minimum=float(v.min()),maximum=float(v.max())))
    summary=pd.DataFrame(aggregate);summary.to_csv(outdir/'box_igd_summary.csv',index=False)
    effects=[]
    for metric in metrics:
        pivot=frame.pivot(index='seed',columns='method',values=metric).sort_index()
        for a,b in combinations(METHODS,2):
            margin=MARGINS.get(metric)
            if margin is None:
                diff=pivot[b].to_numpy()-pivot[a].to_numpy()
                boot=diff[np.random.default_rng(20261116).integers(0,20,(bootstrap_draws,20))].mean(axis=1)
                result=dict(mean_difference=float(diff.mean()),median_difference=float(np.median(diff)),
                            ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)))
            else:result=paired_statistics(pivot[a],pivot[b],margin,bootstrap_draws)
            effects.append(dict(metric=metric,method_a=a,method_b=b,n_pairs=20,**result))
    paired=pd.DataFrame(effects);paired['tost_p_holm']=np.nan;paired['equivalent_proposed_margin_holm']=False
    estimable=paired.index[paired.tost_p.notna()]
    previous=0.
    for rank,index in enumerate(paired.loc[estimable].sort_values('tost_p').index):
        previous=max(previous,min(1.,(len(estimable)-rank)*paired.at[index,'tost_p']))
        paired.at[index,'tost_p_holm']=previous
        paired.at[index,'equivalent_proposed_margin_holm']=previous<.05
    paired.to_csv(outdir/'box_igd_paired_effects.csv',index=False)
    plot_results(frame,outdir)
    after=hashes(study,outdir)
    if before!=after:raise RuntimeError('An existing study file changed during analysis')
    verification=dict(reference_solutions=len(ref),paired_seeds=20,fronts=60,
        complete_paired_keys=True,existing_study_files_unchanged=True,optimization_rerun=False,
        objective_boxes_built_with_existing_model=True,
        bvls_fallbacks=sum(d['bvls_fallback'] for d in solver_diagnostics),
        nominal_candidates_included=sum(d['nominal_candidate_included'] for d in solver_diagnostics),
        source_sha256=before,
        analysis_sha256={name:hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                         for name in ['analyze_box_igd.py','src/metrics/box_igd.py']})
    (outdir/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return frame,summary,paired,settings


def plot_results(frame,outdir):
    """Regenerate figures from saved per-seed measurements only."""
    outdir=Path(outdir)
    from plot_alsf_elsevier import style
    style()
    def distribution(ax,metric,label,percent=False):
        data=[frame.loc[frame.method==m,metric].to_numpy()*(100 if percent else 1) for m in METHODS]
        result=ax.boxplot(data,positions=range(3),widths=.45,patch_artist=True,showfliers=False,
                          medianprops={'color':'#222','linewidth':1.2})
        for i,(method,values) in enumerate(zip(METHODS,data)):
            result['boxes'][i].set(facecolor=COLORS[method],alpha=.22)
            ax.scatter(i+np.linspace(-.13,.13,20),values,s=10,c=COLORS[method],alpha=.85)
        ax.set_xticks(range(3),['NSGA-II','Resample','Drop']);ax.set_ylabel(label)
        ax.spines[['top','right']].set_visible(False)
    for space,title,unit in [('decision','Decision boxes: half-width h = 0.1 MA/m² (full width 0.2)','MA/m²'),
                             ('objective','Objective rectangles from J ±0.1 MA/m²; reference-range normalization','normalized')]:
        fig,axes=plt.subplots(2,2,figsize=(7.2,5.2),layout='constrained')
        for ax,metric,label,percent in [
            (axes[0,0],space+'_box_igd','Box-IGD ('+unit+')',False),
            (axes[0,1],space+'_tolerance_coverage','Tolerance coverage (%)',True),
            (axes[1,0],space+'_p95_box_distance','95th-percentile excess ('+unit+')',False),
            (axes[1,1],space+'_max_box_distance','Maximum excess ('+unit+')',False)]:distribution(ax,metric,label,percent)
        fig.suptitle(title,fontsize=10)
        for ext in ['png','pdf']:fig.savefig(outdir/f'box_igd_{space}_space.{ext}',dpi=320,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    for ax,metric,title,unit in [(axes[0],'decision_box_igd','10D half-width h = 0.1 MA/m²','MA/m²'),
                               (axes[1],'objective_box_igd','Uncertainty-induced objective rectangles','reference-normalized units')]:
        for method in METHODS:
            group=frame[frame.method==method]
            ax.scatter(group.actual_model_calls,group[metric],s=17,c=COLORS[method],label=LABELS[method],alpha=.75)
        ax.set(xlabel='Actual optimization model calls',ylabel='Box-IGD ('+unit+')',title=title)
    axes[0].legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(outdir/f'box_igd_vs_evaluations.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('docs/twenty_seed_study'))
    parser.add_argument('--outdir',type=Path,default=Path('docs/twenty_seed_study/box_igd'))
    args=parser.parse_args();frame,summary,paired,settings=analyze(args.study,args.outdir)
    write_report(args.outdir,frame,summary,paired,settings)
    print(summary[summary.metric.isin(MARGINS)].to_string(index=False))


def write_report(outdir,frame,summary,paired,settings):
    # Kept separate so tests can exercise computation without modifying source data.
    rows=[]
    for method in METHODS:
        g=frame[frame.method==method]
        rows.append(f'| {LABELS[method]} | {g.decision_box_igd.mean():.6f} / {g.decision_box_igd.median():.6f} | {100*g.decision_tolerance_coverage.mean():.2f}% / {100*g.decision_tolerance_coverage.median():.2f}% | {g.objective_box_igd.mean():.6f} / {g.objective_box_igd.median():.6f} | {100*g.objective_tolerance_coverage.mean():.2f}% / {100*g.objective_tolerance_coverage.median():.2f}% |')
    effect_rows=[]
    for metric in MARGINS:
        r=paired[(paired.metric==metric)&(paired.method_a=='baseline')&(paired.method_b=='drop')].iloc[0]
        effect_rows.append(f'| {metric} | {r.mean_difference:.7g} | [{r.ci95_low:.7g}, {r.ci95_high:.7g}] |')
    equivalence_rows=[]
    for metric in MARGINS:
        r=paired[(paired.metric==metric)&(paired.method_a=='baseline')&(paired.method_b=='drop')].iloc[0]
        result='not estimable (zero paired variance)' if pd.isna(r.tost_p) else ('passes' if r.equivalent_proposed_margin_holm else 'does not pass')
        equivalence_rows.append(f'| {metric} | ±{MARGINS[metric]} | {result} |')
    verification=json.loads((Path(outdir)/'verification.json').read_text())
    numerical_note=f"Across all reference points, BVLS fallbacks: {verification['bvls_fallbacks']}; nominal feasible-candidate inclusions: {verification['nominal_candidates_included']}. All fallback KKT checks passed. See per-reference solver diagnostics."
    base=frame[frame.method=='baseline'];drop=frame[frame.method=='drop'];resample=frame[frame.method=='resample']
    ed=paired[(paired.metric=='decision_box_igd')&(paired.method_a=='baseline')&(paired.method_b=='drop')].iloc[0]
    ef=paired[(paired.metric=='objective_box_igd')&(paired.method_a=='baseline')&(paired.method_b=='drop')].iloc[0]
    passed=paired[(paired.metric.isin(MARGINS))&(paired.method_a=='baseline')&(paired.method_b=='drop')&paired.equivalent_proposed_margin_holm]
    passed_text=', '.join(passed.metric) if len(passed) else 'none'
    saved_percent=100*(1-drop.actual_model_calls.mean()/base.actual_model_calls.mean())
    text=f'''# Tolerance-aware Box-IGD: existing 20-seed solenoid results

This is a supplementary **post-processing diagnostic**, not IGD+ and not a new
optimization experiment. The existing {settings['reference_solutions']} reference designs
and 60 final fronts (20 paired seeds per method) were used without modification.
Every reference row receives equal weight. Existing HV, IGD+ and actual model-call
values are retained and independently checked against the stored common reference.

## Definitions and units

For each reference design r, the decision-space half-width is **h_i=0.1** in all
ten coordinates x_i=J_i/(1 MA/m²). This corresponds to independent **±0.1 MA/m²**
physical current-density tolerance; the full box side is 0.2, not 0.1. The distance
is `max_i max(0,abs(p_i-r_i)-h_i)`, minimized over obtained designs and averaged
over reference rows. Coverage counts exactly zero distances, including boundaries.
Values approach symmetric decision-space L-infinity IGD at zero half-width, not
Euclidean IGD+. The diagnostic is not automatically Pareto-compliant.

Objective-space rectangles use the existing `box_objective_extrema` routine on
[r-0.1,r+0.1] intersected with [-2,2]^10. f1 minima are bounded least squares,
maxima exhaust 1,024 vertices; f2 extrema are separable. If the existing TRF
minimum solver does not converge, the analysis uses bounded BVLS with a checked
projected-gradient KKT residual, on the identical least-squares problem. This
fallback is logged per reference design. The nominal feasible design is included
as a minimum/maximum candidate, with no tolerance padding or hidden coverage epsilon. The existing routines
were previously validated and are regression-tested here. The lower/upper bounds
are normalized by **the reference objective ranges only**, identically for every
method: scales={settings['objective_scales']}, ideal={settings['objective_ideal']}.
Distance is `max_j max(lower_j-p_j,0,p_j-upper_j)` in these normalized units.
The rectangle is a **coordinatewise bounding approximation** to the joint
attainable objectives. Different coordinates' extrema may require different
perturbations; an interior vector need not be jointly attainable.

{numerical_note}

## Results

Each cell is **mean / median across 20 seeds**. Decision distances are numerically
in normalized-current units (MA/m²); objective distances are reference-normalized.

| Method | Decision Box-IGD | Decision coverage | Objective Box-IGD | Objective coverage |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Within-run 95th-percentile and maximum reference distances are in the per-seed
CSV and both distribution figures, alongside all 20 individual outcomes. These
are tail distances **within one reference set**, not confidence intervals over
seeds. HV, IGD+ and actual evaluation counts are also in that CSV.

Paired Drop-minus-baseline differences, with 20,000-resample paired-bootstrap
95% mean-difference intervals (fixed seed 20261116):

| Metric | Mean difference | Pointwise 95% CI |
|---|---:|---:|
{chr(10).join(effect_rows)}

All three method pairs and all requested per-run metrics are exported. Bootstrap
CIs are pointwise, not simultaneous multiplicity-adjusted intervals. A CI
containing zero is not proof of practical equivalence. Reference rows are not
bootstrap replicates: the independent paired unit is the optimizer seed.

## What the tolerance-aware results show

In decision space, baseline covers **{100*base.decision_tolerance_coverage.mean():.2f}%**
of reference boxes and Drop **{100*drop.decision_tolerance_coverage.mean():.2f}%**.
Mean excess distances are **{base.decision_box_igd.mean():.5f} versus
{drop.decision_box_igd.mean():.5f} MA/m²**. These are excesses beyond the allowed
box, not distances from its center. The paired Drop-minus-baseline difference is
{ed.mean_difference:.6f}, CI [{ed.ci95_low:.6f}, {ed.ci95_high:.6f}]. The
ill-conditioned field map makes design matching and performance matching
fundamentally different questions.

Objective-rectangle coverage averages **{100*base.objective_tolerance_coverage.mean():.2f}%
baseline, {100*resample.objective_tolerance_coverage.mean():.2f}% Resample and
{100*drop.objective_tolerance_coverage.mean():.2f}% Drop**. Mean objective Box-IGD
is **{base.objective_box_igd.mean():.6f} baseline versus
{drop.objective_box_igd.mean():.6f} Drop**; the paired difference is
{ef.mean_difference:.6f}, CI [{ef.ci95_low:.6f}, {ef.ci95_high:.6f}]. Interpret
this interval on its own scale; neither matching nor a nonsignificant difference
establishes guaranteed robustness or general practical equivalence.

The tails still matter: the mean within-run maximum objective excess is
**{base.objective_max_box_distance.mean():.4f} baseline and
{drop.objective_max_box_distance.mean():.4f} Drop**. Drop's worst run has a
maximum of **{drop.objective_max_box_distance.max():.4f}** reference-normalized
units. Small means can hide missed reference-endpoint regions. All reference
distances are retained; no endpoint was trimmed for a favorable score.

The following Drop-versus-baseline outcomes pass the illustrative, Holm-adjusted
equivalence margins: **{passed_text}**. Outcomes not listed do not pass the
adjusted family test (or are not estimable). This is not an overall engineering-
equivalence certificate: the margins are proposals, and the objective rectangles
bound rather than exactly characterize the jointly attainable uncertainty set.
The measured evaluation saving is **{saved_percent:.2f}%**; its engineering
acceptability must consider all metrics, tails and approved acceptance margins.

## Explicit, illustrative equivalence analysis

Engineering acceptance margins were not supplied. The following are **proposals
fixed in the analysis code before computing new Box metrics**, not validated
engineering specifications: mean decision Box-IGD difference ±0.01 MA/m² (10% of
one physical half-width); mean objective Box-IGD difference ±0.001 normalized
units (0.1% of one reference range); coverage difference ±0.05 (five percentage
points). The same definitions apply to every pair. They require engineering
approval and cannot establish physical equivalence by themselves.

An exploratory paired t-TOST at alpha=0.05 evaluates the paired population mean
against both sides of each margin, assuming a t approximation for the paired
mean. Its matching 90% t-interval and p-value are saved. Holm correction covers
all estimable proposed-margin tests across both spaces and three method pairs.
Zero paired variance is marked not estimable, rather than inventing population
certainty from constant observations. This is a separate equivalence question,
not a reinterpretation of nonsignificance in a difference test.

| Drop versus baseline outcome | Proposed symmetric margin | Holm-adjusted TOST result |
|---|---:|---|
{chr(10).join(equivalence_rows)}

## Interpretation constraints

Decision matching within manufacturing tolerance does not guarantee identical
objective performance or robustness. Conversely, the ill-conditioned solenoid
field matrix permits different current distributions with similar objectives,
so poor decision-box matching need not imply poor objective performance.
Objective-rectangle matching is more permissive and can hide differences through
its joint-set bounding approximation. Neither diagnostic overrides the previous
exact-sensitivity result that sampled final designs are sensitive at the 10%
criterion. These metrics supplement, rather than replace, conventional HV/IGD+.
The mean optimization call saving of Drop is {saved_percent:.2f}%.

The stored reference is adaptively sampled in weight/objective space, not uniformly
in decision coordinates or arclength. The requested uniform stored-row average
therefore reflects that sampling density, including its low-error endpoint.
`reference_distribution.csv`, per-reference distances and rectangle bounds allow
audit of this effect. No method-specific normalization or favorable relabeling
was performed. Differences and tail distances must be considered alongside
coverage; Box-IGD alone is not an engineering acceptance decision.

## Reproduce and audit

```
python analyze_box_igd.py --study docs/twenty_seed_study --outdir docs/twenty_seed_study/box_igd
```

The output directory must not already exist. Use a new `--outdir` for repeat
analysis. No NSGA-II optimization or new Pareto-front construction is launched. The only new model work
is independent post-processing of reference uncertainty extrema and consistency
checks of stored objective values; these calls are not optimization evaluations.
All pre-existing study files are SHA-256 checked before/after analysis.

- [Per-seed results](box_igd_per_seed.csv), [summaries](box_igd_summary.csv),
  [paired effects/equivalence checks](box_igd_paired_effects.csv).
- [Decision-space PNG](box_igd_decision_space.png) / [PDF](box_igd_decision_space.pdf).
- [Objective-space PNG](box_igd_objective_space.png) / [PDF](box_igd_objective_space.pdf).
- [Box-IGD versus model calls PNG](box_igd_vs_evaluations.png) / [PDF](box_igd_vs_evaluations.pdf).
- [Exact rectangle bounds](objective_tolerance_rectangles.csv),
  [per-reference solver diagnostics](objective_extrema_solver_diagnostics.csv),
  [settings and proposed margins](settings.json), [source integrity](verification.json).
- `per_reference_distances.npz`: ordered `run_keys` and decision/objective arrays,
  each of shape (60, number_of_reference_rows).
'''
    (Path(outdir)/'REPORT.md').write_text(text)

if __name__=='__main__':main()
