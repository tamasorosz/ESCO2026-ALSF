"""Measured baseline/Resample/Drop comparison, with explicit evaluation budgets."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.drop_metrics import indicators,polyline_distances

COLORS={'baseline':'#356B9A','resample':'#C17B3A','drop':'#237C69'}
LABELS={'baseline':'NSGA-II','resample':'ALSF-Resample','drop':'ALSF-Drop'}


def compare(study,out):
    out.mkdir(parents=True,exist_ok=False)
    from plot_alsf_elsevier import style
    style()
    reference=pd.read_csv(study/'reference/reference.csv')
    protocol=json.loads((study/'protocol.json').read_text());target=protocol['target_hv']
    runs=[];rows=[];attainments=[];checkpoints=[]
    for directory in sorted((study/'full').glob('*_seed*')):
        mode,seed=directory.name.split('_seed');seed=int(seed)
        meta=json.loads((directory/'summary.json').read_text())
        h=pd.read_csv(directory/'history.csv');front=pd.read_csv(directory/'pareto.csv')
        metrics=indicators(front[['Obj1','Obj2']],reference)
        calls=int(meta.get('actual_model_calls',meta.get('model_calls')))
        hit=h.loc[h.hypervolume>=target]
        distance=polyline_distances(front[['Obj1','Obj2']],reference)
        exact=pd.read_csv(directory/'pareto_sensitivity.csv')
        rows.append(dict(mode=mode,seed=seed,actual_model_calls=calls,**metrics,
            reference_distance_mean=float(distance.mean()),run_seconds=meta['run_seconds'],
            archive_processing_seconds=meta['archive_processing_seconds'],
            exact_sensitive_fraction=float(exact.exact_sensitive.mean()),
            median_R1=float(exact.R1_exact.median()),median_R2=float(exact.R2_exact.median()),
            saved_evaluations=int(meta.get('saved_evaluations',0)),
            total_dropped=int(meta.get('total_dropped',0)),drop_fraction=float(meta.get('drop_fraction',0))))
        attainments.append(dict(mode=mode,seed=seed,target_hv=target,reached=bool(len(hit)),
            first_attainment_evaluations=int(hit.iloc[0].evaluations) if len(hit) else np.nan,
            first_attainment_generation=int(hit.iloc[0].generation) if len(hit) else np.nan,
            censoring_evaluations=calls))
        runs.append((mode,seed,h,front,directory))
    result=pd.DataFrame(rows)
    for seed in protocol['seeds']:
        if set(result.loc[result.seed==seed,'mode'])!=set(COLORS):raise ValueError('Unpaired comparison')
    # A fixed allowed budget uses only completed algorithm states at <= budget.
    # Batches have different sizes: the actual checkpoint and unused budget are
    # exported, never interpolated or counted as additional ALSF savings.
    shared=int(result.actual_model_calls.min())
    for mode,seed,h,front,directory in runs:
        for budget in sorted(set(protocol['evaluation_checkpoints']+[shared])):
            row=h.loc[h.evaluations<=budget].iloc[-1]
            checkpoints.append(dict(mode=mode,seed=seed,allowed_evaluation_budget=budget,
                actual_checkpoint_evaluations=int(row.evaluations),unused_budget=budget-int(row.evaluations),
                generation=int(row.generation),hypervolume=row.hypervolume,igd_plus=row.igd_plus))
    result.to_csv(out/'fixed_generations.csv',index=False)
    pd.DataFrame(attainments).to_csv(out/'target_hv_attainment.csv',index=False)
    pd.DataFrame(checkpoints).to_csv(out/'fixed_evaluation_checkpoints.csv',index=False)
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=320,bbox_inches='tight')
        plt.close(fig)
    # Actual evaluation axes and unsmoothed per-seed trajectories.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    seen=set()
    for mode,seed,h,front,directory in runs:
        for ax,col in zip(axes,['hypervolume','igd_plus']):
            ax.plot(h.evaluations,h[col],c=COLORS[mode],lw=.9,alpha=.65,
                    label=LABELS[mode] if mode not in seen else None)
            ax.set_xlabel('Actual objective evaluations')
        seen.add(mode)
    axes[0].axhline(target,c='.3',ls='--',lw=.8,label='Predefined target HV')
    axes[0].set_ylabel('Hypervolume');axes[1].set_ylabel('IGD+');axes[1].set_yscale('log')
    axes[0].legend(fontsize=7);save(fig,'01_quality_vs_actual_evaluations')
    # Reference / final fronts at the same 200-generation budget.
    fig,ax=plt.subplots(figsize=(6,3.5),layout='constrained')
    ax.plot(reference.Obj1,reference.Obj2,c='.2',lw=1,label='Convex reference')
    seen=set()
    for mode,seed,h,front,directory in runs:
        ax.scatter(front.Obj1,front.Obj2,s=7,c=COLORS[mode],alpha=.5,
                   label=LABELS[mode] if mode not in seen else None)
        seen.add(mode)
    ax.set(xlabel='Field-error objective',ylabel='Loss objective');ax.legend(fontsize=7)
    save(fig,'02_final_fronts_fixed_generations')
    # Target attainment: marks indicate actual calls at first measured crossing.
    fig,ax=plt.subplots(figsize=(6,3.2),layout='constrained')
    attainment=pd.DataFrame(attainments)
    for i,mode in enumerate(COLORS):
        data=attainment[attainment['mode']==mode]
        for _,row in data.iterrows():
            ax.scatter(row.first_attainment_evaluations if row.reached else row.censoring_evaluations,
                i+(row.seed-42)*.001,c=COLORS[mode],marker='o' if row.reached else '>',s=25)
        misses=int((~data.reached).sum())
        ax.text(1.01,i,f'{misses}/{len(data)} not reached',transform=ax.get_yaxis_transform(),va='center',fontsize=8)
    ax.set_yticks(range(3),[LABELS[x] for x in COLORS]);ax.set_xlabel('Actual evaluations ( > = censored)')
    ax.set_title('Evaluations to the predefined target hypervolume')
    save(fig,'03_evaluations_to_target_hv')
    # Savings and per-generation true drop activity.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
    for mode,seed,h,front,directory in runs:
        if mode!='drop':continue
        p=pd.read_csv(directory/'process.csv')
        axes[0].plot(p.generation,p.cumulative_dropped,lw=1,label=f'Seed {seed}')
        axes[1].plot(p.generation,100*p.n_dropped/p.n_proposed,alpha=.6,lw=.8)
    axes[0].set(ylabel='Cumulative saved evaluations',xlabel='Generation');axes[0].legend()
    axes[1].set(ylabel='Dropped proposals (%)',xlabel='Generation')
    save(fig,'04_true_drop_savings')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    for i,mode in enumerate(COLORS):
        data=result.loc[result['mode']==mode]
        axes[0].scatter(np.full(len(data),i),data.hypervolume,c=COLORS[mode],s=20)
        axes[1].scatter(np.full(len(data),i),data.actual_model_calls,c=COLORS[mode],s=20)
    for ax in axes:ax.set_xticks(range(3),['NSGA-II','Resample','Drop'])
    axes[0].set_ylabel('Final hypervolume');axes[1].set_ylabel('Actual evaluations')
    save(fig,'05_quality_and_evaluations_fixed_generations')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    args=p.parse_args();compare(args.study,args.outdir)
if __name__=='__main__':main()
