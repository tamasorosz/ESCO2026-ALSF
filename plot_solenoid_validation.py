"""Figures A-E from measured phased solenoid-validation artifacts only."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solenoid_validation import normalization,indicators

COLORS={'baseline':'#356B9A','alsf':'#23846F','reference':'#343434'}


def save(fig,out,name):
    for ext in ('png','pdf'):
        fig.savefig(out/f'{name}.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)


def plot_study(study,out):
    out.mkdir(parents=True,exist_ok=False)
    from plot_alsf_elsevier import style
    style()
    reference=pd.read_csv(study/'reference/reference.csv')
    ideal,span=normalization(reference)
    runs=[];rows=[]
    for directory in sorted((study/'full').glob('*_seed*')):
        summary=json.loads((directory/'summary.json').read_text())
        runs.append((directory,summary))
        rows.append({k:v for k,v in summary.items() if not isinstance(v,(dict,list))})
    if not runs:raise ValueError('No measured full runs found')
    summary=pd.DataFrame(rows);summary.to_csv(out/'run_summary.csv',index=False)
    paired=[]
    for seed,group in summary.groupby('seed'):
        if set(group['mode'])!={'baseline','alsf'}:raise ValueError('Unpaired seed')
        a=group.set_index('mode').loc['alsf'];b=group.set_index('mode').loc['baseline']
        paired.append(dict(seed=int(seed),delta_hypervolume=a.hypervolume-b.hypervolume,
            delta_igd_plus=a.igd_plus-b.igd_plus,delta_reference_distance=a.reference_distance_mean-b.reference_distance_mean,
            delta_sensitive_fraction=a.exact_sensitive_fraction-b.exact_sensitive_fraction,
            delta_R1_median=a.exact_R1_median-b.exact_R1_median,
            delta_R2_median=a.exact_R2_median-b.exact_R2_median,runtime_ratio=a.run_seconds/b.run_seconds))
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    coarse=pd.read_csv(study/'reference/reference_coarse.csv')
    resolutions=[]
    for directory,info in runs:
        f=pd.read_csv(directory/'pareto.csv')[['Obj1','Obj2']]
        a=indicators(f,reference);b=indicators(f,coarse)
        resolutions.append(dict(run=directory.name,delta_hv_fine_minus_coarse=a['hypervolume']-b['hypervolume'],
                                delta_igd_fine_minus_coarse=a['igd_plus']-b['igd_plus']))
    pd.DataFrame(resolutions).to_csv(out/'indicator_reference_resolution.csv',index=False)
    # A: full linear overview retains endpoints; log detail explicitly omits zero.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    curve=(reference[['Obj1','Obj2']].to_numpy()-ideal)/span
    for ax in axes:
        ax.plot(curve[:,0],curve[:,1],color=COLORS['reference'],lw=1.2,label='Convex reference')
        seen=set()
        for directory,info in runs:
            mode=info['mode'];front=pd.read_csv(directory/'pareto.csv')
            f=(front[['Obj1','Obj2']].to_numpy()-ideal)/span
            ax.scatter(f[:,0],f[:,1],s=6,alpha=.45,color=COLORS[mode],
                marker='o' if mode=='baseline' else 'x',linewidths=.5,
                label=('NSGA-II' if mode=='baseline' else 'ALSF') if mode not in seen else None)
            seen.add(mode)
        ax.set(xlabel=r'Normalized field error $f_1$',ylabel=r'Normalized loss $f_2$')
    axes[0].set_title('(a) Full trade-off');axes[1].set_title('(b) Positive-error detail')
    axes[1].set_xscale('log');axes[1].set_xlim(1e-8,1.05)
    axes[0].legend();save(fig,out,'A_reference_and_final_fronts')
    # B: evaluation budgets, median and IQR across actual seeds.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
    for mode in ('baseline','alsf'):
        histories=[pd.read_csv(d/'history.csv') for d,i in runs if i['mode']==mode]
        evaluations=histories[0].evaluations.to_numpy()
        if any(not np.array_equal(h.evaluations,evaluations) for h in histories):raise ValueError('Unmatched evaluation axes')
        for ax,col in zip(axes,['hypervolume','igd_plus']):
            values=np.array([h[col].to_numpy() for h in histories])
            ax.plot(evaluations,np.median(values,axis=0),c=COLORS[mode],label=mode.upper() if mode=='alsf' else 'NSGA-II')
            ax.fill_between(evaluations,*np.quantile(values,[.25,.75],axis=0),color=COLORS[mode],alpha=.15)
            ax.set_xlabel('Objective evaluations')
    axes[0].axhline(indicators(reference[['Obj1','Obj2']],reference)['hypervolume'],c='.4',lw=.7,ls=':',label='Reference HV')
    axes[0].set(title='(a) Hypervolume',ylabel='HV (fixed normalization)')
    axes[1].set(title='(b) IGD+',ylabel='IGD+ (fixed reference)',yscale='log')
    axes[0].legend();axes[1].legend();save(fig,out,'B_convergence_vs_evaluations')
    # C: independently solved extrema on the sampled reference.
    audit=pd.read_csv(study/'sensitivity/reference_sensitivity.csv').sort_values('Obj1')
    fig,ax=plt.subplots(figsize=(5.8,3.2),layout='constrained')
    for col,label,color in [('R1_exact',r'$R_1$ (field error)','#B46C36'),('R2_exact',r'$R_2$ (loss)','#356B9A')]:
        ax.plot((audit.Obj1-ideal[0])/span[0],100*audit[col],'.-',c=color,ms=3,lw=1,label=label)
    ax.axhline(10,color='.3',ls='--',lw=.8,label='10% threshold')
    ax.set(xlabel='Normalized reference field error',ylabel='Exact worst relative variation (%)',
           title=f'{len(audit)} sampled convex-reference designs')
    ax.legend();save(fig,out,'C_exact_reference_sensitivity')
    # D: independently audited optimization archives, with explicit sampling.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained')
    for ax,mode in zip(axes,('baseline','alsf')):
        directories=[d for d,i in runs if i['mode']==mode]
        files=[study/'archive_audit'/(d.name+'.csv') for d in directories]
        stratified=all(f.exists() for f in files)
        if not stratified:
            files=[d/'pareto_sensitivity.csv' for d in directories]
        data=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
        table=np.array([[int(((data.flag==flag)&(data.exact_sensitive==actual)).sum())
            for flag in ['robust','non-robust','NEI']] for actual in [False,True]])
        ax.imshow(table,cmap='Blues',vmin=0,vmax=max(int(table.max()),1),aspect='auto')
        for (i,j),n in np.ndenumerate(table):
            ax.text(j,i,str(n),ha='center',va='center',color='white' if n>table.max()/2 else '#222')
        ax.set_xticks([0,1,2],['Stable','Sensitive','NEI'])
        ax.set_yticks([0,1],['Within threshold','Sensitive'])
        label='NSGA-II' if mode=='baseline' else 'ALSF'
        selection='stratified archive audit' if stratified else 'Pareto audit'
        ax.set(xlabel='Empirical archive label',ylabel='Exact box test',
               title=f'{label}: {selection} (n={len(data)})')
    save(fig,out,'D_empirical_vs_exact')
    # E: evidence coverage and real repair activity; no fabricated warmup labels.
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained')
    for directory,info in runs:
        if info['mode']!='alsf':continue
        p=pd.read_csv(directory/'process.csv');valid=p.evidence_available.astype(bool)
        known=p.loc[valid]
        axes[0,0].plot(p.generation,100*p.n_regenerated/p.n_offspring_seen,c=COLORS['alsf'],alpha=.45,lw=.8)
        for col,color,label in [('n_nei','#9DA5AE','NEI'),('n_non_robust','#B46C36','Verified sensitive'),('n_robust',COLORS['alsf'],'Stable')]:
            axes[0,1].plot(known.generation,100*known[col]/known.n_archive,c=color,alpha=.6,lw=.8,
                          label=label if directory==next(d for d,i in runs if i['mode']=='alsf') else None)
        axes[1,0].plot(known.generation,known.neighbor_p95,c=COLORS['alsf'],alpha=.5,lw=.8)
        axes[1,1].plot(known.generation,known.n_clusters,c=COLORS['alsf'],alpha=.5,lw=.8)
    axes[0,0].set(title='(a) Regenerated candidates',ylabel='Fraction checked (%)')
    axes[0,1].set(title='(b) Local evidence coverage',ylabel='Archive samples (%)',ylim=(0,100));axes[0,1].legend()
    axes[1,0].set(title='(c) Local observation counts',ylabel='95th percentile (includes self)')
    axes[1,1].set(title='(d) DBSCAN connectivity',ylabel='Components')
    for ax in axes.ravel():ax.set_xlabel('Generation')
    save(fig,out,'E_repair_and_evidence')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    args=p.parse_args();plot_study(args.study,args.outdir)
if __name__=='__main__':main()
