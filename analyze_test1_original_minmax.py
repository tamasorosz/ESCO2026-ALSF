"""Paired 20-seed analysis: real outcomes, bootstrap CIs and Holm adjustment."""
from pathlib import Path
from itertools import combinations
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.drop_metrics import indicators

ORDER=['baseline','resample','drop']
LABEL={'baseline':'NSGA-II','resample':'ALSF-Resample','drop':'ALSF-Drop'}
COLOR={'baseline':'#356B9A','resample':'#C17B3A','drop':'#237C69'}


def paired_effect(a,b,draws=10000):
    difference=np.asarray(b)-np.asarray(a)
    rng=np.random.default_rng(20261115)
    bootstrap=difference[rng.integers(0,len(difference),(draws,len(difference)))].mean(axis=1)
    p=1. if np.all(difference==0) else float(wilcoxon(difference,alternative='two-sided').pvalue)
    return dict(mean_difference=float(difference.mean()),median_difference=float(np.median(difference)),
        ci95_low=float(np.quantile(bootstrap,.025)),ci95_high=float(np.quantile(bootstrap,.975)),p_raw=p)


def holm(pvalues):
    pvalues=np.asarray(pvalues);order=np.argsort(pvalues);adjusted=np.empty(len(order));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(order)-rank)*pvalues[index]));adjusted[index]=previous
    return adjusted


def analyze(root,out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text());protocol['benchmarks']=['test1']
    expected={(b,m,s) for b in protocol['benchmarks'] for m in protocol['modes'] for s in protocol['seeds']}
    summaries=[];histories={};attainment=[];checkpoints=[]
    for path in (root/'runs').glob('*/*/seed*/summary.json'):
        r=json.loads(path.read_text());key=(r['benchmark'],r['mode'],r['seed'])
        if key not in expected:continue
        process=pd.read_csv(path.parent/'process.csv')
        rejected_initial=int(process.n_forbidden_candidates.sum())
        repair_inputs=int(process.n_offspring_seen.sum())
        replacement_checks=int(process.n_attempts.sum())
        resample_rejected_checks=rejected_initial+replacement_checks-int(process.n_regenerated.sum())
        summaries.append({**{k:v for k,v in r.items() if not isinstance(v,(dict,list))},
            **{'archive_'+k:v for k,v in r['final_labels'].items()},
            'rejected_fraction':r['dropped']/r['proposed'] if key[1]=='drop' else (rejected_initial/repair_inputs if key[1]=='resample' else 0.),
            'repair_inputs':repair_inputs,'replacement_checks':replacement_checks,
            'resample_total_rejected_checks':resample_rejected_checks,
            'resample_rejected_check_fraction':resample_rejected_checks/(repair_inputs+replacement_checks),
            'resample_forbidden_checks':int(pd.read_csv(path.parent/'process.csv').n_forbidden_candidates.sum())})
        h=pd.read_csv(path.parent/'history.csv');histories[key]=h
        ref=pd.read_csv(root/'reference'/f'{key[0]}.csv')
        reference_hv=protocol['reference_hv']
        for fraction in [.99,.995,.999]:
            target=fraction*reference_hv;hit=h[h.hypervolume>=target]
            attainment.append(dict(benchmark=key[0],mode=key[1],seed=key[2],target_fraction=fraction,target_hv=target,
                reached=bool(len(hit)),first_evaluations=float(hit.iloc[0].evaluations) if len(hit) else np.nan,
                censoring_evaluations=r['actual_model_calls']))
    if set(histories)!=expected:raise ValueError(f'Missing or extra run keys: {expected-set(histories)}')
    frame=pd.DataFrame(summaries).sort_values(['benchmark','seed','mode'])
    frame.to_csv(out/'all_runs.csv',index=False)
    all_hits=pd.DataFrame(attainment);all_hits.to_csv(out/'target_hv_attainment.csv',index=False)
    hits=all_hits[all_hits.target_fraction==.995].copy()
    descriptive=[];effects=[]
    for (benchmark,mode),group in frame.groupby(['benchmark','mode']):
        for metric in ['actual_model_calls','hypervolume','igd_plus','mean_reference_distance','run_seconds','sensitive_audit_fraction','R1_median','R2_median']:
            values=group[metric].to_numpy()
            descriptive.append(dict(benchmark=benchmark,mode=mode,metric=metric,n=len(values),
                mean=float(values.mean()),std=float(values.std(ddof=1)),median=float(np.median(values)),
                q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),
                minimum=float(values.min()),maximum=float(values.max())))
    pd.DataFrame(descriptive).to_csv(out/'descriptive.csv',index=False)
    for benchmark in protocol['benchmarks']:
        sub=frame[frame.benchmark==benchmark]
        for metric in ['hypervolume','igd_plus','actual_model_calls']:
            pivot=sub.pivot(index='seed',columns='mode',values=metric).sort_index()
            for a,b in combinations(ORDER,2):
                if a in pivot and b in pivot:
                    effects.append(dict(benchmark=benchmark,metric=metric,method_a=a,method_b=b,
                        difference='B minus A',n_pairs=len(pivot),**paired_effect(pivot[a],pivot[b])))
        shared=int(sub.actual_model_calls.min())
        for (bench,mode,seed),h in histories.items():
            if bench!=benchmark:continue
            for budget in sorted(set(v for v in [5000,10000,15000,shared] if v<=shared)):
                if budget>h.evaluations.iloc[-1]:continue
                row=h[h.evaluations<=budget].iloc[-1]
                checkpoints.append(dict(benchmark=bench,mode=mode,seed=seed,budget=budget,
                    actual_evaluations=int(row.evaluations),unused_budget=budget-int(row.evaluations),
                    hypervolume=row.hypervolume,igd_plus=row.igd_plus))
    effects=pd.DataFrame(effects);effects['p_holm']=holm(effects.p_raw)
    effects.to_csv(out/'paired_effects.csv',index=False)
    pd.DataFrame(checkpoints).to_csv(out/'evaluation_checkpoints.csv',index=False)
    # Target effects only for seeds where BOTH methods actually attained it.
    # Never replace a censored observation with its evaluation limit.
    target_pairs=[]
    for benchmark in protocol['benchmarks']:
        sub=hits[hits.benchmark==benchmark]
        pivot=sub.pivot(index='seed',columns='mode',values='first_evaluations')
        for a,b in combinations(ORDER,2):
            if a not in pivot or b not in pivot:continue
            both=pivot[[a,b]].dropna()
            row=dict(benchmark=benchmark,method_a=a,method_b=b,n_total=len(pivot),n_both_reached=len(both),
                     interpretation='Conditional on both reaching; selection-biased if censored. Descriptive only.')
            if len(both):row.update(mean_difference=float((both[b]-both[a]).mean()),median_difference=float((both[b]-both[a]).median()))
            target_pairs.append(row)
    pd.DataFrame(target_pairs).to_csv(out/'conditional_target_differences.csv',index=False)
    from plot_alsf_elsevier import style
    style()
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=320,bbox_inches='tight')
        plt.close(fig)
    for benchmark in protocol['benchmarks']:
        sub=frame[frame.benchmark==benchmark];shared=int(sub.actual_model_calls.min())
        # Common allowed evaluation budgets, step hold from observed complete
        # generations only. Every curve retains the same full 20-seed cohort.
        grid=np.arange(int(sub.pop_size.min()),shared+1,min(100,int(sub.pop_size.min())))
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
        for mode in ORDER:
            if mode not in protocol['modes']:continue
            for ax,metric in zip(axes,['hypervolume','igd_plus']):
                values=[]
                for seed in protocol['seeds']:
                    h=histories[(benchmark,mode,seed)]
                    indices=np.searchsorted(h.evaluations.to_numpy(),grid,side='right')-1
                    values.append(h[metric].to_numpy()[indices])
                values=np.array(values)
                ax.plot(grid,np.median(values,axis=0),c=COLOR[mode],label=LABEL[mode],lw=1.2)
                ax.fill_between(grid,*np.quantile(values,[.25,.75],axis=0),color=COLOR[mode],alpha=.15)
                ax.set_xlabel('Allowed objective-evaluation budget')
        axes[0].set_ylabel('Hypervolume');axes[1].set_ylabel('IGD+');axes[1].set_yscale('log')
        axes[0].legend(fontsize=7);save(fig,f'{benchmark}_01_convergence_median_IQR')
        fig,axes=plt.subplots(1,3,figsize=(8,3),layout='constrained')
        for ax,metric,title in zip(axes,['hypervolume','igd_plus','actual_model_calls'],['Final HV','Final IGD+','Actual evaluations']):
            data=[sub.loc[sub['mode']==m,metric].to_numpy() for m in ORDER if m in protocol['modes']]
            modes=[m for m in ORDER if m in protocol['modes']]
            boxes=ax.boxplot(data,positions=np.arange(len(modes)),widths=.45,patch_artist=True,showfliers=False)
            for box,mode in zip(boxes['boxes'],modes):box.set_facecolor(COLOR[mode]);box.set_alpha(.25)
            for i,(mode,values) in enumerate(zip(modes,data)):
                ax.scatter(i+np.linspace(-.13,.13,len(values)),values,s=8,c=COLOR[mode],alpha=.8)
            ax.set_xticks(range(len(modes)),['NSGA-II' if m=='baseline' else m.capitalize() for m in modes],rotation=20)
            ax.set_title(title)
        save(fig,f'{benchmark}_02_final_distributions')
        fig,ax=plt.subplots(figsize=(6.7,3),layout='constrained')
        for i,mode in enumerate(ORDER):
            group=hits[(hits.benchmark==benchmark)&(hits['mode']==mode)].sort_values('seed')
            for j,(_,r) in enumerate(group.iterrows()):
                ax.scatter(r.first_evaluations if r.reached else r.censoring_evaluations,
                    i+(j-(len(group)-1)/2)*.018,c=COLOR[mode],s=14,marker='o' if r.reached else '>')
            ax.text(1.01,i,f'{int(group.reached.sum())}/{len(group)} reached',transform=ax.get_yaxis_transform(),va='center',fontsize=8)
        ax.set_yticks(range(3),[LABEL[m] for m in ORDER]);ax.set_xlabel('Evaluations to predefined target HV ( > = censored)')
        save(fig,f'{benchmark}_03_target_attainment')
        fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
        pivot_hv=sub.pivot(index='seed',columns='mode',values='hypervolume')
        pivot_igd=sub.pivot(index='seed',columns='mode',values='igd_plus')
        pivot_calls=sub.pivot(index='seed',columns='mode',values='actual_model_calls')
        for mode in ['resample','drop']:
            if mode not in protocol['modes']:continue
            saved=100*(pivot_calls.baseline-pivot_calls[mode])/pivot_calls.baseline
            axes[0].scatter(saved,100*(pivot_hv[mode]-pivot_hv.baseline)/pivot_hv.baseline,
                            c=COLOR[mode],s=16,label=LABEL[mode])
            axes[1].scatter(saved,pivot_igd[mode]-pivot_igd.baseline,c=COLOR[mode],s=16)
        for ax in axes:ax.axhline(0,c='.6',lw=.7);ax.set_xlabel('Evaluation savings vs baseline (%)')
        axes[0].set_ylabel('Final HV change (%)');axes[1].set_ylabel('Final IGD+ change');axes[0].legend(fontsize=7)
        save(fig,f'{benchmark}_04_savings_quality_tradeoff')
    return frame,effects


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    args=p.parse_args();frame,effects=analyze(args.root,args.outdir)
    from report_test1_minmax import supplement
    supplement(args.root,args.outdir,frame,effects)
if __name__=='__main__':main()
