#!/usr/bin/env python3
"""Recreate the three Test 1 ALSF figures from actual 2-D archive/process CSV files.

Example:
  python plot_three_figures.py --archive results/test1_alsf_seed42_full/archive.csv \
      --process results/test1_alsf_seed42_full/process.csv --outdir figures/test1

No optimization is performed. Outputs reflect the supplied run, not a screenshot.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


COLUMN_ALIASES = {
    'generation': ['generation', 'gen', 'n_gen'],
    'robust': ['n_robust_clusters', 'robust_clusters', 'n_robust', 'robust'],
    'nonrobust': ['n_non_robust_clusters', 'n_nonrobust_clusters',
                  'non_robust_clusters', 'n_non_robust', 'non_robust', 'non-robust'],
    'regenerated': ['n_regenerated', 'regenerated', 'n_regenerated_offspring',
                    'regenerated_offspring', 'offspring_regenerated', 'rejected_offspring'],
    'attempts': ['n_attempts', 'attempts', 'resample_attempts',
                 'n_resample_attempts', 'resampling_attempts'],
}


def resolve(df: pd.DataFrame, variants, required=True):
    """Resolve historic and Pavel-version process column aliases."""
    lower={str(c).strip().lower():c for c in df.columns}
    for v in variants:
        if v.lower() in lower:
            return lower[v.lower()]
    if required:
        raise ValueError(f'Missing column from {variants}; found {list(df.columns)}')
    return None


def read_archive(path: Path) -> pd.DataFrame:
    df=pd.read_csv(path)
    for col in ['X1','X2','Obj1','Obj2']:
        if col not in df:
            raise ValueError(f'{path}: missing {col}; available: {list(df.columns)}')
    # This figure is for the ORIGINAL 2-D test, not a 10-D projection.
    xs=sorted((c for c in df if re.fullmatch(r'X\d+',str(c))),key=lambda s:int(s[1:]))
    if xs != ['X1','X2']:
        raise ValueError(f'Expected 2 decision variables (X1, X2), found {xs}. '
                         'This script must not be used on the 10-D TEAM35 archive.')
    if 'flag' not in df:
        raise ValueError('Archive has no flag column. Run ALSF and its callback first.')
    df=df.copy()
    df['flag']=df.flag.astype(str).str.strip().str.lower().str.replace('_','-',regex=False)
    df['flag']=df.flag.replace({'nonrobust':'non-robust', 'nei':'NEI',
                                'insufficiently explored':'NEI','noise':'NEI'})
    allowed={'NEI','robust','non-robust'}
    invalid=sorted(set(df.flag)-allowed)
    if invalid:
        raise ValueError(f'Unknown flags: {invalid}; expected {sorted(allowed)}')
    if not np.isfinite(df[['X1','X2','Obj1','Obj2']].to_numpy(dtype=float)).all():
        raise ValueError('Archive contains non-finite decision/objective values')
    return df


def representative_indices(df: pd.DataFrame) -> np.ndarray:
    """Find medoids of robust points IN each DBSCAN component.

    For old ALSF honour explicit is_medoid when it refers to a robust row.
    For Pavel version, is_medoid is for the whole mixed component, so it need
    not be robust: compute the medoid restricted to robust points instead.
    This is used for plotting ONLY, not for determining robustness.
    """
    if 'cluster_id' not in df:
        return np.array([],dtype=int)
    result=[]
    cls=pd.to_numeric(df['cluster_id'],errors='coerce').fillna(-1).to_numpy()
    robust=(df.flag=='robust').to_numpy()
    X=df[['X1','X2']].to_numpy(dtype=float)
    medoid=(df['is_medoid'].astype(str).str.lower().isin(['true','1','yes']).to_numpy()
            if 'is_medoid' in df else np.zeros(len(df),dtype=bool))
    for cid in np.unique(cls[cls >= 0]):
        members=np.flatnonzero(cls == cid)
        # The *entire* component must be robust. A locally robust point
        # inside a mixed/non-robust component is not a robust medoid.
        if not robust[members].all():
            continue
        if 'cluster_status' in df and not (df.iloc[members]['cluster_status'] == 'robust').all():
            continue
        ids=members
        old_med=ids[medoid[ids]]
        if len(old_med)==1:
            result.append(int(old_med[0]))
            continue
        pts=X[ids]
        if len(ids)>1500:
            # Avoid quadratic memory for exceptionally large clusters.
            mu=np.mean(pts,axis=0)
            result.append(int(ids[np.argmin(np.max(np.abs(pts-mu),axis=1))]))
        else:
            result.append(int(ids[np.argmin(cdist(pts,pts,metric='chebyshev').sum(axis=1))]))
    return np.asarray(result,dtype=int)


def prepare_process(path: Path) -> pd.DataFrame:
    raw=pd.read_csv(path)
    c_gen=resolve(raw,COLUMN_ALIASES['generation'])
    gen=pd.to_numeric(raw[c_gen],errors='raise').astype(int).to_numpy()
    if len(np.unique(gen))!=len(gen):
        raise ValueError('Process has duplicate generation indices')
    result=pd.DataFrame({'generation':gen})
    for key in ('robust','nonrobust','regenerated','attempts'):
        # Missing repair statistics must NEVER be silently replaced with
        # zeros: that would disguise a logging failure as zero regeneration.
        col=resolve(raw,COLUMN_ALIASES[key],required=True)
        vals=pd.to_numeric(raw[col],errors='raise').to_numpy(dtype=float)
        if not np.isfinite(vals).all() or np.any(vals != np.round(vals)):
            raise ValueError(f'{key}: counts must be finite integers')
        result[key]=vals.astype(int)
    if (result.attempts < result.regenerated).any():
        raise ValueError('n_attempts < n_regenerated: inconsistent repair counters')
    for optional in ('n_repair_calls','n_offspring_seen',
                     'n_forbidden_candidates','n_exclusion_centers'):
        if optional in raw:
            result[optional]=pd.to_numeric(raw[optional],errors='raise')
            if (result[optional]<0).any() or not np.isfinite(result[optional]).all():
                raise ValueError(f'{optional}: invalid counts')
    if (result[['robust','nonrobust','regenerated','attempts']]<0).any().any():
        raise ValueError('Process counts must be nonnegative')
    # The ALSF callback only records rows from FLAG_START_GEN onward.
    result=result.set_index('generation').sort_index()
    last=int(result.index.max())
    if last>100000: raise ValueError('Implausible generation counter')
    result=result.reindex(range(0,last+1),fill_value=0).rename_axis('generation').reset_index()
    return result


def plot_colors():
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':12,
        'axes.labelsize':15, 'axes.titlesize':16,
        'xtick.labelsize':12,'ytick.labelsize':12,
        'legend.fontsize':11, 'axes.spines.top':True,
        'axes.spines.right':True,'savefig.facecolor':'white',
    })


def make_scatter(ax,archive,space,original_axes,draw_ab=False):
    xcol,ycol=('X1','X2') if space=='decision' else ('Obj1','Obj2')
    X=archive[xcol].to_numpy(dtype=float)
    Y=archive[ycol].to_numpy(dtype=float)
    is_bad=(archive.flag=='non-robust').to_numpy()
    # NEI plus non-medoid robust points: gray as in the original plots.
    ax.scatter(X,Y,s=6,c='#c9c9c9',alpha=.37,edgecolors='none',label='noise',zorder=1,rasterized=True)
    ax.scatter(X[is_bad],Y[is_bad],s=10,c='#ffad32',alpha=.77,
               edgecolors='none',label='non-robust point',zorder=2,rasterized=True)
    representatives=representative_indices(archive)
    if len(representatives):
        palette=plt.get_cmap('gist_rainbow')
        # Color-coded robust medoids; faint matching crosses emulate reference.
        for j,k in enumerate(representatives):
            c=palette((j*0.61803398875)%1)
            ax.scatter([X[k]],[Y[k]],marker='x',s=82,linewidths=1.8,
                       color=c,zorder=4)
            ax.scatter([X[k]],[Y[k]],marker='x',s=27,linewidths=1.3,
                       color='black',zorder=5)
        ax.scatter([],[],s=85,color='black',marker='x',linewidths=1.8,
                   label='robust medoid')
    else:
        ax.scatter([],[],s=85,color='black',marker='x',label='robust medoid (none)')
        print('NOTICE: no wholly robust DBSCAN component: no robust medoids plotted')
    ax.set_xlabel('X1' if space=='decision' else 'Objective 1')
    ax.set_ylabel('X2' if space=='decision' else 'Objective 2')
    ax.grid(False)
    if original_axes:
        if space=='decision':
            ax.set_xlim(-0.4,6.15)
            ax.set_ylim(-0.8,5.55)
            ax.axvline(2.5,color='0.35',ls=':',lw=1,alpha=.65)
            ax.axhline(2.5,color='0.35',ls=':',lw=1,alpha=.65)
        else:
            ax.set_xlim(-3.5,62)
            ax.set_ylim(-3.0,57.5)
        shown=((X>=ax.get_xlim()[0])&(X<=ax.get_xlim()[1])&
               (Y>=ax.get_ylim()[0])&(Y<=ax.get_ylim()[1]))
        if (~shown).sum():
            print(f'WARNING: {int((~shown).sum())} archived points clipped in {space} space by --reference-axes')
    else:
        if space=='decision':
            ax.axvline(2.5,color='0.6',ls=':',lw=0.8,alpha=.5)
            ax.axhline(2.5,color='0.6',ls=':',lw=0.8,alpha=.5)
        ax.margins(x=.045,y=.07)
    if draw_ab:
        if space=='decision':
            ax.add_patch(Circle((2.5,2.5),radius=.18,fill=False,
                                edgecolor='blue',lw=2.0,zorder=6))
            ax.text(2.7,2.55,'A',fontsize=24,zorder=7)
            ax.add_patch(Ellipse((1.8,1.87),.48,.95,angle=40,
                                 fill=False,edgecolor='blue',lw=2,zorder=6))
            ax.text(1.5,1.3,'B',fontsize=24,zorder=7)
        else:
            ax.add_patch(Ellipse((11.0,14.5),4,8,angle=55,
                                 fill=False,edgecolor='blue',lw=2,zorder=6))
            ax.text(5,15,'B',fontsize=25,zorder=7)
    loc='lower right' if space=='decision' else 'lower left'
    ax.legend(loc=loc,frameon=True,edgecolor='0.3',facecolor='white',framealpha=.94)


def save(fig,outdir,stem):
    outdir.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        path=outdir/f'{stem}.{ext}'
        fig.savefig(path,dpi=260,bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


def plots(archive,process,outdir,reference_axes=False,annotate_ab=False,watermark=None):
    plot_colors()
    fig,ax=plt.subplots(figsize=(10.66,8.46),dpi=100)
    make_scatter(ax,archive,'decision',reference_axes,annotate_ab)
    fig.tight_layout()
    if watermark: fig.text(.5,.5,watermark,ha='center',va='center',fontsize=33,rotation=25,color='crimson',alpha=.19,zorder=20)
    save(fig,outdir,'design_space_clusters')

    if process is not None:
        gen=process.generation.to_numpy()
        fig,axes=plt.subplots(2,1,figsize=(15,12),sharex=True,
                               gridspec_kw={'height_ratios':[1.0,1.0]})
        r=process.robust.to_numpy()
        nr=process.nonrobust.to_numpy()
        axes[0].stackplot(gen,r,nr,colors=['#4caf50','#e45151'],alpha=.98,
                          labels=['Robust','Non-robust'])
        axes[0].set_title('Robust / Non-robust per generation')
        axes[0].set_ylabel('Clusters')
        axes[0].legend(loc='upper left')
        reg=process.regenerated.to_numpy()
        att=process.attempts.to_numpy()
        axes[1].bar(gen,reg,width=.9,color='#638fb9',label='Regenerated')
        # The Pavel repair counts EVERY resampling attempt; each successful
        # regeneration therefore already uses at least one attempt. Plot only
        # the additional attempts as a red cap, so total height = attempts.
        extra=np.maximum(att-reg,0)
        axes[1].bar(gen,extra,width=.9,bottom=reg,color='#ed9292',alpha=.9,
                    label='Additional attempts',zorder=2)
        axes[1].set_title('Archive repair — regenerated offspring & resample attempts per generation')
        axes[1].set_ylabel('Count')
        if reg.sum() == 0 and att.sum() == 0:
            if 'n_offspring_seen' in process:
                seen=int(process.n_offspring_seen.sum())
                forbidden=int(process.n_forbidden_candidates.sum())
                annotation=f'0 regenerated; {seen} offspring checked; {forbidden} forbidden candidates'
            else:
                annotation='0 regenerated; repair invocation was not logged'
            axes[1].text(.5,.55,annotation,ha='center',va='center',
                         transform=axes[1].transAxes,fontsize=12,
                         bbox={'facecolor':'white','edgecolor':'#aaaaaa','alpha':.95})
            axes[1].set_ylim(0,1)
        axes[1].set_xlabel('Generation')
        axes[1].legend(loc='upper left')
        for ax in axes:
            ax.grid(True,ls='--',color='0.78',alpha=.75)
            ax.set_axisbelow(True)
        fig.tight_layout(h_pad=1.1)
        if watermark: fig.text(.5,.5,watermark,ha='center',va='center',fontsize=40,rotation=25,color='crimson',alpha=.19,zorder=20)
        save(fig,outdir,'test_evolution')

    fig,ax=plt.subplots(figsize=(10.44,8.45),dpi=100)
    make_scatter(ax,archive,'objectives',reference_axes,annotate_ab)
    fig.tight_layout()
    if watermark: fig.text(.5,.5,watermark,ha='center',va='center',fontsize=33,rotation=25,color='crimson',alpha=.19,zorder=20)
    save(fig,outdir,'pareto_front')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--archive',type=Path,required=True,help='2D ALSF archive CSV containing flag, cluster_id, is_medoid')
    ap.add_argument('--process',type=Path,required=True,help='2D ALSF per-generation process CSV')
    ap.add_argument('--outdir',type=Path,default=Path('figures/test1_reproduced'))
    ap.add_argument('--reference-axes',action='store_true',help='Match screenshot axes; may clip data!')
    ap.add_argument('--annotate-ab',action='store_true',help='Add A/B reference circles (visual aids, not algorithm output)')
    args=ap.parse_args(argv)
    a=read_archive(args.archive)
    p=prepare_process(args.process)
    print('Archive:',len(a),'points; flags:',a.flag.value_counts().to_dict())
    print('Process:',len(p),'generation records (including zero-filled warmup)')
    print('Recorded repair totals:',
          {k:int(p[k].sum()) for k in ('regenerated','attempts')})
    if 'n_offspring_seen' in p:
        print('Repair diagnostics:',
              {k:int(p[k].sum()) for k in ('n_repair_calls','n_offspring_seen',
                                           'n_forbidden_candidates') if k in p})
    plots(a,p,args.outdir,args.reference_axes,args.annotate_ab)

if __name__=='__main__':
    main()
