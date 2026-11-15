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
    'robust': ['n_robust_clusters', 'robust_clusters'],
    'nonrobust': ['n_non_robust_clusters', 'n_nonrobust_clusters',
                  'non_robust_clusters'],
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
    """Actual locally stable members of bounded regions, never certificates."""
    if 'is_stable_representative' in df:
        chosen = df.is_stable_representative.astype(str).str.lower().isin(['true', '1'])
        if (chosen & df.flag.ne('robust')).any():
            raise ValueError('Stable representative has a non-stable pointwise label')
        return np.flatnonzero(chosen)
    # Historic archives: retain only whole-component stable representatives.
    if 'cluster_id' not in df:
        return np.array([], dtype=int)
    chosen = []
    for cid, group in df.reset_index(drop=True).groupby('cluster_id'):
        if cid >= 0 and group.flag.eq('robust').all():
            pts = group[['X1', 'X2']].to_numpy(float)
            chosen.append(group.index[np.argmin(np.max(abs(pts-pts.mean(axis=0)), axis=1))])
    return np.asarray(chosen, dtype=int)


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
                     'n_forbidden_candidates','n_exclusion_centers',
                     'n_robust','n_non_robust','n_nei','n_mixed_clusters',
                     'n_nei_clusters','n_regions','n_mixed_regions','largest_cluster'):
        if optional in raw:
            result[optional]=pd.to_numeric(raw[optional],errors='raise')
            if (result[optional]<0).any() or not np.isfinite(result[optional]).all():
                raise ValueError(f'{optional}: invalid counts')
    if (result[['robust','nonrobust','regenerated','attempts']]<0).any().any():
        raise ValueError('Process counts must be nonnegative')
    # Unrecorded warmup and gaps are unknown, never fabricated zeros.
    return result.sort_values('generation').reset_index(drop=True)


def plot_colors():
    from plot_alsf_elsevier import style
    style()


def make_scatter(ax,archive,space,original_axes,draw_ab=False):
    xcol,ycol=('X1','X2') if space=='decision' else ('Obj1','Obj2')
    X=archive[xcol].to_numpy(dtype=float)
    Y=archive[ycol].to_numpy(dtype=float)
    is_bad=(archive.flag=='non-robust').to_numpy()
    # NEI plus non-medoid robust points: gray as in the original plots.
    ax.scatter(X,Y,s=6,c='#c9c9c9',alpha=.37,edgecolors='none',label='Archived solutions',zorder=1,rasterized=True)
    ax.scatter(X[is_bad],Y[is_bad],s=10,c='#ffad32',alpha=.77,
               edgecolors='none',label='Verified violation endpoint',zorder=2,rasterized=True)
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
                   label='Locally stable representative')
    else:
        ax.scatter([],[],s=85,color='black',marker='x',label='Locally stable representative (none)')
        print('NOTICE: no locally stable representatives available')
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
            # Map the exact same archive rows, not a hand-drawn objective ellipse.
            from plot_original_three_ALSF import masks
            for name, selected in masks(archive).items():
                ax.scatter(X[selected], Y[selected], facecolors='none',
                           edgecolors='#345E89', s=20, linewidths=.6,
                           label=f'Region {name} archive samples', zorder=6)
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
        fig,axes=plt.subplots(3,1,figsize=(7.2,8.5),sharex=True)
        for col, label, color in [('robust','All locally stable','#217A6B'),
                ('nonrobust','All verified sensitive','#C35B36'),
                ('n_nei_clusters','All NEI','#B5B8BC'),
                ('n_mixed_clusters','Mixed','#7563A8')]:
            if col in process:
                axes[0].plot(gen,process[col],label=label,color=color)
        for col, label in [('n_regions','Bounded regions'), ('n_mixed_regions','Mixed regions')]:
            if col in process:
                axes[0].plot(gen,process[col],ls='--',label=label)
        axes[0].set_ylabel('Components / regions')
        axes[0].legend(loc='upper left',ncol=2)
        for col,label,color in [('n_robust','Locally stable','#217A6B'),
                ('n_non_robust','Verified sensitive','#C35B36'),
                ('n_nei','NEI','#B5B8BC')]:
            if col in process:
                axes[1].plot(gen,process[col],label=label,color=color)
        if 'n_robust' not in process:
            axes[1].text(.5,.5,'Pointwise history not recorded',ha='center',transform=axes[1].transAxes)
        else:
            axes[1].legend(loc='upper left')
        axes[1].set_ylabel('Archive samples')
        reg=process.regenerated.to_numpy()
        att=process.attempts.to_numpy()
        axes[2].bar(gen,reg,width=.9,color='#638fb9',label='Regenerated')
        # The Pavel repair counts EVERY resampling attempt; each successful
        # regeneration therefore already uses at least one attempt. Plot only
        # the additional attempts as a red cap, so total height = attempts.
        extra=np.maximum(att-reg,0)
        axes[2].bar(gen,extra,width=.9,bottom=reg,color='#ed9292',alpha=.9,
                    label='Additional attempts',zorder=2)
        axes[2].set_title('Archive repair — regenerated offspring & resample attempts per generation')
        axes[2].set_ylabel('Count')
        if reg.sum() == 0 and att.sum() == 0:
            if 'n_offspring_seen' in process:
                seen=int(process.n_offspring_seen.sum())
                forbidden=int(process.n_forbidden_candidates.sum())
                annotation=f'0 regenerated; {seen} offspring checked; {forbidden} forbidden candidates'
            else:
                annotation='0 regenerated; repair invocation was not logged'
            axes[2].text(.5,.55,annotation,ha='center',va='center',
                         transform=axes[2].transAxes,fontsize=12,
                         bbox={'facecolor':'white','edgecolor':'#aaaaaa','alpha':.95})
            axes[2].set_ylim(0,1)
        axes[2].set_xlabel('Generation')
        axes[2].legend(loc='upper left')
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
    print('Process:',len(p),'recorded generation records')
    print('Recorded repair totals:',
          {k:int(p[k].sum()) for k in ('regenerated','attempts')})
    if 'n_offspring_seen' in p:
        print('Repair diagnostics:',
              {k:int(p[k].sum()) for k in ('n_repair_calls','n_offspring_seen',
                                           'n_forbidden_candidates') if k in p})
    plots(a,p,args.outdir,args.reference_axes,args.annotate_ab)

if __name__=='__main__':
    main()
