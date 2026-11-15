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


COLORS = {'NEI':'#B6BDC4', 'robust':'#237C69', 'non-robust':'#D17A32'}
POINT_LABELS = {'NEI':'Insufficient evidence', 'robust':'Locally stable (empirical)',
                'non-robust':'Verified sensitive'}


def display_representatives(archive, limit=32):
    """Thin markers for display only; never change evidence or representatives.

    The identical deterministic subset of actual archive rows is used in both
    spaces. The full representative column remains untouched.
    """
    indices = representative_indices(archive)
    if len(indices) <= limit:
        return indices
    coords = archive.iloc[indices][['X1','X2']].to_numpy(float)
    order = np.lexsort((coords[:,1],coords[:,0]))
    return indices[order[np.linspace(0,len(indices)-1,limit,dtype=int)]]


def make_scatter(ax,archive,space,original_axes=False,draw_ab=False,
                 representatives=None,pareto=None,detail=False):
    xcol,ycol=('X1','X2') if space=='decision' else ('Obj1','Obj2')
    X=archive[xcol].to_numpy(float); Y=archive[ycol].to_numpy(float)
    for category in ('NEI','robust','non-robust'):
        chosen=archive.flag.eq(category).to_numpy()
        ax.scatter(X[chosen],Y[chosen],s=5 if not detail else 11,
            color=COLORS[category],alpha=.28 if category=='NEI' else .68,
            edgecolors='none',rasterized=True,label=POINT_LABELS[category],
            zorder={'NEI':1,'robust':2,'non-robust':3}[category])
    if representatives is None:
        representatives=display_representatives(archive)
    if len(representatives) and not detail:
        ax.scatter(X[representatives],Y[representatives],s=18,marker='D',
            facecolors='white',edgecolors=COLORS['robust'],linewidths=.7,
            label=f'Stable representatives (display subset: {len(representatives)})',zorder=4)
    if pareto is not None:
        ax.scatter(pareto[xcol],pareto[ycol],s=12,marker='o',
            facecolors='none',edgecolors='#244A73',linewidths=.65,
            label='Final NSGA-II Pareto set',zorder=5)
    ax.set_xlabel(r'$x_1$' if space=='decision' else r'$f_1$')
    ax.set_ylabel(r'$x_2$' if space=='decision' else r'$f_2$')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(top=False,right=False)
    ax.margins(.04)
    if space=='decision':
        ax.set_aspect('equal',adjustable='box')
    if original_axes:
        ax.set_xlim(*((-.4,6.15) if space=='decision' else (-3.5,62)))
        ax.set_ylim(*((-.8,5.55) if space=='decision' else (-3.,57.5)))
        outside=(X<ax.get_xlim()[0])|(X>ax.get_xlim()[1])|(Y<ax.get_ylim()[0])|(Y>ax.get_ylim()[1])
        print(f'Reference window: {outside.sum()} archived points outside {space} panel')
    if draw_ab and space=='decision':
        ax.add_patch(Circle((2.5,2.5),.18,fill=False,edgecolor='#244A73',lw=1,zorder=6))
        ax.add_patch(Ellipse((1.8,1.87),.48,.95,angle=40,fill=False,
                            edgecolor='#244A73',lw=1,zorder=6))
        for label,pos in [('A',(2.75,2.65)),('B',(1.15,1.65))]:
            ax.annotate(label,pos,color='#244A73',fontweight='bold',zorder=7,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))


def scatter_figure(archive,space,reference_axes=False,annotate_ab=False,pareto=None):
    """Full archive plus explicit detail windows: no hidden full-view clipping."""
    fig=plt.figure(figsize=(9,5.1),layout='constrained')
    grid=fig.add_gridspec(2,2,width_ratios=[1.8,1])
    full=fig.add_subplot(grid[:,0])
    reps=display_representatives(archive)
    make_scatter(full,archive,space,reference_axes,annotate_ab,reps,pareto)
    full.set_title('(a) Full archive' if not reference_axes else '(a) Reference window',loc='left')
    if space=='decision':
        from plot_original_three_ALSF import masks
        region_masks=masks(archive)
        for row,(label,center,width) in enumerate([('A',(2.5,2.5),.32),('B',(1.8,1.87),.55)]):
            ax=fig.add_subplot(grid[row,1])
            make_scatter(ax,archive,space,False,True,reps,pareto,detail=True)
            ax.set_xlim(center[0]-width,center[0]+width)
            ax.set_ylim(center[1]-width,center[1]+width)
            selected=archive.loc[region_masks[label],'flag'].value_counts()
            ax.set_title(f'({chr(98+row)}) Region {label}',loc='left')
            text=' / '.join(str(int(selected.get(c,0))) for c in ('robust','non-robust','NEI'))
            ax.text(.02,.98,'Stable / sensitive / NEI: '+text,transform=ax.transAxes,
                    va='top',fontsize=7,bbox=dict(fc='white',ec='none',alpha=.9,pad=2),zorder=8)
    else:
        # Same archive rows, with explicitly restricted detail panels.
        from plot_original_three_ALSF import masks
        for row,label in enumerate(('A','B')):
            ax=fig.add_subplot(grid[row,1])
            make_scatter(ax,archive,space,False,False,reps,pareto,detail=True)
            selected=masks(archive)[label]
            pts=archive.loc[selected,['Obj1','Obj2']].to_numpy(float)
            if len(pts):
                low=pts.min(axis=0); high=pts.max(axis=0)
                pad=np.maximum((high-low)*.18,.25)
                ax.set_xlim(low[0]-pad[0],high[0]+pad[0])
                ax.set_ylim(low[1]-pad[1],high[1]+pad[1])
                ax.scatter(pts[:,0],pts[:,1],s=20,facecolors='none',
                           edgecolors='#765C91',marker='s',linewidths=.55,zorder=6)
            else:
                ax.text(.5,.5,'No archive points in region',ha='center',transform=ax.transAxes)
            ax.set_title(f'({chr(98+row)}) Region {label} in objective space',loc='left')
    handles,labels=full.get_legend_handles_labels()
    if space=='objectives':
        from matplotlib.lines import Line2D
        handles.append(Line2D([],[],marker='s',markersize=4,markerfacecolor='none',
                              markeredgecolor='#765C91',linestyle='none'))
        labels.append('Samples from indicated region')
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,
               markerscale=1.4,fontsize=8)
    return fig


def save(fig,outdir,stem):
    outdir.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        path=outdir/f'{stem}.{ext}'
        fig.savefig(path,dpi=260,bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


def plots(archive,process,outdir,reference_axes=False,annotate_ab=False,watermark=None,pareto=None):
    plot_colors()
    fig=scatter_figure(archive,'decision',reference_axes,annotate_ab,pareto)
    if watermark: fig.text(.5,.5,watermark,ha='center',va='center',fontsize=33,rotation=25,color='crimson',alpha=.19,zorder=20)
    save(fig,outdir,'design_space_clusters')

    if process is not None:
        gen=process.generation.to_numpy()
        fig,axs=plt.subplots(2,2,figsize=(9,6),sharex=True,layout='constrained')
        axes=[axs[0,0],axs[0,1],axs[1,0],axs[1,1]]
        for col,label,color in [('robust','All stable','#237C69'),
            ('nonrobust','All sensitive','#D17A32'),('n_nei_clusters','All NEI','#B6BDC4'),
            ('n_mixed_clusters','Mixed','#7563A8')]:
            if col in process:
                axes[0].plot(gen,process[col],label=label,color=color,lw=1.3)
        axes[0].set(title='(a) DBSCAN components',ylabel='Components')
        axes[0].legend(ncol=2)
        for col,label,color in [('n_regions','Occupied cells','#244A73'),
                ('n_mixed_regions','Mixed cells','#7563A8')]:
            if col in process:
                axes[1].plot(gen,process[col],label=label,color=color,lw=1.3)
        axes[1].set(title='(b) Bounded spatial regions',ylabel='Cells')
        if 'n_regions' in process:
            axes[1].legend()
        else:
            axes[1].text(.5,.5,'Cell history not recorded',ha='center',transform=axes[1].transAxes)
        if all(c in process for c in ('n_robust','n_non_robust','n_nei')):
            total=process[['n_robust','n_non_robust','n_nei']].sum(axis=1).to_numpy(float)
            if np.any(total<=0):
                raise ValueError('Pointwise history must have positive archive totals')
            for col,label,color in [('n_robust','Locally stable','#237C69'),
                    ('n_non_robust','Verified sensitive','#D17A32'),('n_nei','NEI','#B6BDC4')]:
                axes[3].plot(gen,100*process[col]/total,label=label,color=color,lw=1.3)
            axes[3].set_ylim(0,100)
            axes[3].legend()
        else:
            axes[3].text(.5,.5,'Pointwise history not recorded',ha='center',transform=axes[3].transAxes)
        axes[3].set(title='(d) Pointwise archive evidence',ylabel='Archive samples (%)',xlabel='Generation')
        reg=process.regenerated.to_numpy()
        att=process.attempts.to_numpy()
        axes[2].bar(gen,reg,width=.9,color='#638fb9',label='Regenerated')
        # The Pavel repair counts EVERY resampling attempt; each successful
        # regeneration therefore already uses at least one attempt. Plot only
        # the additional attempts as a red cap, so total height = attempts.
        extra=np.maximum(att-reg,0)
        axes[2].bar(gen,extra,width=.9,bottom=reg,color='#ed9292',alpha=.9,
                    label='Additional attempts',zorder=2)
        axes[2].set_title('(c) Offspring repair')
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
            ax.grid(axis='y',color='0.9',lw=.5)
            ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
        if watermark: fig.text(.5,.5,watermark,ha='center',va='center',fontsize=40,rotation=25,color='crimson',alpha=.19,zorder=20)
        save(fig,outdir,'test_evolution')

    fig=scatter_figure(archive,'objectives',reference_axes,annotate_ab,pareto)
    if watermark: fig.text(.5,.5,watermark,ha='center',va='center',fontsize=33,rotation=25,color='crimson',alpha=.19,zorder=20)
    save(fig,outdir,'pareto_front')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--archive',type=Path,required=True,help='2D ALSF archive CSV containing flag, cluster_id, is_medoid')
    ap.add_argument('--pareto',type=Path,help='Optional final Pareto CSV from the same run')
    ap.add_argument('--process',type=Path,required=False,help='2D ALSF per-generation process CSV')
    ap.add_argument('--outdir',type=Path,default=Path('figures/test1_reproduced'))
    ap.add_argument('--reference-axes',action='store_true',help='Match screenshot axes; may clip data!')
    ap.add_argument('--annotate-ab',action='store_true',help='Add A/B reference circles (visual aids, not algorithm output)')
    args=ap.parse_args(argv)
    a=read_archive(args.archive)
    p=prepare_process(args.process) if args.process else None
    front=read_archive(args.pareto) if args.pareto else None
    print('Archive:',len(a),'points; flags:',a.flag.value_counts().to_dict())
    if p is None:
        print('No process.csv supplied: evolution is not generated.')
    else:
        print('Process:',len(p),'recorded generation records')
    plots(a,p,args.outdir,args.reference_axes,args.annotate_ab,pareto=front)

if __name__=='__main__':
    main()
