#!/usr/bin/env python3
"""Reproduce original Test 1 ALSF 3-panel figure style from REAL 2D run CSV files.

Requirements: Python 3.9+, numpy, pandas, matplotlib.

Input:
  results/test1_alsf_seed42_repair_verified_full/archive.csv
  results/test1_alsf_seed42_repair_verified_full/process.csv

No optimization is executed. Colors are faithful to archive flags, not inferred
from screenshot pixels. Colored X symbols represent a locally-stable *member*
selected as a representative of each density-connected component. They DO NOT
prove that the whole DBSCAN component is robust.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D

A_CENTER = (2.5, 2.5)
B_CENTER = (1.8, 1.87)
PALETTE = {'background': '#c7c7c7', 'violation': '#f4a43d',
           'stable': '#23877c', 'outline': '#2d43ca',
           'robust': '#46aa57', 'nonrobust': '#df5055',
           'regen': '#547faa', 'extra': '#e69899'}


def normalize_flags(series):
    s = series.fillna('').astype(str).str.strip().str.lower().str.replace('_','-',regex=False)
    s = s.replace({'nonrobust':'non-robust', 'nei':'NEI', 'noise':'NEI',
                   'insufficiently explored':'NEI',
                   'insufficient evidence':'NEI'})
    invalid=set(s)-{'robust','non-robust','NEI'}
    if invalid:
        raise ValueError('Invalid flag values in archive: %s' % sorted(invalid))
    return s


def load_data(run):
    apath, ppath = run/'archive.csv', run/'process.csv'
    if not apath.exists() or not ppath.exists():
        raise FileNotFoundError('Need BOTH archive.csv and process.csv in %s' % run)
    a = pd.read_csv(apath)
    p = pd.read_csv(ppath)
    required_a=['X1','X2','Obj1','Obj2','flag','cluster_id']
    required_p=['generation','n_robust_clusters','n_non_robust_clusters',
                'n_regenerated','n_attempts']
    for df, required, name in [(a,required_a,'archive'),(p,required_p,'process')]:
        missing=set(required)-set(df.columns)
        if missing:
            raise ValueError('%s.csv missing columns: %s\nFound: %s' %
                             (name,sorted(missing),list(df.columns)))
    xs=sorted(k for k in a if k.startswith('X') and k[1:].isdigit())
    if xs != ['X1','X2']:
        raise ValueError('This program needs the ORIGINAL two-variable Test 1 archive, not a 10D archive')
    a=a.copy(); p=p.copy()
    a['flag']=normalize_flags(a['flag'])
    for col in ['X1','X2','Obj1','Obj2']:
        a[col]=pd.to_numeric(a[col],errors='raise')
    if not np.isfinite(a[['X1','X2','Obj1','Obj2']].to_numpy(float)).all():
        raise ValueError('Nonfinite positions or objective values')
    a['cluster_id']=pd.to_numeric(a['cluster_id'],errors='coerce').fillna(-1).astype(int)
    for col in required_p:
        p[col]=pd.to_numeric(p[col],errors='raise')
    p=p.sort_values('generation')
    if p['generation'].duplicated().any():
        raise ValueError('Duplicate generation')
    if (p[required_p[1:]]<0).any().any():
        raise ValueError('Negative process count')
    if (p.n_attempts < p.n_regenerated).any():
        raise ValueError('n_attempts must be >= n_regenerated')
    return a,p


def style():
    plt.rcParams.update({
        'font.family':'STIXGeneral','mathtext.fontset':'stix',
        'font.size':10.5,'axes.labelsize':12.0,'axes.titlesize':11.0,
        'xtick.labelsize':10,'ytick.labelsize':10,
        'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in',
        'xtick.minor.visible':True,'ytick.minor.visible':True,
        'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white',
        'savefig.facecolor':'white',
    })


def masks(a):
    x=a.X1.to_numpy(float); y=a.X2.to_numpy(float)
    A=(x-2.5)**2+(y-2.5)**2 <= .18**2
    th=np.deg2rad(40); dx=x-B_CENTER[0]; dy=y-B_CENTER[1]
    B=((np.cos(th)*dx+np.sin(th)*dy)/.24)**2 + \
      ((-np.sin(th)*dx+np.cos(th)*dy)/.475)**2 <= 1
    return {'A':A,'B':B}


def representatives(a, min_stable_in_component=3):
    from plot_three_figures import representative_indices
    return representative_indices(a)


def save(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        f=out/(stem+'.'+ext)
        fig.savefig(f,dpi=350,bbox_inches='tight',pad_inches=.07)
        print('Saved',f)
    plt.close(fig)


def scatter_fig(a,out,which,rep,ab):
    decision=which=='design_space_clusters'
    xx,yy=('X1','X2') if decision else ('Obj1','Obj2')
    X=a[xx].to_numpy(float);Y=a[yy].to_numpy(float)
    bad=(a.flag=='non-robust').to_numpy()
    fig,ax=plt.subplots(figsize=(8.45,6.55),dpi=110)
    # Original gray/orange/colored-X look, accurate classification.
    ax.scatter(X,Y,c=PALETTE['background'],s=6.2,alpha=.34,linewidths=0,
               zorder=1,rasterized=True)
    ax.scatter(X[bad],Y[bad],c=PALETTE['violation'],s=8.0,alpha=.80,
               linewidths=0,zorder=2,rasterized=True)

    # Stronger emphasis for archive samples inside user-marked A/B regions.
    # NOTE: this does not change flags or make regions into DBSCAN clusters.
    for title,mask in ab.items():
        for flag,color in [('robust',PALETTE['stable']),
                           ('non-robust','#db6037'), ('NEI','#b5b5b5')]:
            selected=mask&(a.flag.to_numpy()==flag)
            if selected.sum():
                ax.scatter(X[selected],Y[selected],s=16,
                           c=color,alpha=.98,edgecolors='white',linewidths=.28,
                           zorder=4,rasterized=True)

    if len(rep):
        palette=plt.get_cmap('gist_rainbow')
        for j,i in enumerate(rep):
            color=palette(((j*.61803399)%1.0))
            ax.scatter(X[i],Y[i],marker='x',s=69,c=[color],lw=2.1,zorder=7)
            ax.scatter(X[i],Y[i],marker='x',s=23,c='black',lw=.9,zorder=8)

    if decision:
        ax.set(xlabel=r'$x_1$',ylabel=r'$x_2$',xlim=(-.4,6.15),ylim=(-.8,5.55))
        ax.axvline(2.5,c='.65',ls=':',lw=.8,zorder=0)
        ax.axhline(2.5,c='.65',ls=':',lw=.8,zorder=0)
        ax.add_patch(Circle(A_CENTER,.18,fill=False,lw=2.2,ec=PALETTE['outline'],zorder=10))
        ax.add_patch(Ellipse(B_CENTER,.48,.95,angle=40,fill=False,lw=2.2,
                             ec=PALETTE['outline'],zorder=10))
        ax.text(2.73,2.58,'A',fontsize=22,va='center',zorder=10)
        ax.text(1.52,1.28,'B',fontsize=22,va='center',zorder=10)
        ax.set_xticks(np.arange(0,7,1));ax.set_yticks(np.arange(0,6,1))
    else:
        ax.set(xlabel=r'$f_1(\mathbf{x})$',ylabel=r'$f_2(\mathbf{x})$',
               xlim=(-3,62),ylim=(-2.5,57.5))
        # Same target as the manually outlined B region in the design plot;
        # its objective-space counterpart is highlighted rather than assuming
        # the old screenshot's hand-drawn ellipse is mathematically exact.
        sel=ab['B']
        if sel.any():
            ax.scatter(X[sel],Y[sel],marker='o',s=18,
                       facecolors='none',edgecolors=PALETTE['outline'],
                       linewidths=.40,alpha=.85,zorder=9,rasterized=True)
        ax.set_xticks(np.arange(0,61,10));ax.set_yticks(np.arange(0,51,10))

    handles=[
        Line2D([],[],marker='o',color='none',markerfacecolor=PALETTE['background'],
               markeredgecolor='none',markersize=5,label='Archive (background)'),
        Line2D([],[],marker='o',color='none',markerfacecolor=PALETTE['violation'],
               markeredgecolor='none',markersize=5,label='Observed violation'),
        Line2D([],[],marker='x',color='black',markersize=7,linestyle='',
               label='Locally stable representative')
    ]
    ax.legend(handles=handles,loc='lower right' if decision else 'lower left',
              fontsize=9,frameon=True,edgecolor='.4',facecolor='white',framealpha=.98)
    visible=(X>=ax.get_xlim()[0])&(X<=ax.get_xlim()[1])&\
            (Y>=ax.get_ylim()[0])&(Y<=ax.get_ylim()[1])
    if (~visible).sum():
        print('%s: %d archive points outside original screenshot window' % (which,(~visible).sum()))
    fig.tight_layout()
    save(fig,out,which)


def evolution_fig(p,a,out):
    # Shared evolution implementation; pointwise and component counts separated.
    from plot_three_figures import plots
    prepared = p.rename(columns={'n_robust_clusters':'robust',
        'n_non_robust_clusters':'nonrobust', 'n_regenerated':'regenerated',
        'n_attempts':'attempts'})
    plots(a,prepared,out,annotate_ab=True)


def main():
    q=argparse.ArgumentParser(description='Three ORIGINAL-style Test 1 ALSF figures from real CSV')
    q.add_argument('--run',type=Path,default=Path('results/test1_alsf_seed42_repair_verified_full'))
    q.add_argument('--outdir',type=Path,default=Path('figures/original_three_recreated'))
    args=q.parse_args()
    a,p=load_data(args.run)
    style()
    rep=representatives(a); ab=masks(a)
    print('Actual archive labels:',a.flag.value_counts().to_dict())
    print('Stable representative counts:',len(rep))
    for name,sel in ab.items():
        print(name,'points:',sel.sum(),a.flag[sel].value_counts().to_dict())
    evolution_fig(p,a,args.outdir)
    print('Finished: exact original 3-file naming, updated labels, no fabricated data.')

if __name__=='__main__':
    main()
