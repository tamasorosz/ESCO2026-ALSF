#!/usr/bin/env python3
"""Elsevier/JCAM-inspired, reproducible Test 1 figures from real ALSF CSVs.

Requires Python >=3.9; numpy pandas matplotlib.

Usage (existing run; no new optimization):
  python plot_alsf_elsevier.py --run results/test1_alsf_seed42_repair_verified_full \\
    --outdir figures/elsevier --reference-axes --annotate-ab

If only a console log is available, the point-level evolution figure can be
created from it. The program does not fabricate missing scatterplot data:
  python plot_alsf_elsevier.py --log full_run_log.txt --outdir figures/elsevier

Outputs real-data PNG (400 dpi), PDF (vector text/axes), and SVG figures.
Point categories are archive-based EMPIRICAL labels, not proven worst-case
robustness. Blue A and B contours are manual visual annotations, not clusters.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
import pandas as pd

# Restrained, print-friendly palette; different luminance for grayscale.
C = {'robust': '#217A6B', 'non-robust': '#C35B36', 'NEI': '#B5B8BC',
     'repair': '#52799F', 'extra': '#BB8D82', 'annotation': '#345E89'}
ORDER = ('NEI', 'robust', 'non-robust')
LAB = {'NEI': 'Insufficient evidence (NEI)',
       'robust': 'Locally stable (empirical)',
       'non-robust': 'Observed threshold violation'}


def style():
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
        'font.size': 9.0, 'axes.labelsize': 10.0, 'axes.titlesize': 9.5,
        'xtick.labelsize': 8.3, 'ytick.labelsize': 8.3,
        'axes.edgecolor': '#262626', 'axes.linewidth': 0.75,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 3.2, 'ytick.major.size': 3.2,
        'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
        'legend.fontsize': 8.2, 'legend.frameon': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


def normalise_flag(s):
    z = s.fillna('').astype(str).str.strip().str.lower().str.replace('_','-',regex=False)
    return z.replace({'nonrobust':'non-robust', 'noise':'NEI', 'nei':'NEI',
                      'insufficiently explored':'NEI', 'not enough information':'NEI'})


def load_archive(path):
    df = pd.read_csv(path)
    needed = {'X1','X2','Obj1','Obj2','flag'}
    missing=needed-set(df.columns)
    if missing: raise ValueError('Archive missing required columns: '+', '.join(sorted(missing)))
    # Prevent silently using a 10D projection, which would be misleading.
    xcols=sorted((c for c in df if re.fullmatch(r'X\d+',str(c))),
                  key=lambda c:int(c[1:]))
    if xcols!=['X1','X2']:
        raise ValueError('Expected 2D Test 1 (X1,X2); found '+str(xcols))
    df=df.copy()
    df['flag']=normalise_flag(df['flag'])
    unknown=set(df.flag)-set(ORDER)
    if unknown: raise ValueError('Unknown flag labels: '+str(unknown))
    if not np.isfinite(df[['X1','X2','Obj1','Obj2']].to_numpy(float)).all():
        raise ValueError('Non-finite archive coordinates or objectives')
    return df


def load_process(path):
    df=pd.read_csv(path)
    required=['generation','n_robust','n_non_robust','n_nei']
    missing=[col for col in required if col not in df]
    if missing: raise ValueError('Process CSV lacks: '+str(missing))
    df=df.sort_values('generation').reset_index(drop=True)
    for name in required:
        df[name]=pd.to_numeric(df[name],errors='raise')
        if (not np.isfinite(df[name]).all() or (df[name]<0).any()):
            raise ValueError('Invalid process column: '+name)
    if df.generation.duplicated().any():
        raise ValueError('Duplicate generations in process.csv')
    totals=df[['n_robust','n_non_robust','n_nei']].sum(axis=1)
    if (totals==0).any(): raise ValueError('Zero point total in process.csv')
    if 'n_archive' in df and not np.allclose(totals,df['n_archive'].values):
        raise ValueError('n_archive differs from the three flag counts')
    repair_cols=['n_regenerated','n_attempts']
    if any(c in df for c in repair_cols) and not all(c in df for c in repair_cols):
        raise ValueError('Both n_regenerated and n_attempts are required')
    if all(c in df for c in repair_cols):
        for name in repair_cols:
            df[name]=pd.to_numeric(df[name],errors='raise')
            if (not np.isfinite(df[name]).all() or (df[name]<0).any()):
                raise ValueError('Invalid repair column: '+name)
        if (df.n_attempts<df.n_regenerated).any():
            raise ValueError('n_attempts < n_regenerated')
    return df


def load_log(path):
    pat=re.compile(r'Local L-inf filter \((\d+) evaluations, (\d+) DBSCAN components\): '
                   r'(\d+) verified-violation endpoints, (\d+) locally stable samples, (\d+) NEI')
    out=[]
    for m in pat.finditer(path.read_text(encoding='utf-8')):
        n_eval,n_clusters,n_sens,n_stable,n_nei=map(int,m.groups())
        if n_eval != n_sens+n_stable+n_nei:
            raise ValueError('Invalid log counters at evaluation '+str(n_eval))
        out.append((n_eval,n_clusters,n_sens,n_stable,n_nei))
    if not out: raise ValueError('No Local L-inf filter entries in log')
    # The generation number is inferred from the n_gen|n_eval table rather than
    # assuming a population size of 100; this also handles different sizes.
    pairpat=re.compile(r'^\s*(\d+)\s*\|\s*(\d+)\s*\|', re.M)
    gen_by_eval={int(ev):int(gen) for gen,ev in pairpat.findall(path.read_text(encoding='utf-8'))}
    if not all(ev in gen_by_eval for ev,*rest in out):
        raise ValueError('Cannot map some evaluations to generations; use process.csv instead')
    rows=[{'generation':gen_by_eval[ev], 'n_archive':ev,
           'n_robust':st, 'n_non_robust':sn, 'n_nei':nei,
           'n_dbscan_clusters':nc} for ev,nc,sn,st,nei in out]
    return pd.DataFrame(rows).drop_duplicates('generation',keep='last').sort_values('generation')


def save(fig, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ('png','pdf','svg'):
        output=folder/(stem+'.'+ext)
        fig.savefig(output,dpi=400,bbox_inches='tight',pad_inches=.05)
        print('Saved: '+str(output))
    plt.close(fig)


def scatter(df, folder, decision, zoom=False, annotate_ab=False):
    a,b,stem=(('X1','X2','decision_space_elsevier') if decision else
              ('Obj1','Obj2','objective_space_elsevier'))
    fig,ax=plt.subplots(figsize=(7.10,4.95),constrained_layout=True)
    for flag in ORDER:
        d=df[df.flag==flag]
        ax.scatter(d[a],d[b],s=3.8 if flag=='NEI' else 5.5,
                   color=C[flag],edgecolors='none',
                   alpha=.27 if flag=='NEI' else .76,
                   zorder={'NEI':1,'robust':2,'non-robust':3}[flag],
                   rasterized=True,
                   label='{} (n={:,})'.format(LAB[flag],len(d)))
    if decision:
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        if zoom:
            ax.set_xlim(-.4,6.15);ax.set_ylim(-.8,5.55)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MultipleLocator(1))
        else:
            ax.set_xlim(-5.1,10.1);ax.set_ylim(-5.1,10.1)
        ax.axvline(2.5,ls=(0,(2,3)),color='#858585',lw=.6,zorder=0)
        ax.axhline(2.5,ls=(0,(2,3)),color='#858585',lw=.6,zorder=0)
        if annotate_ab:
            ax.add_patch(Circle((2.5,2.5),.18,fill=False,edgecolor=C['annotation'],lw=1.25,zorder=5))
            ax.text(2.7,2.55,'A',fontsize=11,ha='left',va='center')
            ax.add_patch(Ellipse((1.8,1.87),.48,.95,angle=40,fill=False,
                                 edgecolor=C['annotation'],lw=1.25,zorder=5))
            ax.text(1.5,1.32,'B',fontsize=11,ha='center',va='center')
    else:
        ax.set_xlabel(r'$f_1(\mathbf{x})$');ax.set_ylabel(r'$f_2(\mathbf{x})$')
        if zoom: ax.set_xlim(-3,62);ax.set_ylim(-2,57)
        else: ax.margins(.025)
    ax.minorticks_on()
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,
              columnspacing=1.1,handletextpad=.35,markerscale=2.2,fontsize=7.7)
    if zoom:
        n_clipped=int((~(df[a].between(*ax.get_xlim()) & df[b].between(*ax.get_ylim()))).sum())
        if n_clipped: print('INFO: {} archived points outside the displayed {} zoom.'.format(n_clipped,'decision' if decision else 'objective'))
    save(fig,folder,stem)


def evolution(df,folder):
    gen=df.generation.to_numpy(float)
    stable=df.n_robust.to_numpy(float)
    nonrob=df.n_non_robust.to_numpy(float)
    nei=df.n_nei.to_numpy(float)
    totals=stable+nonrob+nei
    # Percentages avoid conflating archive growth with the number of stable designs.
    frac=np.vstack((stable,nonrob,nei))/totals*100
    has_repair='n_regenerated' in df and 'n_attempts' in df
    if has_repair:
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7.10,5.55),sharex=True,
               gridspec_kw={'height_ratios':[1.17,.83]},constrained_layout=True)
    else:
        fig,ax1=plt.subplots(figsize=(7.10,3.25),constrained_layout=True)
        ax2=None
    # Put NEI first on top of the stack, then stable then observed violations.
    ax1.stackplot(gen,frac[0],frac[1],frac[2],
                  colors=[C['robust'],C['non-robust'],C['NEI']],
                  labels=['Locally stable','Observed violation','NEI'],
                  alpha=.92,linewidth=0)
    ax1.set_ylabel('Archive fraction (%)')
    ax1.set_ylim(0,100)
    ax1.set_yticks([0,25,50,75,100])
    ax1.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.18),
               columnspacing=.9,handlelength=1.3,fontsize=8.0)
    ax1.text(.015,.93,'(a)',transform=ax1.transAxes,fontsize=9.2,ha='left',va='top')
    for ax in (ax1,ax2) if has_repair else (ax1,):
        ax.grid(which='major',color='#D8D8D8',lw=.5,ls='--')
        ax.set_axisbelow(True)
        ax.minorticks_on()
    if has_repair:
        regen=df.n_regenerated.to_numpy(float)
        extra=(df.n_attempts-df.n_regenerated).to_numpy(float)
        # For dense generations, vertical bars remain legible as thin columns.
        ax2.bar(gen,regen,width=.86,color=C['repair'],label='Regenerated offspring',linewidth=0)
        ax2.bar(gen,extra,bottom=regen,width=.86,color=C['extra'],label='Additional attempts',linewidth=0)
        ax2.set_ylabel('Count per generation')
        ax2.legend(ncol=2,loc='upper left',fontsize=7.8)
        ax2.text(.015,.93,'(b)',transform=ax2.transAxes,fontsize=9.2,ha='left',va='top')
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel('Generation')
    else:
        ax1.set_xlabel('Generation')
    ax1.set_xlim(0,max(gen)+1)
    save(fig,folder,'evolution_elsevier')
    df[['generation','n_robust','n_non_robust','n_nei']].to_csv(
        folder/'evolution_source_counts.csv',index=False)
    if has_repair:
        df[['generation','n_regenerated','n_attempts']].to_csv(
            folder/'offspring_repair_counts.csv',index=False)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run',type=Path,help='Results directory containing archive.csv and process.csv')
    ap.add_argument('--archive',type=Path,help='Path to 2D archive.csv')
    ap.add_argument('--process',type=Path,help='Path to process.csv')
    ap.add_argument('--log',type=Path,help='Console log; evolution only when process.csv missing')
    ap.add_argument('--outdir',type=Path,default=Path('figures/elsevier'))
    ap.add_argument('--reference-axes',action='store_true',help='Zoom like previous screenshots; clips outside points')
    ap.add_argument('--annotate-ab',action='store_true',help='Mark A/B areas (manual illustration only)')
    args=ap.parse_args(argv)
    if not any((args.run,args.archive,args.process,args.log)):
        ap.error('Use --run / --archive+--process or --log')
    archive=args.archive or (args.run/'archive.csv' if args.run else None)
    process=args.process or (args.run/'process.csv' if args.run else None)
    style()
    had_any=False
    if archive is not None:
        if not archive.is_file(): raise FileNotFoundError('Archive CSV not found: '+str(archive))
        ar=load_archive(archive)
        scatter(ar,args.outdir,True,zoom=args.reference_axes,annotate_ab=args.annotate_ab)
        scatter(ar,args.outdir,False,zoom=args.reference_axes)
        print('Archive classifications: '+str(ar.flag.value_counts().to_dict()))
        had_any=True
    if process is not None and process.is_file():
        pr=load_process(process)
        evolution(pr,args.outdir)
        had_any=True
    elif args.log:
        evolution(load_log(args.log),args.outdir)
        had_any=True
    elif process is not None:
        raise FileNotFoundError('Process CSV not found: '+str(process))
    if not had_any: raise ValueError('No input was loaded')

if __name__=='__main__':
    main()
