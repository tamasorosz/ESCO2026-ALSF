"""Plot Misi Test 1: objective surfaces, sampled Pareto reference, run outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_test1 import evaluate,sample_reference

ROOT=Path(__file__).resolve().parent
FIGS=ROOT/'figures'


def save(fig,basename):
    FIGS.mkdir(exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(FIGS/(basename+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    print('Saved:',FIGS/(basename+'.png'))


def plot():
    t=np.linspace(-5,10,401)
    xx, yy=np.meshgrid(t,t)
    z=evaluate(np.stack((xx,yy),axis=-1))
    fig, axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    for k,ax in enumerate(axes):
        v=z[...,k]
        levels=np.linspace(0,min(100,np.percentile(v,90)),25)
        ax.contourf(xx,yy,v,levels=levels,extend='max',cmap='viridis')
        ax.contour(xx,yy,v,levels=levels[::4],colors='k',alpha=.24,linewidths=.55)
        ax.set(xlabel='$x_1$',ylabel='$x_2$',title='$f_{}$'.format(k+1),
               xlim=(-5,10),ylim=(-5,10),aspect='equal')
    axes[1].plot(2.5,2.5,'rx',markersize=9,markeredgewidth=2,label='Gaussian peak')
    axes[1].legend(frameon=False)
    save(fig,'01_objective_surfaces')

    xr,fr=sample_reference(n_grid=501)
    FIGS.mkdir(exist_ok=True)
    pd.DataFrame(np.column_stack((xr,fr)),columns=['X1','X2','Obj1','Obj2']).to_csv(
        FIGS/'sampled_pareto_reference.csv',index=False)

    all_runs=sorted((ROOT/'results').glob('test1_*/*pareto.csv'))
    fig,ax=plt.subplots(figsize=(7.7,5.1))
    ax.plot(fr[:,0],fr[:,1],color='0.4',lw=1.4,alpha=.9,
            label='Sampled grid reference (approximate)')
    for file in all_runs:
        data=pd.read_csv(file)
        if len(data):
            ax.scatter(data['Obj1'],data['Obj2'],s=14,alpha=.75,
                       label=file.parent.name)
    ax.set(xlabel='$f_1$ (min)',ylabel='$f_2$ (min)',title='Test 1: Pareto-front comparison')
    ax.grid(alpha=.3)
    ax.legend(frameon=False,fontsize=8)
    save(fig,'02_pareto_comparison')

    fig,ax=plt.subplots(figsize=(6.7,5.6))
    ax.plot(xr[:,0],xr[:,1],'.',ms=2.1,color='0.5',label='Grid reference (approximate)')
    for file in all_runs:
        data=pd.read_csv(file)
        if len(data):
            ax.scatter(data['X1'],data['X2'],s=13,alpha=.75,label=file.parent.name)
    ax.set(xlabel='$x_1$',ylabel='$x_2$',title='Test 1: decision-space Pareto sets',
           xlim=(-5,10),ylim=(-5,10),aspect='equal')
    ax.grid(alpha=.3)
    ax.legend(frameon=False,fontsize=8)
    save(fig,'03_pareto_decision_space')

    for file in sorted((ROOT/'results').glob('test1_alsf_*/archive.csv')):
        data=pd.read_csv(file)
        if 'flag' not in data:
            continue
        fig,ax=plt.subplots(figsize=(7,5.5))
        colors={'NEI':'0.65','robust':'tab:green','non-robust':'tab:red'}
        for flag in ['NEI','robust','non-robust']:
            g=data[data.flag==flag]
            ax.scatter(g.X1,g.X2,s=6 if flag=='NEI' else 10,alpha=.55,
                       c=colors[flag],label='{} (n={})'.format(flag,len(g)))
        ax.set(xlabel='$x_1$',ylabel='$x_2$',title=file.parent.name+' — local flags',
               xlim=(-5,10),ylim=(-5,10),aspect='equal')
        ax.legend(frameon=False,fontsize=8)
        save(fig,'04_flags_'+file.parent.name)

if __name__=='__main__':
    plot()
