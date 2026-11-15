"""Independent finite perturbation diagnostics; never feeds evaluations to ALSF.

Compare DBSCAN and bounded reporting regions on an existing archive without
modifying it. All additional benchmark evaluations are diagnostic only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_test1 import evaluate, LOWER, UPPER
from src.local_variation import relative_pairwise_variation
from src.find_robust_clusters import classify_dataframe
from plot_original_three_ALSF import masks


def perturbation_test(points, tolerances=(.1,.1), grid_size=21):
    """Center-to-perturbation differences on a finite tensor grid, not a proof."""
    points = np.asarray(points, dtype=float).reshape(-1,2)
    axis = np.linspace(-1,1,grid_size)
    u,v = np.meshgrid(axis,axis)
    offsets = np.column_stack([u.ravel(),v.ravel()])*np.asarray(tolerances)
    maxima = []
    for point in points:
        trial = point+offsets
        trial = trial[np.all((trial >= LOWER)&(trial <= UPPER),axis=1)]
        variations = relative_pairwise_variation(evaluate(point),evaluate(trial),[1e-3]*2)
        maxima.append(variations.max(axis=0))
    values = np.asarray(maxima).reshape(-1,2)
    return values, np.any(values > .1,axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path)
    p.add_argument('--outdir', type=Path, required=True)
    args=p.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=False)
    report={'diagnostic_grid_size':21,'tolerance':[.1,.1],
            'thresholds':[.1,.1],'floors':[1e-3,1e-3],
            'interpretation':'Finite perturbations establish counterexamples, not guaranteed robustness.'}
    for label,center in [('A',[2.5,2.5]),('B',[1.8,1.87])]:
        maxima,bad=perturbation_test([center])
        report[label]={'center':center,'max_relative_variations':maxima[0].tolist(),
                       'violating_perturbation_found':bool(bad[0])}
    if args.archive:
        raw=pd.read_csv(args.archive)
        classified,stats=classify_dataframe(raw,[.1,.1],.1,.1)
        evidence=classified[['flag','verified_violation','local_n','R1_local','R2_local']]
        comparisons=[]
        for eps in [1.,1.25,1.5,1.75,2.]:
            other,summary=classify_dataframe(raw,[.1,.1],.1,.1,dbscan_eps=eps)
            pd.testing.assert_frame_equal(evidence,other[evidence.columns])
            comparisons.append({'eps':eps,**summary})
        pd.DataFrame(comparisons).to_csv(args.outdir/'partition_comparison.csv',index=False)
        classified.to_csv(args.outdir/'classified_archive.csv',index=False)
        # Audit ALL empirically stable samples in the marked A/B regions.
        audit=[]
        for label,mask in masks(classified).items():
            ids=np.flatnonzero(mask & classified.flag.eq('robust').to_numpy())
            maxima,bad=perturbation_test(classified.iloc[ids][['X1','X2']].to_numpy())
            report[label]['archive_stable_points_tested']=len(ids)
            report[label]['archive_stable_points_with_counterexample']=int(bad.sum())
            for idx,maximum,violation in zip(ids,maxima,bad):
                audit.append({'archive_row':int(idx),'region':label,
                              'R1_perturbation':maximum[0],'R2_perturbation':maximum[1],
                              'counterexample_found':bool(violation)})
        pd.DataFrame(audit,columns=['archive_row','region','R1_perturbation',
            'R2_perturbation','counterexample_found']).to_csv(args.outdir/'perturbation_audit.csv',index=False)
    (args.outdir/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
