from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from src.find_robust_clusters import classify_dataframe, flag_by_clusters
from src.local_variation import LocalVariationState
from src.offspring_policy import VerifiedViolationFilter
from validate_alsf import perturbation_test


def archive():
    return pd.DataFrame({'X1':[0,.08,.16,.24,1], 'X2':[0]*5,
                         'Obj1':[1,1,1.02,1.5,1], 'Obj2':[1]*5})


def test_geometry_changes_never_change_evidence_or_exclusion():
    state=LocalVariationState([.1,.1])
    first,stats=classify_dataframe(archive(),[.1,.1],.1,.1,min_samples=2,
                                 dbscan_min_samples=2,state=state)
    cols=['flag','verified_violation','local_n','R1_local','R2_local']
    assert stats['n_mixed_clusters']==1
    for eps,minimum,width in [(.2,2,.5),(2.,1,2.),(20.,5,1.)]:
        other,_=classify_dataframe(archive(),[.1,.1],.1,.1,min_samples=2,
                    dbscan_eps=eps,dbscan_min_samples=minimum,partition_width=width,state=state)
        pd.testing.assert_frame_equal(first[cols],other[cols])
        policy=VerifiedViolationFilter([.1,.1])
        policy.update(other[['X1','X2']].to_numpy(),other.verified_violation)
        assert not policy.forbidden([0,0])  # same mixed chain, outside bad boxes
        assert policy.forbidden([.2,0])
    mixed=first[first.cluster_status=='mixed']
    assert mixed.cluster_fraction_robust.eq(.5).all()
    assert mixed.cluster_fraction_non_robust.eq(.5).all()
    assert mixed.cluster_fraction_nei.eq(0).all()


def test_bounded_regions_have_diameter_and_keep_noise():
    out,_=classify_dataframe(archive(),[.1,.1],.1,.1,partition_width=1.)
    assert out.cluster_id.eq(-1).all()
    assert out.region_id.notna().all()
    for _,group in out.groupby('region_id'):
        assert np.max(np.ptp(group[['X1','X2']].to_numpy()/.1,axis=0))<=1.+1e-12
    assert out[out.is_stable_representative].flag.eq('robust').all()


def test_file_interface_separates_dbscan_and_local_policy(tmp_path):
    path=tmp_path/'archive.csv'
    archive().to_csv(path,index=False)
    first,_=flag_by_clusters(path,.3,.1,.1,min_samples=2,tolerances=[.1,.1],local_min_samples=2)
    other,_=flag_by_clusters(path,3.,.1,.1,min_samples=4,tolerances=[.1,.1],local_min_samples=2)
    assert first.flag.equals(other.flag)
    assert not first.cluster_id.equals(other.cluster_id)


def test_reused_cache_with_different_local_policy_is_rejected():
    with pytest.raises(ValueError,match='parameters differ'):
        classify_dataframe(archive(),[.1,.1],.1,.1,state=LocalVariationState([.1,.1],radius=2.))


def test_finite_perturbations_find_A_and_B_counterexamples():
    maxima,bad=perturbation_test([[2.5,2.5],[1.8,1.87]])
    assert bad.all()
    assert maxima[0,1]>.7
    assert maxima[1,0]>.1
    # A sparse benign archive can miss B sensitivity despite ten observations.
    x=np.array([1.8,1.87])+np.linspace(-.001,.001,10)[:,None]*np.ones((1,2))
    from benchmark_test1 import evaluate
    f=evaluate(x)
    out,_=classify_dataframe(pd.DataFrame(np.column_stack([x,f]),
                columns=['X1','X2','Obj1','Obj2']),[.1,.1],.1,.1)
    assert out.flag.eq('robust').all()
    assert perturbation_test(x)[1].all()  # empirical label is not a certificate


def test_actual_nsga2_repair_rejects_forbidden_samples_deterministically():
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from src.preprocessing import OffspringRepair
    class P(Problem):
        def __init__(self):
            super().__init__(n_var=2,n_obj=2,xl=np.full(2,-1.),xu=np.ones(2))
            self.seen=[]
        def _evaluate(self,X,out,*args,**kwargs):
            self.seen.extend(X.copy())
            out['F']=np.column_stack([np.sum(X**2,axis=1),np.sum((X-.8)**2,axis=1)])
    runs=[]
    for _ in range(2):
        repair=OffspringRepair('unused',{},1.,tolerances=[.1,.1],rng=np.random.default_rng(42))
        repair.policy.update(np.array([[0.,0.]]),np.array([True]))
        problem=P()
        initial=np.array([[.01,.01],[.02,.02],[-.03,.03],[.04,-.04]])
        algorithm=NSGA2(pop_size=4,sampling=initial,repair=repair)
        minimize(problem,algorithm,('n_gen',3),seed=42,copy_algorithm=False)
        seen=np.asarray(problem.seen)
        assert not any(repair.policy.forbidden(x) for x in seen)
        info=repair.pop_diagnostics()
        assert info['n_regenerated']>=4
        assert info['n_forbidden_candidates']==info['n_regenerated']
        assert info['n_attempts']>=info['n_regenerated']
        runs.append(seen)
    np.testing.assert_array_equal(*runs)


@pytest.mark.parametrize('script',['plot_three_figures.py','plot_original_three_ALSF.py','plot_alsf_elsevier.py','plot_original_3.py'])
def test_plotting_entrypoints_with_measured_csv_schema(tmp_path,script):
    out,_=classify_dataframe(archive(),[.1,.1],.1,.1,min_samples=2,dbscan_min_samples=2)
    out.to_csv(tmp_path/'archive.csv',index=False)
    pd.DataFrame([dict(generation=3,n_archive=5,n_robust=2,n_non_robust=2,n_nei=1,
        n_robust_clusters=0,n_non_robust_clusters=0,n_mixed_clusters=1,n_nei_clusters=0,
        n_regions=4,n_mixed_regions=0,n_regenerated=2,n_attempts=3)]).to_csv(tmp_path/'process.csv',index=False)
    target=tmp_path/'plots'
    args=['--archive',str(tmp_path/'archive.csv'),'--process',str(tmp_path/'process.csv')] if script=='plot_three_figures.py' else ['--run',str(tmp_path)]
    subprocess.run([sys.executable,script,*args,'--outdir',str(target)],check=True,capture_output=True)
    assert len(list(target.glob('*.png')))==3
    assert len(list(target.glob('*.pdf')))==3
