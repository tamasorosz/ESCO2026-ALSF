import numpy as np
import pandas as pd
from benchmark_test1 import evaluate
from benchmark_test1_minmax import to_optimization,to_physical,nondominated_indices
from src.test1_minmax_reference import reference_front,branches,peak_features
from src.local_variation import relative_pairwise_variation,LocalVariationState


def test_physical_peak_and_explicit_adapter():
    np.testing.assert_allclose(evaluate([2.5,2.5]),[12.5,112.5])
    np.testing.assert_allclose(to_optimization(evaluate([2.5,2.5])),[12.5,-112.5])
    np.testing.assert_array_equal(to_physical(to_optimization([[1,2],[3,4]])),[[1,2],[3,4]])


def test_mixed_dominance_and_center_is_not_assumed_efficient():
    f=np.array([[1,1],[1,2],[2,1],[2,3]])
    np.testing.assert_array_equal(nondominated_indices(f),[1,3])
    features=peak_features();point=np.full(2,np.sqrt(features['peak_local_maximum_q']/2))
    f=evaluate(point);center=evaluate([2.5,2.5])
    assert f[0]<center[0] and f[1]>center[1]


def test_full_domain_upper_envelope_and_peak_resolution():
    ref,features=reference_front()
    assert ref.peak_branch.any()
    assert ref.loc[ref.peak_branch,'Obj1'].between(12.47,12.5).all()
    assert np.all(np.diff(ref.Obj1)>0) and np.all(np.diff(ref.Obj2)>0)
    x=np.random.default_rng(13).uniform(-5,10,(10000,2));f=evaluate(x)
    near=f[:,0]<=50
    n,p=branches(f[near,0])
    assert np.all(f[near,1]<=np.maximum(n,p)+1e-10)
    assert np.all(f[~near,1]<200)
    np.testing.assert_allclose(ref.iloc[[0,-1]][['Obj1','Obj2']].to_numpy(float),[[0,50],[50,200]])


def test_sensitivity_is_sign_invariant_including_cached_flags():
    rng=np.random.default_rng(8);x=2.5+rng.uniform(-.12,.12,(50,2));f=evaluate(x);g=to_optimization(f)
    np.testing.assert_array_equal(relative_pairwise_variation(f[0],f,[.001]*2),
                                  relative_pairwise_variation(g[0],g,[.001]*2))
    a=LocalVariationState([.1,.1]);b=LocalVariationState([.1,.1])
    np.testing.assert_array_equal(a.update(x,f),b.update(x,g))
    np.testing.assert_array_equal(a.verified_violation,b.verified_violation)
    np.testing.assert_array_equal(a.max_observed_variation,b.max_observed_variation)


def test_pymoo_adapter_and_physical_exports(tmp_path):
    import json
    import pandas as pd
    from run_test1_original_minmax import run_case
    from src.drop_metrics import indicators
    reference,_=reference_front(501,101)
    (tmp_path/'reference').mkdir();reference.to_csv(tmp_path/'reference/test1.csv',index=False)
    for mode in ['baseline','resample','drop']:
        summary=run_case(tmp_path,'test1',mode,1,8,24,3)
        out=tmp_path/'runs/test1'/mode/'seed1'
        front=pd.read_csv(out/'pareto.csv');archive=pd.read_csv(out/'archive.csv')
        np.testing.assert_allclose(front[['Obj1','Obj2']],evaluate(front[['X1','X2']]),rtol=1e-12)
        np.testing.assert_allclose(archive[['Obj1','Obj2']],evaluate(archive[['X1','X2']]),rtol=1e-12)
        # Actual pymoo survivor front obeys corrected dominance, including exports.
        assert len(nondominated_indices(front[['Obj1','Obj2']]))==len(front)
        gref=reference.copy();gref[['Obj1','Obj2']]=to_optimization(reference[['Obj1','Obj2']])
        expected=indicators(to_optimization(front[['Obj1','Obj2']]),gref)
        np.testing.assert_allclose(summary['hypervolume'],expected['hypervolume'])
        assert summary['actual_model_calls']==summary['evaluator_calls']
        if mode=='drop':assert summary['actual_model_calls']+summary['dropped']==192
        else:assert summary['actual_model_calls']==192


def test_observation_does_not_change_policy_decisions():
    from src.test1_minmax_observation import ObservedDrop,ObservedRepair
    from src.offspring_drop import OffspringDropFilter
    from src.preprocessing import OffspringRepair
    from pymoo.core.problem import Problem
    coords=np.array([[2.5,2.5],[0,0]])
    observation=dict(coords=coords,generation=51);events=[]
    observed=ObservedDrop(observation,events,[.1,.1]);plain=OffspringDropFilter([.1,.1])
    for p in [observed.policy,plain.policy]:p.update(coords,[True,False])
    np.testing.assert_array_equal(observed.keep_mask(coords),plain.keep_mask(coords))
    assert events[0]['rejected']
    problem=Problem(n_var=2,xl=[-5,-5],xu=[10,10])
    args=('memory',{},1.)
    o=ObservedRepair(observation,events,*args,tolerances=[.1,.1],rng=np.random.default_rng(1))
    p=OffspringRepair(*args,tolerances=[.1,.1],rng=np.random.default_rng(1))
    for a in [o,p]:a.policy.update(coords,[True,False])
    np.testing.assert_array_equal(o._do(problem,coords),p._do(problem,coords))
    assert o.pop_diagnostics()==p.pop_diagnostics()
