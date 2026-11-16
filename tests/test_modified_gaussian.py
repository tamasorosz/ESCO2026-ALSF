"""Independent checks for the separate modified Gaussian experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import differential_evolution
from benchmark_test1_modified_gaussian import evaluate,to_optimization,to_physical
from src.modified_gaussian_reference import endpoint,reference,box_extrema,sensitivity
from src.local_variation import LocalVariationState
from src.drop_metrics import indicators


def test_physical_and_mixed_direction():
    np.testing.assert_allclose(evaluate([0,0]),[0,-50])
    np.testing.assert_allclose(evaluate([2.5,2.5]),[12.5,87.5])
    np.testing.assert_allclose(evaluate([5,5]),[50,0],atol=1e-12)
    f=evaluate([[2.5,2.5],[5,5]])
    g=to_optimization(f)
    assert np.all(g[0]<g[1])
    np.testing.assert_array_equal(to_physical(g),f)


def test_reference_endpoint_and_global_domain():
    end=endpoint();assert abs(end-2.50024998)<1e-8
    np.testing.assert_allclose(evaluate([end,end]),[12.5025,87.50125],atol=3e-7)
    r=reference(101)
    assert (np.diff(r.Obj1)>0).all() and (np.diff(r.Obj2)>0).all()
    rng=np.random.default_rng(771)
    x=rng.uniform(-5,10,(20000,2));q=(x*x).sum(axis=1);f=evaluate(x)
    t=np.minimum(np.sqrt(q/2),end)
    dominating=evaluate(np.column_stack((t,t)))
    assert np.all(dominating[:,0]<=f[:,0]+1e-10)
    assert np.all(dominating[:,1]>=f[:,1]-1e-10)


@pytest.mark.parametrize('x',[[2.5,2.5],[2.4,2.6],[0,0],[-5,10],[2.7,2.5],[2.50025,2.50025]])
def test_complete_extrema_against_independent_optimizer_and_grid(x):
    x=np.array(x);lo=np.maximum(-5,x-.1);hi=np.minimum(10,x+.1)
    low,high=box_extrema(x)
    bounds=list(zip(lo,hi))
    for sign,expected in [(1,low[1]),(-1,high[1])]:
        result=differential_evolution(lambda p:sign*evaluate(p)[1],bounds,seed=33,tol=1e-11,x0=np.clip([2.5,2.5],lo,hi))
        assert abs(sign*result.fun-expected)<1e-7
    grid=np.stack(np.meshgrid(np.linspace(lo[0],hi[0],101),np.linspace(lo[1],hi[1],101)),axis=-1).reshape(-1,2)
    f=evaluate(grid)
    assert np.all(f>=low-1e-9) and np.all(f<=high+1e-9)
    a,r=sensitivity(x)
    assert np.all(a>=0) and np.all((r>=0)&(r<=2+1e-10))


def test_sign_invariance_of_archive_evidence():
    x=np.array([[2.5,2.5],[2.55,2.55],[2.59,2.59]])
    a=LocalVariationState([.1,.1]);b=LocalVariationState([.1,.1])
    a.update(x,evaluate(x));b.update(x,to_optimization(evaluate(x)))
    np.testing.assert_array_equal(a.verified_violation,b.verified_violation)
    np.testing.assert_allclose(a.max_observed_variation,b.max_observed_variation)
    assert a.verified_violation.all()


def test_integration_reproducible_accounting_and_indicators(tmp_path):
    from run_test1_modified_gaussian import run_case
    results=[]
    for run in range(2):
        root=tmp_path/str(run);(root/'reference').mkdir(parents=True)
        reference(51).to_csv(root/'reference/test1.csv',index=False)
        for mode in ['baseline','resample','drop']:
            result=run_case(root,'test1',mode,42,generations=8,pop_size=20,flag_start=2)
            out=root/'runs/test1'/mode/'seed42'
            h=pd.read_csv(out/'history.csv');p=pd.read_csv(out/'process.csv')
            e=pd.read_csv(out/'evaluated.csv.gz');events=pd.read_csv(out/'screening.csv.gz')
            assert len(h)==8 and len(e)==result['actual_model_calls']==result['evaluator_calls']
            assert len(pd.read_csv(out/'final_population.csv'))==20
            if mode=='drop':
                assert len(e)==160-int(p.n_dropped.sum())
                assert int(events.rejected.sum())==int(p.n_dropped.sum())
            f=pd.read_csv(out/'pareto.csv');ref=pd.read_csv(root/'reference/test1.csv')
            ref[['Obj1','Obj2']]=to_optimization(ref[['Obj1','Obj2']])
            metrics=indicators(to_optimization(f[['Obj1','Obj2']]),ref)
            for key,value in metrics.items():assert abs(value-result[key])<1e-12
            results.append((mode,run,f,result['actual_model_calls']))
    for mode in ['baseline','resample','drop']:
        a,b=[r for r in results if r[0]==mode]
        pd.testing.assert_frame_equal(a[2],b[2]);assert a[3]==b[3]


def test_observed_drop_rejects_before_evaluation():
    from src.modified_gaussian_observation import ObservedDrop
    from pymoo.core.population import Population
    events=[]
    policy=ObservedDrop(dict(generation=51),events,[.1,.1])
    policy.policy.update(np.array([[2.5,2.5]]),np.array([True]))
    proposal=Population.new(X=np.array([[2.52,2.52],[1.,1.]]))
    accepted=policy.filter_population(proposal)
    np.testing.assert_array_equal(accepted.get('X'),[[1.,1.]])
    assert policy.last_counts==dict(n_proposed=2,n_dropped=1,n_accepted=1)
    assert events[0][-1] and not events[1][-1]


def test_paired_statistics_and_multiplicity():
    from analyze_modified_gaussian import paired_effect,holm
    a=np.arange(20,dtype=float)
    result=paired_effect(a,a+2)
    assert result['mean_difference']==result['ci95_low']==result['ci95_high']==2
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    null=paired_effect(a,a)
    assert null['p_raw']==1 and null['mean_difference']==0


def test_shipped_primary_results_are_complete_and_consistent():
    root=Path(__file__).resolve().parents[1]/'docs/test1_modified_gaussian'
    if not (root/'analysis/all_runs.csv').exists():pytest.skip('Primary artifact bundle not available')
    protocol=json.loads((root/'protocol.json').read_text())
    data=pd.read_csv(root/'analysis/all_runs.csv')
    assert len(data)==60
    expected={(mode,seed) for mode in ['baseline','resample','drop'] for seed in protocol['seeds']}
    assert set(zip(data['mode'],data.seed))==expected
    assert (data.generations==200).all()
    np.testing.assert_allclose(data.saved_percent,100*(20000-data.actual_model_calls)/20000)
    effects=pd.read_csv(root/'analysis/paired_effects.csv')
    for row in effects.itertuples():
        pivot=data.pivot(index='seed',columns='mode',values=row.metric)
        assert abs(row.mean_difference-(pivot[row.method_b]-pivot[row.method_a]).mean())<1e-9
        assert row.p_holm>=row.p_raw-1e-12
    for metric in ['hypervolume','igd_plus']:
        for mode in ['baseline','resample','drop']:
            subset=data[data['mode']==mode]
            for row in subset.itertuples():
                folder=root/'runs/test1'/mode/f'seed{row.seed}'
                h=pd.read_csv(folder/'history.csv')
                assert len(h)==200 and h.evaluations.iloc[-1]==row.actual_model_calls
                assert abs(h[metric].iloc[-1]-getattr(row,metric))<1e-12
