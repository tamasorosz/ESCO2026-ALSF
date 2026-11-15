import numpy as np
import pandas as pd
import pytest
from scipy.optimize import lsq_linear
from di_barba_solenoid_benchmark import DiBarbaSolenoid,SolenoidSettings
from src.find_robust_clusters import classify_dataframe
from src.local_variation import relative_pairwise_variation


def test_quadratic_hessians_and_all_gradient_coordinates():
    model=DiBarbaSolenoid()
    for hessian in model.hessians():
        np.testing.assert_allclose(hessian,hessian.T,atol=1e-15)
        assert np.linalg.eigvalsh(hessian).min()>-1e-12
    x=np.linspace(-1.3,1.4,10)
    for i in range(10):
        step=np.eye(10)[i]*1e-6
        np.testing.assert_allclose((np.array(model.evaluate(x+step))-model.evaluate(x-step))/2e-6,
                                  model.gradient(x)[:,i],rtol=1e-6,atol=1e-8)


def test_reference_solves_bounds_kkt_and_nondominance():
    model=DiBarbaSolenoid()
    front=model.reference_pareto(41,adaptive_gap=.04,max_points=300)
    assert front.solver_success.all()
    assert model.reference_diagnostics.bound_violation.max()<1e-12
    assert model.reference_diagnostics.kkt_residual.max()<1e-7
    xy=front[['Obj1','Obj2']].to_numpy()
    assert np.all(np.diff(xy[:,0])>=0)
    assert np.all(np.diff(xy[:,1])<0)
    assert front.iloc[-1].Obj2==0
    assert len(front)>30
    # Independent analytic unconstrained weighted solution for a positive weight.
    lam=.1
    x=np.linalg.solve(model.K.T@model.K/41+lam*np.eye(10)/10,model.K.T@np.ones(41)/41)
    assert np.all(abs(x)<2)
    got=model._weighted_solution(lam)
    np.testing.assert_allclose([got[f'X{i+1}'] for i in range(10)],x,atol=1e-10)


def test_exact_sensitivity_against_exhaustive_2d_and_random_10d():
    # Two independent currents permit an independent exhaustive dense grid.
    model=DiBarbaSolenoid(SolenoidSettings(n_sections=4))
    x=np.array([1.98,-.03])
    lo=np.maximum(model.lower,x-.1);hi=np.minimum(model.upper,x+.1)
    u,v=np.meshgrid(np.linspace(lo[0],hi[0],301),np.linspace(lo[1],hi[1],301))
    points=np.column_stack([u.ravel(),v.ravel()])
    values=np.column_stack([np.mean((points@model.K.T-1)**2,axis=1),np.mean(points**2,axis=1)])
    low,high=model.box_objective_extrema(x)
    assert np.all(low<=values.min(axis=0)+1e-10)
    np.testing.assert_allclose(high,values.max(axis=0),atol=1e-12)
    exact=model.exact_relative_box_sensitivity(x)
    sampled=relative_pairwise_variation(model.evaluate(x),values,[.001,.001]).max(axis=0)
    np.testing.assert_allclose(exact,sampled,atol=1e-5)
    model=DiBarbaSolenoid()
    for x in [np.zeros(10),np.full(10,2.),np.linspace(-1,1,10)]:
        trials=np.clip(x+np.random.default_rng(4).uniform(-.1,.1,(500,10)),model.lower,model.upper)
        low,high=model.box_objective_extrema(x)
        values=np.array([model.evaluate(y) for y in trials])
        assert np.all(values>=low-1e-9) and np.all(values<=high+1e-9)
        assert np.all(model.exact_relative_box_sensitivity(x)>=
            relative_pairwise_variation(model.evaluate(x),values,[.001]*2).max(axis=0)-1e-9)
    assert model.exact_relative_box_sensitivity(np.zeros(10))[1]==pytest.approx(1.)
    assert model.exact_relative_box_sensitivity(np.zeros(10),delta_j=100)[1]==pytest.approx(1e-5)
    assert model.exact_box_sensitivity(np.zeros(10),delta_j=0)==(0.,0.)
    with pytest.raises(ValueError):model.box_objective_extrema(np.full(10,2.01))


def test_disabling_sparse_cells_preserves_all_local_evidence():
    model=DiBarbaSolenoid()
    x=np.random.default_rng(4).uniform(-1,1,(30,10))
    df=pd.DataFrame(np.column_stack([x,[model.evaluate(y) for y in x]]),columns=[f'X{i+1}' for i in range(10)]+['Obj1','Obj2'])
    before,_=classify_dataframe(df,.1,.1,.1)
    after,stats=classify_dataframe(df,.1,.1,.1,partition_width=None)
    columns=['flag','verified_violation','local_n','R1_local','R2_local','cluster_id']
    pd.testing.assert_frame_equal(before[columns],after[columns])
    assert 'region_id' not in after
    assert stats['n_regions']==0


def test_classification_metrics_keep_abstentions_and_undefined_rates():
    from solenoid_validation import classification_metrics
    audit=pd.DataFrame({'flag':['robust','robust','non-robust','NEI'],
                        'exact_sensitive':[True,False,True,True]})
    metrics=classification_metrics(audit)
    assert metrics['false_stable_rate']==.5
    assert metrics['sensitive_precision']==1.
    assert metrics['sensitive_recall_classified']==.5
    assert metrics['sensitive_detection_all']==pytest.approx(1/3)
    assert metrics['classification_coverage']==.75
    metrics=classification_metrics(audit.iloc[3:])
    assert metrics['false_stable_rate'] is None
    assert metrics['sensitive_precision'] is None


def test_box_extrema_vertices_and_independent_minimum():
    from itertools import product
    from scipy.optimize import minimize
    model=DiBarbaSolenoid();center=np.linspace(-.3,.7,10)
    lo=center-.1;hi=center+.1
    corners=np.where(np.array(list(product([False,True],repeat=10))),hi,lo)
    values=np.array([model.evaluate(x) for x in corners])
    low,high=model.box_objective_extrema(center)
    np.testing.assert_allclose(high,values.max(axis=0),atol=1e-12)
    result=minimize(lambda x:model.evaluate(x)[0],center,
        jac=lambda x:model.gradient(x)[0],bounds=list(zip(lo,hi)),
        method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-12,'maxiter':2000})
    assert result.success
    assert low[0]==pytest.approx(result.fun,abs=1e-9)


def test_matched_smoke_runs_budget_and_inactive_policy(tmp_path):
    from solenoid_validation import run_optimization,sensitivity_phase,archive_audit_phase
    model=DiBarbaSolenoid();reference=model.reference_pareto(41)
    for mode in ['baseline','alsf']:
        result=run_optimization(reference,tmp_path/'full'/f'{mode}_seed3',mode,seed=3,generations=4,
                                pop_size=12,offsprings=12,flag_start=2)
        assert result['evaluations']==48
        assert result['model_calls']==48
        assert result['total_regenerated']==0
        assert result['total_offspring_seen']>0
        assert (tmp_path/'full'/f'{mode}_seed3'/'pareto_sensitivity.csv').exists()
    a=pd.read_csv(tmp_path/'full/baseline_seed3/archive.csv');b=pd.read_csv(tmp_path/'full/alsf_seed3/archive.csv')
    pd.testing.assert_frame_equal(a,b)
    with pytest.raises(FileExistsError):
        run_optimization(reference,tmp_path/'full/baseline_seed3','baseline',generations=1)

    # The complete plotting workflow uses measured smoke artifacts and exact audits.
    (tmp_path/'reference').mkdir()
    reference.to_csv(tmp_path/'reference/reference.csv',index=False)
    reference.to_csv(tmp_path/'reference/reference_coarse.csv',index=False)
    sensitivity_phase(reference,tmp_path/'sensitivity')
    archive_audit_phase(tmp_path,tmp_path/'archive_audit')
    from plot_solenoid_validation import plot_study
    plot_study(tmp_path,tmp_path/'figures')
    assert len(list((tmp_path/'figures').glob('*.png')))==5
    assert len(list((tmp_path/'figures').glob('*.pdf')))==5
