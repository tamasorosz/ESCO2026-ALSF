import numpy as np
import pytest
from src.metrics.box_igd import box_igd,rectangle_igd,nearest_box_distances,nearest_rectangle_distances


def test_exact_inside_boundary_and_excess_not_center_distance():
    r=np.array([[0.,0.]])
    assert box_igd([[.1,-.1]],r,[.1,.1])['box_igd']==0
    assert box_igd([[.09,0]],r,.1)['tolerance_coverage']==1
    assert box_igd([[.3,.12]],r,.1)['box_igd']==pytest.approx(.2)
    assert box_igd([[.100001,0]],r,.1)['box_igd']==pytest.approx(.000001)


def test_anisotropic_widths_nearest_not_first_and_uniform_average():
    r=np.array([[0,0],[4,0]],float);p=np.array([[9,9],[.2,.4],[4.5,.1]])
    distances=nearest_box_distances(p,r,[.1,.5])
    np.testing.assert_allclose(distances,[.1,.4])
    result=box_igd(p,r,[.1,.5])
    assert result['box_igd']==pytest.approx(.25)
    assert result['tolerance_coverage']==0
    assert result['max_box_distance']==pytest.approx(.4)
    # h is half-width; a full side length .2 means h=.1.
    assert box_igd([[.15,0]],[[0,0]],[.2/2,.2/2])['box_igd']==pytest.approx(.05)


def test_asymmetric_rectangles_and_objective_scaling():
    lo=np.array([[1,10],[4,20]],float);hi=np.array([[3,14],[5,22]],float)
    np.testing.assert_allclose(nearest_rectangle_distances([[2,12],[6,23]],lo,hi),[0,1])
    scale=np.array([10,2])
    np.testing.assert_allclose(nearest_rectangle_distances([[6,23]]/scale,lo/scale,hi/scale),[4.5,.5])


def test_zero_width_limit_is_symmetric_chebyshev_not_igd_plus():
    # A dominating solution still has nonzero symmetric distance.
    assert box_igd([[0,0]],[[1,1]],0)['box_igd']==1
    assert box_igd([[0,0]],[[1,1]],.5)['box_igd']==.5
    assert box_igd([[0,0]],[[1,1]],1)['box_igd']==0
    assert box_igd([[1,2],[3,4]],[[1,2],[3,4]],0)['tolerance_coverage']==1


def test_empty_solutions_and_empty_references():
    result=box_igd(np.empty((0,2)),[[0,0]],.1)
    assert np.isinf(result['box_igd']) and np.isinf(result['p95_box_distance'])
    assert result['tolerance_coverage']==0
    with pytest.raises(ValueError):box_igd([[0,0]],np.empty((0,2)),.1)


@pytest.mark.parametrize('width',[-.1,np.inf,np.nan,[.1],[[.1,.1]],[.1,np.nan]])
def test_invalid_tolerances(width):
    with pytest.raises(ValueError):box_igd([[0,0]],[[0,0]],width)


@pytest.mark.parametrize('p,r', [([0,0],[[0,0]]),([[0,0]],[[0]]),([[np.inf,0]],[[0,0]]),([[0,0]],[[np.nan,0]])])
def test_invalid_coordinates(p,r):
    with pytest.raises(ValueError):box_igd(p,r,.1)


def test_chunking_matches_independent_scalar_loop_arbitrary_dimension():
    rng=np.random.default_rng(8);r=rng.normal(size=(17,7));p=rng.normal(size=(23,7));h=np.linspace(0,.3,7)
    expected=np.array([min(max(max(abs(float(x)-float(y))-float(w),0) for x,y,w in zip(point,ref,h)) for point in p) for ref in r])
    for chunk in [1,3,128]:np.testing.assert_allclose(nearest_box_distances(p,r,h,chunk_size=chunk),expected)
    for chunk in [0,-1,1.5,True]:
        with pytest.raises(ValueError):nearest_box_distances(p,r,h,chunk_size=chunk)
    with pytest.raises(ValueError):rectangle_igd(p,np.ones((2,7)),np.zeros((2,7)))


def test_plain_empty_list_is_an_empty_solution_set():
    assert np.isinf(box_igd([],[[0,0]],.1)['box_igd'])


def test_checked_bvls_fallback_matches_independent_dense_box(monkeypatch):
    from analyze_box_igd import checked_objective_extrema
    from di_barba_solenoid_benchmark import DiBarbaSolenoid,SolenoidSettings
    model=DiBarbaSolenoid(SolenoidSettings(n_sections=4))
    center=np.array([.3,.7])
    def fail(*args):raise RuntimeError('Sensitivity minimum solve failed: forced test nonconvergence')
    monkeypatch.setattr(model,'box_objective_extrema',fail)
    lo,hi,diagnostic=checked_objective_extrema(model,center)
    axis=np.linspace(-.1,.1,301);a,b=np.meshgrid(axis,axis)
    x=center+np.column_stack([a.ravel(),b.ravel()])
    f=np.column_stack([np.mean((x@model.K.T-model.b_target)**2,axis=1),np.mean(x**2,axis=1)])
    assert diagnostic['bvls_fallback']
    assert diagnostic['projected_kkt']<1e-8
    assert np.all(lo<=f.min(axis=0)+1e-12)
    np.testing.assert_allclose(hi,f.max(axis=0),atol=1e-12)
    np.testing.assert_allclose(lo,f.min(axis=0),atol=1e-6)


def test_paired_equivalence_is_distinct_from_nonsignificance():
    from analyze_box_igd import paired_statistics
    a=np.arange(20,dtype=float)
    small=paired_statistics(a,a+np.linspace(-.001,.001,20),.01,draws=1000)
    assert small['tost_p']<.05
    big=paired_statistics(a,a+.02+np.linspace(-.001,.001,20),.01,draws=1000)
    assert big['tost_p']>.05
    constant=paired_statistics(a,a,.01,draws=1000)
    assert np.isnan(constant['tost_p'])
    assert not constant['equivalent_proposed_margin']


def test_nominal_feasible_candidate_is_included_without_epsilon(monkeypatch):
    from analyze_box_igd import checked_objective_extrema
    from di_barba_solenoid_benchmark import DiBarbaSolenoid
    model=DiBarbaSolenoid();x=np.full(10,.4);f=np.asarray(model.evaluate(x))
    monkeypatch.setattr(model,'box_objective_extrema',lambda point:(f+1e-14,f+1))
    low,high,info=checked_objective_extrema(model,x)
    np.testing.assert_array_equal(low,f)
    assert info['nominal_candidate_included']


def test_complete_postprocessing_preserves_inputs_and_never_runs_optimizer(tmp_path,monkeypatch):
    import json
    import pandas as pd
    import pymoo.optimize
    from analyze_box_igd import analyze,write_report,hashes
    from di_barba_solenoid_benchmark import DiBarbaSolenoid
    from src.drop_metrics import indicators
    study=tmp_path/'study';(study/'reference').mkdir(parents=True);(study/'analysis').mkdir()
    model=DiBarbaSolenoid();x=np.array([np.zeros(10),np.full(10,.5)])
    cols=[f'X{i+1}' for i in range(10)]+['Obj1','Obj2']
    ref=pd.DataFrame(np.column_stack([x,[model.evaluate(v) for v in x]]),columns=cols)
    ref.to_csv(study/'reference/solenoid.csv',index=False)
    seeds=list(range(20));methods=['baseline','resample','drop'];rows=[]
    (study/'protocol.json').write_text(json.dumps({'seeds':seeds}))
    for method in methods:
        for seed in seeds:
            d=study/'runs/solenoid'/method/f'seed{seed}';d.mkdir(parents=True)
            ref.to_csv(d/'pareto.csv',index=False)
            rows.append(dict(benchmark='solenoid',mode=method,seed=seed,actual_model_calls=20000,
                             **indicators(ref[['Obj1','Obj2']],ref)))
    pd.DataFrame(rows).to_csv(study/'analysis/all_runs.csv',index=False)
    def forbidden(*args,**kwargs):raise AssertionError('Optimization must not run')
    monkeypatch.setattr(pymoo.optimize,'minimize',forbidden)
    monkeypatch.setattr(DiBarbaSolenoid,'reference_pareto',forbidden)
    out=study/'box_igd';before=hashes(study,out)
    frame,summary,effects,settings=analyze(study,out,bootstrap_draws=100)
    write_report(out,frame,summary,effects,settings)
    assert before==hashes(study,out)
    assert len(frame)==60 and frame.decision_box_igd.eq(0).all()
    assert frame.objective_box_igd.eq(0).all()
    assert frame.decision_tolerance_coverage.eq(1).all()
    assert frame.objective_tolerance_coverage.eq(1).all()
    assert (out/'REPORT.md').exists()
    assert len(list(out.glob('*.png')))==3 and len(list(out.glob('*.pdf')))==3
    with pytest.raises(FileExistsError):analyze(study,out)
