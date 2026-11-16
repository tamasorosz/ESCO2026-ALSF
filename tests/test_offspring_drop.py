import numpy as np
import pandas as pd
import pytest
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from src.offspring_drop import OffspringDropFilter,DropNSGA2
from src.find_robust_clusters import classify_dataframe


def test_filter_uses_verified_points_not_component_flags():
    f=OffspringDropFilter([.1,.1])
    data=pd.DataFrame({'X1':[0,2],'X2':[0,2],'verified_violation':[True,False],
                       'flag':['non-robust','robust'],'cluster_id':[0,0]})
    f.update_flags(data)
    pop=Population.new(X=np.array([[.05,.05],[2,2],[.2,0]]))
    out=f.filter_population(pop)
    np.testing.assert_array_equal(out.get('X'),[[2,2],[.2,0]])
    assert f.last_counts==dict(n_proposed=3,n_dropped=1,n_accepted=2)
    assert len(f.filter_population(Population.empty()))==0


class InstrumentedProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2,n_obj=2,xl=np.full(2,-2.),xu=np.full(2,2.))
        self.calls=[]
    def _evaluate(self,X,out,*args,**kwargs):
        self.calls.extend(X.copy())
        out['F']=np.column_stack([np.sum(X**2,axis=1),np.sum((X-1)**2,axis=1)])


class FixedMating:
    def __init__(self,points):self.points=np.asarray(points);self.calls=0
    def do(self,*args,**kwargs):
        self.calls+=1
        return Population.new(X=self.points.copy())


@pytest.mark.parametrize('all_dropped',[False,True])
def test_actual_pymoo_never_evaluates_dropped_and_never_refills(all_dropped):
    policy=OffspringDropFilter([.1,.1]);problem=InstrumentedProblem()
    initial=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    proposals=np.array([[0.,0.],[.04,.04],[1.5,1.5],[1.7,1.7]])
    if all_dropped:proposals=np.array([[0.,0.],[.04,.04],[-.04,0.],[.02,-.03]])
    mating=FixedMating(proposals)
    algorithm=DropNSGA2(policy,pop_size=4,n_offsprings=4,sampling=initial)
    algorithm.mating=mating
    counts=[];sizes=[];initialized=[]
    class Monitor(Callback):
        def notify(self,a):
            sizes.append(len(a.pop));counts.append(a.evaluator.n_eval)
            if a.n_gen==1:
                initialized.append(a.pop.get('X').copy())
                policy.policy.update(np.array([[0.,0.]]),np.array([True]))
    result=minimize(problem,algorithm,('n_gen',5),seed=42,callback=Monitor(),copy_algorithm=False)
    assert mating.calls==4  # exactly one proposal batch/generation; no refill
    assert algorithm.mating_batches==4
    assert len(sizes)==5 and sizes==[4]*5
    accepted=0 if all_dropped else 2
    assert result.algorithm.evaluator.n_eval==4+4*accepted
    assert len(problem.calls)==4+4*accepted
    # The instrumented objective records EVERY evaluated coordinate. None of
    # the forbidden proposals after initialization may appear in this record.
    assert not any(policy.policy.forbidden(x) for x in problem.calls[4:])
    for rejected in proposals[:(4 if all_dropped else 2)]:
        assert not any(np.array_equal(rejected,x) for x in problem.calls)
    if all_dropped:
        np.testing.assert_array_equal(result.algorithm.pop.get('X'),initialized[0])
        assert counts==[4]*5
    assert not algorithm.termination.force_termination


def test_real_archive_updates_change_drop_decisions():
    f=OffspringDropFilter([.1,.1])
    data=pd.DataFrame({'X1':[0,1],'X2':[0,1],'Obj1':[1.,1.],'Obj2':[1.,1.]})
    out,_=classify_dataframe(data,[.1,.1],.1,.1)
    f.update_flags(out)
    assert f.keep_mask([[0,0]]).tolist()==[True]
    data.loc[2]=[.05,.05,2.,1.]
    out,_=classify_dataframe(data,[.1,.1],.1,.1)
    f.update_flags(out)
    assert f.keep_mask([[0,0]]).tolist()==[False]


def test_seed_reproducibility_and_empty_policy_matches_nsga2():
    from pymoo.algorithms.moo.nsga2 import NSGA2
    outputs=[]
    for kind in ['drop','drop','baseline']:
        p=InstrumentedProblem()
        a=DropNSGA2(OffspringDropFilter([.1,.1]),pop_size=12) if kind=='drop' else NSGA2(pop_size=12)
        minimize(p,a,('n_gen',5),seed=13,copy_algorithm=False)
        outputs.append(np.array(p.calls))
    np.testing.assert_array_equal(outputs[0],outputs[1])
    np.testing.assert_array_equal(outputs[0],outputs[2])


def test_runner_accounting_and_no_overwrite(tmp_path):
    from run_di_barba_drop import run
    target=tmp_path/'run'
    summary=run(target,generations=3,pop_size=8,offsprings=8,flag_start=2,seed=3)
    assert summary['actual_model_calls']==24
    assert summary['saved_evaluations']==summary['total_dropped']==0
    assert summary['total_proposed']==24
    before=(target/'settings.json').read_bytes()
    with pytest.raises(FileExistsError):run(target,generations=3,pop_size=8,offsprings=8)
    assert before==(target/'settings.json').read_bytes()


def test_cell_reporting_optout_does_not_change_filter_evidence():
    data=pd.DataFrame({'X1':[0,.05,1],'X2':[0,.05,1],
                       'Obj1':[1,2,1],'Obj2':[1,1,1]})
    a,_=classify_dataframe(data,.1,.1,.1)
    b,_=classify_dataframe(data,.1,.1,.1,partition_width=None)
    pd.testing.assert_frame_equal(a[['flag','verified_violation','local_n','cluster_id']],
                                  b[['flag','verified_violation','local_n','cluster_id']])


def test_independent_drop_audit_uses_the_same_relative_floor():
    from di_barba_solenoid_benchmark import DiBarbaSolenoid
    from src.drop_metrics import exact_relative_sensitivity
    from src.local_variation import relative_pairwise_variation
    model=DiBarbaSolenoid()
    assert exact_relative_sensitivity(model,np.zeros(10))[1]==pytest.approx(1.)
    center=np.linspace(-.3,.3,10)
    trials=center+np.random.default_rng(13).uniform(-.1,.1,(100,10))
    observed=relative_pairwise_variation(model.evaluate(center),
        np.array([model.evaluate(x) for x in trials]),[.001,.001]).max(axis=0)
    assert np.all(exact_relative_sensitivity(model,center)>=observed-1e-9)


def test_comparison_uses_only_observed_states_and_marks_censoring(tmp_path):
    import json
    from plot_di_barba_drop_comparison import compare
    root=tmp_path/'study';(root/'reference').mkdir(parents=True)
    pd.DataFrame({'Obj1':[0.,1.],'Obj2':[1.,0.]}).to_csv(root/'reference/reference.csv',index=False)
    (root/'protocol.json').write_text(json.dumps(dict(target_hv=.15,seeds=[42],evaluation_checkpoints=[1500])))
    for mode in ['baseline','resample','drop']:
        d=root/'full'/f'{mode}_seed42';d.mkdir(parents=True)
        (d/'summary.json').write_text(json.dumps(dict(actual_model_calls=2000,run_seconds=1.,archive_processing_seconds=.1)))
        pd.DataFrame({'Obj1':[.2],'Obj2':[.3]}).to_csv(d/'pareto.csv',index=False)
        pd.DataFrame({'generation':[1,2],'evaluations':[1000,2000],
                      'hypervolume':[.1,.2 if mode!='drop' else .14],'igd_plus':[.1,.01]}).to_csv(d/'history.csv',index=False)
        pd.DataFrame({'exact_sensitive':[True],'R1_exact':[.2],'R2_exact':[.3]}).to_csv(d/'pareto_sensitivity.csv',index=False)
        if mode=='drop':
            pd.DataFrame({'generation':[1,2],'cumulative_dropped':[0,1],
                          'n_dropped':[0,1],'n_proposed':[10,10]}).to_csv(d/'process.csv',index=False)
    out=tmp_path/'plots';compare(root,out)
    assert len(list(out.glob('*.png')))==5 and len(list(out.glob('*.pdf')))==5
    checkpoints=pd.read_csv(out/'fixed_evaluation_checkpoints.csv')
    assert checkpoints.loc[checkpoints.allowed_evaluation_budget==1500,'actual_checkpoint_evaluations'].eq(1000).all()
    hits=pd.read_csv(out/'target_hv_attainment.csv').set_index('mode')
    assert not hits.loc['drop','reached']
    assert pd.isna(hits.loc['drop','first_attainment_evaluations'])
    assert hits.loc['baseline','first_attainment_evaluations']==2000
