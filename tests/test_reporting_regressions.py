"""Regression checks for the two visible figure inconsistencies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.find_robust_clusters import classify_dataframe
from plot_three_figures import representative_indices, prepare_process


def dataframe(points, objectives):
    return pd.DataFrame(np.column_stack([points, objectives]),
                        columns=['X1','X2','Obj1','Obj2'])


def test_no_robust_medoid_from_mixed_dbscan_component():
    data = dataframe([[0,0],[0.08,0],[0.16,0],[0.24,0]],
                     [[1,1],[1,1],[1.02,1],[1.5,1]])
    out, stats = classify_dataframe(data,[.1,.1],.1,.1,
                                    min_samples=2,floors=(.01,.01))
    assert stats['n_clusters'] == 1
    assert stats['n_non_robust_clusters'] == 1
    assert stats['n_robust_clusters'] == 0
    assert (out.flag == 'robust').sum() == 2  # locally stable samples exist
    assert not out.is_medoid.any()  # but the component is not robust!
    assert len(representative_indices(out)) == 0


def test_wholly_robust_dbscan_component_has_one_medoid():
    data = dataframe([[0,0],[0.08,0],[1,1]],
                     [[1,1],[1.02,1.02],[2,2]])
    out, stats = classify_dataframe(data,[.1,.1],.1,.1,
                                    min_samples=2,floors=(.01,.01))
    assert stats['n_robust_clusters'] == 1
    assert int(out.is_medoid.sum()) == 1
    assert len(representative_indices(out)) == 1


def test_missing_regeneration_columns_not_silently_zero(tmp_path):
    source=tmp_path/'process.csv'
    pd.DataFrame({'generation':[50,51],
                  'n_robust_clusters':[1,0],
                  'n_non_robust_clusters':[0,1]}).to_csv(source,index=False)
    with pytest.raises(ValueError,match='Missing column'):
        prepare_process(source)


def test_reported_zeros_are_preserved_when_explicit(tmp_path):
    source=tmp_path/'process.csv'
    pd.DataFrame({'generation':[50,51],
                  'n_robust_clusters':[1,0],
                  'n_non_robust_clusters':[0,1],
                  'n_regenerated':[0,0],
                  'n_attempts':[0,0],
                  'n_repair_calls':[1,1],
                  'n_offspring_seen':[100,100],
                  'n_forbidden_candidates':[0,0],
                  'n_exclusion_centers':[4,10]}).to_csv(source,index=False)
    p=prepare_process(source)
    assert p.regenerated.sum() == 0
    assert p.n_offspring_seen.sum() == 200
    assert p.n_exclusion_centers.iloc[-1] == 10


def test_inconsistent_attempt_counts_raise(tmp_path):
    source=tmp_path/'process.csv'
    pd.DataFrame({'generation':[50],
                  'n_robust_clusters':[1],
                  'n_non_robust_clusters':[3],
                  'n_regenerated':[5],
                  'n_attempts':[2]}).to_csv(source,index=False)
    with pytest.raises(ValueError,match='inconsistent'):
        prepare_process(source)


def test_real_pymoo_repair_hook_counts_forbidden_candidates():
    """This check runs on a machine with pymoo installed; otherwise skipped."""
    pytest.importorskip('pymoo')
    from types import SimpleNamespace
    from src.preprocessing import OffspringRepair
    from src.offspring_policy import VerifiedViolationFilter
    repair=OffspringRepair('unused.csv', {}, 1.0, tolerances=[0.1,0.1],
                           rng=np.random.default_rng(42))
    repair.policy.update(np.array([[0.,0.]]), np.array([True]))
    problem=SimpleNamespace(xl=np.array([-5.,-5.]), xu=np.array([10.,10.]))
    before=np.array([[0.04,0.05], [3.,3.]])
    after=repair._do(problem,before)
    info=repair.pop_diagnostics()
    assert info['n_repair_calls']==1
    assert info['n_offspring_seen']==2
    assert info['n_forbidden_candidates']==1
    assert info['n_regenerated']==1
    assert info['n_attempts']>=1
    assert not repair.policy.forbidden(after[0])
    assert (after[1]==before[1]).all()
    assert repair.pop_diagnostics()['n_offspring_seen']==0
