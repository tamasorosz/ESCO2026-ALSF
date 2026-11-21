import numpy as np
from scipy.spatial import cKDTree
from src.local_variation import LocalVariationState,relative_pairwise_variation


def test_incremental_evidence_against_independent_full_pair_oracle():
    rng=np.random.default_rng(731)
    for dimension in [2,10]:
        x=rng.normal(0,.15,(70,dimension));x[10]=x[0]
        f=rng.uniform(0,2,(70,2));state=LocalVariationState(.1)
        for indices in [np.arange(12),np.arange(35),np.arange(70),np.arange(70),np.arange(70)[::-1],np.arange(5)]:
            points=x[indices];scores=f[indices]
            counts=state.update(points,scores)
            near=cKDTree(points/.1).query_ball_point(points/.1,r=1.+1e-12,p=np.inf)
            np.testing.assert_array_equal(counts,[len(ids) for ids in near])
            variation=np.array([relative_pairwise_variation(scores[i],scores[ids],[.001]*2).max(axis=0) for i,ids in enumerate(near)])
            np.testing.assert_allclose(state.max_observed_variation,variation)
            np.testing.assert_array_equal(state.verified_violation,(variation>.1).any(axis=1))


def test_gaussian_reference_matches_independent_scalar_minimization():
    from scipy.optimize import minimize_scalar
    from src.study_reference import test1_reference
    reference=test1_reference(1001)
    assert (np.diff(reference.Obj1)>=0).all()
    assert (np.diff(reference.Obj2)<0).all()
    for i in [0,100,210,230,250,270,290,500,999,1000]:
        row=reference.iloc[i];q=row.Obj1
        if q==0:continue
        bound=np.sqrt(2*q)
        result=minimize_scalar(lambda s:q-10*s+50+100*np.exp(-(q-5*s+12.5)/.01),
                               bounds=(-bound,bound),method='bounded',options={'xatol':1e-13})
        assert result.fun>=row.Obj2-1e-8
        assert abs(result.fun-row.Obj2)<1e-5
    np.testing.assert_allclose(reference.iloc[[0,-1]][['Obj1','Obj2']],[[0,50],[50,0]],atol=1e-10)
