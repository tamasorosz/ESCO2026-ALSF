from __future__ import annotations
import numpy as np
from benchmark_test1 import evaluate,gradient,sample_reference

def test_objective_minima():
    assert np.allclose(evaluate([0,0]), [0,50], atol=1e-10)
    assert np.allclose(evaluate([5,5]), [50,0], atol=1e-10)

def test_gaussian_peak():
    assert np.allclose(evaluate([2.5,2.5]), [12.5,112.5])

def test_vectorization():
    x=np.array([[0,0],[2.5,2.5],[5,5]])
    assert np.allclose(evaluate(x),np.array([evaluate(y) for y in x]))

def test_gradients():
    for x in ([1.,1.], [2.45,2.51], [4.,4.]):
        x=np.array(x)
        delta=1e-6
        fd=np.column_stack([(evaluate(x+np.eye(2)[i]*delta)-evaluate(x-np.eye(2)[i]*delta))/(2*delta) for i in range(2)])
        assert np.allclose(gradient(x),fd,rtol=1e-5,atol=1e-5)

def test_reference_nondominated():
    x,f=sample_reference(101)
    assert len(x)>2
    assert np.all(np.diff(f[:,0])>=-1e-12)
    assert np.all(np.diff(f[:,1])<-1e-12)

def test_legacy_equation_aliases():
    from src.equations import equation01, equation02
    for x in [[0, 0], [2.5, 2.5], [5, 5]]:
        assert np.allclose([equation01(*x),equation02(*x)],evaluate(x))
