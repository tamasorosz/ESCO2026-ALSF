"""Original JCAM Test 1: minimize f1, maximize f2.

The unchanged raw model remains benchmark_test1.evaluate. Historical min/min
runners/results are preserved; this adapter alone maps to pymoo minimization.
"""
import numpy as np
from benchmark_test1 import evaluate,gradient,LOWER,UPPER


def to_optimization(physical):
    f=np.asarray(physical,dtype=float)
    if f.ndim<1 or f.shape[-1]!=2 or not np.isfinite(f).all():
        raise ValueError('Expected finite objective vectors [...,2]')
    return f*np.array([1.,-1.])


def to_physical(optimization):
    return to_optimization(optimization)


def evaluate_optimization(x):
    return to_optimization(evaluate(x))


def nondominated_indices(physical):
    f=np.asarray(physical,float)
    to_optimization(f) # validation
    order=np.lexsort((-f[:,1],f[:,0]));best=-np.inf;keep=[]
    for i in order:
        if f[i,1]>best:
            keep.append(int(i));best=f[i,1]
    return np.asarray(keep,dtype=int)
