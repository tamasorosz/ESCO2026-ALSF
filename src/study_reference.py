"""Independent Pareto envelope for unchanged Gaussian Test 1.

For q=x1^2+x2^2 and s=x1+x2, f2=q-10s+50+100exp(-(q-5s+12.5)/.01).
At fixed q in [0,50], feasible s<=sqrt(2q). The convex-in-s minimum is
s=min(sqrt(2q),(q+12.5-.01log(5000))/5). Both reconstructed coordinates are
feasible. The resulting envelope decreases to f2=0, so every sampled q is Pareto.
"""
import numpy as np
import pandas as pd
from benchmark_test1 import evaluate


def test1_reference(n=2001):
    q=np.linspace(0,50,n)
    total=np.minimum(np.sqrt(2*q),(q+12.5-.01*np.log(5000))/5)
    gap=np.sqrt(np.maximum(2*q-total**2,0))
    x=np.column_stack([(total+gap)/2,(total-gap)/2])
    return pd.DataFrame(np.column_stack([x,evaluate(x)]),columns=['X1','X2','Obj1','Obj2'])
