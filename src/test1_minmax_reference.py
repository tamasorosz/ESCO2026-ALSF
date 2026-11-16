"""Validated fixed-f1 upper envelope over the complete feasible Test 1 domain.

For q<=50, f2 is convex in s=x1+x2, so its maximum occurs at one of the
feasible diagonals s=+-sqrt(2q). q>50 is dominated by (-5,-5): it requires a
coordinate >5, giving quadratic background <=125 and Gaussian <=100 exp(-1250).
The peak-efficient segment is narrow and NOT assumed to include its center.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from benchmark_test1 import evaluate
from benchmark_test1_minmax import nondominated_indices


def branches(q):
    q=np.asarray(q,float);a=np.sqrt(q/2)
    positive=np.stack([a,a],axis=-1);negative=-positive
    return evaluate(negative)[...,1],evaluate(positive)[...,1]


def peak_features():
    def diff(q):
        a,b=branches(q);return float(b-a)
    left=brentq(diff,12.4,12.499,xtol=1e-13)
    def derivative(q):
        distance=(np.sqrt(q)-np.sqrt(12.5))**2
        return 1-10/np.sqrt(2*q)-10000*np.exp(-distance/.01)*(1-np.sqrt(12.5/q))
    maximum=brentq(derivative,left,12.5,xtol=1e-13)
    maximum_value=float(branches(maximum)[1])
    catch=brentq(lambda q:float(branches(q)[0])-maximum_value,12.5,12.6,xtol=1e-13)
    return dict(peak_branch_start_q=left,peak_local_maximum_q=maximum,
                peak_maximum_f2=maximum_value,negative_branch_catch_q=catch,
                peak_center_f1=12.5,peak_center_f2=112.5)


def reference_front(n_global=5001,n_peak=1001):
    features=peak_features()
    q=np.unique(np.r_[np.linspace(0,50,n_global),
        np.linspace(features['peak_branch_start_q']-.0001,features['negative_branch_catch_q']+.0001,n_peak),
        features['peak_branch_start_q'],features['peak_local_maximum_q'],features['negative_branch_catch_q'],12.5])
    a,b=branches(q);positive=b>a
    x=np.sqrt(q/2)[:,None]*np.ones((1,2))*np.where(positive[:,None],1,-1)
    f=evaluate(x)
    out=pd.DataFrame(np.column_stack([x,f]),columns=['X1','X2','Obj1','Obj2'])
    out['peak_branch']=positive
    return out.iloc[nondominated_indices(f)].reset_index(drop=True),features
