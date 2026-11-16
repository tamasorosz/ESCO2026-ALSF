"""Independent analytical reduction and exhaustive box-stationarity enumeration.

All f2 interior stationary points have x1=x2=2.5+z and 0<z<2.5.
On an edge a free coordinate also has 0<z<2.5. Its derivative is
h(z)=5-2z-200/w*z*exp(-(k*z*z+d2)/w), k=1 (edge), 2 (diagonal).
h' has at most two positive roots: with u=k*z*z/w its root equation is
log(2u-1)-u=log(w/100)+d2/w. The left side increases to u=1.5,
then decreases strictly. Splitting there enumerates EVERY h' root, then
bracketing h on its monotone intervals enumerates EVERY stationary point.
This is analytical completeness with double-precision root solving, not
interval-arithmetic certification. Degenerate roots are included explicitly.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from benchmark_test1_modified_gaussian import evaluate, WIDTH, to_optimization
from src.local_variation import relative_pairwise_variation


def stationary(k=2, d2=0.):
    w=WIDTH; rhs=np.log(w/100)+d2/w
    critical=[]
    def logeq(u): return np.log(2*u-1)-u-rhs
    if rhs <= np.log(2)-1.5:
        if abs(logeq(1.5)) < 1e-13: critical=[np.sqrt(1.5*w/k)]
        else:
            upper=max(10., -rhs+10.)
            while logeq(upper)>0: upper*=2
            for a,b in [(np.nextafter(.5,1.),1.5),(1.5,upper)]:
                # Underflow close to u=.5 means this critical point is
                # indistinguishable from sqrt(w/(2k)) in float64.
                u=brentq(logeq,a,b,xtol=1e-14) if logeq(a)*logeq(b)<0 else .5
                critical.append(np.sqrt(u*w/k))
    def h(z): return 5-2*z-200/w*z*np.exp(-(k*z*z+d2)/w)
    split=sorted(set([0.,2.5]+[z for z in critical if 0<z<2.5]))
    roots=[]
    for z in split:
        if abs(h(z))<1e-11:roots.append(z)
    for a,b in zip(split[:-1],split[1:]):
        if h(a)*h(b)<0:roots.append(brentq(h,a,b,xtol=5e-15))
    return np.unique(roots)


def endpoint():
    t=np.r_[0.,5.,2.5+stationary()]
    return float(t[np.argmax(evaluate(np.column_stack((t,t)))[:,1])])


def reference(n=1001):
    """Adaptive knots; concentrate three quarters of resolution near the peak."""
    end=endpoint()
    t=np.unique(np.r_[np.linspace(0,2.2,n),np.linspace(2.2,2.4,n),
        np.linspace(2.4,2.48,n),np.linspace(2.48,end,n)])
    x=np.column_stack((t,t));f=evaluate(x);g=to_optimization(f)
    return pd.DataFrame(np.column_stack((x,f,g)),columns=['X1','X2','Obj1','Obj2','G1','G2'])


def box_extrema(x, tolerance=.1):
    """All corners, edge stationary points and feasible interior stationary points."""
    x=np.asarray(x,float);lo=np.maximum(-5.,x-tolerance);hi=np.minimum(10.,x+tolerance)
    points=[np.array([a,b]) for a in (lo[0],hi[0]) for b in (lo[1],hi[1])]
    for fixed in (0,1):
        free=1-fixed
        for value in (lo[fixed],hi[fixed]):
            for z in stationary(1,(value-2.5)**2):
                p=np.empty(2);p[fixed]=value;p[free]=2.5+z
                if lo[free]<=p[free]<=hi[free]:points.append(p)
    for z in stationary():
        p=np.full(2,2.5+z)
        if np.all(p>=lo) and np.all(p<=hi):points.append(p)
    vals=evaluate(points)
    low=vals.min(axis=0);high=vals.max(axis=0)
    # f1 convex separable: nearest-to-zero point supplies its exact minimum.
    low[0]=np.sum(np.clip(np.zeros(2),lo,hi)**2)
    return low,high


def sensitivity(x):
    center=evaluate(x);low,high=box_extrema(x)
    absolute=np.maximum(abs(low-center),abs(high-center))
    # Symmetric relative deviation can have an interior maximum at -f(x).
    candidates=np.vstack((low,high,np.clip(-center,low,high),
        np.clip(np.full(2,.001),low,high),np.clip(np.full(2,-.001),low,high)))
    relative=relative_pairwise_variation(center,candidates,[.001,.001]).max(axis=0)
    return absolute,relative
