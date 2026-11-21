"""Symmetric L-infinity distance to reference boxes, not IGD+.

All coordinates/tolerances must use the same units. Rectangles may be asymmetric.
Both reference and solution axes are chunked; the full [R,P,D] tensor is never
allocated. Empty obtained sets have infinite distance and zero coverage.
"""
from __future__ import annotations
import numpy as np


def _points(value,name):
    result=np.asarray(value,dtype=float)
    if result.ndim!=2 or result.shape[1]==0:
        raise ValueError(f'{name} must have shape [n,d] with d>=1')
    if not np.isfinite(result).all():raise ValueError(f'{name} must be finite')
    return result


def nearest_rectangle_distances(solutions,lower,upper,*,chunk_size=128):
    """Per-reference min_p max_j max(lower_j-p_j,0,p_j-upper_j).

    Nonempty references are required. Empty solutions must retain shape (0,d).
    Boundaries count as inside, using exact <= with no undisclosed epsilon.
    """
    lo=_points(lower,'lower');hi=_points(upper,'upper')
    p=np.asarray(solutions,dtype=float)
    if p.ndim==1 and p.size==0:p=p.reshape(0,lo.shape[1])
    p=_points(p,'solutions')
    if lo.shape!=hi.shape or p.shape[1]!=lo.shape[1]:
        raise ValueError('Dimension/shape mismatch')
    if not len(lo):raise ValueError('Reference rectangles must be nonempty')
    if np.any(lo>hi):raise ValueError('Lower bounds exceed upper bounds')
    if isinstance(chunk_size,bool) or not isinstance(chunk_size,(int,np.integer)) or chunk_size<1:
        raise ValueError('chunk_size must be a positive integer')
    output=np.full(len(lo),np.inf)
    for i in range(0,len(lo),chunk_size):
        a=lo[i:i+chunk_size];b=hi[i:i+chunk_size]
        best=np.full(len(a),np.inf)
        for j in range(0,len(p),chunk_size):
            block=p[j:j+chunk_size]
            excess=np.maximum(np.maximum(a[:,None,:]-block[None,:,:],
                                          block[None,:,:]-b[:,None,:]),0.)
            best=np.minimum(best,excess.max(axis=2).min(axis=1))
        output[i:i+len(a)]=best
    return output


def nearest_box_distances(solutions,reference,half_widths,*,chunk_size=128):
    """Per-reference box distance with scalar or length-d nonnegative half-widths.

    If an input tolerance is a full width, callers must divide it by TWO first.
    At zero half-widths this equals symmetric L-infinity IGD, not Euclidean IGD+.
    """
    r=_points(reference,'reference')
    h=np.asarray(half_widths,dtype=float)
    if h.ndim==0:h=np.full(r.shape[1],float(h))
    if h.shape!=(r.shape[1],) or not np.isfinite(h).all() or np.any(h<0):
        raise ValueError('half_widths must be finite nonnegative scalar or length-d vector')
    return nearest_rectangle_distances(solutions,r-h,r+h,chunk_size=chunk_size)


def summarize_distances(distances):
    values=np.asarray(distances,dtype=float)
    if values.ndim!=1 or not len(values) or np.isnan(values).any() or np.any(values<0):
        raise ValueError('Expected nonempty nonnegative distances without NaN')
    # All-inf is the explicit convention for no obtained solutions; avoid
    # percentile interpolation warnings when the input contains infinity.
    p95=np.inf if np.isinf(values).all() else float(np.quantile(values,.95))
    return dict(box_igd=float(values.mean()),tolerance_coverage=float(np.mean(values==0.)),
                p95_box_distance=float(p95),max_box_distance=float(values.max()))


def box_igd(solutions,reference,half_widths,*,chunk_size=128):
    """Mean nearest-reference-box distance, with uniform reference-row weights."""
    return summarize_distances(nearest_box_distances(solutions,reference,half_widths,
                                                    chunk_size=chunk_size))


def rectangle_igd(solutions,lower,upper,*,chunk_size=128):
    return summarize_distances(nearest_rectangle_distances(solutions,lower,upper,
                                                          chunk_size=chunk_size))
