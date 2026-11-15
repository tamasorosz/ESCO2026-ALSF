"""Misi's Test 1: 2-D bi-objective problem from the ESCO / JCAM manuscript.

Both objectives are MINIMIZED, including the Gaussian-peak objective.
The positive bump in f2 is centered at (2.5, 2.5), amplitude 100, width
parameter 0.01 in the exponent's denominator (not standard-deviation sigma).
"""
from __future__ import annotations
import numpy as np

LOWER = np.array([-5.0, -5.0])
UPPER = np.array([10.0, 10.0])


def evaluate(x):
    """Return array of shape (...,2) for x with last dimension two."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != 2:
        raise ValueError('Expected decision-vector shape (..., 2)')
    a, b = x[..., 0], x[..., 1]
    f1 = a*a + b*b
    f2 = ((a-5.0)**2 + (b-5.0)**2
          + 100.0*np.exp(-((a-2.5)**2+(b-2.5)**2)/0.01))
    return np.stack((f1, f2), axis=-1)


def sample_reference(n_grid=501):
    """Approximate nondominated frontier of a regular 2-D decision grid.

    NOT an exact Pareto reference. Suitable for visual diagnostic only.
    """
    xx = np.linspace(-5, 10, n_grid)
    yy = np.linspace(-5, 10, n_grid)
    u, v = np.meshgrid(xx, yy)
    points = np.stack((u.ravel(), v.ravel()), axis=-1)
    scores = evaluate(points)
    # 2 objectives, both minimizing. For equal f1 keep smallest f2 first.
    order = np.lexsort((scores[:, 1], scores[:, 0]))
    best_f2 = np.inf
    indices = []
    for i in order:
        if scores[i, 1] < best_f2 - 1e-12:
            indices.append(i)
            best_f2 = scores[i, 1]
    return points[indices], scores[indices]


def gradient(x):
    x = np.asarray(x, dtype=float)
    a,b = x[...,0],x[...,1]
    e = np.exp(-((a-2.5)**2+(b-2.5)**2)/0.01)
    g1 = np.stack([2*a,2*b],axis=-1)
    g2 = np.stack([2*(a-5)-20000*(a-2.5)*e,
                   2*(b-5)-20000*(b-2.5)*e],axis=-1)
    return np.stack([g1,g2],axis=-2)
