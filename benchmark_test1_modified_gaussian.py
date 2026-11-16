"""Separate fixed-width physical min/max benchmark; historical models untouched."""
import numpy as np
LOWER = np.full(2, -5.)
UPPER = np.full(2, 10.)
WIDTH = .01
AMPLITUDE = 100.
CENTER = np.full(2, 2.5)


def evaluate(x):
    x = np.asarray(x, dtype=float)
    return np.stack((np.sum(x*x, axis=-1),
        -np.sum((x-5)**2, axis=-1) + AMPLITUDE*np.exp(-np.sum((x-CENTER)**2, axis=-1)/WIDTH)), axis=-1)


def to_optimization(f):
    return np.asarray(f, dtype=float) * np.array([1., -1.])


def to_physical(g):
    return to_optimization(g)
