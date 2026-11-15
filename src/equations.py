"""Compatibility aliases for the old ESCO two-variable Test 1 runner."""
from __future__ import annotations
import numpy as np


def equation01(x1, x2):
    return x1*x1 + x2*x2


def equation02(x1, x2):
    return ((x1-5.0)**2 + (x2-5.0)**2
            + 100.0*np.exp(-((x1-2.5)**2 + (x2-2.5)**2)/0.01))
