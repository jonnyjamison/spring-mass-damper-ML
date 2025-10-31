# Optional baseline: direct ODE solve at inference (for validation / comparison)
# Not used by the API in production path, but helpful for unit tests.
from __future__ import annotations
import numpy as np
from ..data.generator import simulate


def baseline_displacement(m: float, c: float, k: float, t_end: float = 10.0, num_points: int = 1000):
    t = np.linspace(0.0, t_end, num_points)
    return simulate(m, c, k, t)