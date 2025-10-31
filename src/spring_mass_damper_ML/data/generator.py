from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp


@dataclass
class GenConfig:
    t_start: float
    t_end: float
    num_points: int
    n_samples: int
    m_range: tuple[float, float]
    c_range: tuple[float, float]
    k_range: tuple[float, float]


def simulate(m: float, c: float, k: float, t: np.ndarray) -> np.ndarray:
    # Simple deterministic forcing (sum of sines) to avoid stochastic labels
    def forcing(tt: float) -> float:
        return np.sin(2 * np.pi * 1.0 * tt) + 0.3 * np.sin(2 * np.pi * 0.3 * tt)

    def dyn(_t, y):
        x, xdot = y
        xddot = (forcing(_t) - c * xdot - k * x) / m
        return [xdot, xddot]

    sol = solve_ivp(dyn, [t[0], t[-1]], [0.0, 0.0], t_eval=t, rtol=1e-7, atol=1e-9)
    return sol.y[0]


def generate_dataset(cfg: GenConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    t = np.linspace(cfg.t_start, cfg.t_end, cfg.num_points)

    X = np.zeros((cfg.n_samples, 3), dtype=np.float32)
    Y = np.zeros((cfg.n_samples, cfg.num_points), dtype=np.float32)

    for i in range(cfg.n_samples):
        m = rng.uniform(*cfg.m_range)
        c = rng.uniform(*cfg.c_range)
        k = rng.uniform(*cfg.k_range)
        X[i] = (m, c, k)
        Y[i] = simulate(m, c, k, t)
    return X, Y