from __future__ import annotations
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from .generator import GenConfig, generate_dataset


@dataclass
class DataConfig:
    t_start: float
    t_end: float
    num_points: int
    n_samples: int
    m_range: list[float]
    c_range: list[float]
    k_range: list[float]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)




def build_training_data(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    gen = GenConfig(
        t_start=cfg.t_start,
        t_end=cfg.t_end,
        num_points=cfg.num_points,
        n_samples=cfg.n_samples,
        m_range=tuple(cfg.m_range),
        c_range=tuple(cfg.c_range),
        k_range=tuple(cfg.k_range),
    )
    return generate_dataset(gen)