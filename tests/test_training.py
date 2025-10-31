from __future__ import annotations
import tempfile
from pathlib import Path
from src.spring_mass_damper_ML.pipelines.train import train
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_training_runs():
    # Use default config; ensure it completes and writes artifact
    train("configs/default.yaml")
    assert Path("model_registry/v1/model.pt").exists()