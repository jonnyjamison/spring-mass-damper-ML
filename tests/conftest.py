import os
import pytest
from fastapi.testclient import TestClient
from src.spring_mass_damper_ML.api.main import app
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="session")
def client():
    # ensure a model path exists for tests; default artifact is checked in
    os.environ.setdefault("MODEL_PATH", "model_registry/v1/model.pt")
    return TestClient(app)