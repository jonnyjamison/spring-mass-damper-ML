from __future__ import annotations
import os
import logging
import torch
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from ..logging_conf import configure_logging
from ..models.surrogate import MLP
from .schemas import PredictRequest, PredictResponse

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(title="Spring-Mass-Damper Surrogate API")
configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants and globals
# ---------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model_registry/v1/model.pt")
NUM_POINTS = 1000  # must match training config
_model = None

# ---------------------------------------------------------------------
# Helper: lazy model loader
# ---------------------------------------------------------------------
def load_model():
    global _model
    if _model is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = MLP(input_dim=3, hidden_sizes=[64, 128], output_dim=NUM_POINTS)
        try:
            state = torch.load(MODEL_PATH, map_location="cpu")
        except FileNotFoundError as e:
            logger.error(f"Model not found at {MODEL_PATH}")
            raise RuntimeError(f"Model not found at {MODEL_PATH}") from e
        model.load_state_dict(state)
        model.eval()
        _model = model
    return _model

# ---------------------------------------------------------------------
# Prometheus instrumentation (must be BEFORE startup)
# ---------------------------------------------------------------------
Instrumentator().instrument(app).expose(app)

# ---------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------
@app.on_event("startup")
def _startup():
    logger.info("Application startup complete.")
    try:
        load_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.warning(f"Model preload failed: {e}")

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    try:
        load_model()
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "error": str(e)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = load_model()
    with torch.no_grad():
        x = torch.tensor([[req.m, req.c, req.k]], dtype=torch.float32)
        y = model(x).numpy().flatten().tolist()
    return PredictResponse(displacement=y, num_points=len(y))
