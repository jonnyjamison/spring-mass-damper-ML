from __future__ import annotations
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    m: float = Field(..., gt=0, description="Mass")
    c: float = Field(..., ge=0, description="Damping coefficient")
    k: float = Field(..., gt=0, description="Stiffness")


class PredictResponse(BaseModel):
    displacement: list[float]
    num_points: int