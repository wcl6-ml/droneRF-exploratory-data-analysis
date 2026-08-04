"""
schemas.py
==========
Defines the HTTP contract: what a request body must look like, what a
response body will look like. 

"""

from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    raw_h: List[float] = Field(..., description="Flat interleaved I/Q samples, H-band")
    raw_l: List[float] = Field(..., description="Flat interleaved I/Q samples, L-band")


class PredictResponse(BaseModel):
    predicted_class: str
    class_index: int
    confidence: float
    inference_time_ms: float
    model_version: str


class ModelInfoResponse(BaseModel):
    classes: List[str]
    model_version: str
    segment_length: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool