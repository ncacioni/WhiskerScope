from __future__ import annotations

from pydantic import BaseModel


class BoundingBoxResponse(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    model_config = {"json_schema_extra": {"examples": [{"x1": 120.5, "y1": 80.2, "x2": 340.1, "y2": 290.7}]}}


class DetectionItem(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBoxResponse


class DetectResponse(BaseModel):
    cat_count: int
    detections: list[DetectionItem]
    request_id: str
    processing_time_ms: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cat_count": 1,
                    "detections": [{"label": "cat", "confidence": 0.92, "bbox": {"x1": 120, "y1": 80, "x2": 340, "y2": 290}}],
                    "request_id": "req_abc123",
                    "processing_time_ms": 45.2,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_secs: float
    version: str


class StatsResponse(BaseModel):
    total_detections: int
    max_simultaneous: int
    session_duration_secs: float
    daily_summary: dict


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: str | None = None
    request_id: str
