"""Whiskerscope — FastAPI REST API."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from whiskerscope.adapters.api_models import (
    DetectResponse,
    DetectionItem,
    BoundingBoxResponse,
    ErrorResponse,
    HealthResponse,
    StatsResponse,
)
from whiskerscope.adapters.sqlite_store import SQLiteStore
from whiskerscope.adapters.yolo_detector import YOLOv8Detector
from whiskerscope.application.event_counter import EventCounter
from whiskerscope.config import WhiskerscopeConfig
from whiskerscope.domain.models import Frame
from whiskerscope.logging_setup import correlation_id, setup_logging

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    config = WhiskerscopeConfig.from_env()
    setup_logging(config)
    app.state.detector = YOLOv8Detector(config.model_name, config.confidence)
    app.state.store = SQLiteStore(config.db_path)
    app.state.counter = EventCounter(store=app.state.store, session_id=config.session_id)
    app.state.config = config
    _start_time = time.monotonic()
    logger.info("API started, model=%s", config.model_name)
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Whiskerscope API",
    version="0.1.0",
    description="Real-time cat detection API powered by YOLOv8",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    token = correlation_id.set(req_id)
    t0 = time.monotonic()
    try:
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{(time.monotonic() - t0) * 1000:.1f}"
        return response
    finally:
        correlation_id.reset(token)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = correlation_id.get()
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(code="INTERNAL_ERROR", message="An unexpected error occurred", request_id=req_id).model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=hasattr(app.state, "detector"),
        uptime_secs=round(time.monotonic() - _start_time, 1),
        version="0.1.0",
    )


@app.post("/v1/detect", response_model=DetectResponse, tags=["Detection"], summary="Detect cats in an image")
async def detect(file: UploadFile):
    req_id = correlation_id.get()
    t0 = time.monotonic()

    if file.content_type and file.content_type not in ALLOWED_TYPES:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(code="INVALID_FILE_TYPE", message=f"Expected image, got {file.content_type}", request_id=req_id).model_dump(),
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(code="FILE_TOO_LARGE", message="File exceeds 10 MB limit", request_id=req_id).model_dump(),
        )

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(code="INVALID_IMAGE", message="Could not decode image", request_id=req_id).model_dump(),
        )

    h, w = img.shape[:2]
    frame = Frame(data=img, timestamp=datetime.now(timezone.utc), width=w, height=h)
    detections = app.state.detector.detect_cats(frame)

    max_conf = max((d.bbox.confidence for d in detections), default=0.0)
    app.state.counter.update(len(detections), max_confidence=max_conf)

    processing_ms = (time.monotonic() - t0) * 1000
    return DetectResponse(
        cat_count=len(detections),
        detections=[
            DetectionItem(
                label=d.label,
                confidence=round(d.bbox.confidence, 3),
                bbox=BoundingBoxResponse(x1=d.bbox.x1, y1=d.bbox.y1, x2=d.bbox.x2, y2=d.bbox.y2),
            )
            for d in detections
        ],
        request_id=req_id,
        processing_time_ms=round(processing_ms, 1),
    )


@app.get("/v1/stats", response_model=StatsResponse, tags=["Detection"], summary="Session statistics")
async def stats():
    counter: EventCounter = app.state.counter
    return StatsResponse(
        **counter.stats(),
        daily_summary=counter.daily_summary(),
    )


@app.websocket("/v1/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    from whiskerscope.adapters.opencv_camera import OpenCVCamera
    config: WhiskerscopeConfig = app.state.config
    camera = OpenCVCamera(config.camera_index)

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                await websocket.send_json({"error": "Camera frame unavailable"})
                break

            detections = app.state.detector.detect_cats(frame)
            await websocket.send_json({
                "cat_count": len(detections),
                "detections": [
                    {"label": d.label, "confidence": round(d.bbox.confidence, 3),
                     "bbox": {"x1": d.bbox.x1, "y1": d.bbox.y1, "x2": d.bbox.x2, "y2": d.bbox.y2}}
                    for d in detections
                ],
                "timestamp": frame.timestamp.isoformat(),
            })

            try:
                msg = await websocket.receive_text()
                if msg == '{"command":"stop"}':
                    break
            except WebSocketDisconnect:
                break
    finally:
        camera.release()
        logger.info("WebSocket client disconnected")
