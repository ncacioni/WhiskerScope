"""Whiskerscope — FastAPI REST API."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile

from whiskerscope.adapters.yolo_detector import YOLOv8Detector
from whiskerscope.application.event_counter import EventCounter
from whiskerscope.domain.models import Frame

app = FastAPI(title="Whiskerscope API", version="0.1.0")

detector = YOLOv8Detector()
counter = EventCounter()


@app.post("/detect")
async def detect(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    frame = Frame(data=img, timestamp=datetime.now(timezone.utc), width=w, height=h)
    detections = detector.detect_cats(frame)
    counter.update(len(detections))

    return {
        "cat_count": len(detections),
        "detections": [
            {
                "label": d.label,
                "confidence": round(d.bbox.confidence, 3),
                "bbox": {"x1": d.bbox.x1, "y1": d.bbox.y1, "x2": d.bbox.x2, "y2": d.bbox.y2},
            }
            for d in detections
        ],
    }


@app.get("/status")
async def status():
    return counter.stats()
